/// bedrockformation
/// GUI layer (iced) for the Bedrock Formation Finder.
///
/// All search/computation logic lives in `crate::core`; this module holds
/// only the application state, message handling, and view rendering.

use std::sync::{Arc, atomic::{AtomicBool, AtomicI64, Ordering}};

use iced::{
    Application, Command, Element, Event, Length, Subscription, Theme, Alignment,
    executor, theme, time,
    keyboard::{self, Key},
    event,
    widget::{button, checkbox, container, horizontal_rule, mouse_area, radio, row, scrollable, text, text_input, Column, Row, Space},
};

use crate::core::{
    BedrockType, Block,
    compute_probability, prob_to_threshold,
    generate_rotations, area_label_from_l, run_search,
};
use crate::gpu::GpuContext;

// GUI - types

/// State of one cell in the constraint grid.
/// Cycles through Unknown -> NonBedrock -> Bedrock -> Unknown on each click.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
enum CellState { #[default] Unknown, NonBedrock, Bedrock }

impl CellState {
    fn next(self) -> Self {
        match self {
            CellState::Unknown    => CellState::NonBedrock,
            CellState::NonBedrock => CellState::Bedrock,
            CellState::Bedrock    => CellState::Unknown,
        }
    }
    fn prev(self) -> Self {
        match self {
            CellState::Unknown    => CellState::Bedrock,
            CellState::NonBedrock => CellState::Unknown,
            CellState::Bedrock    => CellState::NonBedrock,
        }
    }
}

/// The four Y values containing probabilistic bedrock for each layer type,
/// ordered with the most-air end first (-60 ... -63 for floor).
/// Y=-64 (always solid) and Y=-59 (always air) are excluded as redundant.
fn y_values(bt: BedrockType) -> [i32; 4] {
    match bt {
        BedrockType::Floor => [-60, -61, -62, -63],
        BedrockType::Roof  => [124, 125, 126, 127],
    }
}

/// Allocate a fresh 4-layer * rows * cols grid, all Unknown.
fn make_grid(rows: usize, cols: usize) -> Vec<Vec<Vec<CellState>>> {
    vec![vec![vec![CellState::Unknown; cols]; rows]; 4]
}

// Grid rotation helpers

/// Rotate all Y-layers 90 degrees clockwise.
/// col -> X, row -> Z, so CW: new_col = rows-1-row, new_row = col.
/// Resulting dimensions swap: new_rows = old_cols, new_cols = old_rows.
fn rotate_grid_cw(
    cells: &[Vec<Vec<CellState>>],
    rows: usize,
    cols: usize,
) -> (Vec<Vec<Vec<CellState>>>, usize, usize) {
    let new_rows = cols;
    let new_cols = rows;
    let new_cells = cells
        .iter()
        .map(|layer| {
            let mut new_layer = vec![vec![CellState::Unknown; new_cols]; new_rows];
            for r in 0..rows {
                for c in 0..cols {
                    // CW: (r, c) -> (c, rows-1-r)
                    new_layer[c][rows - 1 - r] = layer[r][c];
                }
            }
            new_layer
        })
        .collect();
    (new_cells, new_rows, new_cols)
}

/// Rotate all Y-layers 90 degrees counter-clockwise.
/// CCW: (r, c) -> (cols-1-c, r).
/// Resulting dimensions swap: new_rows = old_cols, new_cols = old_rows.
fn rotate_grid_ccw(
    cells: &[Vec<Vec<CellState>>],
    rows: usize,
    cols: usize,
) -> (Vec<Vec<Vec<CellState>>>, usize, usize) {
    let new_rows = cols;
    let new_cols = rows;
    let new_cells = cells
        .iter()
        .map(|layer| {
            let mut new_layer = vec![vec![CellState::Unknown; new_cols]; new_rows];
            for r in 0..rows {
                for c in 0..cols {
                    // CCW: (r, c) -> (cols-1-c, r)
                    new_layer[cols - 1 - c][r] = layer[r][c];
                }
            }
            new_layer
        })
        .collect();
    (new_cells, new_rows, new_cols)
}

#[derive(Debug, Clone, PartialEq)]
enum SearchStatus {
    Idle,
    /// Actively searching; carries a human-readable area label like "10k x 10k".
    Searching(String),
    /// Search was cancelled; carries elapsed seconds.
    Cancelled(f64),
    /// Match found; carries coordinates and elapsed seconds.
    Found(i32, i32, f64),
    Error(String),
}

pub struct App {
    seed:          String,
    center_x:      String,
    center_z:      String,
    bedrock_type:  BedrockType,
    // Grid dimensions (1-16 each)
    grid_cols:     usize,
    grid_rows:     usize,
    grid_cols_str: String,
    grid_rows_str: String,
    // Which Y-layer tab is active
    grid_y_idx:    usize,
    // Top-left corner offset (relative block coords)
    grid_offset_x: String,
    grid_offset_z: String,
    // [y_layer 0..4][row 0..grid_rows][col 0..grid_cols]
    grid_cells:          Vec<Vec<Vec<CellState>>>,
    /// When true the search tests all 4 rotations of the pattern at every
    /// candidate position, so the result is found regardless of which
    /// compass direction the user was facing when they captured the pattern.
    search_all_rotations: bool,
    status:        SearchStatus,
    cancel_flag:   Option<Arc<AtomicBool>>,
    /// Wall-clock instant when the current search started (None when idle).
    search_start:  Option<std::time::Instant>,
    /// Shared atomic updated by the search thread with the latest spiral index.
    /// None when not searching.
    progress_pos:  Option<Arc<AtomicI64>>,
    /// UI zoom level: 1.0 = default, range 0.5-2.0 in steps of 0.1.
    ui_scale:      f32,
    /// Cached spiral shell number `L` for the current search's progress
    /// label, updated incrementally in `Message::SearchProgress` so the
    /// per-callback handler doesn't need to call `sqrt` every time.
    area_label_l:           i64,
    /// Spiral index at which `area_label_l` next needs to increment
    /// (= 4L^2+4L+1 for the current `area_label_l`).
    area_label_next_thresh: i64,
    /// GPU compute context. Initialisation is deferred until the user first
    /// enables GPU search, since `GpuContext::new()` probes the graphics
    /// driver and may spin up background threads that compete with Rayon.
    /// See `Message::ToggleGpu` and `Message::GpuInitialized`.
    gpu_init: GpuInitState,
    /// Whether the GPU search path is currently enabled by the user.
    use_gpu:  bool,
    /// Set while a GPU probe triggered by enabling the checkbox is in
    /// flight, so that `Message::GpuInitialized` knows whether to flip
    /// `use_gpu` on once the probe completes.
    pending_gpu_enable: bool,
}

/// Lazy GPU initialisation state. The GPU adapter/device is only probed the
/// first time the user enables the "Use GPU" checkbox.
#[derive(Debug, Default)]
enum GpuInitState {
    /// No probe has been attempted yet.
    #[default]
    NotProbed,
    /// A probe is currently running (`GpuContext::new()` in flight).
    Probing,
    /// A suitable GPU adapter was found and initialised.
    Ready(Arc<GpuContext>),
    /// GPU was probed and found, but the user has disabled it. The device is
    /// dropped so its driver threads do not compete with Rayon. Re-enabling
    /// GPU search will re-create the context.
    Capable,
    /// No suitable GPU adapter was found; don't probe again.
    Unavailable,
}

impl Default for App {
    fn default() -> Self {
        let cols = 8usize;
        let rows = 8usize;
        Self {
            seed:          String::new(),
            center_x:      "0".into(),
            center_z:      "0".into(),
            bedrock_type:  BedrockType::Floor,
            grid_cols:     cols,
            grid_rows:     rows,
            grid_cols_str: cols.to_string(),
            grid_rows_str: rows.to_string(),
            grid_y_idx:    0,
            grid_offset_x: "0".into(),
            grid_offset_z: "0".into(),
            grid_cells:          make_grid(rows, cols),
            search_all_rotations: false,
            status:        SearchStatus::Idle,
            cancel_flag:   None,
            search_start:  None,
            progress_pos:  None,
            ui_scale:      1.0,
            area_label_l:           0,
            area_label_next_thresh: 1,
            gpu_init: GpuInitState::NotProbed,
            use_gpu:  false,
            pending_gpu_enable: false,
        }
    }
}

#[derive(Debug, Clone)]
pub enum Message {
    SeedChanged(String),
    CenterXChanged(String),
    CenterZChanged(String),
    TypeChanged(BedrockType),
    GridColsChanged(String),
    GridRowsChanged(String),
    GridYChanged(usize),
    GridOffsetXChanged(String),
    GridOffsetZChanged(String),
    /// Cycle the state of cell (row, col) in the active Y-layer.
    GridCellClicked(usize, usize),
    /// Cycle the state of cell (row, col) in reverse (right-click).
    GridCellRightClicked(usize, usize),
    /// Rotate all Y-layers 90 degrees clockwise (X->Z, Z->-X).
    RotateCW,
    /// Rotate all Y-layers 90 degrees counter-clockwise (X->-Z, Z->X).
    RotateCCW,
    /// Toggle whether the search tries all 4 rotations of the pattern.
    ToggleAllRotations(bool),
    /// Toggle whether the GPU compute path is used for the coarse search.
    ToggleGpu(bool),
    /// Result of an async GPU probe triggered by first enabling `ToggleGpu`.
    /// `None` means no compatible GPU adapter/device was found.
    GpuInitialized(Option<Arc<GpuContext>>),
    Search,
    Cancel,
    /// Fired periodically during a search with the farthest spiral index checked so far.
    SearchProgress(i64),
    SearchDone(Result<Option<(i32, i32)>, String>),
    ZoomIn,
    ZoomOut,
    /// Set every Unknown cell in the focused Y-layer to NonBedrock.
    FillUnknownNonBedrockLayer,
    /// Clear all cells across all Y-layers back to Unknown.
    ClearGrid,
}

// GUI - Application impl

impl Application for App {
    type Message = Message;
    type Theme   = Theme;
    type Executor = executor::Default;
    type Flags   = ();

    fn new(_flags: ()) -> (Self, Command<Message>) {
        // GPU initialisation is deferred until the user enables GPU search
        // (see `Message::ToggleGpu`) so that CPU-only runs never pay the
        // cost - or risk the driver-thread contention - of probing for a
        // GPU adapter.
        (App::default(), Command::none())
    }

    fn title(&self) -> String { String::from("Bedrock Formation Finder") }

    fn theme(&self) -> Theme { Theme::GruvboxDark }

    fn subscription(&self) -> Subscription<Message> {
        let keyboard_sub = event::listen_with(|event, _| {
            if let Event::Keyboard(keyboard::Event::KeyPressed {
                key,
                modifiers,
                ..
            }) = event
                && modifiers.control()
                && let Key::Character(c) = &key
            {
                return match c.as_str() {
                    "+" | "=" => Some(Message::ZoomIn),
                    "-"       => Some(Message::ZoomOut),
                    _         => None,
                };
            }
            None
        });

        if let Some(progress_pos) = self.progress_pos.clone() {
            // While searching, tick every 150 ms and read the shared atomic.
            let tick_sub = time::every(std::time::Duration::from_millis(150))
                .map(move |_| {
                    Message::SearchProgress(progress_pos.load(Ordering::Relaxed))
                });
            Subscription::batch([keyboard_sub, tick_sub])
        } else {
            keyboard_sub
        }
    }

    fn update(&mut self, message: Message) -> Command<Message> {
        match message {
            Message::SeedChanged(s)    => { self.seed     = s; Command::none() }
            Message::CenterXChanged(s) => { self.center_x = s; Command::none() }
            Message::CenterZChanged(s) => { self.center_z = s; Command::none() }
            Message::TypeChanged(t) => {
                // Y values change between floor/roof, so reset the grid.
                self.bedrock_type = t;
                self.grid_cells   = make_grid(self.grid_rows, self.grid_cols);
                self.grid_y_idx   = 0;
                Command::none()
            }

            Message::GridColsChanged(s) => {
                self.grid_cols_str = s.clone();
                if let Ok(n) = s.parse::<usize>() {
                    let n = n.clamp(1, 16);
                    for layer in &mut self.grid_cells {
                        for row in &mut *layer {
                            row.resize(n, CellState::Unknown);
                        }
                    }
                    self.grid_cols = n;
                }
                Command::none()
            }
            Message::GridRowsChanged(s) => {
                self.grid_rows_str = s.clone();
                if let Ok(n) = s.parse::<usize>() {
                    let n = n.clamp(1, 16);
                    for layer in &mut self.grid_cells {
                        layer.resize(n, vec![CellState::Unknown; self.grid_cols]);
                    }
                    self.grid_rows = n;
                }
                Command::none()
            }
            Message::GridYChanged(idx)     => { self.grid_y_idx    = idx; Command::none() }
            Message::GridOffsetXChanged(s) => { self.grid_offset_x = s;   Command::none() }
            Message::GridOffsetZChanged(s) => { self.grid_offset_z = s;   Command::none() }
            Message::GridCellClicked(r, c) => {
                self.grid_cells[self.grid_y_idx][r][c] =
                    self.grid_cells[self.grid_y_idx][r][c].next();
                Command::none()
            }
            Message::GridCellRightClicked(r, c) => {
                self.grid_cells[self.grid_y_idx][r][c] =
                    self.grid_cells[self.grid_y_idx][r][c].prev();
                Command::none()
            }

            Message::RotateCW => {
                let (new_cells, new_rows, new_cols) =
                    rotate_grid_cw(&self.grid_cells, self.grid_rows, self.grid_cols);
                self.grid_cells    = new_cells;
                self.grid_rows     = new_rows;
                self.grid_cols     = new_cols;
                self.grid_rows_str = new_rows.to_string();
                self.grid_cols_str = new_cols.to_string();
                // Keep y-index in bounds (it always stays valid since we never
                // change the number of Y-layers, just rows/cols).
                Command::none()
            }

            Message::RotateCCW => {
                let (new_cells, new_rows, new_cols) =
                    rotate_grid_ccw(&self.grid_cells, self.grid_rows, self.grid_cols);
                self.grid_cells    = new_cells;
                self.grid_rows     = new_rows;
                self.grid_cols     = new_cols;
                self.grid_rows_str = new_rows.to_string();
                self.grid_cols_str = new_cols.to_string();
                Command::none()
            }

            Message::ToggleAllRotations(v) => {
                self.search_all_rotations = v;
                Command::none()
            }

            Message::ToggleGpu(v) => {
                if !v {
                    self.use_gpu = false;
                    // If the GPU context is alive, drop it now so that the
                    // wgpu driver background threads are torn down and stop
                    // competing with Rayon workers for CPU cores. Transition
                    // to `Capable` so we know we can re-create the context
                    // without probing again if the user re-enables GPU search.
                    if matches!(self.gpu_init, GpuInitState::Ready(_)) {
                        self.gpu_init = GpuInitState::Capable;
                    }
                    return Command::none();
                }
                match &self.gpu_init {
                    GpuInitState::Ready(_) => {
                        self.use_gpu = true;
                        Command::none()
                    }
                    GpuInitState::Unavailable => {
                        // Already probed once and found nothing; don't probe again.
                        self.use_gpu = false;
                        Command::none()
                    }
                    GpuInitState::Probing => Command::none(),
                    // GPU was available before but the context was released
                    // when the user disabled it. Re-create it now.
                    GpuInitState::Capable |
                    GpuInitState::NotProbed => {
                        // First enable (or re-enable after a prior disable):
                        // probe for a GPU adapter asynchronously so the UI
                        // does not block. `use_gpu` is set once the probe
                        // completes successfully.
                        self.gpu_init = GpuInitState::Probing;
                        self.pending_gpu_enable = true;
                        Command::perform(
                            async { GpuContext::new().await.map(Arc::new) },
                            Message::GpuInitialized,
                        )
                    }
                }
            }

            Message::GpuInitialized(ctx) => {
                match ctx {
                    Some(ctx) => {
                        self.gpu_init = GpuInitState::Ready(ctx);
                        if self.pending_gpu_enable { self.use_gpu = true; }
                    }
                    None => {
                        self.gpu_init = GpuInitState::Unavailable;
                        self.use_gpu = false;
                        if self.pending_gpu_enable {
                            self.status = SearchStatus::Error(
                                "No compatible GPU adapter was found.".into());
                        }
                    }
                }
                self.pending_gpu_enable = false;
                Command::none()
            }

            Message::Search => {
                let seed = match self.seed.parse::<i64>() {
                    Ok(s)  => s,
                    Err(_) => { self.status = SearchStatus::Error("Seed must be a 64-bit integer".into()); return Command::none(); }
                };
                let start_x = match self.center_x.parse::<i32>() {
                    Ok(v)  => v,
                    Err(_) => { self.status = SearchStatus::Error("Invalid center X".into()); return Command::none(); }
                };
                let start_z = match self.center_z.parse::<i32>() {
                    Ok(v)  => v,
                    Err(_) => { self.status = SearchStatus::Error("Invalid center Z".into()); return Command::none(); }
                };
                let offset_x = self.grid_offset_x.parse::<i32>().unwrap_or(0);
                let offset_z = self.grid_offset_z.parse::<i32>().unwrap_or(0);
                let bt  = self.bedrock_type;
                let ys  = y_values(bt);
                let mut blocks_vec: Vec<Block> = Vec::new();
                for (y_idx, &y) in ys.iter().enumerate() {
                    for row in 0..self.grid_rows {
                        for col in 0..self.grid_cols {
                            let state = self.grid_cells[y_idx][row][col];
                            if state == CellState::Unknown { continue; }
                            let prob = compute_probability(y, bt);
                            blocks_vec.push(Block {
                                x: offset_x + col as i32,
                                y,
                                z: offset_z + row as i32,
                                should_be_bedrock: state == CellState::Bedrock,
                                probability:    prob,
                                prob_threshold: prob_to_threshold(prob),
                            });
                        }
                    }
                }
                let all_rotations = self.search_all_rotations;
                let cancel = Arc::new(AtomicBool::new(false));
                self.cancel_flag = Some(cancel.clone());
                self.search_start = Some(std::time::Instant::now());
                // Shared atomic updated by the worker; polled by the subscription tick.
                let progress_pos = Arc::new(AtomicI64::new(0));
                self.progress_pos = Some(progress_pos.clone());
                self.status = SearchStatus::Searching("0 x 0".into());
                self.area_label_l = 0;
                self.area_label_next_thresh = 1;
                // Clone the GPU context Arc for the worker thread (cheap ref-count bump).
                let gpu_ctx = if self.use_gpu {
                    match &self.gpu_init {
                        GpuInitState::Ready(ctx) => Some(ctx.clone()),
                        _ => None,
                    }
                } else {
                    None
                };
                Command::perform(
                    async move {
                        tokio::task::spawn_blocking(move || {
                            // Build the list of block-sets to search: either just
                            // the entered pattern, or all 4 rotations of it. The
                            // spiral is traversed exactly once regardless of how
                            // many rotations are present - each candidate position
                            // is checked against every rotation's block-set before
                            // moving on, so a match against *any* rotation ends
                            // the search immediately.
                            let rotations: Vec<Vec<Block>> = if all_rotations {
                                generate_rotations(blocks_vec)
                            } else {
                                vec![blocks_vec]
                            };

                            run_search(
                                seed, start_x, start_z, bt,
                                rotations,
                                cancel,
                                Some(&|idx| { progress_pos.store(idx, Ordering::Relaxed); }),
                                gpu_ctx,
                            )
                        })
                            .await
                            .unwrap_or_else(|e| Err(format!("Thread panic: {e}")))
                    },
                    Message::SearchDone,
                )
            }

            Message::Cancel => {
                if let Some(flag) = &self.cancel_flag { flag.store(true, Ordering::Relaxed); }
                // Flip status immediately so the UI responds; SearchDone will
                // report the precise elapsed time when it arrives.
                self.status = SearchStatus::Cancelled(
                    self.search_start.map(|t| t.elapsed().as_secs_f64()).unwrap_or(0.0)
                );
                Command::none()
            }

            Message::SearchProgress(idx) => {
                if matches!(self.status, SearchStatus::Searching(_)) {
                    if idx <= 0 {
                        self.status = SearchStatus::Searching("0 x 0".into());
                    } else {
                        // The shell L only ever increases as idx grows, so step
                        // it forward with integer arithmetic until idx falls
                        // within the current shell's range.
                        while idx >= self.area_label_next_thresh {
                            self.area_label_l += 1;
                            let l = self.area_label_l;
                            self.area_label_next_thresh = 4 * l * l + 4 * l + 1;
                        }
                        self.status = SearchStatus::Searching(area_label_from_l(self.area_label_l));
                    }
                }
                Command::none()
            }

            Message::ZoomIn => {
                self.ui_scale = (self.ui_scale + 0.1).min(2.0);
                Command::none()
            }
            Message::ZoomOut => {
                self.ui_scale = (self.ui_scale - 0.1).max(0.5);
                Command::none()
            }

            Message::FillUnknownNonBedrockLayer => {
                let layer = &mut self.grid_cells[self.grid_y_idx];
                for row in layer {
                    for cell in row {
                        if *cell == CellState::Unknown {
                            *cell = CellState::NonBedrock;
                        }
                    }
                }
                Command::none()
            }

            Message::ClearGrid => {
                self.grid_cells = make_grid(self.grid_rows, self.grid_cols);
                Command::none()
            }

            Message::SearchDone(result) => {
                let elapsed = self.search_start
                    .take()
                    .map(|t| t.elapsed().as_secs_f64())
                    .unwrap_or(0.0);
                self.cancel_flag = None;
                self.progress_pos = None;
                self.status = match result {
                    Ok(Some((x, z))) => SearchStatus::Found(x, z, elapsed),
                    Ok(None)         => SearchStatus::Cancelled(elapsed),
                    Err(e)           => SearchStatus::Error(e),
                };
                Command::none()
            }
        }
    }

    fn view(&self) -> Element<'_, Message> {
        let is_searching = matches!(self.status, SearchStatus::Searching(_));
        let s = self.ui_scale;
        // Scale a fixed pixel value by the current zoom factor.
        let sc = |v: f32| v * s;

        // ── Zoom controls (top-right corner) ───────────────────────────────
        let zoom_row = row![
            text(format!("Zoom: {:.0}%", self.ui_scale * 100.0)).size(sc(12.0) as u16),
            Space::with_width(Length::Fixed(sc(6.0))),
            button(text("\u{2212}").size(sc(14.0) as u16))
                .on_press(Message::ZoomOut)
                .style(theme::Button::Secondary)
                .padding([sc(3.0) as u16, sc(10.0) as u16]),
            button(text("+").size(sc(14.0) as u16))
                .on_press(Message::ZoomIn)
                .style(theme::Button::Secondary)
                .padding([sc(3.0) as u16, sc(10.0) as u16]),
        ].spacing(sc(4.0) as u16).align_items(Alignment::Center);

        // ── Section: Search parameters ──────────────────────────────────────
        let seed_row = row![
            text("World Seed").size(sc(14.0) as u16).width(Length::Fixed(sc(120.0))),
            text_input("e.g. 124352345", &self.seed)
                .on_input(Message::SeedChanged)
                .size(sc(15.0) as u16)
                .width(Length::Fill)
                .padding(sc(8.0) as u16),
        ].spacing(sc(10.0) as u16).align_items(Alignment::Center);

        let center_row = row![
            text("Search Center").size(sc(14.0) as u16).width(Length::Fixed(sc(120.0))),
            text("X").size(sc(13.0) as u16),
            text_input("0", &self.center_x)
                .on_input(Message::CenterXChanged)
                .size(sc(15.0) as u16)
                .width(Length::Fixed(sc(90.0)))
                .padding(sc(8.0) as u16),
            Space::with_width(Length::Fixed(sc(8.0))),
            text("Z").size(sc(13.0) as u16),
            text_input("0", &self.center_z)
                .on_input(Message::CenterZChanged)
                .size(sc(15.0) as u16)
                .width(Length::Fixed(sc(90.0)))
                .padding(sc(8.0) as u16),
        ].spacing(sc(8.0) as u16).align_items(Alignment::Center);

        let type_row = row![
            text("Bedrock Layer").size(sc(14.0) as u16).width(Length::Fixed(sc(120.0))),
            radio("Floor  (Y \u{2212}64 to \u{2212}59)", BedrockType::Floor, Some(self.bedrock_type), Message::TypeChanged)
                .text_size(sc(14.0) as u16),
            Space::with_width(Length::Fixed(sc(24.0))),
            radio("Roof  (Y 123 to 128)", BedrockType::Roof, Some(self.bedrock_type), Message::TypeChanged)
                .text_size(sc(14.0) as u16),
        ].spacing(sc(10.0) as u16).align_items(Alignment::Center);

        // ── Section: Pattern grid ───────────────────────────────────────────

        // Grid size + offset on one compact row, with a visual gap between the
        // two groups (size vs. offset) instead of just spacing.
        let grid_controls = row![
            text("Grid Size").size(sc(13.0) as u16).width(Length::Fixed(sc(68.0))),
            text("Cols").size(sc(13.0) as u16),
            text_input("8", &self.grid_cols_str)
                .on_input(Message::GridColsChanged)
                .size(sc(14.0) as u16)
                .width(Length::Fixed(sc(46.0)))
                .padding(sc(6.0) as u16),
            text("Rows").size(sc(13.0) as u16),
            text_input("8", &self.grid_rows_str)
                .on_input(Message::GridRowsChanged)
                .size(sc(14.0) as u16)
                .width(Length::Fixed(sc(46.0)))
                .padding(sc(6.0) as u16),
            Space::with_width(Length::Fixed(sc(24.0))),
            text("Offset").size(sc(13.0) as u16).width(Length::Fixed(sc(46.0))),
            text("X").size(sc(13.0) as u16),
            text_input("0", &self.grid_offset_x)
                .on_input(Message::GridOffsetXChanged)
                .size(sc(14.0) as u16)
                .width(Length::Fixed(sc(58.0)))
                .padding(sc(6.0) as u16),
            text("Z").size(sc(13.0) as u16),
            text_input("0", &self.grid_offset_z)
                .on_input(Message::GridOffsetZChanged)
                .size(sc(14.0) as u16)
                .width(Length::Fixed(sc(58.0)))
                .padding(sc(6.0) as u16),
        ].spacing(sc(8.0) as u16).align_items(Alignment::Center);

        // Y-layer tab strip. Tabs marked with * contain at least one non-Unknown cell.
        let ys = y_values(self.bedrock_type);
        let mut y_row: Row<'_, Message> = Row::new()
            .spacing(sc(6.0) as u16)
            .align_items(Alignment::Center)
            .push(text("Y Layer").size(sc(13.0) as u16).width(Length::Fixed(sc(60.0))));
        for (i, &y) in ys.iter().enumerate() {
            let has_data = self.grid_cells[i].iter()
                .any(|r| r.iter().any(|&c| c != CellState::Unknown));
            let label = if has_data { format!("{}*", y) } else { y.to_string() };
            let btn = if i == self.grid_y_idx {
                // Active tab: no on_press so clicking it again is a no-op.
                button(text(label).size(sc(13.0) as u16))
                    .style(theme::Button::Primary)
                    .padding([sc(5.0) as u16, sc(14.0) as u16])
            } else {
                button(text(label).size(sc(13.0) as u16))
                    .style(theme::Button::Secondary)
                    .on_press(Message::GridYChanged(i))
                    .padding([sc(5.0) as u16, sc(14.0) as u16])
            };
            y_row = y_row.push(btn);
        }

        // Cell grid
        let mut grid_col: Column<'_, Message> = Column::new().spacing(sc(3.0) as u16);
        for row_idx in 0..self.grid_rows {
            let mut grid_row: Row<'_, Message> = Row::new().spacing(sc(3.0) as u16);
            for col_idx in 0..self.grid_cols {
                let state = self.grid_cells[self.grid_y_idx][row_idx][col_idx];
                let (label, style) = match state {
                    CellState::Unknown    => ("?", theme::Button::Secondary),
                    CellState::NonBedrock => ("O", theme::Button::Primary),
                    CellState::Bedrock    => ("X", theme::Button::Destructive),
                };
                let cell = mouse_area(
                    button(
                        container(text(label).size(sc(14.0) as u16))
                            .width(Length::Fill)
                            .height(Length::Fill)
                            .center_x()
                            .center_y()
                    )
                    .on_press(Message::GridCellClicked(row_idx, col_idx))
                    .style(style)
                    .width(Length::Fixed(sc(32.0)))
                    .height(Length::Fixed(sc(32.0)))
                    .padding(0)
                ).on_right_press(Message::GridCellRightClicked(row_idx, col_idx));
                grid_row = grid_row.push(cell);
            }
            grid_col = grid_col.push(grid_row);
        }

        // Grid toolbar: rotate buttons on the left, action buttons on the right.
        // Consolidating what were two separate rows into one saves vertical space
        // and groups related controls more logically.
        let grid_toolbar = row![
            text("Rotate:").size(sc(12.0) as u16).width(Length::Fixed(sc(54.0))),
            button(text("+90\u{00b0} CW").size(sc(12.0) as u16))
                .on_press(Message::RotateCW)
                .style(theme::Button::Secondary)
                .padding([sc(4.0) as u16, sc(10.0) as u16]),
            button(text("\u{2212}90\u{00b0} CCW").size(sc(12.0) as u16))
                .on_press(Message::RotateCCW)
                .style(theme::Button::Secondary)
                .padding([sc(4.0) as u16, sc(10.0) as u16]),
            Space::with_width(Length::Fill),
            button(text("Fill ? \u{2192} O").size(sc(12.0) as u16))
                .on_press(Message::FillUnknownNonBedrockLayer)
                .style(theme::Button::Secondary)
                .padding([sc(4.0) as u16, sc(10.0) as u16]),
            button(text("Clear all").size(sc(12.0) as u16))
                .on_press(Message::ClearGrid)
                .style(theme::Button::Secondary)
                .padding([sc(4.0) as u16, sc(10.0) as u16]),
        ].spacing(sc(8.0) as u16).align_items(Alignment::Center);

        // Legend: just the visual key. Actions moved to grid_toolbar above.
        let legend = row![
            text("Click:").size(sc(11.0) as u16),
            Space::with_width(Length::Fixed(sc(6.0))),
            text("?  Unknown").size(sc(11.0) as u16),
            Space::with_width(Length::Fixed(sc(14.0))),
            text("O  Non-bedrock").size(sc(11.0) as u16),
            Space::with_width(Length::Fixed(sc(14.0))),
            text("X  Bedrock").size(sc(11.0) as u16),
            Space::with_width(Length::Fixed(sc(18.0))),
            text("(right-click reverses)").size(sc(11.0) as u16),
        ].align_items(Alignment::Center);

        // ── Section: Search options ─────────────────────────────────────────
        let all_rotations_row = row![
            checkbox(
                "Search all 4 rotations  (north direction unknown)",
                self.search_all_rotations,
            ).on_toggle(Message::ToggleAllRotations).text_size(sc(13.0) as u16),
        ].align_items(Alignment::Center);

        // GPU toggle row — always shown; probed lazily on first enable.
        let gpu_label = match &self.gpu_init {
            GpuInitState::Probing     => "Checking for a GPU\u{2026}".to_string(),
            GpuInitState::Unavailable => "GPU not available on this system".to_string(),
            _ if self.use_gpu => "GPU search enabled (faster)".to_string(),
            _ => "Use GPU for search".to_string(),
        };
        let gpu_row: Row<'_, Message> = row![
            checkbox(gpu_label, self.use_gpu)
                .on_toggle(Message::ToggleGpu)
                .text_size(sc(13.0) as u16),
        ].align_items(Alignment::Center);

        // ── Search / Cancel buttons ─────────────────────────────────────────
        // Search gets Primary style so it stands out; Cancel gets Destructive
        // only while a search is actually running (communicates urgency).
        let search_btn = if is_searching {
            button(text("Searching\u{2026}").size(sc(16.0) as u16))
                .padding([sc(10.0) as u16, sc(32.0) as u16])
        } else {
            button(text("Search").size(sc(16.0) as u16))
                .on_press(Message::Search)
                .style(theme::Button::Primary)
                .padding([sc(10.0) as u16, sc(32.0) as u16])
        };
        let cancel_btn = if is_searching {
            button(text("Cancel").size(sc(15.0) as u16))
                .on_press(Message::Cancel)
                .style(theme::Button::Destructive)
                .padding([sc(10.0) as u16, sc(22.0) as u16])
        } else {
            button(text("Cancel").size(sc(15.0) as u16))
                .style(theme::Button::Secondary)
                .padding([sc(10.0) as u16, sc(22.0) as u16])
        };

        // ── Status bar ──────────────────────────────────────────────────────
        let status_msg = match &self.status {
            SearchStatus::Idle              => text("Ready.").size(sc(15.0) as u16),
            SearchStatus::Searching(area)   => text(format!("Searching {}\u{2026}", area)).size(sc(15.0) as u16),
            SearchStatus::Cancelled(secs)   => text(format!("Cancelled after {:.1}s.", secs)).size(sc(15.0) as u16),
            SearchStatus::Found(x, z, secs) => text(format!("Found at  X: {}   Z: {}   ({:.1}s)", x, z, secs)).size(sc(17.0) as u16),
            SearchStatus::Error(e)          => text(format!("Error: {}", e)).size(sc(15.0) as u16),
        };

        // ── Layout assembly ─────────────────────────────────────────────────
        let content = Column::new()
            .spacing(0)
            .padding(sc(18.0) as u16)
            .max_width(sc(760.0))

            // Title bar
            .push(
                row![
                    text("Bedrock Formation Finder").size(sc(24.0) as u16),
                    Space::with_width(Length::Fill),
                    zoom_row,
                ].align_items(Alignment::Center)
            )
            .push(Space::with_height(Length::Fixed(sc(12.0))))
            .push(horizontal_rule(1))
            .push(Space::with_height(Length::Fixed(sc(12.0))))

            // Search parameters
            .push(seed_row)
            .push(Space::with_height(Length::Fixed(sc(8.0))))
            .push(center_row)
            .push(Space::with_height(Length::Fixed(sc(8.0))))
            .push(type_row)
            .push(Space::with_height(Length::Fixed(sc(14.0))))
            .push(horizontal_rule(1))
            .push(Space::with_height(Length::Fixed(sc(12.0))))

            // Pattern grid
            .push(grid_controls)
            .push(Space::with_height(Length::Fixed(sc(10.0))))
            .push(y_row)
            .push(Space::with_height(Length::Fixed(sc(6.0))))
            .push(grid_col)
            .push(Space::with_height(Length::Fixed(sc(8.0))))
            .push(grid_toolbar)
            .push(Space::with_height(Length::Fixed(sc(6.0))))
            .push(legend)
            .push(Space::with_height(Length::Fixed(sc(14.0))))
            .push(horizontal_rule(1))
            .push(Space::with_height(Length::Fixed(sc(12.0))))

            // Search options
            .push(all_rotations_row)
            .push(Space::with_height(Length::Fixed(sc(8.0))))
            .push(gpu_row)
            .push(Space::with_height(Length::Fixed(sc(16.0))))
            .push(horizontal_rule(1))
            .push(Space::with_height(Length::Fixed(sc(16.0))))

            // Search actions
            .push(
                container(
                    row![
                        search_btn,
                        Space::with_width(Length::Fixed(sc(12.0))),
                        cancel_btn,
                    ].align_items(Alignment::Center)
                )
                .width(Length::Fill)
                .center_x()
            )
            .push(Space::with_height(Length::Fixed(sc(12.0))))
            .push(
                container(status_msg)
                    .width(Length::Fill)
                    .center_x()
                    .padding([sc(10.0) as u16, sc(16.0) as u16])
            );

        container(scrollable(content)).width(Length::Fill).height(Length::Fill).center_x().into()
    }
}
