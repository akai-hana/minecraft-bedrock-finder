/// bedrockformation
/// GUI layer (iced) for the Bedrock Formation Finder.
///
/// All search/computation logic lives in `crate::core`; this module holds
/// only the application state, message handling, and view rendering.

use std::sync::{Arc, Mutex, atomic::{AtomicBool, AtomicI64, Ordering}};

use iced::{
    Application, Command, Element, Event, Length, Subscription, Theme, Alignment, Color,
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

// GUI - theme
//
// A custom dark palette, replacing the built-in GruvboxDark theme. Note
// `warning` and `danger` intentionally share the same teal (#007373) rather
// than danger being red, `primary` (a deep red, #ac3232) is reserved for
// the app's main affordances (the Search button, the active Y-layer tab,
// and "non-bedrock" grid cells), while the teal marks the Cancel button and
// "bedrock" grid cells.
fn custom_palette() -> theme::Palette {
    theme::Palette {
        background: Color::from_rgb8(0x0d, 0x0d, 0x14),
        text:       Color::from_rgb8(0xb9, 0xc0, 0xa8),
        primary:    Color::from_rgb8(0xac, 0x32, 0x32),
        // secondary:    Color::from_rgb8(0x00, 0x73, 0x73),
        success:    Color::from_rgb8(0x52, 0x26, 0x3e),
        danger:     Color::from_rgb8(0x00, 0x73, 0x73),
    }
}

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
    /// Search was cancelled; carries how many coordinates had been found so
    /// far and the elapsed seconds.
    Cancelled(usize, f64),
    /// Search ran to completion (found the requested number of matches, or
    /// exhausted the configured limit); carries the count found and elapsed
    /// seconds.
    Found(usize, f64),
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
    /// Raw text of the "number of matches to find" input.
    dup_count_str: String,
    /// Parsed target match count. 0 means unlimited (keep searching until
    /// cancelled or the spiral search space is exhausted).
    dup_count:     usize,
    /// Coordinates found by the current (or most recently completed) search,
    /// in the order they were discovered. Displayed in a scrollable list.
    found_coords:  Vec<(i32, i32)>,
    /// Shared with the search worker thread while a search is running: the
    /// worker pushes each new match here as it's found, and the GUI polls it
    /// on the same tick subscription used for progress, so the results list
    /// updates live instead of only at the very end.
    found_shared:  Option<Arc<Mutex<Vec<(i32, i32)>>>>,
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

// Pattern rarity helpers
//
// Computes the expected number of times the user's current pattern would
// appear somewhere in a full Minecraft Java Edition world.
//
// IMPORTANT: this is an a-priori expectation over a randomly generated
// world, not a claim about whether the pattern can exist. If the pattern
// was copied from a real, already-inspected world, it already occurs there
// at least once, a tiny E does not contradict that, it just means a
// *second* (duplicate) occurrence elsewhere is essentially impossible.
// See `fmt_rarity` for how this is worded for the user.
//
// Model
// -----
// Each filled cell (Bedrock or NonBedrock) at Y-layer y contributes an
// independent probability factor:
//   - Bedrock cell:     p_i  = compute_probability(y, bt)      ∈ {0.2, 0.4, 0.6, 0.8}
//   - NonBedrock cell:  p_i  = 1 - compute_probability(y, bt)  ∈ {0.2, 0.4, 0.6, 0.8}
//
// The joint probability that a single world position (x, z) perfectly matches
// the entire multi-layer pattern is:
//   p_match = ∏ p_i
//
// Computed in log-space to handle patterns with dozens of constrained cells
// without floating-point underflow.
//
// The expected number of occurrences across the world is then:
//   E = N x p_match,  where N ≈ 60 000 000² ≈ 3.6 x 10¹⁵ (Java world border)
//
// Returns None when no cells are filled (pattern is completely unconstrained).
// Returns Some((n_filled, ln_E)) otherwise, where ln_E may be ±∞ in edge
// cases (handled by the display layer).
fn pattern_occurrence_stats(
    cells: &[Vec<Vec<CellState>>],
    bt: BedrockType,
) -> Option<(usize, f64)> {
    let ys = y_values(bt);
    let mut log_p: f64 = 0.0;
    let mut n = 0usize;

    for (y_idx, &y) in ys.iter().enumerate() {
        let p = compute_probability(y, bt);
        // p is strictly in (0,1) for the 4 probabilistic Y-layers shown in
        // the grid; ln(p) and ln(1-p) are therefore both finite and negative.
        for row in &cells[y_idx] {
            for &cell in row {
                match cell {
                    CellState::Unknown => {}
                    CellState::Bedrock    => { n += 1; log_p += p.ln(); }
                    CellState::NonBedrock => { n += 1; log_p += (1.0 - p).ln(); }
                }
            }
        }
    }

    if n == 0 {
        return None;
    }

    // Java Edition world border: ±29 999 984 blocks ⟹ ~60 M x 60 M positions.
    let ln_world: f64 = (60_000_000_f64 * 60_000_000_f64).ln(); // ≈ 35.82
    Some((n, ln_world + log_p))
}

/// Format a large or small positive number in a compact, readable way.
/// For E ≥ 1   : "~{value}x per world"
/// For E < 1   : "1 in ~{1/E} worlds"
fn format_duration(secs: f64) -> String {
    let total = secs.round().max(0.0) as u64;
    let h = total / 3600;
    let m = (total % 3600) / 60;
    let s = total % 60;
    if h > 0 {
        format!("{:02}:{:02}:{:02}", h, m, s)
    } else {
        format!("{:02}:{:02}", m, s)
    }
}

fn fmt_rarity(ln_e: f64) -> (String, String) {
    // Thresholds in log-space (all values are ln of the boundary)
    const LN_1E6:  f64 =  13.815; // ln(1 000 000)
    const LN_1E3:  f64 =   6.908; // ln(1 000)
    const LN_10:   f64 =   2.303;
    const LN_0_01: f64 =  -4.605; // ln(0.01)
    const LN_1E_6: f64 = -13.816; // ln(1e-6)  ← "essentially unique" boundary

    if ln_e > LN_1E6 {
        let e = ln_e.exp();
        ("Extremely common".into(), format!("~{:.2e}x per world", e))
    } else if ln_e > LN_1E3 {
        let e = ln_e.exp();
        ("Very common".into(), format!("~{:.0}x per world", e))
    } else if ln_e > LN_10 {
        let e = ln_e.exp();
        ("Common".into(), format!("~{:.0}x per world", e))
    } else if ln_e >= 0.0 {
        let e = ln_e.exp();
        ("Uncommon".into(), format!("~{:.1}x per world", e))
    } else if ln_e > LN_0_01 {
        // E between 0.01 and 1
        let inv = (-ln_e).exp();
        ("Rare".into(), format!("1 in ~{:.0} worlds", inv))
    } else if ln_e > LN_1E_6 {
        // E between 1e-6 and 0.01
        let inv = (-ln_e).exp();
        ("Very rare".into(), format!("1 in ~{:.2e} worlds", inv))
    } else {
        // Below one-in-a-million worlds. This does NOT mean the pattern
        // can't exist, if you typed it in from a real world, it already
        // does. It means a *second* occurrence of this exact pattern,
        // anywhere else, is astronomically unlikely.
        ("ESSENTIALLY".into(), "UNIQUE".into())
    }
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
            dup_count_str: "1".into(),
            dup_count:     1,
            found_coords:  Vec::new(),
            found_shared:  None,
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
    /// Number of matching coordinates to search for before stopping
    /// ("0" = unlimited / keep going until cancelled).
    DupCountChanged(String),
    /// Toggle whether the GPU compute path is used for the coarse search.
    ToggleGpu(bool),
    /// Result of an async GPU probe triggered by first enabling `ToggleGpu`.
    /// `None` means no compatible GPU adapter/device was found.
    GpuInitialized(Option<Arc<GpuContext>>),
    Search,
    Cancel,
    /// Fired periodically during a search with the farthest spiral index checked so far.
    SearchProgress(i64),
    /// Fired periodically during a search with a fresh snapshot of every
    /// coordinate found so far, so the results list updates live.
    FoundUpdate(Vec<(i32, i32)>),
    /// `Ok((coords, was_cancelled))` once the worker stops: either the
    /// requested number of matches were found, the search was cancelled, or
    /// (for a "0 = unlimited" search) the spiral space was exhausted.
    SearchDone(Result<(Vec<(i32, i32)>, bool), String>),
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

    fn theme(&self) -> Theme { Theme::custom("Bedrock Dark".to_string(), custom_palette()) }

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

        let mut subs = vec![keyboard_sub];

        if let Some(progress_pos) = self.progress_pos.clone() {
            // While searching, tick every 150 ms and read the shared atomic.
            let tick_sub = time::every(std::time::Duration::from_millis(150))
                .map(move |_| {
                    Message::SearchProgress(progress_pos.load(Ordering::Relaxed))
                });
            subs.push(tick_sub);
        }

        if let Some(found_shared) = self.found_shared.clone() {
            // Separate tick: snapshot the live results list so the
            // scrollable updates as new duplicates are found, not only once
            // the whole search finishes.
            let found_tick = time::every(std::time::Duration::from_millis(150))
                .map(move |_| {
                    let snapshot = found_shared.lock().map(|g| g.clone()).unwrap_or_default();
                    Message::FoundUpdate(snapshot)
                });
            subs.push(found_tick);
        }

        Subscription::batch(subs)
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

            Message::DupCountChanged(s) => {
                self.dup_count_str = s.clone();
                if s.is_empty() {
                    self.dup_count = 1;
                } else if let Ok(n) = s.parse::<usize>() {
                    self.dup_count = n;
                }
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
                    Err(_) => { self.status = SearchStatus::Error("Invalid seed. Please set a valid 64-bit integer.".into()); return Command::none(); }
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
                let dup_target = self.dup_count;
                let cancel = Arc::new(AtomicBool::new(false));
                self.cancel_flag = Some(cancel.clone());
                self.search_start = Some(std::time::Instant::now());
                // Shared atomic updated by the worker; polled by the subscription tick.
                let progress_pos = Arc::new(AtomicI64::new(0));
                self.progress_pos = Some(progress_pos.clone());
                // Shared list of matches found so far, updated by the worker
                // as it goes and polled by the subscription tick so the
                // results scrollable fills in live.
                let found_shared: Arc<Mutex<Vec<(i32, i32)>>> = Arc::new(Mutex::new(Vec::new()));
                self.found_shared = Some(found_shared.clone());
                self.found_coords.clear();
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

                            let progress_cb = |idx: i64| { progress_pos.store(idx, Ordering::Relaxed); };

                            // Repeatedly search, resuming just past each match,
                            // until `dup_target` matches are found (0 = keep
                            // going until cancelled / the spiral is exhausted).
                            let mut start_group: i64 = 0;
                            let mut results: Vec<(i32, i32)> = Vec::new();
                            let mut was_cancelled = false;
                            loop {
                                if cancel.load(Ordering::Relaxed) {
                                    was_cancelled = true;
                                    break;
                                }
                                let outcome = run_search(
                                    seed, start_x, start_z, bt,
                                    rotations.clone(),
                                    cancel.clone(),
                                    Some(&progress_cb),
                                    gpu_ctx.clone(),
                                    start_group,
                                );
                                match outcome {
                                    Ok(Some((x, z, next_group))) => {
                                        results.push((x, z));
                                        if let Ok(mut g) = found_shared.lock() {
                                            g.push((x, z));
                                        }
                                        start_group = next_group;
                                        if dup_target != 0 && results.len() >= dup_target {
                                            break;
                                        }
                                    }
                                    Ok(None) => {
                                        was_cancelled = true;
                                        break;
                                    }
                                    Err(e) => return Err(e),
                                }
                            }
                            Ok((results, was_cancelled))
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
                    self.found_coords.len(),
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

            Message::FoundUpdate(coords) => {
                self.found_coords = coords;
                Command::none()
            }

            Message::SearchDone(result) => {
                let elapsed = self.search_start
                    .take()
                    .map(|t| t.elapsed().as_secs_f64())
                    .unwrap_or(0.0);
                self.cancel_flag  = None;
                self.progress_pos = None;
                self.found_shared = None;
                self.status = match result {
                    Ok((coords, was_cancelled)) => {
                        let count = coords.len();
                        self.found_coords = coords;
                        if was_cancelled {
                            SearchStatus::Cancelled(count, elapsed)
                        } else {
                            SearchStatus::Found(count, elapsed)
                        }
                    }
                    Err(e) => SearchStatus::Error(e),
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

        // Zoom controls (top-right corner)
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

        // Section: Search parameters
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

        // Section: Pattern grid
        // Laid out as [grid] | [panel], the grid sits on the left where it
        // has room to breathe, and every control that configures or acts on
        // it (size/offset, Y layer, rotate/fill/clear, legend) is gathered
        // into a single compact panel on the right, instead of trailing
        // below as a series of full-width rows.

        // A label column width shared by every row in the side panel, so the
        // mini-labels ("Size", "Offset", "Y Layer"...) line up like a ruler.
        let panel_label_w = Length::Fixed(sc(54.0));

        let size_row = row![
            text("Size").size(sc(12.0) as u16).width(panel_label_w),
            text("Cols").size(sc(12.0) as u16),
            text_input("8", &self.grid_cols_str)
                .on_input(Message::GridColsChanged)
                .size(sc(13.0) as u16)
                .width(Length::Fixed(sc(40.0)))
                .padding(sc(5.0) as u16),
            text("Rows").size(sc(12.0) as u16),
            text_input("8", &self.grid_rows_str)
                .on_input(Message::GridRowsChanged)
                .size(sc(13.0) as u16)
                .width(Length::Fixed(sc(40.0)))
                .padding(sc(5.0) as u16),
        ].spacing(sc(6.0) as u16).align_items(Alignment::Center);

        let offset_row = row![
            text("Offset").size(sc(12.0) as u16).width(panel_label_w),
            text("X").size(sc(12.0) as u16),
            text_input("0", &self.grid_offset_x)
                .on_input(Message::GridOffsetXChanged)
                .size(sc(13.0) as u16)
                .width(Length::Fixed(sc(48.0)))
                .padding(sc(5.0) as u16),
            text("Z").size(sc(12.0) as u16),
            text_input("0", &self.grid_offset_z)
                .on_input(Message::GridOffsetZChanged)
                .size(sc(13.0) as u16)
                .width(Length::Fixed(sc(48.0)))
                .padding(sc(5.0) as u16),
        ].spacing(sc(6.0) as u16).align_items(Alignment::Center);

        // Y-layer tab strip. Tabs marked with * contain at least one non-Unknown cell.
        let ys = y_values(self.bedrock_type);
        let mut y_row: Row<'_, Message> = Row::new()
            .spacing(sc(5.0) as u16)
            .align_items(Alignment::Center)
            .push(text("Y Layer").size(sc(12.0) as u16).width(panel_label_w));
        for (i, &y) in ys.iter().enumerate() {
            let has_data = self.grid_cells[i].iter()
                .any(|r| r.iter().any(|&c| c != CellState::Unknown));
            let label = if has_data { format!("{}*", y) } else { y.to_string() };
            let btn = if i == self.grid_y_idx {
                // Active tab: no on_press so clicking it again is a no-op.
                button(text(label).size(sc(12.0) as u16))
                    .style(theme::Button::Primary)
                    .padding([sc(5.0) as u16, sc(10.0) as u16])
            } else {
                button(text(label).size(sc(12.0) as u16))
                    .style(theme::Button::Secondary)
                    .on_press(Message::GridYChanged(i))
                    .padding([sc(5.0) as u16, sc(10.0) as u16])
            };
            y_row = y_row.push(btn);
        }

        // Cell grid
        // The visual centerpiece, kept on the left with nothing crowding it.
        let mut grid_col: Column<'_, Message> = Column::new().spacing(sc(3.0) as u16);
        for row_idx in 0..self.grid_rows {
            let mut grid_row: Row<'_, Message> = Row::new().spacing(sc(3.0) as u16);
            for col_idx in 0..self.grid_cols {
                let state = self.grid_cells[self.grid_y_idx][row_idx][col_idx];
                let (label, style) = match state {
                    CellState::Unknown    => ("?", theme::Button::Secondary),
                    CellState::NonBedrock => ("O", theme::Button::Destructive),
                    CellState::Bedrock    => ("X", theme::Button::Primary),
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

        let rotate_row = row![
            text("Rotate").size(sc(12.0) as u16).width(panel_label_w),
            button(text("+90º").size(sc(12.0) as u16))
                .on_press(Message::RotateCW)
                .style(theme::Button::Secondary)
                .padding([sc(4.0) as u16, sc(10.0) as u16]),
            button(text("-90º").size(sc(12.0) as u16))
                .on_press(Message::RotateCCW)
                .style(theme::Button::Secondary)
                .padding([sc(4.0) as u16, sc(10.0) as u16]),
        ].spacing(sc(6.0) as u16).align_items(Alignment::Center);

        let edit_row = row![
            text("Edit").size(sc(12.0) as u16).width(panel_label_w),
            button(text("? -> O").size(sc(12.0) as u16))
                .on_press(Message::FillUnknownNonBedrockLayer)
                .style(theme::Button::Secondary)
                .padding([sc(4.0) as u16, sc(10.0) as u16]),
            button(text("O/X -> ?").size(sc(12.0) as u16))
                .on_press(Message::ClearGrid)
                .style(theme::Button::Secondary)
                .padding([sc(4.0) as u16, sc(10.0) as u16]),
        ].spacing(sc(6.0) as u16).align_items(Alignment::Center);

        // Legend
        // Stacked vertically rather than strung out horizontally;
        // reads naturally as a short key in a narrow side panel
        let legend = Column::new()
            .spacing(sc(3.0) as u16)
            .push(text("Legend").size(sc(12.0) as u16))
            .push(Space::with_height(Length::Fixed(sc(6.0))))
            .push(text("?   Unknown").size(sc(11.0) as u16))
            .push(text("O  Non-bedrock").size(sc(11.0) as u16))
            .push(text("X   Bedrock").size(sc(11.0) as u16))
            .push(Space::with_height(Length::Fixed(sc(6.0))))
            .push(text("Right-click to reverse").size(sc(11.0) as u16));

        // Pattern rarity panel
        // Shown to the right of the legend.
        // Updates in real time as the user fills cells, using only the cells that are
        // actually set (not the grid dimensions), so a 2x8 pattern and a 4x4
        // pattern with the same number of constrained cells show the same odds.
        let hsz = sc(12.0) as u16; // header size, matches the "Legend" header
        let rsz = sc(11.0) as u16; // same font size as the legend entries
        let mut rarity_col: Column<'_, Message> = Column::new()
            .spacing(sc(3.0) as u16)
            .push(text("Pattern rarity").size(hsz))
            .push(text("To avoid duplicates,").size(rsz))
            .push(text("keep odds below 'Common'.").size(rsz));

        match pattern_occurrence_stats(&self.grid_cells, self.bedrock_type) {
            None => {
                // No cells filled yet; gently prompt the user.
                rarity_col = rarity_col
                    .push(text("(Fill grid cells").size(rsz))
                    .push(text("to see odds)").size(rsz));
            }
            Some((_n, ln_e)) => {
                let (label, detail) = fmt_rarity(ln_e);
                rarity_col = rarity_col
                    .push(text(label).size(rsz))
                    .push(text(detail).size(rsz));

                // When E is below one-in-a-million, a *second* occurrence of
                // this exact pattern anywhere else is essentially impossible.
                // This doesn't mean the pattern itself can't exist; if it
                // came from a real world, it already does, once.
                if ln_e <= -13.816 {
                    rarity_col = rarity_col
                        .push(text("No duplicate of this").size(rsz))
                        .push(text("will ever be found.").size(rsz));
                }
            }
        }

        // Legend and rarity side by side, top-aligned.
        let legend_and_rarity: Row<'_, Message> = Row::new()
            .spacing(sc(20.0) as u16)
            .align_items(Alignment::Start)
            .push(legend)
            .push(rarity_col);

        // The full right-hand panel: everything that configures or acts on
        // the grid, grouped into clearly separated sub-blocks.
        let grid_panel: Column<'_, Message> = Column::new()
            .spacing(sc(16.0) as u16)
            .width(Length::Shrink)
            .push(Column::new().spacing(sc(6.0) as u16).push(size_row).push(offset_row))
            .push(y_row)
            .push(Column::new().spacing(sc(6.0) as u16).push(rotate_row).push(edit_row))
            .push(legend_and_rarity);

        // Grid + panel side by side, top-aligned so the panel starts level
        // with the first row of cells.
        let grid_section: Row<'_, Message> = Row::new()
            .spacing(sc(28.0) as u16)
            .align_items(Alignment::Start)
            .push(grid_col)
            .push(grid_panel);

        // Section: Search options
        let all_rotations_row = row![
            checkbox(
                "Search all 4 rotations  (north direction unknown)",
                self.search_all_rotations,
            ).on_toggle(Message::ToggleAllRotations).text_size(sc(13.0) as u16),
        ].align_items(Alignment::Center);

        // GPU toggle row
        // Always shown; probed lazily on first enable.
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

        // How many matching coordinates to look for before stopping. "0"
        // means keep searching indefinitely (until cancelled), useful for
        // hunting down every duplicate of a pattern.
        let dup_row = row![
            text("Find").size(sc(13.0) as u16).width(Length::Fixed(sc(40.0))),
            text_input("1", &self.dup_count_str)
                .on_input(Message::DupCountChanged)
                .size(sc(14.0) as u16)
                .width(Length::Fixed(sc(60.0)))
                .padding(sc(6.0) as u16),
            text("coordinate(s)  (0 = scan until cancelled)").size(sc(13.0) as u16),
        ].spacing(sc(8.0) as u16).align_items(Alignment::Center);

        // Search / Cancel buttons
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

        // Pluralize "N coordinate(s)" consistently across status messages.
        let fmt_coord_count = |n: usize| {
            if n == 1 { "1 coordinate".to_string() } else { format!("{} coordinates", n) }
        };

        // Status bar
        let status_msg = match &self.status {
            SearchStatus::Idle                => text("Ready.").size(sc(15.0) as u16),
            SearchStatus::Searching(area)     => text(format!("Searching {}\u{2026}", area)).size(sc(15.0) as u16),
            SearchStatus::Cancelled(n, secs)  => text(format!(
                "Cancelled after {}; {} found.", format_duration(*secs), fmt_coord_count(*n)
            )).size(sc(15.0) as u16),
            SearchStatus::Found(n, secs)      => text(format!(
                "{} found  ({})", fmt_coord_count(*n), format_duration(*secs)
            )).size(sc(17.0) as u16),
            SearchStatus::Error(e)            => text(format!("Error: {}", e)).size(sc(15.0) as u16),
        };

        // Results list
        // Scrollable so any number of duplicate matches can be
        // browsed without growing the rest of the window. Updates live while
        // a search with "Find" > 1 (or 0/unlimited) is in progress.
        let mut results_col: Column<'_, Message> = Column::new().spacing(sc(4.0) as u16);
        if self.found_coords.is_empty() {
            results_col = results_col.push(
                text("No coordinates found yet.").size(sc(13.0) as u16)
            );
        } else {
            for (i, (x, z)) in self.found_coords.iter().enumerate() {
                results_col = results_col.push(
                    text(format!("{:>4}.   X: {:<10}  Z: {}", i + 1, x, z))
                        .size(sc(14.0) as u16)
                );
            }
        }
        let results_panel = container(
            scrollable(results_col).height(Length::Fixed(sc(140.0)))
        )
            .width(Length::Fill)
            .padding(sc(10.0) as u16)
            .style(theme::Container::Box);

        // Layout assembly
        let content = Column::new()
            .spacing(0)
            .padding(sc(18.0) as u16)
            .max_width(sc(760.0))

            // Title bar
            .push(
                row![
                    text("Minecraft Bedrock Finder").size(sc(24.0) as u16),
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

            // Pattern grid (grid on the left, all grid controls on the right)
            .push(grid_section)
            .push(Space::with_height(Length::Fixed(sc(16.0))))
            .push(horizontal_rule(1))
            .push(Space::with_height(Length::Fixed(sc(12.0))))

            // Search options
            .push(all_rotations_row)
            .push(Space::with_height(Length::Fixed(sc(8.0))))
            .push(gpu_row)
            .push(Space::with_height(Length::Fixed(sc(8.0))))
            .push(dup_row)
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
            )
            .push(Space::with_height(Length::Fixed(sc(8.0))))
            .push(results_panel);

        container(scrollable(content)).width(Length::Fill).height(Length::Fill).center_x().into()
    }
}
