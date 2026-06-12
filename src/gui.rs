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

// GUI - types

/// State of one cell in the constraint grid.
/// Cycles Unknown -> NonBedrock -> Bedrock -> Unknown on each click.
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

/// The four Y values that can contain probabilistic bedrock for each layer type.
/// Ordered left-to-right on the tab strip: most-air end first (-60 ... -63 for floor).
/// -64 (always bedrock) and -59 (always air) are excluded as redundant.
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

/// Rotate all Y-layers 90º clockwise.
/// In the grid, col maps to X and row maps to Z, so CW means: new_col = rows−1−row, new_row = col.
/// The resulting grid has new_rows = old_cols, new_cols = old_rows.
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
                    // CW: (r, c) -> new position (c, rows−1−r)
                    new_layer[c][rows - 1 - r] = layer[r][c];
                }
            }
            new_layer
        })
        .collect();
    (new_cells, new_rows, new_cols)
}

/// Rotate all Y-layers 90º counter-clockwise.
/// CCW: (r, c) -> new position (cols−1−c, r).
/// The resulting grid has new_rows = old_cols, new_cols = old_rows.
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
                    // CCW: (r, c) -> new position (cols−1−c, r)
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
    /// Actively searching; carries a human-readable area label like "10k × 10k".
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
    /// (= 4L²+4L+1 for the current `area_label_l`).
    area_label_next_thresh: i64,
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
    /// Rotate all Y-layers 90º clockwise (X->Z, Z->−X).
    RotateCW,
    /// Rotate all Y-layers 90º counter-clockwise (X->−Z, Z->X).
    RotateCCW,
    /// Toggle whether the search tries all 4 rotations of the pattern.
    ToggleAllRotations(bool),
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
                self.status = SearchStatus::Searching("0 × 0".into());
                self.area_label_l = 0;
                self.area_label_next_thresh = 1;
                Command::perform(
                    async move {
                        tokio::task::spawn_blocking(move || {
                            // Build the list of block-sets to search: either just
                            // the entered pattern, or all 4 rotations of it. The
                            // spiral is traversed exactly once regardless of how
                            // many rotations are present — each candidate position
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
                // We don't know elapsed precisely on cancel; SearchDone will carry it.
                // Just flip status so the UI reflects cancellation immediately.
                self.status = SearchStatus::Cancelled(
                    self.search_start.map(|t| t.elapsed().as_secs_f64()).unwrap_or(0.0)
                );
                Command::none()
            }

            Message::SearchProgress(idx) => {
                if matches!(self.status, SearchStatus::Searching(_)) {
                    if idx <= 0 {
                        self.status = SearchStatus::Searching("0 × 0".into());
                    } else {
                        // batch_start_group (and thus idx) grows monotonically,
                        // so the shell L only ever increases. Step it forward
                        // algebraically (no sqrt) until idx is back within the
                        // current shell's range.
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
        // Helper: scale a fixed pixel value by the zoom factor.
        let sc = |v: f32| v * s;

        let seed_row = row![
            text("World Seed").size(sc(16.0) as u16).width(Length::Fixed(sc(130.0))),
            text_input("e.g. 124352345", &self.seed)
                .on_input(Message::SeedChanged)
                .size(sc(16.0) as u16)
                .width(Length::Fill)
                .padding(sc(8.0) as u16),
        ].spacing(sc(12.0) as u16).align_items(Alignment::Center);

        let center_row = row![
            text("Search Center").size(sc(16.0) as u16).width(Length::Fixed(sc(130.0))),
            text("X").size(sc(16.0) as u16),
            text_input("0", &self.center_x).on_input(Message::CenterXChanged).size(sc(16.0) as u16).width(Length::Fixed(sc(90.0))).padding(sc(8.0) as u16),
            text("Z").size(sc(16.0) as u16),
            text_input("0", &self.center_z).on_input(Message::CenterZChanged).size(sc(16.0) as u16).width(Length::Fixed(sc(90.0))).padding(sc(8.0) as u16),
        ].spacing(sc(10.0) as u16).align_items(Alignment::Center);

        let type_row = row![
            text("Bedrock Layer").size(sc(16.0) as u16).width(Length::Fixed(sc(130.0))),
            radio("Floor (Y -64 to -59)", BedrockType::Floor, Some(self.bedrock_type), Message::TypeChanged).text_size(sc(16.0) as u16),
            Space::with_width(Length::Fixed(sc(20.0))),
            radio("Roof  (Y 123 to 128)", BedrockType::Roof,  Some(self.bedrock_type), Message::TypeChanged).text_size(sc(16.0) as u16),
        ].spacing(sc(10.0) as u16).align_items(Alignment::Center);

        // Grid size + offset controls
        let grid_controls = row![
            text("Grid Size").size(sc(16.0) as u16).width(Length::Fixed(sc(80.0))),
            text("Cols").size(sc(16.0) as u16),
            text_input("8", &self.grid_cols_str)
                .on_input(Message::GridColsChanged)
                .size(sc(16.0) as u16)
                .width(Length::Fixed(sc(46.0)))
                .padding(sc(7.0) as u16),
            text("Rows").size(sc(16.0) as u16),
            text_input("8", &self.grid_rows_str)
                .on_input(Message::GridRowsChanged)
                .size(sc(16.0) as u16)
                .width(Length::Fixed(sc(46.0)))
                .padding(sc(7.0) as u16),
            Space::with_width(Length::Fixed(sc(20.0))),
            text("Offset").size(sc(16.0) as u16).width(Length::Fixed(sc(48.0))),
            text("X").size(sc(16.0) as u16),
            text_input("0", &self.grid_offset_x)
                .on_input(Message::GridOffsetXChanged)
                .size(sc(16.0) as u16)
                .width(Length::Fixed(sc(58.0)))
                .padding(sc(7.0) as u16),
            text("Z").size(sc(16.0) as u16),
            text_input("0", &self.grid_offset_z)
                .on_input(Message::GridOffsetZChanged)
                .size(sc(16.0) as u16)
                .width(Length::Fixed(sc(58.0)))
                .padding(sc(7.0) as u16),
        ].spacing(sc(8.0) as u16).align_items(Alignment::Center);

        // Y-layer tab strip
        // Tabs marked with * have at least one non-Unknown cell.
        let ys = y_values(self.bedrock_type);
        let mut y_row: Row<'_, Message> = Row::new()
            .spacing(sc(6.0) as u16)
            .align_items(Alignment::Center)
            .push(text("Y Layer").size(sc(16.0) as u16).width(Length::Fixed(sc(70.0))));
        for (i, &y) in ys.iter().enumerate() {
            let has_data = self.grid_cells[i].iter()
                .any(|r| r.iter().any(|&c| c != CellState::Unknown));
            let label = if has_data { format!("{}*", y) } else { y.to_string() };
            let btn = if i == self.grid_y_idx {
                // Active tab: no on_press so it is not re-clickable
                button(text(label).size(sc(13.0) as u16))
                    .style(theme::Button::Primary)
                    .padding([sc(5.0) as u16, sc(10.0) as u16])
            } else {
                button(text(label).size(sc(13.0) as u16))
                    .style(theme::Button::Secondary)
                    .on_press(Message::GridYChanged(i))
                    .padding([sc(5.0) as u16, sc(10.0) as u16])
            };
            y_row = y_row.push(btn);
        }

        // Cell grid
        let mut grid_col: Column<'_, Message> = Column::new().spacing(sc(2.0) as u16);
        for row_idx in 0..self.grid_rows {
            let mut grid_row: Row<'_, Message> = Row::new().spacing(sc(2.0) as u16);
            for col_idx in 0..self.grid_cols {
                let state = self.grid_cells[self.grid_y_idx][row_idx][col_idx];
                let (label, style) = match state {
                    CellState::Unknown    => ("?", theme::Button::Secondary),
                    CellState::NonBedrock => ("O", theme::Button::Primary),
                    CellState::Bedrock    => ("X", theme::Button::Destructive),
                };
                let cell = mouse_area(
                    button(
                            container(text(label).size(sc(15.0) as u16))
                                .width(Length::Fill)
                                .height(Length::Fill)
                                .center_x()
                                .center_y()
                        )
                        .on_press(Message::GridCellClicked(row_idx, col_idx))
                        .style(style)
                        .width(Length::Fixed(sc(30.0)))
                        .height(Length::Fixed(sc(30.0)))
                        .padding(0)
                ).on_right_press(Message::GridCellRightClicked(row_idx, col_idx));
                grid_row = grid_row.push(cell);
            }
            grid_col = grid_col.push(grid_row);
        }

        let rotate_row = row![
            text("Rotate grid:").size(sc(12.0) as u16).width(Length::Fixed(sc(80.0))),
            button(text("+90º (Clockwise)").size(sc(13.0) as u16))
                .on_press(Message::RotateCW)
                .style(theme::Button::Secondary)
                .padding([sc(4.0) as u16, sc(10.0) as u16]),
            button(text("−90º (Counter-clockwise)").size(sc(13.0) as u16))
                .on_press(Message::RotateCCW)
                .style(theme::Button::Secondary)
                .padding([sc(4.0) as u16, sc(10.0) as u16]),
        ].spacing(sc(8.0) as u16).align_items(Alignment::Center);

        let legend = row![
            text("Click to cycle:").size(sc(12.0) as u16),
            Space::with_width(Length::Fixed(sc(8.0))),
            text("? Unknown").size(sc(12.0) as u16),
            Space::with_width(Length::Fixed(sc(12.0))),
            text("O Non-bedrock").size(sc(12.0) as u16),
            Space::with_width(Length::Fixed(sc(12.0))),
            text("X Bedrock").size(sc(12.0) as u16),
            Space::with_width(Length::Fixed(sc(16.0))),
            button(text("Set ? -> O for current layer").size(sc(12.0) as u16))
                .on_press(Message::FillUnknownNonBedrockLayer)
                .style(theme::Button::Secondary)
                .padding([sc(4.0) as u16, sc(10.0) as u16]),
            Space::with_width(Length::Fixed(sc(8.0))),
            button(text("Clear grid").size(sc(12.0) as u16))
                .on_press(Message::ClearGrid)
                .style(theme::Button::Secondary)
                .padding([sc(4.0) as u16, sc(10.0) as u16]),
        ].align_items(Alignment::Center);

        let all_rotations_row = row![
            checkbox(
                "Search all 4 rotations (if north direction is unknown)",
                self.search_all_rotations,
            ).on_toggle(Message::ToggleAllRotations).text_size(sc(13.0) as u16),
        ].align_items(Alignment::Center);

        let search_btn = if is_searching {
            button(text("Searching...").size(sc(16.0) as u16)).padding([sc(10.0) as u16, sc(28.0) as u16])
        } else {
            button(text("Search").size(sc(16.0) as u16)).on_press(Message::Search).padding([sc(10.0) as u16, sc(28.0) as u16])
        };
        let cancel_btn = if is_searching {
            button(text("Cancel").size(sc(16.0) as u16)).on_press(Message::Cancel).padding([sc(10.0) as u16, sc(20.0) as u16])
        } else {
            button(text("Cancel").size(sc(16.0) as u16)).padding([sc(10.0) as u16, sc(20.0) as u16])
        };

        let status_msg = match &self.status {
            SearchStatus::Idle              => text("Ready when you are.").size(sc(16.0) as u16),
            SearchStatus::Searching(area)   => text(format!("Searching {}...", area)).size(sc(16.0) as u16),
            SearchStatus::Cancelled(secs)   => text(format!("Search cancelled after {:.1}s.", secs)).size(sc(16.0) as u16),
            SearchStatus::Found(x, z, secs) => text(format!("Found at X: {}   Z: {}   ({:.1}s)", x, z, secs)).size(sc(18.0) as u16),
            SearchStatus::Error(e)          => text(format!("Error: {}", e)).size(sc(16.0) as u16),
        };

        let zoom_row = row![
            text(format!("Zoom: {:.0}%", self.ui_scale * 100.0)).size(sc(12.0) as u16),
            Space::with_width(Length::Fixed(sc(8.0))),
            button(text("−").size(sc(14.0) as u16))
                .on_press(Message::ZoomOut)
                .style(theme::Button::Secondary)
                .padding([sc(3.0) as u16, sc(10.0) as u16]),
            button(text("+").size(sc(14.0) as u16))
                .on_press(Message::ZoomIn)
                .style(theme::Button::Secondary)
                .padding([sc(3.0) as u16, sc(10.0) as u16]),
        ].spacing(sc(4.0) as u16).align_items(Alignment::Center);

        let content = Column::new()
            .spacing(sc(2.0) as u16)
            .padding(sc(16.0) as u16)
            .max_width(sc(760.0))
            .push(
                row![
                    text("Bedrock Formation Finder").size(sc(26.0) as u16),
                    Space::with_width(Length::Fill),
                    zoom_row,
                ].align_items(Alignment::Center)
            )
            .push(Space::with_height(Length::Fixed(sc(4.0))))
            .push(horizontal_rule(1))
            .push(Space::with_height(Length::Fixed(sc(6.0))))
            .push(seed_row)
            .push(center_row)
            .push(type_row)
            .push(Space::with_height(Length::Fixed(sc(6.0))))
            .push(horizontal_rule(1))
            .push(Space::with_height(Length::Fixed(sc(6.0))))
            .push(grid_controls)
            .push(Space::with_height(Length::Fixed(sc(6.0))))
            .push(y_row)
            .push(Space::with_height(Length::Fixed(sc(4.0))))
            .push(grid_col)
            .push(Space::with_height(Length::Fixed(sc(4.0))))
            .push(rotate_row)
            .push(Space::with_height(Length::Fixed(sc(4.0))))
            .push(legend)
            .push(Space::with_height(Length::Fixed(sc(6.0))))
            .push(all_rotations_row)
            .push(Space::with_height(Length::Fixed(sc(6.0))))
            .push(
                container(row![search_btn, cancel_btn].spacing(16).align_items(Alignment::Center))
                    .width(Length::Fill)
                    .center_x()
            )
            .push(Space::with_height(Length::Fixed(sc(8.0))))
            .push(container(status_msg).width(Length::Fill).padding([sc(8.0) as u16, sc(14.0) as u16]));

        container(scrollable(content)).width(Length::Fill).height(Length::Fill).center_x().into()
    }
}
