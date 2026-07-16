//! The application shell (the "Environment"): owns the backend, camera, interaction
//! state, and the whole interactive canvas. All drawing is hand-rolled so pointer feel
//! — hover, snapping, dragging, wiring — is precise and smooth.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use eframe::egui::{
    self, Align, Button, CursorIcon, Key, Layout, PointerButton, Pos2, Rect, Response, RichText,
    Sense, Stroke, StrokeKind, Vec2,
};

use llmc::backend::{CircuitManager, Simulation};
use llmc::io::{AppConfig, CameraState, Project};
use llmc::model::{
    Block, BlockId, BlockType, ChipDef, ChipId, Circuit, ConnId, Connection, Orientation, Port,
    PortKind, Pos, Vec2f,
};

use super::glyphs::{draw_block, BlockStyle};
use super::theme::{apply_style, Theme};

const MIN_ZOOM: f32 = 7.0;
const MAX_ZOOM: f32 = 130.0;
const DEFAULT_ZOOM: f32 = 26.0;

#[derive(Clone, Copy)]
struct Camera {
    pan: Vec2,
    zoom: f32,
}

impl Camera {
    fn to_screen(self, w: Vec2f, origin: Pos2) -> Pos2 {
        origin + self.pan + Vec2::new(w.x * self.zoom, w.y * self.zoom)
    }

    fn to_world(self, s: Pos2, origin: Pos2) -> Vec2f {
        let d = s - origin - self.pan;
        Vec2f::new(d.x / self.zoom, d.y / self.zoom)
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Tool {
    Select,
    Place(BlockType),
}

enum Interaction {
    Idle,
    Panning,
    BoxSelect {
        start_screen: Pos2,
    },
    DragBlocks {
        grab: Vec2f,
        offset: Pos,
        original: Vec<(BlockId, Pos)>,
        moved: bool,
    },
    DrawWire {
        from: Port,
        from_output: bool,
    },
}

struct Clipboard {
    blocks: Vec<Block>,
    connections: Vec<Connection>,
}

/// Active "edit this chip's internals" session. The working circuit is swapped for the
/// chip's, and the previous circuit/path are stashed to restore on save or cancel.
struct ChipEdit {
    id: ChipId,
    prev_circuit: Circuit,
    prev_path: Option<PathBuf>,
}

pub struct LlmcApp {
    manager: CircuitManager,
    sim: Simulation,
    sim_dirty: bool,
    running: bool,
    config: AppConfig,
    dark: bool,
    camera: Camera,
    tool: Tool,
    interaction: Interaction,
    selection: BTreeSet<BlockId>,
    selected_conns: BTreeSet<ConnId>,
    path: Option<PathBuf>,
    clipboard: Option<Clipboard>,
    status: String,
    show_chip_dialog: bool,
    chip_name: String,
    space_down: bool,
    /// A click-to-wire in progress: click a port, then click the target port.
    pending_wire: Option<(Port, bool)>,
    /// World position where the last context menu was opened (for paste-at-cursor).
    menu_world: Vec2f,
    /// Chip being renamed, with its editable name buffer.
    rename_chip: Option<(ChipId, String)>,
    /// Active chip-internals editing session.
    chip_edit: Option<ChipEdit>,
    /// Block whose Properties window is open, plus its editable buffers.
    props_for: Option<BlockId>,
    props_name: String,
    props_color: [u8; 3],
    props_has_color: bool,
    props_hz: f32,
}

impl LlmcApp {
    pub fn new(
        cc: &eframe::CreationContext<'_>,
        config: AppConfig,
        initial: Option<PathBuf>,
    ) -> Self {
        let dark = config.dark_mode;
        apply_style(&cc.egui_ctx, dark);
        let manager = CircuitManager::new();
        let sim = Simulation::build(&manager.circuit, &manager.chips);
        let mut app = Self {
            manager,
            sim,
            sim_dirty: false,
            running: false,
            dark,
            camera: Camera {
                pan: Vec2::new(140.0, 90.0),
                zoom: DEFAULT_ZOOM,
            },
            tool: Tool::Select,
            interaction: Interaction::Idle,
            selection: BTreeSet::new(),
            selected_conns: BTreeSet::new(),
            path: None,
            clipboard: None,
            status: "Ready".to_string(),
            show_chip_dialog: false,
            chip_name: String::new(),
            space_down: false,
            pending_wire: None,
            menu_world: Vec2f::ZERO,
            rename_chip: None,
            chip_edit: None,
            props_for: None,
            props_name: String::new(),
            props_color: [0x5b, 0x9d, 0xf9],
            props_has_color: false,
            props_hz: 1.0,
            config,
        };
        if let Some(path) = initial {
            app.load_path(&path);
        }
        app
    }

    fn theme(&self) -> Theme {
        if self.dark {
            Theme::dark()
        } else {
            Theme::light()
        }
    }

    fn rebuild_sim(&mut self) {
        self.sim = Simulation::build(&self.manager.circuit, &self.manager.chips);
        // Settle once so combinational state (lit LEDs/wires) shows even when paused.
        self.sim.step(0.0);
        self.sim_dirty = false;
    }

    // ----- editing operations -----

    fn undo(&mut self) {
        if self.manager.undo() {
            self.after_structural_edit();
        }
    }

    fn redo(&mut self) {
        if self.manager.redo() {
            self.after_structural_edit();
        }
    }

    fn after_structural_edit(&mut self) {
        self.sim_dirty = true;
        let c = &self.manager.circuit;
        self.selection.retain(|id| c.block(*id).is_some());
        self.selected_conns.retain(|id| c.connection(*id).is_some());
    }

    fn delete_selection(&mut self) {
        let blocks: Vec<_> = self.selection.iter().copied().collect();
        let conns: Vec<_> = self.selected_conns.iter().copied().collect();
        if blocks.is_empty() && conns.is_empty() {
            return;
        }
        self.manager.delete(&blocks, &conns);
        self.selection.clear();
        self.selected_conns.clear();
        self.sim_dirty = true;
    }

    fn duplicate_selection(&mut self) {
        let ids: Vec<_> = self.selection.iter().copied().collect();
        if ids.is_empty() {
            return;
        }
        let new_ids = self.manager.duplicate(&ids, Pos::new(2, 2));
        self.selection = new_ids.into_iter().collect();
        self.selected_conns.clear();
        self.sim_dirty = true;
    }

    fn copy_selection(&mut self) {
        let ids: BTreeSet<_> = self.selection.iter().copied().collect();
        let blocks: Vec<Block> = ids
            .iter()
            .filter_map(|id| self.manager.circuit.block(*id).cloned())
            .collect();
        if blocks.is_empty() {
            return;
        }
        let connections: Vec<Connection> = self
            .manager
            .circuit
            .iter_connections()
            .filter(|c| ids.contains(&c.from.block) && ids.contains(&c.to.block))
            .cloned()
            .collect();
        self.clipboard = Some(Clipboard {
            blocks,
            connections,
        });
    }

    fn paste(&mut self) {
        let Some(clip) = &self.clipboard else {
            return;
        };
        let blocks = clip.blocks.clone();
        let connections = clip.connections.clone();
        let new_ids = self
            .manager
            .insert_group(&blocks, &connections, Pos::new(2, 2));
        self.selection = new_ids.into_iter().collect();
        self.selected_conns.clear();
        self.sim_dirty = true;
    }

    /// Paste the clipboard so its top-left lands near `world` (used by the context menu).
    fn paste_at(&mut self, world: Vec2f) {
        let Some(clip) = &self.clipboard else {
            return;
        };
        let blocks = clip.blocks.clone();
        let connections = clip.connections.clone();
        let min_x = blocks.iter().map(|b| b.pos.x).min().unwrap_or(0);
        let min_y = blocks.iter().map(|b| b.pos.y).min().unwrap_or(0);
        let offset = Pos::new(
            world.x.round() as i32 - min_x,
            world.y.round() as i32 - min_y,
        );
        let new_ids = self.manager.insert_group(&blocks, &connections, offset);
        self.selection = new_ids.into_iter().collect();
        self.selected_conns.clear();
        self.sim_dirty = true;
    }

    // ----- chip management -----

    fn delete_chip(&mut self, id: ChipId) {
        // Remove any instances of this chip in the working circuit first.
        let instances: Vec<BlockId> = self
            .manager
            .circuit
            .iter_blocks()
            .filter(|b| matches!(b.ty, BlockType::Chip(cid) if cid == id))
            .map(|b| b.id)
            .collect();
        if !instances.is_empty() {
            self.manager.delete(&instances, &[]);
        }
        let name = self
            .manager
            .chips
            .get(id)
            .map(|c| c.name.clone())
            .unwrap_or_default();
        self.manager.chips.remove(id);
        if matches!(self.tool, Tool::Place(BlockType::Chip(cid)) if cid == id) {
            self.tool = Tool::Select;
        }
        self.after_structural_edit();
        self.status = format!("Deleted chip '{name}'");
    }

    /// Open a chip's internal circuit for editing (stashing the current circuit).
    fn begin_chip_edit(&mut self, id: ChipId) {
        if self.chip_edit.is_some() {
            return;
        }
        let Some(def) = self.manager.chips.get(id) else {
            return;
        };
        let name = def.name.clone();
        let circuit = def.circuit.clone();
        let prev_circuit = std::mem::replace(&mut self.manager.circuit, circuit);
        self.manager.undo.clear();
        self.chip_edit = Some(ChipEdit {
            id,
            prev_circuit,
            prev_path: self.path.take(),
        });
        self.tool = Tool::Select;
        self.selection.clear();
        self.selected_conns.clear();
        self.interaction = Interaction::Idle;
        self.pending_wire = None;
        self.sim_dirty = true;
        self.status = format!("Editing chip '{name}' — Update or Cancel in the toolbar");
    }

    fn save_chip_edit(&mut self) {
        let Some(edit) = self.chip_edit.take() else {
            return;
        };
        let name = self
            .manager
            .chips
            .get(edit.id)
            .map(|c| c.name.clone())
            .unwrap_or_else(|| "chip".to_string());
        let def = ChipDef::from_circuit(edit.id, name.clone(), self.manager.circuit.clone());
        self.manager.chips.insert(def);
        self.manager.circuit = edit.prev_circuit;
        self.path = edit.prev_path;
        self.manager.undo.clear();
        self.selection.clear();
        self.selected_conns.clear();
        self.sim_dirty = true;
        self.status = format!("Updated chip '{name}'");
    }

    fn cancel_chip_edit(&mut self) {
        let Some(edit) = self.chip_edit.take() else {
            return;
        };
        self.manager.circuit = edit.prev_circuit;
        self.path = edit.prev_path;
        self.manager.undo.clear();
        self.selection.clear();
        self.selected_conns.clear();
        self.sim_dirty = true;
        self.status = "Cancelled chip edit".to_string();
    }

    // ----- block properties -----

    /// Open the Properties window for the single selected block, loading its buffers.
    fn open_properties(&mut self) {
        if self.selection.len() != 1 {
            return;
        }
        let Some(&id) = self.selection.iter().next() else {
            return;
        };
        if let Some(b) = self.manager.circuit.block(id) {
            self.props_name = b.label.clone().unwrap_or_default();
            self.props_has_color = b.color.is_some();
            self.props_color = b.color.unwrap_or([0x5b, 0x9d, 0xf9]);
            self.props_hz = b.freq_hz.unwrap_or(1.0);
            self.props_for = Some(id);
        }
    }

    fn properties_window(&mut self, ctx: &egui::Context) {
        let Some(id) = self.props_for else {
            return;
        };
        let Some(block) = self.manager.circuit.block(id) else {
            self.props_for = None;
            return;
        };
        let ty = block.ty;
        let is_clock = matches!(ty, BlockType::Clock);
        let type_label = match ty {
            BlockType::Chip(cid) => self
                .manager
                .chips
                .get(cid)
                .map(|c| c.name.clone())
                .unwrap_or_else(|| "Chip".to_string()),
            other => palette_name(other).to_string(),
        };

        let mut open = true;
        egui::Window::new("Properties")
            .collapsible(false)
            .resizable(false)
            .open(&mut open)
            .show(ctx, |ui| {
                ui.label(
                    RichText::new(format!("Type:  {type_label}")).color(self.theme().label_dim),
                );
                ui.separator();
                ui.horizontal(|ui| {
                    ui.label("Name:");
                    ui.text_edit_singleline(&mut self.props_name);
                });
                ui.horizontal(|ui| {
                    ui.checkbox(&mut self.props_has_color, "Custom color");
                    ui.add_enabled_ui(self.props_has_color, |ui| {
                        ui.color_edit_button_srgb(&mut self.props_color);
                    });
                });
                if is_clock {
                    ui.horizontal(|ui| {
                        ui.label("Frequency:");
                        ui.add(
                            egui::Slider::new(&mut self.props_hz, 0.25..=20.0)
                                .suffix(" Hz")
                                .logarithmic(true),
                        );
                    });
                }
            });

        // Apply the edited buffers back to the block every frame.
        if let Some(b) = self.manager.circuit.block_mut(id) {
            let name = self.props_name.trim();
            b.label = if name.is_empty() {
                None
            } else {
                Some(name.to_string())
            };
            b.color = if self.props_has_color {
                Some(self.props_color)
            } else {
                None
            };
            if is_clock {
                let new = Some(self.props_hz);
                if b.freq_hz != new {
                    b.freq_hz = new;
                    self.sim_dirty = true;
                }
            }
        }
        if !open {
            self.props_for = None;
        }
    }

    fn select_all(&mut self) {
        self.selection = self.manager.circuit.iter_blocks().map(|b| b.id).collect();
    }

    fn rotate_selection(&mut self, cw: bool) {
        let items: Vec<(BlockId, Orientation)> = self
            .selection
            .iter()
            .filter_map(|&id| {
                self.manager.circuit.block(id).map(|b| {
                    let o = if cw {
                        b.orientation.cw()
                    } else {
                        b.orientation.ccw()
                    };
                    (id, o)
                })
            })
            .collect();
        if !items.is_empty() {
            self.manager.set_orientations(&items);
        }
    }

    fn flip_selection(&mut self) {
        let items: Vec<(BlockId, Orientation)> = self
            .selection
            .iter()
            .filter_map(|&id| {
                self.manager
                    .circuit
                    .block(id)
                    .map(|b| (id, b.orientation.flip()))
            })
            .collect();
        if !items.is_empty() {
            self.manager.set_orientations(&items);
        }
    }

    // ----- file operations -----

    fn new_circuit(&mut self) {
        self.manager.circuit = Circuit::new("untitled");
        self.manager.undo.clear();
        self.selection.clear();
        self.selected_conns.clear();
        self.path = None;
        self.interaction = Interaction::Idle;
        self.sim_dirty = true;
        self.status = "New circuit".to_string();
    }

    fn save(&mut self) {
        if let Some(p) = self.path.clone() {
            self.write_project(&p);
        } else {
            self.save_as();
        }
    }

    fn save_as(&mut self) {
        let mut dlg = rfd::FileDialog::new()
            .add_filter("LLMC circuit", &["llmc"])
            .set_file_name("circuit.llmc");
        if let Some(dir) = &self.config.last_dir {
            dlg = dlg.set_directory(dir);
        }
        if let Some(path) = dlg.save_file() {
            self.remember_dir(&path);
            self.path = Some(path.clone());
            self.write_project(&path);
        }
    }

    fn write_project(&mut self, path: &Path) {
        let camera = CameraState {
            pan_x: self.camera.pan.x,
            pan_y: self.camera.pan.y,
            zoom: self.camera.zoom,
        };
        let project = Project::new(
            self.manager.circuit.clone(),
            self.manager.chips.clone(),
            camera,
        );
        match project.save(path) {
            Ok(()) => self.status = format!("Saved {}", path.display()),
            Err(e) => self.status = format!("Save failed: {e}"),
        }
    }

    fn open(&mut self) {
        let mut dlg = rfd::FileDialog::new().add_filter("LLMC circuit", &["llmc"]);
        if let Some(dir) = &self.config.last_dir {
            dlg = dlg.set_directory(dir);
        }
        if let Some(path) = dlg.pick_file() {
            self.load_path(&path);
        }
    }

    /// Load a project from a known path (used by the Open dialog and the CLI argument).
    fn load_path(&mut self, path: &Path) {
        match Project::load(path) {
            Ok(p) => {
                self.remember_dir(path);
                self.manager = CircuitManager::from_parts(p.circuit, p.chips);
                self.camera = Camera {
                    pan: Vec2::new(p.camera.pan_x, p.camera.pan_y),
                    zoom: p.camera.zoom.clamp(MIN_ZOOM, MAX_ZOOM),
                };
                self.path = Some(path.to_path_buf());
                self.selection.clear();
                self.selected_conns.clear();
                self.interaction = Interaction::Idle;
                self.sim_dirty = true;
                self.status = format!("Opened {}", path.display());
            }
            Err(e) => self.status = format!("Open failed: {e}"),
        }
    }

    fn remember_dir(&mut self, path: &Path) {
        if let Some(parent) = path.parent() {
            self.config.last_dir = Some(parent.to_path_buf());
            self.config.save();
        }
    }

    fn toggle_theme(&mut self, ctx: &egui::Context) {
        self.dark = !self.dark;
        self.config.dark_mode = self.dark;
        apply_style(ctx, self.dark);
        self.config.save();
    }

    // ----- keyboard -----

    fn handle_shortcuts(&mut self, ctx: &egui::Context) {
        // While the chip dialog is open its text field owns the keyboard.
        if self.show_chip_dialog {
            self.space_down = false;
            return;
        }
        let s = ctx.input(|i| Keys {
            del: i.key_pressed(Key::Delete) || i.key_pressed(Key::Backspace),
            cmd: i.modifiers.command,
            shift: i.modifiers.shift,
            z: i.key_pressed(Key::Z),
            y: i.key_pressed(Key::Y),
            d: i.key_pressed(Key::D),
            c: i.key_pressed(Key::C),
            v: i.key_pressed(Key::V),
            a: i.key_pressed(Key::A),
            r: i.key_pressed(Key::R),
            f: i.key_pressed(Key::F),
            esc: i.key_pressed(Key::Escape),
            space: i.key_down(Key::Space),
        });
        self.space_down = s.space;

        if s.del {
            self.delete_selection();
        }
        if s.cmd && s.z && !s.shift {
            self.undo();
        }
        if s.cmd && (s.y || (s.z && s.shift)) {
            self.redo();
        }
        if s.cmd && s.d {
            self.duplicate_selection();
        }
        if s.cmd && s.c {
            self.copy_selection();
        }
        if s.cmd && s.v {
            self.paste();
        }
        if s.cmd && s.a {
            self.select_all();
        }
        if s.r && !s.cmd {
            self.rotate_selection(!s.shift);
        }
        if s.f && !s.cmd {
            self.flip_selection();
        }
        if s.esc {
            self.tool = Tool::Select;
            self.interaction = Interaction::Idle;
            self.pending_wire = None;
            self.selection.clear();
            self.selected_conns.clear();
        }
    }

    // ----- panels -----

    fn top_toolbar(&mut self, ui: &mut egui::Ui) {
        egui::Panel::top("toolbar").show(ui, |ui| {
            ui.add_space(3.0);
            ui.horizontal(|ui| {
                ui.label(RichText::new("LLMC").strong().size(16.0));
                ui.separator();
                let editing_chip = self.chip_edit.is_some();
                ui.add_enabled_ui(!editing_chip, |ui| {
                    if ui.button("New").clicked() {
                        self.new_circuit();
                    }
                    if ui.button("Open").clicked() {
                        self.open();
                    }
                    if ui.button("Save").clicked() {
                        self.save();
                    }
                    if ui.button("Save As").clicked() {
                        self.save_as();
                    }
                });
                ui.separator();
                let run = if self.running {
                    "\u{23f8}  Stop"
                } else {
                    "\u{25b6}  Run"
                };
                if ui.button(run).clicked() {
                    self.running = !self.running;
                    if self.running {
                        self.sim_dirty = true;
                    }
                }
                if ui.button("Step").clicked() {
                    if self.sim_dirty {
                        self.rebuild_sim();
                    }
                    self.sim.step(0.05);
                }
                ui.separator();
                if ui
                    .add_enabled(self.manager.undo.can_undo(), Button::new("Undo"))
                    .clicked()
                {
                    self.undo();
                }
                if ui
                    .add_enabled(self.manager.undo.can_redo(), Button::new("Redo"))
                    .clicked()
                {
                    self.redo();
                }
                ui.separator();
                if self.chip_edit.is_some() {
                    ui.colored_label(self.theme().accent, "\u{270e} Editing chip");
                    if ui.button("Update chip").clicked() {
                        self.save_chip_edit();
                    }
                    if ui.button("Cancel").clicked() {
                        self.cancel_chip_edit();
                    }
                } else if ui.button("Create Chip").clicked() {
                    self.show_chip_dialog = true;
                }
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    let label = if self.dark { "Light" } else { "Dark" };
                    if ui.button(label).clicked() {
                        let ctx = ui.ctx().clone();
                        self.toggle_theme(&ctx);
                    }
                    if self.sim.has_conflicts() {
                        ui.colored_label(self.theme().conflict, "\u{26a0} multi-driver");
                    }
                });
            });
            ui.add_space(3.0);
        });
    }

    fn left_palette(&mut self, ui: &mut egui::Ui) {
        egui::Panel::left("palette")
            .resizable(false)
            .exact_size(152.0)
            .show(ui, |ui| {
                ui.add_space(6.0);
                if ui
                    .selectable_label(matches!(self.tool, Tool::Select), "  Select")
                    .clicked()
                {
                    self.tool = Tool::Select;
                }
                ui.separator();
                self.palette_section(
                    ui,
                    "INPUTS",
                    &[
                        BlockType::Switch,
                        BlockType::Button,
                        BlockType::ConstantHigh,
                        BlockType::ConstantLow,
                        BlockType::Clock,
                    ],
                );
                self.palette_section(ui, "OUTPUT", &[BlockType::Led]);
                self.palette_section(
                    ui,
                    "GATES",
                    &[
                        BlockType::And,
                        BlockType::Or,
                        BlockType::Not,
                        BlockType::Nand,
                        BlockType::Nor,
                        BlockType::Xor,
                        BlockType::Xnor,
                        BlockType::Buffer,
                    ],
                );

                let chips: Vec<(llmc::model::ChipId, String)> = self
                    .manager
                    .chips
                    .iter()
                    .map(|c| (c.id, c.name.clone()))
                    .collect();
                if !chips.is_empty() {
                    ui.add_space(10.0);
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("CHIPS").small().color(self.theme().label_dim));
                        ui.label(
                            RichText::new("(right-click)")
                                .small()
                                .color(self.theme().label_dim),
                        );
                    });
                    for (id, name) in chips {
                        let selected =
                            matches!(self.tool, Tool::Place(BlockType::Chip(cid)) if cid == id);
                        let resp = ui.selectable_label(selected, format!("  {name}"));
                        if resp.clicked() {
                            self.tool = Tool::Place(BlockType::Chip(id));
                        }
                        resp.context_menu(|ui| {
                            if ui.button("Place").clicked() {
                                self.tool = Tool::Place(BlockType::Chip(id));
                                ui.close();
                            }
                            if ui.button("Edit\u{2026}").clicked() {
                                self.begin_chip_edit(id);
                                ui.close();
                            }
                            if ui.button("Rename\u{2026}").clicked() {
                                self.rename_chip = Some((id, name.clone()));
                                ui.close();
                            }
                            ui.separator();
                            if ui.button("Delete").clicked() {
                                self.delete_chip(id);
                                ui.close();
                            }
                        });
                    }
                }
            });
    }

    fn palette_section(&mut self, ui: &mut egui::Ui, title: &str, items: &[BlockType]) {
        ui.add_space(10.0);
        ui.label(RichText::new(title).small().color(self.theme().label_dim));
        for &ty in items {
            let selected = matches!(self.tool, Tool::Place(t) if t == ty);
            if ui
                .selectable_label(selected, format!("  {}", palette_name(ty)))
                .clicked()
            {
                self.tool = Tool::Place(ty);
            }
        }
    }

    fn status_bar(&mut self, ui: &mut egui::Ui) {
        egui::Panel::bottom("status").show(ui, |ui| {
            ui.add_space(2.0);
            ui.horizontal(|ui| {
                let nb = self.manager.circuit.blocks.len();
                let nc = self.manager.circuit.connections.len();
                ui.label(format!("{nb} blocks \u{00b7} {nc} wires"));
                ui.separator();
                let toolname = if self.chip_edit.is_some() {
                    "Editing chip".to_string()
                } else if let Tool::Place(ty) = self.tool {
                    format!(
                        "Placing {} \u{2014} click to add, right-click/Esc to cancel",
                        palette_name(ty)
                    )
                } else if self.pending_wire.is_some() {
                    "Wiring \u{2014} click a target port, right-click/Esc to cancel".to_string()
                } else {
                    "Select".to_string()
                };
                ui.label(toolname);
                ui.separator();
                ui.label(if self.running { "running" } else { "paused" });
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    ui.label(&self.status);
                });
            });
            ui.add_space(2.0);
        });
    }

    fn chip_dialog(&mut self, ctx: &egui::Context) {
        let mut open = true;
        egui::Window::new("Create chip")
            .collapsible(false)
            .resizable(false)
            .open(&mut open)
            .show(ctx, |ui| {
                ui.label("Package the current circuit as a reusable chip.");
                ui.label(
                    RichText::new("Switches become input pins; LEDs become output pins.")
                        .small()
                        .color(self.theme().label_dim),
                );
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    ui.label("Name:");
                    ui.text_edit_singleline(&mut self.chip_name);
                });
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    let can_create = !self.chip_name.trim().is_empty();
                    if ui.add_enabled(can_create, Button::new("Create")).clicked() {
                        let name = self.chip_name.trim().to_string();
                        self.manager.create_chip_from_active(name.clone());
                        self.chip_name.clear();
                        self.show_chip_dialog = false;
                        self.status = format!("Created chip '{name}'");
                    }
                    if ui.button("Cancel").clicked() {
                        self.show_chip_dialog = false;
                    }
                });
            });
        if !open {
            self.show_chip_dialog = false;
        }
    }

    fn rename_chip_dialog(&mut self, ctx: &egui::Context) {
        let Some((id, mut name)) = self.rename_chip.clone() else {
            return;
        };
        let mut open = true;
        let mut finish = false;
        egui::Window::new("Rename chip")
            .collapsible(false)
            .resizable(false)
            .open(&mut open)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label("Name:");
                    ui.text_edit_singleline(&mut name);
                });
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    let can = !name.trim().is_empty();
                    if ui.add_enabled(can, Button::new("Save")).clicked() {
                        if let Some(chip) = self.manager.chips.chips.get_mut(&id) {
                            chip.name = name.trim().to_string();
                        }
                        self.status = format!("Renamed chip to '{}'", name.trim());
                        finish = true;
                    }
                    if ui.button("Cancel").clicked() {
                        finish = true;
                    }
                });
            });
        if finish || !open {
            self.rename_chip = None;
        } else {
            self.rename_chip = Some((id, name));
        }
    }
}

/// Snapshot of relevant key state read in one `input` call.
struct Keys {
    del: bool,
    cmd: bool,
    shift: bool,
    z: bool,
    y: bool,
    d: bool,
    c: bool,
    v: bool,
    a: bool,
    r: bool,
    f: bool,
    esc: bool,
    space: bool,
}

impl eframe::App for LlmcApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();

        if self.sim_dirty {
            self.rebuild_sim();
        }
        if self.running {
            let dt = ctx.input(|i| i.stable_dt).clamp(0.0, 0.1) as f64;
            self.sim.step(dt);
            ctx.request_repaint();
        }

        self.handle_shortcuts(&ctx);
        self.top_toolbar(ui);
        self.left_palette(ui);
        self.status_bar(ui);

        let bg = self.theme().bg;
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE.fill(bg))
            .show(ui, |ui| {
                self.canvas(ui);
            });

        if self.show_chip_dialog {
            self.chip_dialog(&ctx);
        }
        if self.rename_chip.is_some() {
            self.rename_chip_dialog(&ctx);
        }
        if self.props_for.is_some() {
            self.properties_window(&ctx);
        }
    }
}

// ===================== canvas: geometry, input, drawing =====================

impl LlmcApp {
    fn drag_offset_cells(&self, id: BlockId) -> Pos {
        if let Interaction::DragBlocks {
            offset, original, ..
        } = &self.interaction
        {
            if original.iter().any(|(b, _)| *b == id) {
                return *offset;
            }
        }
        Pos::new(0, 0)
    }

    /// A block's top-left world position, including any live drag offset.
    fn block_base(&self, block: &Block) -> Vec2f {
        let o = self.drag_offset_cells(block.id);
        Vec2f::new(
            block.pos.x as f32 + o.x as f32,
            block.pos.y as f32 + o.y as f32,
        )
    }

    /// A port's world position (drag-aware).
    fn port_world(&self, block: &Block, port: Port) -> Option<Vec2f> {
        let layout = block.ty.layout(&self.manager.chips);
        let locals = match port.kind {
            PortKind::Input => &layout.inputs,
            PortKind::Output => &layout.outputs,
        };
        let local = *locals.get(port.index as usize)?;
        Some(self.block_base(block) + block.orientation.transform(local, layout.size))
    }

    fn port_screen(&self, block: &Block, port: Port, origin: Pos2) -> Option<Pos2> {
        Some(self.camera.to_screen(self.port_world(block, port)?, origin))
    }

    fn block_screen_rect(&self, block: &Block, origin: Pos2) -> Rect {
        let base = self.block_base(block);
        let size = block.footprint(&self.manager.chips);
        Rect::from_min_max(
            self.camera.to_screen(base, origin),
            self.camera.to_screen(base + size, origin),
        )
    }

    /// Outward direction of a port (unit), used to shape wire curves.
    fn port_dir(&self, block: &Block, port: Port) -> Vec2 {
        let size = block.footprint(&self.manager.chips);
        let center = self.block_base(block) + size * 0.5;
        if let Some(pw) = self.port_world(block, port) {
            let d = Vec2::new(pw.x - center.x, pw.y - center.y);
            if d.length() > 1e-3 {
                return d.normalized();
            }
        }
        Vec2::new(1.0, 0.0)
    }

    fn wire_points(&self, conn: &Connection, origin: Pos2) -> Option<Vec<Pos2>> {
        let fb = self.manager.circuit.block(conn.from.block)?;
        let tb = self.manager.circuit.block(conn.to.block)?;
        let p0 = self.port_screen(fb, conn.from, origin)?;
        let p1 = self.port_screen(tb, conn.to, origin)?;
        let d0 = self.port_dir(fb, conn.from);
        let d1 = self.port_dir(tb, conn.to);
        Some(bezier(p0, p1, d0, d1, self.camera.zoom))
    }

    fn hit_port(&self, cursor: Pos2, origin: Pos2) -> Option<(Port, bool)> {
        // Generous pick radius so grabbing a port to start a wire is easy.
        let radius = (self.camera.zoom * 0.5).max(13.0);
        let mut best_d = radius * radius;
        let mut best = None;
        for block in self.manager.circuit.iter_blocks() {
            let layout = block.ty.layout(&self.manager.chips);
            for i in 0..layout.outputs.len() {
                let port = Port::output(block.id, i as u16);
                if let Some(s) = self.port_screen(block, port, origin) {
                    let d = s.distance_sq(cursor);
                    if d < best_d {
                        best_d = d;
                        best = Some((port, true));
                    }
                }
            }
            for i in 0..layout.inputs.len() {
                let port = Port::input(block.id, i as u16);
                if let Some(s) = self.port_screen(block, port, origin) {
                    let d = s.distance_sq(cursor);
                    if d < best_d {
                        best_d = d;
                        best = Some((port, false));
                    }
                }
            }
        }
        best
    }

    fn hit_block(&self, cursor: Pos2, origin: Pos2) -> Option<BlockId> {
        let mut hit = None;
        for block in self.manager.circuit.iter_blocks() {
            if self.block_screen_rect(block, origin).contains(cursor) {
                hit = Some(block.id);
            }
        }
        hit
    }

    fn hit_connection(&self, cursor: Pos2, origin: Pos2) -> Option<ConnId> {
        let thresh = 6.0;
        let mut best_d = thresh * thresh;
        let mut best = None;
        for conn in self.manager.circuit.iter_connections() {
            if let Some(pts) = self.wire_points(conn, origin) {
                for seg in pts.windows(2) {
                    let d = dist_sq_point_seg(cursor, seg[0], seg[1]);
                    if d < best_d {
                        best_d = d;
                        best = Some(conn.id);
                    }
                }
            }
        }
        best
    }

    fn input_high(&self, port: Port) -> bool {
        if let Some(cid) = self.manager.circuit.connection_id_into(port) {
            if let Some(c) = self.manager.circuit.connection(cid) {
                return self.sim.output_value(c.from).unwrap_or(false);
            }
        }
        false
    }

    fn block_on(&self, block: &Block) -> bool {
        match block.ty {
            BlockType::Led => self.sim.led_value(block.id).unwrap_or(false),
            BlockType::Switch | BlockType::Button => block.state,
            _ => self
                .sim
                .output_value(Port::output(block.id, 0))
                .unwrap_or(false),
        }
    }

    fn canvas(&mut self, ui: &mut egui::Ui) {
        let size = ui.available_size();
        let (response, painter) = ui.allocate_painter(size, Sense::click_and_drag());
        let origin = response.rect.min;
        let rect = response.rect;
        let ctx = ui.ctx().clone();
        let cursor = response.hover_pos();

        // Zoom around the cursor.
        if response.hovered() {
            let (scroll, zoomd) = ctx.input(|i| (i.smooth_scroll_delta.y, i.zoom_delta()));
            let anchor = cursor.unwrap_or_else(|| rect.center());
            if scroll.abs() > 0.0 {
                self.apply_zoom(anchor, (scroll * 0.0016).exp(), origin);
            }
            if (zoomd - 1.0).abs() > 1e-3 {
                self.apply_zoom(anchor, zoomd, origin);
            }
        }

        // Eligibility for the right-click menu, captured before input may change the tool.
        let allow_menu = matches!(self.tool, Tool::Select) && self.pending_wire.is_none();
        self.handle_canvas_input(&response, origin, &ctx);

        if let Some(c) = cursor {
            if self.pending_wire.is_some()
                || self.hit_port(c, origin).is_some()
                || matches!(self.tool, Tool::Place(_))
            {
                ctx.set_cursor_icon(CursorIcon::Crosshair);
            } else if self.hit_block(c, origin).is_some() {
                ctx.set_cursor_icon(CursorIcon::Grab);
            }
        }

        let hover_port = cursor.and_then(|c| self.hit_port(c, origin));
        let hover_block = if hover_port.is_some() {
            None
        } else {
            cursor.and_then(|c| self.hit_block(c, origin))
        };

        self.draw_grid(&painter, rect, origin);
        self.draw_wires(&painter, origin);
        self.draw_blocks(&painter, origin, hover_block, hover_port);
        self.draw_overlay(&painter, origin, cursor);

        if allow_menu {
            self.canvas_context_menu(&response);
        }
    }

    fn canvas_context_menu(&mut self, response: &Response) {
        let world = self.menu_world;
        response.context_menu(|ui| {
            let has_sel = !self.selection.is_empty() || !self.selected_conns.is_empty();
            let has_blocks = !self.selection.is_empty();
            let has_clip = self.clipboard.is_some();
            let one_block = self.selection.len() == 1 && self.selected_conns.is_empty();

            if ui
                .add_enabled(one_block, Button::new("Properties\u{2026}"))
                .clicked()
            {
                self.open_properties();
                ui.close();
            }
            ui.separator();

            ui.add_enabled_ui(has_blocks, |ui| {
                if ui.button("Copy").clicked() {
                    self.copy_selection();
                    ui.close();
                }
                if ui.button("Duplicate").clicked() {
                    self.duplicate_selection();
                    ui.close();
                }
            });
            if ui
                .add_enabled(has_clip, Button::new("Paste here"))
                .clicked()
            {
                self.paste_at(world);
                ui.close();
            }
            ui.separator();
            ui.add_enabled_ui(has_blocks, |ui| {
                if ui.button("Rotate").clicked() {
                    self.rotate_selection(true);
                    ui.close();
                }
                if ui.button("Flip").clicked() {
                    self.flip_selection();
                    ui.close();
                }
            });
            if ui.add_enabled(has_sel, Button::new("Delete")).clicked() {
                self.delete_selection();
                ui.close();
            }
            ui.separator();
            if ui.button("Select all").clicked() {
                self.select_all();
                ui.close();
            }
            if ui.add_enabled(has_sel, Button::new("Deselect")).clicked() {
                self.selection.clear();
                self.selected_conns.clear();
                ui.close();
            }
        });
    }

    fn apply_zoom(&mut self, cursor: Pos2, factor: f32, origin: Pos2) {
        let before = self.camera.to_world(cursor, origin);
        let z = (self.camera.zoom * factor).clamp(MIN_ZOOM, MAX_ZOOM);
        self.camera.zoom = z;
        self.camera.pan = Vec2::new(
            cursor.x - origin.x - before.x * z,
            cursor.y - origin.y - before.y * z,
        );
    }

    fn handle_canvas_input(&mut self, response: &Response, origin: Pos2, ctx: &egui::Context) {
        let shift = ctx.input(|i| i.modifiers.shift);
        let pointer = response.interact_pointer_pos();

        // Pan with middle or secondary button anytime.
        if response.dragged_by(PointerButton::Middle)
            || response.dragged_by(PointerButton::Secondary)
        {
            self.camera.pan += response.drag_delta();
        }

        // Right-click (no drag): cancel a held component or click-wire, else prep menu.
        if response.secondary_clicked() {
            let pos = pointer.or_else(|| response.hover_pos());
            if matches!(self.tool, Tool::Place(_)) {
                self.tool = Tool::Select;
            } else if self.pending_wire.is_some() {
                self.pending_wire = None;
            } else if let Some(c) = pos {
                self.menu_world = self.camera.to_world(c, origin);
                if let Some(bid) = self.hit_block(c, origin) {
                    if !self.selection.contains(&bid) {
                        self.selection.clear();
                        self.selected_conns.clear();
                        self.selection.insert(bid);
                    }
                } else if let Some(cid) = self.hit_connection(c, origin) {
                    self.selection.clear();
                    self.selected_conns.clear();
                    self.selected_conns.insert(cid);
                }
            }
        }

        if response.drag_started_by(PointerButton::Primary) {
            if let Some(press) = pointer {
                if self.space_down {
                    self.interaction = Interaction::Panning;
                } else {
                    self.begin_primary(press, origin, shift);
                }
            }
        }
        if response.dragged_by(PointerButton::Primary) {
            if matches!(self.interaction, Interaction::Panning) {
                self.camera.pan += response.drag_delta();
            } else if let Some(p) = pointer {
                self.update_primary(p, origin);
            }
        }
        if response.drag_stopped_by(PointerButton::Primary) {
            self.end_primary(pointer, origin, shift);
        }
        if response.clicked() {
            if let Some(p) = pointer.or_else(|| response.hover_pos()) {
                self.handle_click(p, origin, shift);
            }
        }
    }

    fn begin_primary(&mut self, press: Pos2, origin: Pos2, shift: bool) {
        self.pending_wire = None; // starting a drag cancels any click-to-wire in progress
        if matches!(self.tool, Tool::Place(_)) {
            return; // placement happens on click
        }
        if let Some((port, is_out)) = self.hit_port(press, origin) {
            self.interaction = Interaction::DrawWire {
                from: port,
                from_output: is_out,
            };
            return;
        }
        if let Some(bid) = self.hit_block(press, origin) {
            if !self.selection.contains(&bid) {
                if !shift {
                    self.selection.clear();
                    self.selected_conns.clear();
                }
                self.selection.insert(bid);
            }
            let original: Vec<(BlockId, Pos)> = self
                .selection
                .iter()
                .filter_map(|&id| self.manager.circuit.block(id).map(|b| (id, b.pos)))
                .collect();
            let grab = self.camera.to_world(press, origin);
            self.interaction = Interaction::DragBlocks {
                grab,
                offset: Pos::new(0, 0),
                original,
                moved: false,
            };
            return;
        }
        if !shift {
            self.selection.clear();
            self.selected_conns.clear();
        }
        self.interaction = Interaction::BoxSelect {
            start_screen: press,
        };
    }

    fn update_primary(&mut self, pointer: Pos2, origin: Pos2) {
        let cam = self.camera;
        if let Interaction::DragBlocks {
            grab,
            offset,
            moved,
            ..
        } = &mut self.interaction
        {
            let cur = cam.to_world(pointer, origin);
            let snap = Pos::new(
                (cur.x - grab.x).round() as i32,
                (cur.y - grab.y).round() as i32,
            );
            if snap != *offset {
                *offset = snap;
                if snap.x != 0 || snap.y != 0 {
                    *moved = true;
                }
            }
        }
    }

    fn end_primary(&mut self, pointer: Option<Pos2>, origin: Pos2, shift: bool) {
        let interaction = std::mem::replace(&mut self.interaction, Interaction::Idle);
        match interaction {
            Interaction::DragBlocks {
                original,
                offset,
                moved,
                ..
            } => {
                if moved && (offset.x != 0 || offset.y != 0) {
                    let finals: Vec<(BlockId, Pos)> = original
                        .iter()
                        .map(|(id, op)| (*id, Pos::new(op.x + offset.x, op.y + offset.y)))
                        .collect();
                    self.manager.move_blocks(&finals);
                }
            }
            Interaction::BoxSelect { start_screen } => {
                if let Some(end) = pointer {
                    let box_rect = Rect::from_two_pos(start_screen, end);
                    if !shift {
                        self.selection.clear();
                    }
                    let ids: Vec<BlockId> = self
                        .manager
                        .circuit
                        .iter_blocks()
                        .filter(|b| self.block_screen_rect(b, origin).intersects(box_rect))
                        .map(|b| b.id)
                        .collect();
                    for id in ids {
                        self.selection.insert(id);
                    }
                }
            }
            Interaction::DrawWire { from, from_output } => {
                if let Some(end) = pointer {
                    if let Some((target, to_out)) = self.hit_port(end, origin) {
                        if from_output && !to_out {
                            self.manager.connect(from, target);
                            self.sim_dirty = true;
                        } else if !from_output && to_out {
                            self.manager.connect(target, from);
                            self.sim_dirty = true;
                        }
                    }
                }
            }
            _ => {}
        }
    }

    fn handle_click(&mut self, p: Pos2, origin: Pos2, shift: bool) {
        if let Tool::Place(ty) = self.tool {
            let world = self.camera.to_world(p, origin);
            let sz = ty.layout(&self.manager.chips).size;
            let gp = Pos::new(
                (world.x - sz.x / 2.0).round() as i32,
                (world.y - sz.y / 2.0).round() as i32,
            );
            let id = self.manager.add_block(ty, gp);
            self.sim_dirty = true;
            if !shift {
                self.selection.clear();
                self.selected_conns.clear();
            }
            self.selection.insert(id);
            return;
        }
        // Click-to-wire: click a port, then click a compatible target port. This is an
        // easier alternative to dragging when the drag is fiddly.
        if let Some((port, is_out)) = self.hit_port(p, origin) {
            match self.pending_wire {
                None => self.pending_wire = Some((port, is_out)),
                Some((from, from_out)) => {
                    if from == port {
                        self.pending_wire = None; // clicked the same port again -> cancel
                    } else if from_out && !is_out {
                        self.manager.connect(from, port);
                        self.sim_dirty = true;
                        self.pending_wire = None;
                    } else if !from_out && is_out {
                        self.manager.connect(port, from);
                        self.sim_dirty = true;
                        self.pending_wire = None;
                    } else {
                        // same kind -> restart the wire from the newly clicked port
                        self.pending_wire = Some((port, is_out));
                    }
                }
            }
            return;
        }
        // Clicking away from any port cancels a pending click-wire.
        if self.pending_wire.take().is_some() {
            return;
        }
        if let Some(bid) = self.hit_block(p, origin) {
            let toggle = self.manager.circuit.block(bid).map(|b| {
                (
                    matches!(b.ty, BlockType::Switch | BlockType::Button),
                    b.state,
                )
            });
            if let Some((is_switch, state)) = toggle {
                if is_switch {
                    let ns = !state;
                    if let Some(bm) = self.manager.circuit.block_mut(bid) {
                        bm.state = ns;
                    }
                    self.sim.set_switch(bid, ns);
                }
            }
            if !shift {
                self.selection.clear();
                self.selected_conns.clear();
            }
            self.selection.insert(bid);
            return;
        }
        if let Some(cid) = self.hit_connection(p, origin) {
            if !shift {
                self.selection.clear();
                self.selected_conns.clear();
            }
            self.selected_conns.insert(cid);
            return;
        }
        if !shift {
            self.selection.clear();
            self.selected_conns.clear();
        }
    }

    // ----- drawing -----

    fn draw_grid(&self, painter: &egui::Painter, rect: Rect, origin: Pos2) {
        let t = self.theme();
        let tl = self.camera.to_world(rect.min, origin);
        let br = self.camera.to_world(rect.max, origin);
        let mut step = 1;
        while (self.camera.zoom * step as f32) < 22.0 {
            step *= 2;
        }
        let x0 = (tl.x.floor() as i32).div_euclid(step) * step;
        let y0 = (tl.y.floor() as i32).div_euclid(step) * step;
        let x1 = br.x.ceil() as i32;
        let y1 = br.y.ceil() as i32;
        let mut x = x0;
        while x <= x1 {
            let mut y = y0;
            while y <= y1 {
                let p = self
                    .camera
                    .to_screen(Vec2f::new(x as f32, y as f32), origin);
                let strong = x % (step * 8) == 0 && y % (step * 8) == 0;
                painter.circle_filled(
                    p,
                    if strong { 1.6 } else { 1.0 },
                    if strong { t.grid_strong } else { t.grid },
                );
                y += step;
            }
            x += step;
        }
    }

    fn draw_wires(&self, painter: &egui::Painter, origin: Pos2) {
        let t = self.theme();
        for conn in self.manager.circuit.iter_connections() {
            let Some(pts) = self.wire_points(conn, origin) else {
                continue;
            };
            let high = self.sim.output_value(conn.from).unwrap_or(false);
            let conflict = self.sim.is_conflict(conn.to);
            let selected = self.selected_conns.contains(&conn.id);
            let color = if conflict {
                t.conflict
            } else if selected {
                t.accent
            } else if high {
                t.wire_high
            } else {
                t.wire_low
            };
            let w = if selected {
                3.0
            } else if high {
                2.6
            } else {
                2.0
            };
            let stroke = Stroke::new(w, color);
            for seg in pts.windows(2) {
                painter.line_segment([seg[0], seg[1]], stroke);
            }
        }
    }

    fn draw_blocks(
        &self,
        painter: &egui::Painter,
        origin: Pos2,
        hover_block: Option<BlockId>,
        hover_port: Option<(Port, bool)>,
    ) {
        let t = self.theme();
        for block in self.manager.circuit.iter_blocks() {
            let rect = self.block_screen_rect(block, origin);
            let chip_name = if let BlockType::Chip(id) = block.ty {
                self.manager.chips.get(id).map(|c| c.name.as_str())
            } else {
                None
            };
            let style = BlockStyle {
                rect,
                ty: block.ty,
                on: self.block_on(block),
                selected: self.selection.contains(&block.id),
                hovered: hover_block == Some(block.id),
                running: self.running,
                theme: &t,
                zoom: self.camera.zoom,
                chip_name,
                color: block
                    .color
                    .map(|[r, g, b]| egui::Color32::from_rgb(r, g, b)),
                label: block.label.as_deref(),
            };
            draw_block(painter, &style);

            let layout = block.ty.layout(&self.manager.chips);
            for i in 0..layout.outputs.len() {
                let port = Port::output(block.id, i as u16);
                if let Some(s) = self.port_screen(block, port, origin) {
                    let high = self.sim.output_value(port).unwrap_or(false);
                    self.draw_port(painter, s, true, high, hover_port == Some((port, true)));
                }
            }
            for i in 0..layout.inputs.len() {
                let port = Port::input(block.id, i as u16);
                if let Some(s) = self.port_screen(block, port, origin) {
                    let high = self.input_high(port);
                    self.draw_port(painter, s, false, high, hover_port == Some((port, false)));
                }
            }
        }
    }

    fn draw_port(
        &self,
        painter: &egui::Painter,
        pos: Pos2,
        is_output: bool,
        high: bool,
        hovered: bool,
    ) {
        let t = self.theme();
        let r = (self.camera.zoom * 0.12).clamp(2.5, 5.5);
        let color = if high {
            t.wire_high
        } else if is_output {
            t.port_out
        } else {
            t.port
        };
        painter.circle_filled(pos, r, color);
        if hovered {
            painter.circle_stroke(pos, r + 3.5, Stroke::new(2.0, t.accent));
        }
    }

    fn draw_overlay(&self, painter: &egui::Painter, origin: Pos2, cursor: Option<Pos2>) {
        let t = self.theme();
        match &self.interaction {
            Interaction::BoxSelect { start_screen } => {
                if let Some(c) = cursor {
                    let r = Rect::from_two_pos(*start_screen, c);
                    painter.rect_filled(r, 0.0, t.accent.linear_multiply(0.12));
                    painter.rect_stroke(r, 0.0, Stroke::new(1.0, t.accent), StrokeKind::Inside);
                }
            }
            Interaction::DrawWire { from, from_output } => {
                if let Some(c) = cursor {
                    self.draw_wire_preview(painter, *from, *from_output, c, origin, false);
                }
            }
            _ => {}
        }

        // Click-to-wire in progress (armed from a port).
        if let (Some((from, from_output)), Some(c)) = (self.pending_wire, cursor) {
            self.draw_wire_preview(painter, from, from_output, c, origin, true);
        }

        if let (Tool::Place(ty), Some(c)) = (self.tool, cursor) {
            let world = self.camera.to_world(c, origin);
            let sz = ty.layout(&self.manager.chips).size;
            let tl = Vec2f::new(
                (world.x - sz.x / 2.0).round(),
                (world.y - sz.y / 2.0).round(),
            );
            let r = Rect::from_min_max(
                self.camera.to_screen(tl, origin),
                self.camera.to_screen(tl + sz, origin),
            );
            painter.rect_filled(r, 6.0, t.block_fill.linear_multiply(0.5));
            painter.rect_stroke(r, 6.0, Stroke::new(1.5, t.accent), StrokeKind::Inside);
        }
    }

    fn port_screen_of(&self, port: Port, origin: Pos2) -> Option<Pos2> {
        let block = self.manager.circuit.block(port.block)?;
        self.port_screen(block, port, origin)
    }

    /// Draw a rubber-band wire preview from a source port to the cursor, snapping to a
    /// valid opposite port when hovered. Shared by drag-wiring and click-wiring.
    fn draw_wire_preview(
        &self,
        painter: &egui::Painter,
        from: Port,
        from_output: bool,
        cursor: Pos2,
        origin: Pos2,
        armed: bool,
    ) {
        let t = self.theme();
        let Some(fb) = self.manager.circuit.block(from.block) else {
            return;
        };
        let Some(p0) = self.port_screen(fb, from, origin) else {
            return;
        };
        let (endp, valid) = match self.hit_port(cursor, origin) {
            Some((tp, to_out)) if to_out != from_output => {
                (self.port_screen_of(tp, origin).unwrap_or(cursor), true)
            }
            _ => (cursor, false),
        };
        let d0 = self.port_dir(fb, from);
        let pts = bezier(p0, endp, d0, Vec2::new(-d0.x, -d0.y), self.camera.zoom);
        let color = if valid { t.accent } else { t.port };
        for seg in pts.windows(2) {
            painter.line_segment([seg[0], seg[1]], Stroke::new(2.0, color));
        }
        painter.circle_filled(p0, 4.5, t.accent);
        if armed {
            painter.circle_stroke(p0, 8.0, Stroke::new(1.5, t.accent));
        }
        if valid {
            painter.circle_stroke(endp, 6.0, Stroke::new(2.0, t.accent));
        }
    }
}

fn palette_name(ty: BlockType) -> &'static str {
    match ty {
        BlockType::And => "AND",
        BlockType::Or => "OR",
        BlockType::Not => "NOT",
        BlockType::Nand => "NAND",
        BlockType::Nor => "NOR",
        BlockType::Xor => "XOR",
        BlockType::Xnor => "XNOR",
        BlockType::Buffer => "Buffer",
        BlockType::Switch => "Switch",
        BlockType::Button => "Button",
        BlockType::Led => "LED",
        BlockType::ConstantHigh => "Constant 1",
        BlockType::ConstantLow => "Constant 0",
        BlockType::Clock => "Clock",
        BlockType::Chip(_) => "Chip",
    }
}

fn bezier(p0: Pos2, p1: Pos2, d0: Vec2, d1: Vec2, zoom: f32) -> Vec<Pos2> {
    let dist = (p1 - p0).length();
    let h = (dist * 0.4).clamp(zoom * 0.6, zoom * 6.0);
    let c0 = p0 + d0 * h;
    let c1 = p1 + d1 * h;
    let n = 24;
    let mut v = Vec::with_capacity(n + 1);
    for i in 0..=n {
        let t = i as f32 / n as f32;
        v.push(cubic(p0, c0, c1, p1, t));
    }
    v
}

fn cubic(p0: Pos2, c0: Pos2, c1: Pos2, p1: Pos2, t: f32) -> Pos2 {
    let u = 1.0 - t;
    let (w0, w1, w2, w3) = (u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t);
    Pos2::new(
        w0 * p0.x + w1 * c0.x + w2 * c1.x + w3 * p1.x,
        w0 * p0.y + w1 * c0.y + w2 * c1.y + w3 * p1.y,
    )
}

fn dist_sq_point_seg(p: Pos2, a: Pos2, b: Pos2) -> f32 {
    let ab = b - a;
    let len2 = ab.length_sq();
    if len2 <= 1e-6 {
        return (p - a).length_sq();
    }
    let t = ((p - a).dot(ab) / len2).clamp(0.0, 1.0);
    let proj = a + ab * t;
    (p - proj).length_sq()
}
