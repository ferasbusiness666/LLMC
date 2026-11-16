//! One open project ("document"): its circuit backend, simulation, camera, selection, and all the
//! editing operations the UI and the assistant drive. Every mutation goes through the undoable
//! command layer in `llmc::backend`; the canvas (see `canvas.rs`) only decides *what* to do.

use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeSet, HashSet};
use std::hash::Hasher;
use std::path::{Path, PathBuf};

use eframe::egui::{Pos2, Rect, Vec2};
use serde::{Deserialize, Serialize};

use llmc::ai::{self, ApplyReport, RawOp};
use llmc::backend::{CircuitManager, GroupTransform, Simulation};
use llmc::io::{CameraState, Project};
use llmc::model::{
    Block, BlockId, BlockType, ChipDef, ChipId, Circuit, ConnId, Connection, Port, Pos, Vec2f,
};

use super::assistant::AiSession;

pub const MIN_ZOOM: f32 = 5.0;
pub const MAX_ZOOM: f32 = 140.0;
pub const DEFAULT_ZOOM: f32 = 28.0;

#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub pan: Vec2,
    pub zoom: f32,
}

impl Camera {
    pub fn to_screen(self, w: Vec2f, origin: Pos2) -> Pos2 {
        origin + self.pan + Vec2::new(w.x * self.zoom, w.y * self.zoom)
    }

    pub fn to_world(self, s: Pos2, origin: Pos2) -> Vec2f {
        let d = s - origin - self.pan;
        Vec2f::new(d.x / self.zoom, d.y / self.zoom)
    }
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            pan: Vec2::new(160.0, 110.0),
            zoom: DEFAULT_ZOOM,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Tool {
    Select,
    Place(BlockType),
}

pub enum Interaction {
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

/// A copied selection. Shared across tabs, and mirrored to the system clipboard as JSON so it
/// can be pasted into another LLMC window too. Carries the definitions of any chips it uses, so
/// pasting into a project without them brings them along.
#[derive(Clone, Serialize, Deserialize)]
pub struct Clipboard {
    pub llmc_clipboard: u32,
    pub blocks: Vec<Block>,
    pub connections: Vec<Connection>,
    #[serde(default)]
    pub chips: Vec<ChipDef>,
}

impl Clipboard {
    pub fn to_text(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }

    pub fn from_text(text: &str) -> Option<Self> {
        let c: Clipboard = serde_json::from_str(text.trim()).ok()?;
        (c.llmc_clipboard == 1 && !c.blocks.is_empty()).then_some(c)
    }
}

/// A chip-editing tab: this document holds a working copy of a chip's internals; "Update chip"
/// writes it back into the source project's library.
#[derive(Clone, Copy, Debug)]
pub struct ChipEdit {
    /// [`Document::id`] of the project the chip belongs to.
    pub source_doc: u64,
    pub chip: ChipId,
}

/// Something the user should be told about (shown as a toast by the shell).
#[derive(Clone, Debug)]
pub struct Notice {
    pub text: String,
    pub kind: NoticeKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NoticeKind {
    Info,
    Success,
    Warning,
    Error,
}

/// What the camera should frame on the next canvas pass (it needs the canvas size).
#[derive(Clone, Debug)]
pub enum FrameRequest {
    All,
    Selection,
    /// Only move the camera if these blocks are (partly) off-screen.
    IfOffscreen(Vec<BlockId>),
}

/// Title helper: `adder.llmc` → `adder`.
pub fn title_from_path(path: &Path) -> String {
    path.file_stem()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())
        .unwrap_or("untitled")
        .to_string()
}

pub struct Document {
    /// Stable identity (never reuse an index to refer to a document across frames).
    pub id: u64,
    pub manager: CircuitManager,
    pub sim: Simulation,
    /// The manager revision the simulation was compiled from.
    sim_rev: u64,
    pub running: bool,
    /// Simulation speed multiplier.
    pub speed: f32,
    pub camera: Camera,
    pub tool: Tool,
    pub interaction: Interaction,
    pub selection: BTreeSet<BlockId>,
    pub selected_conns: BTreeSet<ConnId>,
    pub path: Option<PathBuf>,
    saved_fingerprint: u64,
    fingerprint_rev: u64,
    dirty: bool,
    pub title: String,
    pub chip_edit: Option<ChipEdit>,
    pub ai: AiSession,
    pub space_down: bool,
    /// Click-to-wire in progress: click a port, then the target port.
    pub pending_wire: Option<(Port, bool)>,
    /// A momentary Button currently held down.
    pub pressed_button: Option<BlockId>,
    /// World position where the last context menu opened (paste-here).
    pub menu_world: Vec2f,
    /// A palette item being dragged onto the canvas.
    pub palette_drag: Option<BlockType>,
    /// Last known canvas rect and cursor world position (for paste-at-cursor and framing).
    pub canvas_rect: Rect,
    pub cursor_world: Option<Vec2f>,
    pub frame_request: Option<FrameRequest>,
    /// Edit-series counter: property edits with the same series collapse into one undo step.
    pub props_series: u64,
    /// Messages for the shell to show as toasts.
    pub notices: Vec<Notice>,
    /// Successive pastes offset from each other.
    paste_count: i32,
    /// Blocks the assistant just built or pointed at: a fading halo on the canvas (they are no
    /// longer selected, so a stray Backspace can't delete what it built).
    pub glow: Option<Glow>,
    /// Blocks the assistant created in this tab (it may freely change those; the user's own
    /// blocks are protected unless the request asks for changes).
    pub ai_blocks: HashSet<BlockId>,
    /// The "simulation is paused" hint was shown (once per tab).
    paused_hint: bool,
}

/// A temporary highlight drawn around some blocks.
pub struct Glow {
    pub blocks: Vec<BlockId>,
    pub note: Option<String>,
    pub since: std::time::Instant,
    pub duration: f32,
}

impl Document {
    pub fn new(
        manager: CircuitManager,
        path: Option<PathBuf>,
        title: String,
        camera: Camera,
    ) -> Self {
        let sim = Simulation::build(&manager.circuit, &manager.chips);
        let rev = manager.revision();
        let mut doc = Self {
            id: 0,
            manager,
            sim,
            sim_rev: rev,
            running: true,
            speed: 1.0,
            camera,
            tool: Tool::Select,
            interaction: Interaction::Idle,
            selection: BTreeSet::new(),
            selected_conns: BTreeSet::new(),
            path,
            saved_fingerprint: 0,
            fingerprint_rev: u64::MAX,
            dirty: false,
            title,
            chip_edit: None,
            ai: AiSession::default(),
            space_down: false,
            pending_wire: None,
            pressed_button: None,
            menu_world: Vec2f::ZERO,
            palette_drag: None,
            canvas_rect: Rect::from_min_size(Pos2::ZERO, Vec2::new(1000.0, 700.0)),
            cursor_world: None,
            frame_request: None,
            props_series: 1,
            notices: Vec::new(),
            paste_count: 0,
            glow: None,
            ai_blocks: HashSet::new(),
            paused_hint: false,
        };
        // The loaded/empty content is the "saved" baseline. A document opens RUNNING, so a
        // flipped switch lights its LEDs right away (Pause freezes everything).
        doc.saved_fingerprint = doc.content_fingerprint();
        doc.fingerprint_rev = doc.manager.revision();
        doc.sim.step(0.0);
        doc
    }

    pub fn new_empty(title: String) -> Self {
        Self::new(CircuitManager::new(), None, title, Camera::default())
    }

    pub fn from_project(project: Project, path: PathBuf) -> Self {
        let title = title_from_path(&path);
        Self::from_project_at(project, Some(path), title)
    }

    /// A project recovered from an autosave after a crash: it opens with unsaved changes (it's
    /// newer than whatever is on disk).
    pub fn recovered(project: Project, original: Option<PathBuf>, title: String) -> Self {
        let mut doc = Self::from_project_at(project, original, title);
        doc.mark_modified();
        doc
    }

    /// Treat the content as differing from the file on disk.
    pub fn mark_modified(&mut self) {
        self.saved_fingerprint = self.content_fingerprint().wrapping_add(1);
        self.fingerprint_rev = u64::MAX;
    }

    fn from_project_at(project: Project, path: Option<PathBuf>, title: String) -> Self {
        let (mut circuit, mut chips) = (project.circuit, project.chips);
        // Make a damaged or hand-edited file consistent before anything touches it.
        let repairs = llmc::model::repair_project(&mut circuit, &mut chips);
        let manager = CircuitManager::from_parts(circuit, chips);
        let camera = Camera {
            pan: Vec2::new(project.camera.pan_x, project.camera.pan_y),
            zoom: project.camera.zoom.clamp(MIN_ZOOM, MAX_ZOOM),
        };
        let mut doc = Self::new(manager, path, title, camera);
        doc.ai.restore(&project.chat);
        if !repairs.is_empty() {
            doc.notify(
                NoticeKind::Warning,
                format!(
                    "This file had problems that were fixed: {}. Save to keep the repaired version.",
                    repairs.join("; ")
                ),
            );
            // The repaired content differs from the file on disk.
            doc.mark_modified();
        }
        if project.camera.zoom <= 0.0 || !project.camera.pan_x.is_finite() {
            doc.frame_request = Some(FrameRequest::All);
        }
        doc
    }

    /// The project as it would be saved.
    pub fn to_project(&self, save_chat: bool) -> Project {
        let mut p = Project::new(
            self.manager.circuit.clone(),
            self.manager.chips.clone(),
            CameraState {
                pan_x: self.camera.pan.x,
                pan_y: self.camera.pan.y,
                zoom: self.camera.zoom,
            },
        );
        if save_chat {
            p.chat = self.ai.records();
        }
        p
    }

    pub fn notify(&mut self, kind: NoticeKind, text: impl Into<String>) {
        self.notices.push(Notice {
            text: text.into(),
            kind,
        });
    }

    // ----- dirty tracking -----

    /// Fingerprint of the saveable logical content (not the camera nor id counters), so undoing
    /// back to the saved state reads as clean.
    fn content_fingerprint(&self) -> u64 {
        let c = &self.manager.circuit;
        // A chip-editing tab's library is a read-only copy (refreshed when sibling chip tabs
        // update), so only its circuit counts as its content.
        let in_chip_tab = self.chip_edit.is_some();
        let chips: Vec<_> = self
            .manager
            .chips
            .iter()
            .filter(|_| !in_chip_tab)
            .map(|d| {
                (
                    d.id,
                    &d.name,
                    &d.circuit.blocks,
                    &d.circuit.connections,
                    &d.input_blocks,
                    &d.output_blocks,
                )
            })
            .collect();
        let view = (&c.name, &c.blocks, &c.connections, &chips);
        let mut h = DefaultHasher::new();
        match serde_json::to_vec(&view) {
            Ok(bytes) => h.write(&bytes),
            Err(_) => h.write(b"llmc-fingerprint-error"),
        }
        h.finish()
    }

    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Refresh the dirty flag — cheap: only re-hashes when the content actually changed since
    /// the last check (the manager's revision counter moved).
    pub fn refresh_dirty(&mut self) {
        let rev = self.manager.revision();
        if rev != self.fingerprint_rev {
            self.fingerprint_rev = rev;
            self.dirty = self.content_fingerprint() != self.saved_fingerprint;
        }
    }

    pub fn mark_saved(&mut self) {
        self.saved_fingerprint = self.content_fingerprint();
        self.fingerprint_rev = self.manager.revision();
        self.dirty = false;
    }

    // ----- simulation -----

    /// Recompile the simulation if the circuit changed, keeping latch/register/RAM contents.
    /// Paused means frozen: an edit made while paused isn't evaluated until Run or Step.
    pub fn sync_sim(&mut self) {
        if self.sim_rev == self.manager.revision() {
            return;
        }
        let mut fresh = Simulation::build(&self.manager.circuit, &self.manager.chips);
        fresh.carry_state_from(&self.sim);
        if let Some(b) = self.pressed_button {
            fresh.set_switch(b, true);
        }
        self.sim = fresh;
        self.sim_rev = self.manager.revision();
        if self.running {
            self.sim.step(0.0);
        }
    }

    /// Clear all simulated state (memory, clock phase).
    pub fn reset_sim(&mut self) {
        self.sim = Simulation::build(&self.manager.circuit, &self.manager.chips);
        self.sim_rev = self.manager.revision();
        if self.running {
            self.sim.step(0.0);
        }
    }

    pub fn toggle_running(&mut self) {
        self.running = !self.running;
        if self.running {
            self.sync_sim();
            self.sim.step(0.0);
        }
    }

    /// One step: to the next clock edge when there is a clock, else just settle.
    pub fn step_once(&mut self) {
        self.sync_sim();
        match self.sim.time_to_next_edge() {
            Some(dt) => self.sim.step(dt + 1e-9),
            None => self.sim.step(0.0),
        }
    }

    /// Advance the running simulation by a frame's worth of time.
    pub fn tick(&mut self, dt: f64) {
        self.sync_sim();
        if self.running {
            self.sim.step(dt * self.speed as f64);
        }
    }

    /// Set a switch/button's live level (and, for switches, the saved default).
    pub fn set_input(&mut self, id: BlockId, level: bool, persist: bool) {
        // Compile any pending structural edits first, so marking the sim current below never
        // skips them.
        self.sync_sim();
        if persist {
            // Flipping an input is using the circuit, not editing it: a clean file stays clean
            // (the level is still written on the next save).
            self.refresh_dirty();
            let was_clean = !self.dirty;
            if let Some(b) = self.manager.circuit.block_mut(id) {
                b.state = level;
            }
            self.manager.touch();
            // The sim is updated directly; no recompile needed for a level change.
            self.sim_rev = self.manager.revision();
            if was_clean {
                self.mark_saved();
            }
        }
        self.sim.set_switch(id, level);
        if self.running {
            self.sim.step(0.0);
        } else if !self.paused_hint {
            // Otherwise a flipped switch with dark LEDs looks broken.
            self.paused_hint = true;
            self.notify(
                NoticeKind::Info,
                "The simulation is paused \u{2014} press Run (F5) to see the circuit react.",
            );
        }
    }

    // ----- editing -----

    pub fn undo(&mut self) {
        if self.manager.undo() {
            self.after_edit();
        }
    }

    pub fn redo(&mut self) {
        if self.manager.redo() {
            self.after_edit();
        }
    }

    /// Drop selection entries that no longer exist; leave the chip-place tool if its chip went.
    pub fn after_edit(&mut self) {
        let c = &self.manager.circuit;
        self.selection.retain(|id| c.block(*id).is_some());
        self.selected_conns.retain(|id| c.connection(*id).is_some());
        if let Tool::Place(BlockType::Chip(cid)) = self.tool {
            if self.manager.chips.get(cid).is_none() {
                self.tool = Tool::Select;
            }
        }
    }

    pub fn clear_selection(&mut self) {
        self.selection.clear();
        self.selected_conns.clear();
    }

    pub fn select_all(&mut self) {
        self.selection = self.manager.circuit.iter_blocks().map(|b| b.id).collect();
        self.selected_conns = self
            .manager
            .circuit
            .iter_connections()
            .map(|c| c.id)
            .collect();
    }

    pub fn has_selection(&self) -> bool {
        !self.selection.is_empty() || !self.selected_conns.is_empty()
    }

    pub fn delete_selection(&mut self) {
        let blocks: Vec<_> = self.selection.iter().copied().collect();
        let conns: Vec<_> = self.selected_conns.iter().copied().collect();
        if blocks.is_empty() && conns.is_empty() {
            return;
        }
        self.manager.delete(&blocks, &conns);
        self.clear_selection();
    }

    pub fn duplicate_selection(&mut self) {
        let ids: Vec<_> = self.selection.iter().copied().collect();
        if ids.is_empty() {
            return;
        }
        let new_ids = self.manager.duplicate(&ids, Pos::new(2, 2));
        self.selection = new_ids.into_iter().collect();
        self.selected_conns.clear();
    }

    /// Copy the selection (with the chips it uses). Returns `None` when nothing is selected.
    pub fn copy_selection(&self) -> Option<Clipboard> {
        let ids: BTreeSet<_> = self.selection.iter().copied().collect();
        let blocks: Vec<Block> = ids
            .iter()
            .filter_map(|id| self.manager.circuit.block(*id).cloned())
            .collect();
        if blocks.is_empty() {
            return None;
        }
        let connections: Vec<Connection> = self
            .manager
            .circuit
            .iter_connections()
            .filter(|c| ids.contains(&c.from.block) && ids.contains(&c.to.block))
            .cloned()
            .collect();
        let roots: Vec<ChipId> = blocks
            .iter()
            .filter_map(|b| match b.ty {
                BlockType::Chip(c) => Some(c),
                _ => None,
            })
            .collect();
        let chips = self
            .manager
            .chips
            .closure(&roots)
            .into_iter()
            .filter_map(|id| self.manager.chips.get(id).cloned())
            .collect();
        Some(Clipboard {
            llmc_clipboard: 1,
            blocks,
            connections,
            chips,
        })
    }

    /// Paste so the clipboard's top-left lands at `world` (or near the cursor / view center).
    pub fn paste(&mut self, clip: &Clipboard, at: Option<Vec2f>) {
        if clip.blocks.is_empty() {
            return;
        }
        let min_x = clip.blocks.iter().map(|b| b.pos.x).min().unwrap_or(0);
        let min_y = clip.blocks.iter().map(|b| b.pos.y).min().unwrap_or(0);
        let target = at
            .or(self.cursor_world)
            .unwrap_or_else(|| self.view_center());
        // Repeated pastes without moving the mouse cascade instead of stacking.
        let bump = if at.is_none() {
            self.paste_count += 1;
            (self.paste_count - 1) * 2
        } else {
            0
        };
        let offset = Pos::new(
            target.x.round() as i32 - min_x + bump,
            target.y.round() as i32 - min_y + bump,
        );
        // Never paste a chip into its own editing tab (that chip can't contain itself).
        let editing = self.chip_edit.map(|e| e.chip);
        let blocks: Vec<Block> = clip
            .blocks
            .iter()
            .filter(|b| {
                !matches!((b.ty, editing), (BlockType::Chip(c), Some(e)) if c == e || self.manager.chips.contains(c, e))
            })
            .cloned()
            .collect();
        let new_ids = self
            .manager
            .insert_group(&blocks, &clip.connections, offset, &clip.chips);
        self.selection = new_ids.into_iter().collect();
        self.selected_conns.clear();
    }

    /// Note that the cursor moved (resets the paste cascade).
    pub fn cursor_moved(&mut self) {
        self.paste_count = 0;
    }

    pub fn view_center(&self) -> Vec2f {
        self.camera
            .to_world(self.canvas_rect.center(), self.canvas_rect.min)
    }

    /// Rotate (cw/ccw) or flip the selection as one piece.
    pub fn transform_selection(&mut self, kind: GroupTransform) {
        let ids: Vec<_> = self.selection.iter().copied().collect();
        if ids.is_empty() {
            return;
        }
        match kind {
            GroupTransform::RotateCw => self.manager.transform_group(&ids, |o| o.cw(), kind),
            GroupTransform::RotateCcw => self.manager.transform_group(&ids, |o| o.ccw(), kind),
            GroupTransform::Flip => self.manager.transform_group(&ids, |o| o.flip(), kind),
        }
    }

    /// Move the selection by whole cells (arrow keys). Consecutive nudges are one undo step.
    pub fn nudge(&mut self, dx: i32, dy: i32) {
        let moves: Vec<(BlockId, Pos)> = self
            .selection
            .iter()
            .filter_map(|id| self.manager.circuit.block(*id))
            .map(|b| (b.id, Pos::new(b.pos.x + dx, b.pos.y + dy)))
            .collect();
        if moves.is_empty() {
            return;
        }
        let batch = moves
            .iter()
            .map(|&(id, to)| llmc::backend::EditCommand::MoveBlock { id, to })
            .collect();
        self.manager
            .apply_coalesced(batch, 0x6e75_6467_6500 ^ self.props_series);
    }

    /// Commit a property edit for one block (coalesced within the current edit series).
    pub fn update_block(&mut self, block: Block) {
        self.manager.update_block(block, Some(self.props_series));
    }

    /// Start a new edit series (the next property edit is its own undo step).
    pub fn new_series(&mut self) {
        self.props_series = self.props_series.wrapping_add(1);
        self.manager.undo.break_series();
    }

    pub fn add_block_at(&mut self, ty: BlockType, world: Vec2f) -> Option<BlockId> {
        if let (BlockType::Chip(c), Some(e)) = (ty, self.chip_edit) {
            if c == e.chip || self.manager.chips.contains(c, e.chip) {
                self.notify(NoticeKind::Warning, "A chip can't contain itself.");
                return None;
            }
        }
        let sz = ty.layout(&self.manager.chips).size;
        let gp = Pos::new(
            (world.x - sz.x / 2.0).round() as i32,
            (world.y - sz.y / 2.0).round() as i32,
        );
        Some(self.manager.add_block(ty, gp))
    }

    pub fn connect(&mut self, from: Port, to: Port) {
        self.manager.connect(from, to);
    }

    // ----- chips -----

    /// Chips that may be placed here (in a chip-editing tab: not the chip itself, nor anything
    /// that contains it).
    pub fn placeable(&self, id: ChipId) -> bool {
        match self.chip_edit {
            Some(e) => id != e.chip && !self.manager.chips.contains(id, e.chip),
            None => true,
        }
    }

    pub fn chip_name_taken(&self, name: &str, except: Option<ChipId>) -> bool {
        self.manager
            .chips
            .find_by_name(name)
            .is_some_and(|id| Some(id) != except)
    }

    /// Package the whole canvas — or just the selection — as a new chip.
    pub fn create_chip(&mut self, name: &str, from_selection: bool) -> Result<ChipId, String> {
        let name = name.trim();
        if name.is_empty() {
            return Err("Give the chip a name.".into());
        }
        if self.chip_name_taken(name, None) {
            return Err(format!(
                "A chip named \u{201c}{name}\u{201d} already exists."
            ));
        }
        let circuit = if from_selection && !self.selection.is_empty() {
            let mut c = Circuit::new(name);
            for id in &self.selection {
                if let Some(b) = self.manager.circuit.block(*id) {
                    c.insert_block(b.clone());
                }
            }
            for w in self.manager.circuit.iter_connections() {
                if self.selection.contains(&w.from.block) && self.selection.contains(&w.to.block) {
                    c.insert_connection(*w);
                }
            }
            c
        } else {
            self.manager.circuit.clone()
        };
        let has_out = circuit.iter_blocks().any(|b| b.ty == BlockType::Led);
        if !has_out {
            return Err("A chip needs at least one LED (it becomes an output pin).".into());
        }
        let has_in = circuit.iter_blocks().any(|b| b.ty == BlockType::Switch);
        let id = self.manager.chips.allocate_id();
        let def = ChipDef::from_circuit(id, name, circuit);
        self.manager.put_chip(def);
        if has_in {
            self.notify(
                NoticeKind::Success,
                format!(
                    "Created chip \u{201c}{name}\u{201d} \u{2014} find it under Chips in the sidebar"
                ),
            );
        } else {
            self.notify(
                NoticeKind::Warning,
                format!(
                    "Created chip \u{201c}{name}\u{201d}, but it has no input pins (add switches \
                     to make inputs), so its outputs never change."
                ),
            );
        }
        Ok(id)
    }

    pub fn rename_chip(&mut self, id: ChipId, name: &str) -> Result<(), String> {
        let name = name.trim();
        if name.is_empty() {
            return Err("The name can't be empty.".into());
        }
        if self.chip_name_taken(name, Some(id)) {
            return Err(format!(
                "A chip named \u{201c}{name}\u{201d} already exists."
            ));
        }
        if llmc::ai::parse::kind_from_str(name).is_some() {
            return Err(format!("\u{201c}{name}\u{201d} is a built-in block name."));
        }
        self.manager.rename_chip(id, name);
        Ok(())
    }

    /// Chips (by name) that use `id` inside them.
    pub fn chip_users(&self, id: ChipId) -> Vec<String> {
        self.manager
            .chips
            .users_of(id)
            .into_iter()
            .filter_map(|u| self.manager.chips.get(u).map(|d| d.name.clone()))
            .collect()
    }

    pub fn chip_instances(&self, id: ChipId) -> usize {
        self.manager
            .circuit
            .iter_blocks()
            .filter(|b| b.ty == BlockType::Chip(id))
            .count()
    }

    pub fn delete_chip(&mut self, id: ChipId) {
        let name = self
            .manager
            .chips
            .get(id)
            .map(|c| c.name.clone())
            .unwrap_or_default();
        self.manager.delete_chip(id);
        self.after_edit();
        self.notify(
            NoticeKind::Info,
            format!("Deleted chip \u{201c}{name}\u{201d} (Ctrl+Z to undo)"),
        );
    }

    /// Apply an edited chip body back into this (source) project. Wires on placed instances
    /// follow their pins; a body that would contain the chip itself is refused.
    pub fn apply_chip_circuit(&mut self, chip: ChipId, circuit: Circuit) -> Result<String, String> {
        let Some(old) = self.manager.chips.get(chip).cloned() else {
            return Err("The chip was deleted from its project.".into());
        };
        if self.manager.chips.would_cycle(chip, &circuit) {
            return Err(format!("{} can't contain itself.", old.name));
        }
        if !circuit.iter_blocks().any(|b| b.ty == BlockType::Led) {
            return Err("A chip needs at least one LED (output pin).".into());
        }
        let def = old.rederive(circuit);
        let report = self.manager.put_chip(def);
        let mut msg = format!("Updated chip \u{201c}{}\u{201d}", old.name);
        if report.removed > 0 {
            msg.push_str(&format!(
                " \u{2014} {} wire(s) to removed pins were disconnected",
                report.removed
            ));
        }
        Ok(msg)
    }

    // ----- assistant -----

    pub fn system_prompt(&self) -> String {
        let selected: Vec<BlockId> = self.selection.iter().copied().collect();
        ai::system_prompt(&self.manager, &selected)
    }

    /// Apply the assistant's commands (one undo step), make what it built glow, and make sure
    /// it's in view. `allow_changes` says whether the user's request asked to change existing
    /// work; if not, blocks the user placed are protected from deletion.
    pub fn apply_ai_ops(&mut self, ops: Vec<RawOp>, allow_changes: bool) -> ApplyReport {
        let opts = ai::ApplyOptions {
            protected: if allow_changes {
                HashSet::new()
            } else {
                self.manager
                    .circuit
                    .iter_blocks()
                    .map(|b| b.id)
                    .filter(|id| !self.ai_blocks.contains(id))
                    .collect()
            },
        };
        let report = ai::apply_ops_with(&mut self.manager, ops, &opts);
        self.ai_blocks.extend(report.new_blocks.iter().copied());
        if report.changed {
            self.after_edit();
        }
        let (blocks, note, duration) = if !report.highlight.is_empty() {
            (report.highlight.clone(), report.highlight_note.clone(), 9.0)
        } else {
            (report.new_blocks.clone(), None, 3.5)
        };
        if !blocks.is_empty() {
            // What it built is NOT selected (a Backspace meant for the chat must never delete
            // it); it glows for a moment instead.
            self.selection
                .retain(|id| self.manager.circuit.block(*id).is_some());
            self.frame_request = Some(FrameRequest::IfOffscreen(blocks.clone()));
            self.glow = Some(Glow {
                blocks,
                note,
                since: std::time::Instant::now(),
                duration,
            });
        }
        report
    }

    pub fn agent_observation(&self, report: &ApplyReport) -> ai::Observation {
        ai::agent_observation(&self.manager, report)
    }

    // ----- camera -----

    /// World-space bounds of some blocks (or all), as (min, max).
    pub fn bounds_of(&self, ids: Option<&[BlockId]>) -> Option<(Vec2f, Vec2f)> {
        let mut min = Vec2f::new(f32::MAX, f32::MAX);
        let mut max = Vec2f::new(f32::MIN, f32::MIN);
        let mut any = false;
        let mut take = |b: &Block| {
            let sz = b.footprint(&self.manager.chips);
            min.x = min.x.min(b.pos.x as f32);
            min.y = min.y.min(b.pos.y as f32 - 1.0); // room for the caption
            max.x = max.x.max(b.pos.x as f32 + sz.x);
            max.y = max.y.max(b.pos.y as f32 + sz.y);
            any = true;
        };
        match ids {
            Some(ids) => ids
                .iter()
                .filter_map(|id| self.manager.circuit.block(*id))
                .for_each(&mut take),
            None => self.manager.circuit.iter_blocks().for_each(&mut take),
        }
        any.then_some((min, max))
    }

    /// Fit world bounds into the canvas with a margin.
    pub fn frame_bounds(&mut self, min: Vec2f, max: Vec2f, max_zoom: f32) {
        let r = self.canvas_rect;
        let margin = 60.0;
        let w = (max.x - min.x).max(1.0);
        let h = (max.y - min.y).max(1.0);
        let zoom = ((r.width() - margin * 2.0) / w)
            .min((r.height() - margin * 2.0) / h)
            .clamp(MIN_ZOOM, max_zoom);
        let cx = (min.x + max.x) / 2.0;
        let cy = (min.y + max.y) / 2.0;
        self.camera.zoom = zoom;
        self.camera.pan = Vec2::new(r.width() / 2.0 - cx * zoom, r.height() / 2.0 - cy * zoom);
    }

    /// Resolve a pending frame request (called by the canvas once its rect is known).
    pub fn resolve_frame_request(&mut self) {
        let Some(req) = self.frame_request.take() else {
            return;
        };
        let ids: Option<Vec<BlockId>> = match &req {
            FrameRequest::All => None,
            FrameRequest::Selection => Some(self.selection.iter().copied().collect()),
            FrameRequest::IfOffscreen(b) => Some(b.clone()),
        };
        let Some((min, max)) = self.bounds_of(ids.as_deref()) else {
            return;
        };
        if let FrameRequest::IfOffscreen(_) = req {
            let tl = self.camera.to_screen(min, self.canvas_rect.min);
            let br = self.camera.to_screen(max, self.canvas_rect.min);
            if self.canvas_rect.contains(tl) && self.canvas_rect.contains(br) {
                return;
            }
        }
        self.frame_bounds(min, max, DEFAULT_ZOOM * 1.4);
    }

    /// Zoom around a screen point.
    pub fn zoom_at(&mut self, anchor: Pos2, factor: f32) {
        let origin = self.canvas_rect.min;
        let before = self.camera.to_world(anchor, origin);
        let z = (self.camera.zoom * factor).clamp(MIN_ZOOM, MAX_ZOOM);
        self.camera.zoom = z;
        self.camera.pan = Vec2::new(
            anchor.x - origin.x - before.x * z,
            anchor.y - origin.y - before.y * z,
        );
    }

    /// Set an absolute zoom around the view center.
    pub fn set_zoom(&mut self, zoom: f32) {
        let c = self.canvas_rect.center();
        self.zoom_at(c, zoom / self.camera.zoom);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llmc::model::{ChipDef, Port};

    fn doc() -> Document {
        Document::new_empty("t".into())
    }

    #[test]
    fn dirty_reflects_real_content_and_undo_back_to_saved_is_clean() {
        let mut d = doc();
        d.refresh_dirty();
        assert!(!d.is_dirty());
        let id = d.manager.add_block(BlockType::And, Pos::new(0, 0));
        d.refresh_dirty();
        assert!(d.is_dirty());
        d.mark_saved();
        d.manager.delete(&[id], &[]);
        d.refresh_dirty();
        assert!(d.is_dirty());
        d.undo();
        d.refresh_dirty();
        assert!(!d.is_dirty(), "undoing back to the saved state is clean");
    }

    #[test]
    fn paused_sim_is_frozen_until_run_and_keeps_state_across_edits() {
        let mut d = doc();
        assert!(d.running, "documents open running");
        d.toggle_running();
        let s = d.manager.add_block(BlockType::Switch, Pos::new(0, 0));
        let l = d.manager.add_block(BlockType::Led, Pos::new(4, 0));
        d.connect(Port::output(s, 0), Port::input(l, 0));
        d.set_input(s, true, true);
        d.tick(0.016);
        assert_eq!(d.sim.led_value(l), Some(false), "paused: nothing evaluated");
        assert!(
            matches!(d.notices.last().map(|n| &n.kind), Some(NoticeKind::Info)),
            "flipping an input while paused explains why nothing reacts"
        );
        d.toggle_running();
        d.tick(0.016);
        assert_eq!(d.sim.led_value(l), Some(true));
        // An unrelated edit while running keeps everything live.
        d.manager.add_block(BlockType::Not, Pos::new(10, 10));
        d.tick(0.016);
        assert_eq!(d.sim.led_value(l), Some(true));
    }

    #[test]
    fn chip_edit_applies_back_and_updates_instances() {
        let mut src = doc();
        // A pass-through chip.
        let mut inner = CircuitManager::new();
        let i = inner.add_block(BlockType::Switch, Pos::new(0, 0));
        let o = inner.add_block(BlockType::Led, Pos::new(6, 0));
        inner.connect(Port::output(i, 0), Port::input(o, 0));
        let cid = src.manager.chips.allocate_id();
        src.manager
            .put_chip(ChipDef::from_circuit(cid, "Thru", inner.circuit.clone()));
        let s = src.manager.add_block(BlockType::Switch, Pos::new(0, 0));
        let inst = src.manager.add_block(BlockType::Chip(cid), Pos::new(4, 0));
        let led = src.manager.add_block(BlockType::Led, Pos::new(10, 0));
        src.connect(Port::output(s, 0), Port::input(inst, 0));
        src.connect(Port::output(inst, 0), Port::input(led, 0));
        src.set_input(s, true, true);
        src.tick(0.0);
        assert_eq!(src.sim.led_value(led), Some(true));

        // Edit: insert a NOT between the pin markers, then apply back.
        let mut edited = inner.circuit.clone();
        let conn = *edited.iter_connections().next().unwrap();
        edited.remove_connection(conn.id);
        let n = edited.allocate_block_id();
        edited.insert_block(Block::new(n, BlockType::Not, Pos::new(3, 0)));
        let c1 = edited.allocate_conn_id();
        edited.insert_connection(Connection {
            id: c1,
            from: Port::output(i, 0),
            to: Port::input(n, 0),
        });
        let c2 = edited.allocate_conn_id();
        edited.insert_connection(Connection {
            id: c2,
            from: Port::output(n, 0),
            to: Port::input(o, 0),
        });
        src.apply_chip_circuit(cid, edited).unwrap();
        src.tick(0.016);
        assert_eq!(src.sim.led_value(led), Some(false), "instance now inverts");
        // ...and it is one undo step.
        src.undo();
        src.tick(0.016);
        assert_eq!(src.sim.led_value(led), Some(true));
    }

    #[test]
    fn a_chip_cannot_be_edited_to_contain_itself() {
        let mut d = doc();
        let cid = d.manager.chips.allocate_id();
        let mut c = Circuit::new("c");
        let l = c.allocate_block_id();
        c.insert_block(Block::new(l, BlockType::Led, Pos::new(0, 0)));
        d.manager
            .put_chip(ChipDef::from_circuit(cid, "Me", c.clone()));
        let me = c.allocate_block_id();
        c.insert_block(Block::new(me, BlockType::Chip(cid), Pos::new(3, 0)));
        assert!(d.apply_chip_circuit(cid, c).is_err());
    }

    #[test]
    fn copy_paste_carries_chips_into_another_project() {
        let mut a = doc();
        let cid = a.manager.chips.allocate_id();
        let mut c = Circuit::new("c");
        let l = c.allocate_block_id();
        c.insert_block(Block::new(l, BlockType::Led, Pos::new(0, 0)));
        a.manager.put_chip(ChipDef::from_circuit(cid, "Widget", c));
        let inst = a.manager.add_block(BlockType::Chip(cid), Pos::new(0, 0));
        a.selection.insert(inst);
        let clip = a.copy_selection().unwrap();
        // Round-trip through the system-clipboard text form.
        let clip = Clipboard::from_text(&clip.to_text()).unwrap();

        let mut b = doc();
        b.paste(&clip, Some(Vec2f::new(5.0, 5.0)));
        let placed = b.manager.circuit.iter_blocks().next().unwrap();
        let BlockType::Chip(local) = placed.ty else {
            panic!("expected a chip")
        };
        assert_eq!(b.manager.chips.get(local).unwrap().name, "Widget");
        assert_eq!(placed.pos, Pos::new(5, 5));
    }

    #[test]
    fn create_chip_validates_name_and_outputs() {
        let mut d = doc();
        assert!(d.create_chip("", false).is_err());
        assert!(
            d.create_chip("Thing", false).is_err(),
            "no LED means no output pin"
        );
        d.manager.add_block(BlockType::Led, Pos::new(0, 0));
        assert!(d.create_chip("Thing", false).is_ok());
        assert!(d.create_chip("thing", false).is_err(), "names are unique");
        let id = d.manager.chips.find_by_name("Thing").unwrap();
        assert!(
            d.rename_chip(id, "and").is_err(),
            "built-in names are reserved"
        );
    }

    #[test]
    fn ai_edits_glow_and_frame_new_blocks_without_selecting_them() {
        let mut d = doc();
        let ops = llmc::ai::parse_reply(
            r#"{"commands":[{"op":"add","ref":"g","kind":"and","x":500,"y":500}]}"#,
        )
        .ops;
        let r = d.apply_ai_ops(ops, false);
        assert!(r.changed);
        assert!(
            d.selection.is_empty(),
            "a Backspace meant for the chat can't delete what the AI built"
        );
        assert_eq!(d.glow.as_ref().unwrap().blocks, r.new_blocks);
        assert!(matches!(
            d.frame_request,
            Some(FrameRequest::IfOffscreen(_))
        ));
        d.resolve_frame_request();
        let g = r.new_blocks[0];
        let b = d.manager.circuit.block(g).unwrap();
        let p = d.camera.to_screen(b.pos.as_vec(), d.canvas_rect.min);
        assert!(
            d.canvas_rect.contains(p),
            "the new block was brought into view"
        );
    }

    #[test]
    fn flipping_a_switch_does_not_mark_a_clean_file_unsaved() {
        let mut d = doc();
        let s = d.manager.add_block(BlockType::Switch, Pos::new(0, 0));
        d.mark_saved();
        d.set_input(s, true, true);
        d.refresh_dirty();
        assert!(!d.is_dirty());
        assert!(
            d.manager.circuit.block(s).unwrap().state,
            "the level is still saved"
        );
    }

    #[test]
    fn the_assistant_cannot_delete_the_users_blocks_unasked() {
        let mut d = doc();
        let mine = d.manager.add_block(BlockType::Not, Pos::new(0, 0));
        let ops = |j: &str| llmc::ai::parse_reply(j).ops;
        let r = d.apply_ai_ops(ops(r#"{"commands":[{"op":"remove","target":"0"}]}"#), false);
        assert!(d.manager.circuit.block(mine).is_some());
        assert_eq!(r.errors.len(), 1);
        // Its own blocks it may remove.
        let r = d.apply_ai_ops(
            ops(r#"{"commands":[{"op":"add","ref":"x","kind":"or"}]}"#),
            false,
        );
        let theirs = r.new_blocks[0];
        d.apply_ai_ops(
            ops(&format!(
                r#"{{"commands":[{{"op":"remove","target":"{theirs}"}}]}}"#
            )),
            false,
        );
        assert!(d.manager.circuit.block(theirs).is_none());
        // When the user asked for changes, theirs can go too.
        d.apply_ai_ops(ops(r#"{"commands":[{"op":"remove","target":"0"}]}"#), true);
        assert!(d.manager.circuit.block(mine).is_none());
    }

    #[test]
    fn damaged_files_are_repaired_and_flagged() {
        let mut c = llmc::model::Circuit::new("x");
        let a = c.allocate_block_id();
        c.insert_block(llmc::model::Block::new(
            a,
            BlockType::Switch,
            Pos::new(0, 0),
        ));
        let cid = c.allocate_conn_id();
        c.insert_connection(llmc::model::Connection {
            id: cid,
            from: llmc::model::Port::output(a, 0),
            to: llmc::model::Port::input(llmc::model::BlockId(77), 0),
        });
        let p = Project::new(c, Default::default(), Default::default());
        let mut d = Document::from_project(p, PathBuf::from("x.llmc"));
        assert!(d.manager.circuit.connections.is_empty());
        d.refresh_dirty();
        assert!(d.is_dirty(), "the repaired version isn't on disk yet");
        assert!(matches!(d.notices[0].kind, NoticeKind::Warning));
    }
}
