//! The application shell (the "Environment"). `LlmcApp` owns the open projects and the
//! global concerns (theme, clipboard, tab bar, close prompts); each open project is a
//! [`Document`] that owns its own backend, camera, interaction state, and the whole
//! interactive canvas. All canvas drawing is hand-rolled so pointer feel — hover,
//! snapping, dragging, wiring — is precise and smooth.
//!
//! Projects are fully independent: switching tabs never touches another document's state,
//! so long-running work (and, in a later phase, a per-tab AI session) keeps going in the
//! background. The clipboard is shared across tabs so you can copy from one and paste into
//! another.

use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeMap, BTreeSet};
use std::hash::Hasher;
use std::path::{Path, PathBuf};

use eframe::egui::{
    self, Align, Button, CursorIcon, Key, Layout, PointerButton, Pos2, Rect, Response, RichText,
    Sense, Stroke, StrokeKind, Vec2,
};

use llmc::backend::{CircuitManager, EditCommand, Simulation};
use llmc::io::{AppConfig, CameraState, Project};
use llmc::model::{
    Block, BlockId, BlockType, ChipDef, ChipId, Circuit, ConnId, Connection, Orientation, Port,
    PortKind, Pos, Vec2f, MAX_GATE_INPUTS,
};

use super::ai_edit::RawOp;
use super::assistant::{self, AiSession, AiSettings};
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

/// Shared across all tabs so a selection copied in one project can be pasted into another.
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

/// Human-friendly tab label derived from a file path (its stem), e.g. `adder.llmc` → `adder`.
fn title_from_path(path: &Path) -> String {
    path.file_stem()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())
        .unwrap_or("untitled")
        .to_string()
}

// ===================== a single open project =====================

/// One open project: its own circuit/backend, simulation, camera, selection, and all the
/// transient editing state. Everything a tab needs lives here, so tabs are independent.
struct Document {
    /// Stable identity for this open project, so tab actions survive reordering/closing
    /// of other tabs (never reuse an index to refer to a document across frames).
    id: u64,
    manager: CircuitManager,
    sim: Simulation,
    sim_dirty: bool,
    running: bool,
    camera: Camera,
    tool: Tool,
    interaction: Interaction,
    selection: BTreeSet<BlockId>,
    selected_conns: BTreeSet<ConnId>,
    path: Option<PathBuf>,
    /// Fingerprint of the serialized circuit+chips at the last save/load. The project is
    /// unsaved exactly when the current content fingerprint differs from this (see
    /// `content_fingerprint`) — i.e. dirtiness reflects real content, so undoing back to
    /// the saved state, or reverting an edit, correctly clears it.
    saved_fingerprint: u64,
    /// Cached result of the last dirtiness check. Only the active document's content can
    /// change in a frame, so this is recomputed once per frame for it and left frozen (and
    /// correct) for inactive tabs — the tab bar reads it every frame without re-hashing.
    dirty: bool,
    /// Tab label (file stem when saved, otherwise "Untitled N").
    title: String,
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
    props_inputs: u16,
    /// This project's own assistant conversation (independent per tab).
    ai: assistant::AiSession,
    /// Mirrored from the app each frame so canvas colors follow the theme.
    dark: bool,
}

impl Document {
    fn assemble(
        manager: CircuitManager,
        dark: bool,
        path: Option<PathBuf>,
        title: String,
        camera: Camera,
    ) -> Self {
        let sim = Simulation::build(&manager.circuit, &manager.chips);
        let mut doc = Self {
            id: 0,
            manager,
            sim,
            sim_dirty: false,
            running: false,
            camera,
            tool: Tool::Select,
            interaction: Interaction::Idle,
            selection: BTreeSet::new(),
            selected_conns: BTreeSet::new(),
            path,
            saved_fingerprint: 0,
            dirty: false,
            title,
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
            props_inputs: 2,
            ai: AiSession::default(),
            dark,
        };
        // The freshly-loaded/empty content is the "saved" baseline.
        doc.saved_fingerprint = doc.content_fingerprint();
        // Settle once so combinational state (lit LEDs/wires) shows even when paused.
        doc.sim.step(0.0);
        doc
    }

    fn new_empty(dark: bool, title: String) -> Self {
        let camera = Camera {
            pan: Vec2::new(140.0, 90.0),
            zoom: DEFAULT_ZOOM,
        };
        Self::assemble(CircuitManager::new(), dark, None, title, camera)
    }

    fn from_project(project: Project, dark: bool, path: PathBuf) -> Self {
        let manager = CircuitManager::from_parts(project.circuit, project.chips);
        let camera = Camera {
            pan: Vec2::new(project.camera.pan_x, project.camera.pan_y),
            zoom: project.camera.zoom.clamp(MIN_ZOOM, MAX_ZOOM),
        };
        let title = title_from_path(&path);
        Self::assemble(manager, dark, Some(path), title, camera)
    }

    /// A fingerprint of the saveable *logical* content: block/connection/chip data, but
    /// deliberately NOT the camera (so panning never counts as an edit) nor the id-allocator
    /// counters (`next_block`/`next_conn`/`next_id`), which only ever grow and are not rolled
    /// back by undo — hashing them would make "undo to the saved state" look dirty. The
    /// model's deterministic BTreeMap ordering makes equal content yield an equal value.
    fn content_fingerprint(&self) -> u64 {
        let c = &self.manager.circuit;
        let chips: Vec<_> = self
            .manager
            .chips
            .iter()
            .map(|d| {
                (
                    d.id,
                    &d.name,
                    &d.circuit.name,
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
            // Serialization of the model cannot realistically fail; if it ever did, fold in
            // a marker so current and saved fingerprints stay comparable (never falsely clean).
            Err(_) => h.write(b"llmc-fingerprint-error"),
        }
        h.finish()
    }

    /// Whether the project has unsaved changes (cheap: returns the cached result).
    fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Recompute the cached dirty flag from actual content. An in-progress chip edit always
    /// counts as unsaved work (the sub-edit isn't committed to the project yet).
    fn recompute_dirty(&mut self) {
        self.dirty =
            self.chip_edit.is_some() || self.content_fingerprint() != self.saved_fingerprint;
    }

    /// Record the current content as the saved baseline and refresh the dirty flag.
    fn mark_saved(&mut self) {
        self.saved_fingerprint = self.content_fingerprint();
        self.recompute_dirty();
    }

    /// The system prompt handed to the assistant: who it is, the JSON command protocol it uses to
    /// build/edit the circuit, and the user's current circuit as context so edits are grounded in
    /// what's on the canvas.
    fn assistant_system_prompt(&self) -> String {
        format!(
            "You are the built-in AI inside LLMC, a native digital-logic circuit builder and \
             simulator. You can talk to the user AND directly build or edit their circuit.\n\
             \n\
             BLOCK KINDS (use the quoted value as \"kind\"):\n\
             - Gates: \"and\", \"or\", \"not\", \"nand\", \"nor\", \"xor\", \"xnor\", \"buffer\"\n\
             - Inputs: \"switch\" (user-toggle), \"button\" (momentary), \"constant1\", \
             \"constant0\", \"clock\"\n\
             - Output: \"led\"\n\
             and/or/nand/nor/xor/xnor take an optional \"inputs\" count (2-16, default 2); \
             not/buffer have exactly 1 input; sources (switch/button/constant/clock) have no \
             inputs and one output; led has one input and no output.\n\
             \n\
             PORTS: input ports are indexed 0,1,2,… top-to-bottom; every block that has an output \
             uses output port 0. A wire goes from an output port to an input port.\n\
             \n\
             COORDINATES: an integer grid; x increases right, y increases down. Blocks are ~2 \
             cells wide and (number of inputs) tall. Put inputs on the left (small x), gates in \
             the middle, the LED on the right; leave ~4 cells horizontally between stages and ~3 \
             vertically between stacked blocks so wires stay readable.\n\
             \n\
             TO CHANGE THE CIRCUIT, end your reply with ONE fenced ```json block of the form \
             {{\"commands\":[ … ]}}. Commands:\n\
             - {{\"op\":\"add\",\"ref\":\"<name>\",\"kind\":\"<kind>\",\"x\":<int>,\"y\":<int>,\
             \"inputs\":<opt int>,\"label\":\"<opt>\"}}\n\
             - {{\"op\":\"connect\",\"from\":\"<ref-or-id>\",\"to\":\"<ref-or-id>\",\
             \"from_port\":<opt,def 0>,\"to_port\":<opt,def 0>}}\n\
             - {{\"op\":\"remove\",\"target\":\"<ref-or-id>\"}}\n\
             - {{\"op\":\"move\",\"target\":\"<ref-or-id>\",\"x\":<int>,\"y\":<int>}}\n\
             - {{\"op\":\"label\",\"target\":\"<ref-or-id>\",\"text\":\"<string>\"}}\n\
             Give every NEW block a short unique \"ref\" and use it in connects. Reference \
             EXISTING blocks by the numeric id shown in the circuit JSON. Include the json block \
             ONLY when you actually want to change the circuit — for plain questions, just \
             answer. Keep any prose before the json short.\n\
             \n\
             AUTONOMOUS LOOP: after your commands are applied you'll receive an automatic \
             \"auto-check\" — the circuit's real behavior as a truth table (inputs \u{2192} \
             outputs) plus any wiring problems. Compare it against the request. If the circuit is \
             complete and correct, reply with a short summary and NO json block (that ends the \
             task). Otherwise, send more commands to fix it — you can iterate as many times as \
             needed. Always re-check the wiring after you add, remove, or replace a block, since \
             replacing a block drops the wires that were attached to it.\n\
             \n\
             Current circuit:\n{}",
            self.circuit_context()
        )
    }

    /// The automatic post-edit "auto-check" handed back to the agent: what the circuit actually
    /// does (a truth table over its switches → LEDs), plus any multi-driver conflicts. This is
    /// what lets the assistant verify and fix its own work without the user prodding it.
    fn agent_observation(&self) -> String {
        use std::fmt::Write as _;
        let c = &self.manager.circuit;
        let inputs: Vec<&Block> = c
            .iter_blocks()
            .filter(|b| matches!(b.ty, BlockType::Switch | BlockType::Button))
            .collect();
        let outputs: Vec<&Block> = c.iter_blocks().filter(|b| b.ty == BlockType::Led).collect();
        let has_clock = c.iter_blocks().any(|b| b.ty == BlockType::Clock);

        let mut out = String::from("Auto-check after your edits.\n");

        // A fresh sim so the user's live switch settings aren't disturbed.
        let mut sim = Simulation::build(&self.manager.circuit, &self.manager.chips);
        if sim.has_conflicts() {
            out.push_str(
                "WARNING: an input is driven by more than one wire (multi-driver conflict) — \
                 usually a mistake.\n",
            );
        }

        let name = |b: &Block| match &b.label {
            Some(l) if !l.trim().is_empty() => format!("#{}({})", b.id, l.trim()),
            _ => format!("#{}", b.id),
        };

        if inputs.is_empty() || outputs.is_empty() {
            let _ = write!(
                out,
                "Circuit has {} switch input(s) and {} LED output(s); a truth table needs at \
                 least one of each.",
                inputs.len(),
                outputs.len()
            );
            return out;
        }
        if inputs.len() > 6 {
            let _ = write!(
                out,
                "Circuit has {} inputs — too many for a full truth table here. Inputs: {}. \
                 Outputs: {}.",
                inputs.len(),
                inputs
                    .iter()
                    .copied()
                    .map(name)
                    .collect::<Vec<_>>()
                    .join(", "),
                outputs
                    .iter()
                    .copied()
                    .map(name)
                    .collect::<Vec<_>>()
                    .join(", "),
            );
            return out;
        }

        if has_clock {
            out.push_str("(Circuit has a clock; table is a settled snapshot at time 0.)\n");
        }
        let _ = writeln!(
            out,
            "Truth table — inputs [{}] \u{2192} outputs [{}]:",
            inputs
                .iter()
                .copied()
                .map(name)
                .collect::<Vec<_>>()
                .join(", "),
            outputs
                .iter()
                .copied()
                .map(name)
                .collect::<Vec<_>>()
                .join(", "),
        );
        let n = inputs.len();
        for combo in 0..(1u32 << n) {
            for (i, b) in inputs.iter().enumerate() {
                let bit = (combo >> (n - 1 - i)) & 1 == 1;
                sim.set_switch(b.id, bit);
            }
            sim.step(0.0);
            let ins: Vec<&str> = (0..n)
                .map(|i| {
                    if (combo >> (n - 1 - i)) & 1 == 1 {
                        "1"
                    } else {
                        "0"
                    }
                })
                .collect();
            let outs: Vec<&str> = outputs
                .iter()
                .map(|b| {
                    if sim.led_value(b.id).unwrap_or(false) {
                        "1"
                    } else {
                        "0"
                    }
                })
                .collect();
            let _ = writeln!(out, "  {} | {}", ins.join(" "), outs.join(" "));
        }
        out.push_str(
            "If this matches the request, reply with a brief summary and NO commands. Otherwise \
             fix it with more commands.",
        );
        out
    }

    /// A compact JSON view of the active circuit (blocks + connections) for the assistant. Kept
    /// small; an empty circuit is reported in words so the model isn't handed `{}`.
    fn circuit_context(&self) -> String {
        let c = &self.manager.circuit;
        if c.blocks.is_empty() {
            return "The circuit is currently empty (no blocks placed yet).".to_string();
        }
        let view = serde_json::json!({
            "blocks": c.blocks,
            "connections": c.connections,
        });
        serde_json::to_string(&view).unwrap_or_else(|_| "(unavailable)".to_string())
    }

    /// Apply a batch of AI-proposed edits as one undoable step. Handles are resolved (new refs
    /// first, then existing numeric block ids), ports are validated, and any op that can't be
    /// applied is skipped and reported rather than aborting the whole batch. Returns a summary
    /// for the chat "activity" view.
    fn apply_ai_ops(&mut self, ops: Vec<RawOp>) -> AiEditReport {
        let mut refs: std::collections::HashMap<String, BlockId> = std::collections::HashMap::new();
        let mut pending: BTreeMap<BlockId, Block> = BTreeMap::new();
        let mut batch: Vec<EditCommand> = Vec::new();
        let mut activity: Vec<String> = Vec::new();
        let mut errors: Vec<String> = Vec::new();
        let mut new_ids: Vec<BlockId> = Vec::new();

        for op in ops {
            match op {
                RawOp::Add {
                    r#ref,
                    kind,
                    x,
                    y,
                    inputs,
                    label,
                    state,
                } => {
                    let id = self.manager.circuit.allocate_block_id();
                    let mut b = Block::new(id, kind, Pos::new(x, y));
                    if kind.variable_inputs() {
                        if let Some(n) = inputs {
                            b.inputs = Some(n.clamp(2, MAX_GATE_INPUTS as u16));
                        }
                    }
                    if let Some(l) = label {
                        if !l.trim().is_empty() {
                            b.label = Some(l);
                        }
                    }
                    b.state = state;
                    pending.insert(id, b.clone());
                    if let Some(r) = r#ref {
                        refs.insert(r, id);
                    }
                    new_ids.push(id);
                    activity.push(format!("Add {} at ({x}, {y})", palette_name(kind)));
                    batch.push(EditCommand::AddBlock { block: b });
                }
                RawOp::Connect {
                    from,
                    from_port,
                    to,
                    to_port,
                } => {
                    let (Some(fb), Some(tb)) = (
                        resolve_handle(&refs, &self.manager.circuit, &from),
                        resolve_handle(&refs, &self.manager.circuit, &to),
                    ) else {
                        errors.push(format!("connect: unknown block ({from} \u{2192} {to})"));
                        continue;
                    };
                    let out_ok = block_ref(&pending, &self.manager.circuit, fb).is_some_and(|b| {
                        (from_port as usize) < b.output_count(&self.manager.chips)
                    });
                    let in_ok = block_ref(&pending, &self.manager.circuit, tb)
                        .is_some_and(|b| (to_port as usize) < b.input_count(&self.manager.chips));
                    if !out_ok || !in_ok {
                        errors.push(format!(
                            "connect: invalid port ({from}.{from_port} \u{2192} {to}.{to_port})"
                        ));
                        continue;
                    }
                    let to_p = Port::input(tb, to_port);
                    if let Some(existing) = self.manager.circuit.connection_id_into(to_p) {
                        batch.push(EditCommand::RemoveConnection { id: existing });
                    }
                    let cid = self.manager.circuit.allocate_conn_id();
                    activity.push(format!("Wire {from}.{from_port} \u{2192} {to}.{to_port}"));
                    batch.push(EditCommand::AddConnection {
                        conn: Connection {
                            id: cid,
                            from: Port::output(fb, from_port),
                            to: to_p,
                        },
                    });
                }
                RawOp::Remove { target } => {
                    match resolve_handle(&refs, &self.manager.circuit, &target) {
                        Some(id) if self.manager.circuit.block(id).is_some() => {
                            activity.push(format!("Remove block {target}"));
                            batch.push(EditCommand::RemoveBlock { id });
                        }
                        _ => errors.push(format!("remove: unknown block {target}")),
                    }
                }
                RawOp::Move { target, x, y } => {
                    match resolve_handle(&refs, &self.manager.circuit, &target) {
                        Some(id) => {
                            activity.push(format!("Move {target} to ({x}, {y})"));
                            batch.push(EditCommand::MoveBlock {
                                id,
                                to: Pos::new(x, y),
                            });
                        }
                        None => errors.push(format!("move: unknown block {target}")),
                    }
                }
                RawOp::Label { target, text } => {
                    match resolve_handle(&refs, &self.manager.circuit, &target) {
                        Some(id) => {
                            activity.push(match &text {
                                Some(t) => format!("Label {target} \u{201c}{t}\u{201d}"),
                                None => format!("Clear label on {target}"),
                            });
                            batch.push(EditCommand::SetLabel { id, label: text });
                        }
                        None => errors.push(format!("label: unknown block {target}")),
                    }
                }
            }
        }

        let changed = !batch.is_empty();
        if changed {
            self.manager.apply(batch);
            self.after_structural_edit();
            // Select what the AI just created, so it's easy to see/move/delete.
            self.selection = new_ids
                .into_iter()
                .filter(|id| self.manager.circuit.block(*id).is_some())
                .collect();
            self.selected_conns.clear();
        }
        AiEditReport {
            activity,
            errors,
            changed,
        }
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

    /// Per-frame work for the active document: step simulation, keys, panels, canvas, dialogs.
    /// The assistant panel is drawn here (between the other side panels and the canvas) so it
    /// reserves its width before the central canvas fills the remaining space.
    #[allow(clippy::too_many_arguments)]
    fn frame(
        &mut self,
        ui: &mut egui::Ui,
        ctx: &egui::Context,
        clipboard: &mut Option<Clipboard>,
        ai_settings: &mut AiSettings,
        ai_open: bool,
        ai_settings_open: &mut bool,
    ) {
        if self.sim_dirty {
            self.rebuild_sim();
        }
        if self.running {
            let dt = ctx.input(|i| i.stable_dt).clamp(0.0, 0.1) as f64;
            self.sim.step(dt);
            ctx.request_repaint();
        }

        self.handle_shortcuts(ctx, clipboard);
        if ai_open {
            let theme = self.theme();
            let resp = assistant::panel(ui, &theme, ai_settings, &mut self.ai, ai_settings_open);
            if resp.send {
                let prompt = self.assistant_system_prompt();
                self.ai.send(ai_settings, prompt);
            } else if resp.retry {
                let prompt = self.assistant_system_prompt();
                self.ai.resend(ai_settings, prompt);
            }
        }
        self.left_palette(ui);
        self.status_bar(ui);

        let bg = self.theme().bg;
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE.fill(bg))
            .show(ui, |ui| {
                self.canvas(ui, clipboard);
            });

        if self.show_chip_dialog {
            self.chip_dialog(ctx);
        }
        if self.rename_chip.is_some() {
            self.rename_chip_dialog(ctx);
        }
        if self.props_for.is_some() {
            self.properties_window(ctx);
        }
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

    fn copy_selection(&mut self, clipboard: &mut Option<Clipboard>) {
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
        *clipboard = Some(Clipboard {
            blocks,
            connections,
        });
    }

    fn paste(&mut self, clipboard: &Option<Clipboard>) {
        let Some(clip) = clipboard else {
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
    fn paste_at(&mut self, clipboard: &Option<Clipboard>, world: Vec2f) {
        let Some(clip) = clipboard else {
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
            self.props_inputs = b.gate_inputs() as u16;
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
        let is_variable = ty.variable_inputs();
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
                if is_variable {
                    ui.horizontal(|ui| {
                        ui.label("Inputs:");
                        ui.add(egui::Slider::new(
                            &mut self.props_inputs,
                            2..=(MAX_GATE_INPUTS as u16),
                        ));
                    });
                }
            });

        // Apply the edited buffers back to the block (this runs every frame the window is
        // open; the fingerprint picks up any real change for unsaved-state tracking).
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
        // Applying a new input count may orphan wires to removed pins; prune them.
        if is_variable {
            let n = self.props_inputs.clamp(2, MAX_GATE_INPUTS as u16);
            let count_changed = self
                .manager
                .circuit
                .block(id)
                .map(|b| b.inputs != Some(n))
                .unwrap_or(false);
            if count_changed {
                if let Some(b) = self.manager.circuit.block_mut(id) {
                    b.inputs = Some(n);
                }
                let orphans: Vec<ConnId> = self
                    .manager
                    .circuit
                    .iter_connections()
                    .filter(|c| {
                        c.to.block == id && matches!(c.to.kind, PortKind::Input) && c.to.index >= n
                    })
                    .map(|c| c.id)
                    .collect();
                for cid in orphans {
                    self.manager.circuit.remove_connection(cid);
                }
                self.sim_dirty = true;
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

    // ----- keyboard -----

    fn handle_shortcuts(&mut self, ctx: &egui::Context, clipboard: &mut Option<Clipboard>) {
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
            self.copy_selection(clipboard);
        }
        if s.cmd && s.v {
            self.paste(clipboard);
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

    // ----- panels owned by the document -----

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

                let chips: Vec<(ChipId, String)> = self
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

    // ===================== canvas: geometry, input, drawing =====================

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
        let layout = block.layout(&self.manager.chips);
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

    /// True if any part of the wire's drawn path passes through `rect`
    /// (used so box-select grabs wires the same way it grabs blocks).
    fn wire_hits_rect(&self, conn: &Connection, rect: Rect, origin: Pos2) -> bool {
        match self.wire_points(conn, origin) {
            Some(pts) => pts
                .windows(2)
                .any(|seg| seg_intersects_rect(seg[0], seg[1], rect)),
            None => false,
        }
    }

    fn hit_port(&self, cursor: Pos2, origin: Pos2) -> Option<(Port, bool)> {
        // Generous pick radius so grabbing a port to start a wire is easy.
        let radius = (self.camera.zoom * 0.5).max(13.0);
        let mut best_d = radius * radius;
        let mut best = None;
        for block in self.manager.circuit.iter_blocks() {
            let layout = block.layout(&self.manager.chips);
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

    fn canvas(&mut self, ui: &mut egui::Ui, clipboard: &mut Option<Clipboard>) {
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
            self.canvas_context_menu(&response, clipboard);
        }
    }

    fn canvas_context_menu(&mut self, response: &Response, clipboard: &mut Option<Clipboard>) {
        let world = self.menu_world;
        response.context_menu(|ui| {
            let has_sel = !self.selection.is_empty() || !self.selected_conns.is_empty();
            let has_blocks = !self.selection.is_empty();
            let has_clip = clipboard.is_some();
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
                    self.copy_selection(clipboard);
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
                self.paste_at(clipboard, world);
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
                        self.selected_conns.clear();
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
                    // Also select wires whose path passes through the box.
                    let conn_ids: Vec<ConnId> = self
                        .manager
                        .circuit
                        .iter_connections()
                        .filter(|c| self.wire_hits_rect(c, box_rect, origin))
                        .map(|c| c.id)
                        .collect();
                    for cid in conn_ids {
                        self.selected_conns.insert(cid);
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

            let layout = block.layout(&self.manager.chips);
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

// ===================== the application shell =====================

/// What a pending close prompt is about.
#[derive(Clone, Copy)]
enum CloseKind {
    /// A single tab is being closed, identified by its stable document id (not an index,
    /// which could point at a different tab if the vector changes while the prompt is open).
    Tab(u64),
    /// The whole window is being closed.
    Quit,
}

pub struct LlmcApp {
    docs: Vec<Document>,
    active: usize,
    config: AppConfig,
    dark: bool,
    /// Shared across tabs so a selection copied in one project pastes into another.
    clipboard: Option<Clipboard>,
    /// Counter for naming fresh, never-saved projects ("Untitled 1", "Untitled 2", …).
    untitled_count: usize,
    /// Source of stable per-document ids (see [`Document::id`]).
    next_doc_id: u64,
    /// A save/discard/cancel prompt in flight, if any.
    close_confirm: Option<CloseKind>,
    /// Set once the user has resolved the quit prompt so the next close request goes through.
    allow_quit: bool,
    /// Global assistant settings (providers/keys), shared across all tabs.
    ai: AiSettings,
    /// Whether the assistant side panel is open.
    ai_open: bool,
    /// Whether the provider settings modal is open.
    ai_settings_open: bool,
}

impl LlmcApp {
    pub fn new(
        cc: &eframe::CreationContext<'_>,
        config: AppConfig,
        initial: Option<PathBuf>,
    ) -> Self {
        let dark = config.dark_mode;
        apply_style(&cc.egui_ctx, dark);
        let mut app = Self {
            docs: Vec::new(),
            active: 0,
            config,
            dark,
            clipboard: None,
            untitled_count: 0,
            next_doc_id: 0,
            close_confirm: None,
            allow_quit: false,
            // Providers start empty/disconnected; the panel is closed so the base UI stays
            // calm (open it from the toolbar "Assistant" toggle).
            ai: AiSettings::default(),
            ai_open: false,
            ai_settings_open: false,
        };
        // Open the file passed on the command line into the first tab, if any.
        if let Some(path) = initial {
            match Project::load(&path) {
                Ok(project) => {
                    app.remember_dir(&path);
                    let doc = Document::from_project(project, dark, path);
                    app.push_doc(doc);
                }
                Err(e) => {
                    let mut doc = app.fresh_document();
                    doc.status = format!("Open failed: {e}");
                    app.push_doc(doc);
                }
            }
        }
        if app.docs.is_empty() {
            let doc = app.fresh_document();
            app.push_doc(doc);
        }
        app.active = 0;
        // Restore saved providers/keys and reconnect any that were enabled.
        let saved = app.config.ai_providers.clone();
        app.ai.apply_config(&saved);
        app
    }

    fn theme(&self) -> Theme {
        if self.dark {
            Theme::dark()
        } else {
            Theme::light()
        }
    }

    fn alloc_doc_id(&mut self) -> u64 {
        let id = self.next_doc_id;
        self.next_doc_id += 1;
        id
    }

    /// A new empty document with the next "Untitled N" title (id assigned on insertion).
    fn fresh_document(&mut self) -> Document {
        self.untitled_count += 1;
        Document::new_empty(self.dark, format!("Untitled {}", self.untitled_count))
    }

    /// Assign a stable id and append the document. Returns its index; does not change focus.
    fn push_doc(&mut self, mut doc: Document) -> usize {
        doc.id = self.alloc_doc_id();
        self.docs.push(doc);
        self.docs.len() - 1
    }

    fn new_tab(&mut self) {
        let doc = self.fresh_document();
        self.active = self.push_doc(doc);
    }

    fn open(&mut self) {
        let mut dlg = rfd::FileDialog::new().add_filter("LLMC circuit", &["llmc"]);
        if let Some(dir) = &self.config.last_dir {
            dlg = dlg.set_directory(dir);
        }
        if let Some(path) = dlg.pick_file() {
            self.open_path(&path);
        }
    }

    /// Load a project from a known path into a new tab.
    fn open_path(&mut self, path: &Path) {
        // If this file is already open, just focus its tab.
        if let Some(i) = self
            .docs
            .iter()
            .position(|d| d.path.as_deref() == Some(path))
        {
            self.active = i;
            return;
        }
        match Project::load(path) {
            Ok(project) => {
                self.remember_dir(path);
                let mut doc = Document::from_project(project, self.dark, path.to_path_buf());
                doc.status = format!("Opened {}", path.display());
                self.active = self.push_doc(doc);
            }
            Err(e) => {
                self.docs[self.active].status = format!("Open failed: {e}");
            }
        }
    }

    /// Save the given tab, prompting for a location when it has never been saved.
    /// Returns whether the project was actually written — false if the user cancelled the
    /// Save As dialog *or* the write itself failed, so callers must not close a tab whose
    /// save did not succeed.
    fn save_doc(&mut self, i: usize) -> bool {
        if i >= self.docs.len() {
            return false;
        }
        // An in-progress chip edit swaps the live circuit for the chip's internals and
        // stashes the project. Commit it first so we persist the real project (with the
        // chip changes folded in) instead of the chip's sub-circuit, and never lose either.
        if self.docs[i].chip_edit.is_some() {
            self.docs[i].save_chip_edit();
        }
        let path = self.docs[i].path.clone();
        match path {
            Some(p) => self.write_doc(i, &p),
            None => {
                let mut dlg = rfd::FileDialog::new()
                    .add_filter("LLMC circuit", &["llmc"])
                    .set_file_name(format!("{}.llmc", self.docs[i].title));
                if let Some(dir) = &self.config.last_dir {
                    dlg = dlg.set_directory(dir);
                }
                if let Some(path) = dlg.save_file() {
                    self.remember_dir(&path);
                    self.docs[i].path = Some(path.clone());
                    self.docs[i].title = title_from_path(&path);
                    self.write_doc(i, &path)
                } else {
                    false
                }
            }
        }
    }

    /// Force a Save As for the given tab regardless of its current path.
    fn save_doc_as(&mut self, i: usize) {
        if i >= self.docs.len() {
            return;
        }
        if self.docs[i].chip_edit.is_some() {
            self.docs[i].save_chip_edit();
        }
        let mut dlg = rfd::FileDialog::new()
            .add_filter("LLMC circuit", &["llmc"])
            .set_file_name(format!("{}.llmc", self.docs[i].title));
        if let Some(dir) = &self.config.last_dir {
            dlg = dlg.set_directory(dir);
        }
        if let Some(path) = dlg.save_file() {
            self.remember_dir(&path);
            self.docs[i].path = Some(path.clone());
            self.docs[i].title = title_from_path(&path);
            self.write_doc(i, &path);
        }
    }

    /// Serialize the tab to `path`. Returns whether the write succeeded.
    fn write_doc(&mut self, i: usize, path: &Path) -> bool {
        let doc = &mut self.docs[i];
        let camera = CameraState {
            pan_x: doc.camera.pan.x,
            pan_y: doc.camera.pan.y,
            zoom: doc.camera.zoom,
        };
        let project = Project::new(
            doc.manager.circuit.clone(),
            doc.manager.chips.clone(),
            camera,
        );
        match project.save(path) {
            Ok(()) => {
                doc.mark_saved();
                doc.status = format!("Saved {}", path.display());
                true
            }
            Err(e) => {
                doc.status = format!("Save failed: {e}");
                false
            }
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

    /// Close a tab unconditionally (the dirty check happens in [`LlmcApp::request_close_tab`]).
    /// The window always keeps at least one tab: closing the last one resets it to empty.
    fn close_tab(&mut self, i: usize) {
        if i >= self.docs.len() {
            return;
        }
        if self.docs.len() == 1 {
            let mut doc = self.fresh_document();
            doc.id = self.alloc_doc_id();
            self.docs[0] = doc;
            self.active = 0;
            return;
        }
        self.docs.remove(i);
        if self.active > i {
            self.active -= 1;
        }
        if self.active >= self.docs.len() {
            self.active = self.docs.len() - 1;
        }
    }

    /// Begin closing a tab: prompt to save if it has unsaved changes, else close it now.
    fn request_close_tab(&mut self, i: usize) {
        let Some(doc) = self.docs.get(i) else {
            return;
        };
        if doc.is_dirty() {
            let id = doc.id;
            self.active = i;
            self.close_confirm = Some(CloseKind::Tab(id));
        } else {
            self.close_tab(i);
        }
    }

    /// Current index of the document with the given stable id, if it's still open.
    fn doc_index(&self, id: u64) -> Option<usize> {
        self.docs.iter().position(|d| d.id == id)
    }

    // ----- top panels (app-level, delegating per-doc actions to the active document) -----

    fn top_toolbar(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let active = self.active;
        egui::Panel::top("toolbar").show(ui, |ui| {
            ui.add_space(3.0);
            ui.horizontal(|ui| {
                ui.label(RichText::new("LLMC").strong().size(16.0));
                ui.separator();
                let editing_chip = self.docs[active].chip_edit.is_some();
                ui.add_enabled_ui(!editing_chip, |ui| {
                    if ui.button("New").clicked() {
                        self.new_tab();
                    }
                    if ui.button("Open").clicked() {
                        self.open();
                    }
                    if ui.button("Save").clicked() {
                        self.save_doc(self.active);
                    }
                    if ui.button("Save As").clicked() {
                        self.save_doc_as(self.active);
                    }
                });
                ui.separator();
                let running = self.docs[active].running;
                let run = if running {
                    "\u{23f8}  Stop"
                } else {
                    "\u{25b6}  Run"
                };
                if ui.button(run).clicked() {
                    let doc = &mut self.docs[active];
                    doc.running = !doc.running;
                    if doc.running {
                        doc.sim_dirty = true;
                    }
                }
                if ui.button("Step").clicked() {
                    let doc = &mut self.docs[active];
                    if doc.sim_dirty {
                        doc.rebuild_sim();
                    }
                    doc.sim.step(0.05);
                }
                ui.separator();
                let (can_undo, can_redo) = {
                    let u = &self.docs[active].manager.undo;
                    (u.can_undo(), u.can_redo())
                };
                if ui.add_enabled(can_undo, Button::new("Undo")).clicked() {
                    self.docs[active].undo();
                }
                if ui.add_enabled(can_redo, Button::new("Redo")).clicked() {
                    self.docs[active].redo();
                }
                ui.separator();
                if self.docs[active].chip_edit.is_some() {
                    ui.colored_label(self.theme().accent, "\u{270e} Editing chip");
                    if ui.button("Update chip").clicked() {
                        self.docs[active].save_chip_edit();
                    }
                    if ui.button("Cancel").clicked() {
                        self.docs[active].cancel_chip_edit();
                    }
                } else if ui.button("Create Chip").clicked() {
                    self.docs[active].show_chip_dialog = true;
                }
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    // The assistant toggle sits on the right, as its own thing — the base UI
                    // stays calm (just this button and, when open, the side panel).
                    let ai_label = RichText::new("Assistant").color(if self.ai_open {
                        egui::Color32::WHITE
                    } else {
                        self.theme().accent
                    });
                    if ui.selectable_label(self.ai_open, ai_label).clicked() {
                        self.ai_open = !self.ai_open;
                    }
                    ui.separator();
                    let label = if self.dark { "Light" } else { "Dark" };
                    if ui.button(label).clicked() {
                        self.toggle_theme(ctx);
                    }
                    if self.docs[active].sim.has_conflicts() {
                        ui.colored_label(self.theme().conflict, "\u{26a0} multi-driver");
                    }
                });
            });
            ui.add_space(3.0);
        });
    }

    fn tab_bar(&mut self, ui: &mut egui::Ui) {
        let t = self.theme();
        egui::Panel::top("tabs").show(ui, |ui| {
            ui.add_space(2.0);
            let mut select: Option<usize> = None;
            let mut close: Option<usize> = None;
            let mut add = false;
            egui::ScrollArea::horizontal()
                .max_width(f32::INFINITY)
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        for (i, doc) in self.docs.iter().enumerate() {
                            let selected = i == self.active;
                            let dirty = doc.is_dirty();
                            let name = if dirty {
                                format!("{} \u{2022}", doc.title)
                            } else {
                                doc.title.clone()
                            };
                            let text = if selected {
                                RichText::new(name).strong()
                            } else {
                                RichText::new(name).color(t.label_dim)
                            };
                            if ui.selectable_label(selected, text).clicked() {
                                select = Some(i);
                            }
                            if ui
                                .add(Button::new(RichText::new("\u{00d7}").small()).frame(false))
                                .on_hover_text("Close tab")
                                .clicked()
                            {
                                close = Some(i);
                            }
                            ui.separator();
                        }
                        if ui
                            .add(Button::new("+").frame(false))
                            .on_hover_text("New project")
                            .clicked()
                        {
                            add = true;
                        }
                    });
                });
            ui.add_space(2.0);
            if let Some(i) = select {
                self.active = i;
            }
            if add {
                self.new_tab();
            }
            if let Some(i) = close {
                self.request_close_tab(i);
            }
        });
    }

    /// The save/discard/cancel prompt shown before a dirty tab or the window is closed.
    fn close_dialog(&mut self, ctx: &egui::Context) {
        let Some(kind) = self.close_confirm else {
            return;
        };
        // The prompted tab may have been closed/reordered already (the prompt isn't modal);
        // resolve it by stable id and drop the prompt if it's gone.
        if let CloseKind::Tab(id) = kind {
            if self.doc_index(id).is_none() {
                self.close_confirm = None;
                return;
            }
        }
        let mut decision: Option<Decision> = None;
        let title = match kind {
            CloseKind::Tab(id) => format!(
                "Save changes to \u{201c}{}\u{201d} before closing?",
                self.doc_index(id)
                    .map(|i| self.docs[i].title.as_str())
                    .unwrap_or("")
            ),
            CloseKind::Quit => {
                let n = self.docs.iter().filter(|d| d.is_dirty()).count();
                format!("You have unsaved changes in {n} project(s).")
            }
        };
        egui::Window::new("Unsaved changes")
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {
                ui.label(title);
                ui.add_space(4.0);
                ui.label(
                    RichText::new("Your changes will be lost if you don't save them.")
                        .small()
                        .color(self.theme().label_dim),
                );
                ui.add_space(10.0);
                ui.horizontal(|ui| {
                    let save_label = match kind {
                        CloseKind::Quit => "Save all",
                        CloseKind::Tab(_) => "Save",
                    };
                    if ui.button(save_label).clicked() {
                        decision = Some(Decision::Save);
                    }
                    if ui.button("Don't save").clicked() {
                        decision = Some(Decision::Discard);
                    }
                    if ui.button("Cancel").clicked() {
                        decision = Some(Decision::Cancel);
                    }
                });
            });

        let Some(decision) = decision else {
            return;
        };
        self.close_confirm = None;
        match (kind, decision) {
            (_, Decision::Cancel) => {}
            (CloseKind::Tab(id), Decision::Save) => {
                if let Some(i) = self.doc_index(id) {
                    if self.save_doc(i) {
                        // Re-resolve: saving an untitled tab shows a native dialog, during
                        // which nothing here can change the vector, but stay defensive.
                        if let Some(i) = self.doc_index(id) {
                            self.close_tab(i);
                        }
                    }
                    // If the save was cancelled or failed, leave the tab open.
                }
            }
            (CloseKind::Tab(id), Decision::Discard) => {
                if let Some(i) = self.doc_index(id) {
                    self.close_tab(i);
                }
            }
            (CloseKind::Quit, Decision::Save) => {
                let mut all_saved = true;
                for i in 0..self.docs.len() {
                    if self.docs[i].is_dirty() && !self.save_doc(i) {
                        all_saved = false;
                        break;
                    }
                }
                if all_saved {
                    self.allow_quit = true;
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }
            }
            (CloseKind::Quit, Decision::Discard) => {
                self.allow_quit = true;
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
        }
    }

    /// Intercept the window's close button so unsaved work can prompt first.
    fn handle_quit_request(&mut self, ctx: &egui::Context) {
        if !ctx.input(|i| i.viewport().close_requested()) {
            return;
        }
        if self.allow_quit {
            return; // already confirmed — let the close proceed
        }
        if self.docs.iter().any(|d| d.is_dirty()) {
            ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
            self.close_confirm = Some(CloseKind::Quit);
        }
        // No unsaved work: allow the close.
    }
}

/// The user's answer to the unsaved-changes prompt.
enum Decision {
    Save,
    Discard,
    Cancel,
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

        // Drain any provider-connection results from worker threads, and keep animating while a
        // test is in flight so the outcome appears promptly.
        self.ai.poll();
        if self.ai.any_testing() {
            ctx.request_repaint_after(std::time::Duration::from_millis(120));
        }

        // Chat replies can arrive for ANY tab — a request started in one project keeps running
        // in the background when you switch away — so poll every document's session, apply any
        // circuit edits the reply proposed (as one undoable step, into that tab), and keep
        // repainting while any is still awaiting a reply.
        //
        // In agent mode the assistant iterates on its own: after applying its edits we hand it an
        // automatic "auto-check" (the circuit's actual truth table + any problems) and dispatch
        // the next step, until it replies with no edits (done), the user hits Stop, or the step
        // cap is reached.
        let ai_settings = &self.ai;
        let mut any_pending = false;
        for doc in &mut self.docs {
            let reply = doc.ai.poll_chat();
            if let Some(ops) = doc.ai.take_pending_ops() {
                let report = doc.apply_ai_ops(ops);
                if report.changed {
                    doc.recompute_dirty();
                }
                doc.ai.attach_edit_result(report.activity, report.errors);
                if doc.ai.should_continue_or_note() {
                    let feedback = doc.agent_observation();
                    let prompt = doc.assistant_system_prompt();
                    doc.ai.continue_agent(ai_settings, prompt, feedback);
                } else {
                    doc.ai.end_agent();
                }
            } else if reply {
                // A reply with no edits ends the run (final answer, or an error).
                doc.ai.end_agent();
            }
            any_pending |= doc.ai.is_pending();
        }
        if any_pending {
            ctx.request_repaint_after(std::time::Duration::from_millis(120));
        }
        // Persist provider config (never keys) when it changed.
        if self.ai.take_dirty() {
            self.config.ai_providers = self.ai.config_snapshot();
            self.config.save();
        }

        // The toolbar may toggle the theme this frame, so render it (and the tab bar) first…
        self.top_toolbar(ui, &ctx);
        self.tab_bar(ui);

        // …then sync the (possibly just-flipped) theme into the active document, so the
        // canvas paints in the same theme as the chrome on the very same frame.
        let active = self.active;
        self.docs[active].dark = self.dark;

        // Delegate the palette, status bar, assistant panel, canvas, and dialogs to the active
        // document. The assistant panel is drawn inside `frame` (after the other side panels,
        // before the canvas) so all side panels reserve their space before the canvas fills the
        // rest. `docs[active]`, `ai`, `ai_settings_open`, and `clipboard` are disjoint fields.
        let clipboard = &mut self.clipboard;
        let ai_settings = &mut self.ai;
        let ai_open = self.ai_open;
        let ai_settings_open = &mut self.ai_settings_open;
        self.docs[active].frame(ui, &ctx, clipboard, ai_settings, ai_open, ai_settings_open);

        // Only the active document can have changed this frame; refresh its dirty flag from
        // the actual content so the tab bar and close prompts stay accurate (inactive tabs
        // can't change, so their cached flag stays correct without re-hashing).
        self.docs[active].recompute_dirty();

        // App-level modals and window-close handling.
        if self.ai_settings_open {
            let theme = self.theme();
            assistant::settings_window(&ctx, &theme, &mut self.ai, &mut self.ai_settings_open);
        }
        self.close_dialog(&ctx);
        self.handle_quit_request(&ctx);
    }
}

/// The outcome of applying a batch of AI-proposed edits, surfaced in the chat "activity" view.
struct AiEditReport {
    /// One human-readable line per edit that was applied.
    activity: Vec<String>,
    /// One line per edit that was skipped (bad handle / port), so the user can see what didn't
    /// happen.
    errors: Vec<String>,
    /// Whether anything actually changed (empty batches touch nothing).
    changed: bool,
}

/// Resolve an AI block handle to a real block id: a new-block ref first, else an existing numeric
/// id that is actually present in the circuit. A leading `#` is tolerated.
fn resolve_handle(
    refs: &std::collections::HashMap<String, BlockId>,
    circuit: &Circuit,
    handle: &str,
) -> Option<BlockId> {
    if let Some(id) = refs.get(handle) {
        return Some(*id);
    }
    let trimmed = handle.trim().trim_start_matches('#');
    if let Ok(n) = trimmed.parse::<u32>() {
        let id = BlockId(n);
        if circuit.block(id).is_some() {
            return Some(id);
        }
    }
    None
}

/// Look up a block among the not-yet-applied additions first, then the live circuit — so a wire
/// can reference a block created earlier in the same batch.
fn block_ref<'a>(
    pending: &'a BTreeMap<BlockId, Block>,
    circuit: &'a Circuit,
    id: BlockId,
) -> Option<&'a Block> {
    pending.get(&id).or_else(|| circuit.block(id))
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

/// Does segment `a`–`b` touch axis-aligned `rect`? True if either endpoint is
/// inside, or the segment crosses any of the four edges — so a thin selection
/// box still catches a long straight wire between sample points.
fn seg_intersects_rect(a: Pos2, b: Pos2, rect: Rect) -> bool {
    if rect.contains(a) || rect.contains(b) {
        return true;
    }
    let tl = rect.left_top();
    let tr = rect.right_top();
    let br = rect.right_bottom();
    let bl = rect.left_bottom();
    segs_cross(a, b, tl, tr)
        || segs_cross(a, b, tr, br)
        || segs_cross(a, b, br, bl)
        || segs_cross(a, b, bl, tl)
}

/// Standard orientation-sign test for whether open segments p1–p2 and p3–p4 cross.
fn segs_cross(p1: Pos2, p2: Pos2, p3: Pos2, p4: Pos2) -> bool {
    let cross = |o: Pos2, x: Pos2, y: Pos2| (x.x - o.x) * (y.y - o.y) - (x.y - o.y) * (y.x - o.x);
    let d1 = cross(p3, p4, p1);
    let d2 = cross(p3, p4, p2);
    let d3 = cross(p1, p2, p3);
    let d4 = cross(p1, p2, p4);
    ((d1 > 0.0 && d2 < 0.0) || (d1 < 0.0 && d2 > 0.0))
        && ((d3 > 0.0 && d4 < 0.0) || (d3 < 0.0 && d4 > 0.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn box_catches_wire_crossing_it() {
        let rect = Rect::from_min_max(Pos2::new(10.0, 10.0), Pos2::new(20.0, 20.0));
        // Horizontal segment passing straight through the box.
        assert!(seg_intersects_rect(
            Pos2::new(0.0, 15.0),
            Pos2::new(30.0, 15.0),
            rect
        ));
        // Endpoint inside the box.
        assert!(seg_intersects_rect(
            Pos2::new(15.0, 15.0),
            Pos2::new(40.0, 40.0),
            rect
        ));
        // A thin box straddling a long segment still catches it.
        let thin = Rect::from_min_max(Pos2::new(14.0, 0.0), Pos2::new(16.0, 100.0));
        assert!(seg_intersects_rect(
            Pos2::new(0.0, 50.0),
            Pos2::new(500.0, 50.0),
            thin
        ));
    }

    #[test]
    fn box_ignores_wire_that_misses_it() {
        let rect = Rect::from_min_max(Pos2::new(10.0, 10.0), Pos2::new(20.0, 20.0));
        // Well clear of the box.
        assert!(!seg_intersects_rect(
            Pos2::new(0.0, 100.0),
            Pos2::new(30.0, 100.0),
            rect
        ));
        // Parallel and just outside one edge.
        assert!(!seg_intersects_rect(
            Pos2::new(0.0, 25.0),
            Pos2::new(30.0, 25.0),
            rect
        ));
    }

    fn doc() -> Document {
        Document::new_empty(true, "test".to_string())
    }

    #[test]
    fn dirty_reflects_real_content_changes() {
        let mut d = doc();
        d.recompute_dirty();
        assert!(!d.is_dirty(), "a fresh empty project is clean");

        let id = d.manager.add_block(BlockType::Switch, Pos::new(0, 0));
        d.recompute_dirty();
        assert!(d.is_dirty(), "adding a block is an unsaved change");

        d.mark_saved();
        assert!(!d.is_dirty(), "saving clears the dirty flag");

        // A switch's on/off state is part of the serialized project, so toggling it must
        // count as an unsaved change (finding: switch toggle was silently lost on close).
        d.manager.circuit.block_mut(id).unwrap().state = true;
        d.recompute_dirty();
        assert!(d.is_dirty(), "toggling a switch dirties the project");
    }

    #[test]
    fn undo_back_to_saved_state_is_clean() {
        let mut d = doc();
        d.mark_saved();
        d.manager.add_block(BlockType::And, Pos::new(1, 1));
        d.recompute_dirty();
        assert!(d.is_dirty());

        // Undoing all the way back to the saved content must clear dirtiness, not leave it
        // stuck on (the monotonic-counter approach could not detect this).
        assert!(d.manager.undo());
        d.recompute_dirty();
        assert!(!d.is_dirty(), "undo back to the saved state is clean again");
    }

    #[test]
    fn in_progress_chip_edit_counts_as_unsaved() {
        let mut d = doc();
        d.mark_saved();
        assert!(!d.is_dirty());

        // While a chip is being edited the live circuit is the chip's internals; the project
        // has uncommitted sub-work, so the document must read as dirty regardless of content.
        d.chip_edit = Some(ChipEdit {
            id: ChipId(0),
            prev_circuit: Circuit::new("proj"),
            prev_path: None,
        });
        d.recompute_dirty();
        assert!(d.is_dirty(), "an open chip edit is unsaved work");

        // Cancelling restores the exact pre-edit content, which must read clean again
        // (finding: cancel left the project spuriously dirty).
        d.chip_edit = None;
        d.recompute_dirty();
        assert!(
            !d.is_dirty(),
            "reverting the chip edit restores a clean project"
        );
    }

    #[test]
    fn apply_ai_ops_builds_wires_and_reports_bad_ops() {
        let mut d = doc();
        let ops = vec![
            RawOp::Add {
                r#ref: Some("a".into()),
                kind: BlockType::Switch,
                x: 0,
                y: 0,
                inputs: None,
                label: None,
                state: false,
            },
            RawOp::Add {
                r#ref: Some("g".into()),
                kind: BlockType::And,
                x: 6,
                y: 1,
                inputs: None,
                label: None,
                state: false,
            },
            RawOp::Add {
                r#ref: Some("led".into()),
                kind: BlockType::Led,
                x: 12,
                y: 2,
                inputs: None,
                label: None,
                state: false,
            },
            RawOp::Connect {
                from: "a".into(),
                from_port: 0,
                to: "g".into(),
                to_port: 0,
            },
            RawOp::Connect {
                from: "g".into(),
                from_port: 0,
                to: "led".into(),
                to_port: 0,
            },
            // A switch has no input port, so this must be skipped and reported, not applied.
            RawOp::Connect {
                from: "g".into(),
                from_port: 0,
                to: "a".into(),
                to_port: 0,
            },
        ];
        let report = d.apply_ai_ops(ops);
        assert!(report.changed);
        assert_eq!(d.manager.circuit.blocks.len(), 3, "three blocks added");
        assert_eq!(
            d.manager.circuit.connections.len(),
            2,
            "two valid wires; the invalid one skipped"
        );
        assert_eq!(report.errors.len(), 1, "the bad connect is reported");

        // The whole batch is one undo step.
        assert!(d.manager.undo());
        assert_eq!(d.manager.circuit.blocks.len(), 0, "one undo reverts it all");
    }

    #[test]
    fn agent_observation_reports_the_real_truth_table() {
        let mut d = doc();
        let a = d.manager.add_block(BlockType::Switch, Pos::new(0, 0));
        let b = d.manager.add_block(BlockType::Switch, Pos::new(0, 4));
        let g = d.manager.add_block(BlockType::And, Pos::new(6, 1));
        let led = d.manager.add_block(BlockType::Led, Pos::new(12, 2));
        d.manager.connect(Port::output(a, 0), Port::input(g, 0));
        d.manager.connect(Port::output(b, 0), Port::input(g, 1));
        d.manager.connect(Port::output(g, 0), Port::input(led, 0));

        let obs = d.agent_observation();
        assert!(obs.contains("Truth table"), "reports a truth table");
        // AND: high only when both inputs are high.
        assert!(obs.contains("0 0 | 0"));
        assert!(obs.contains("0 1 | 0"));
        assert!(obs.contains("1 0 | 0"));
        assert!(obs.contains("1 1 | 1"));
    }
}
