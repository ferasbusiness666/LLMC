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
use std::collections::{BTreeMap, BTreeSet, HashMap};
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

use super::ai_edit::{AddKind, PortSel, RawOp};
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
    /// A momentary Button currently held down (high while pressed, low on release).
    pressed_button: Option<BlockId>,
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
            pressed_button: None,
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

    /// The system prompt handed to the assistant: the engineer's playbook (plan → build small
    /// verified chips bottom-up → compose → test), the JSON command protocol, and the current
    /// circuit + chip library as context so edits are grounded in what's on the canvas.
    fn assistant_system_prompt(&self) -> String {
        format!(
            "You are the built-in AI of LLMC, a digital-logic builder and simulator. You design, \
             build, edit, and TEST circuits directly on the user's canvas via JSON commands.\n\
             \n\
             METHOD — work like an engineer:\n\
             1. PLAN briefly: decompose the request into sub-circuits.\n\
             2. For anything beyond a few gates, define REUSABLE CHIPS bottom-up with \"defchip\" \
             (each is verified in isolation the moment you define it), then stamp instances with \
             \"add\". A CPU is chips of chips: FullAdder \u{2192} Adder8 \u{2192} ALU \u{2192} \
             CPU. Never build a big design as one flat pile of gates.\n\
             3. After every reply you get an AUTO-CHECK: skipped commands, each new chip's real \
             truth table, unwired inputs, and the circuit's behavior. Compare it with the goal, \
             fix what's wrong, repeat. Trust the auto-check over your assumptions.\n\
             4. Verify sequential logic (latches, registers, memory, CPUs) with the \"test\" \
             command — scripted input sequences with all outputs read after each step.\n\
             \n\
             BLOCK KINDS (\"kind\"): gates \"and\",\"or\",\"not\",\"nand\",\"nor\",\"xor\",\
             \"xnor\",\"buffer\" (multi-input gates take optional \"inputs\":2-16, default 2); \
             sources \"switch\" (click-latching),\"button\" (momentary),\"constant1\",\
             \"constant0\",\"clock\"; output \"led\"; or ANY CHIP NAME from the library.\n\
             \n\
             COMMANDS — reply with exactly ONE fenced ```json block at the END of your reply, \
             containing one object: {{\"commands\":[ \u{2026} ]}}. STRICT JSON: double quotes, \
             NO comments, NO trailing commas, nothing else inside the fence.\n\
             - {{\"op\":\"add\",\"ref\":\"u1\",\"kind\":\"<kind or ChipName>\",\"x\":0,\"y\":0,\
             \"inputs\":3,\"label\":\"A\"}} (\"inputs\"/\"label\" optional)\n\
             - {{\"op\":\"connect\",\"from\":\"u1\",\"to\":\"u2\",\"from_port\":0,\"to_port\":1}}\n\
             - {{\"op\":\"remove\",\"target\":\"\u{2026}\"}}  \
             {{\"op\":\"move\",\"target\":\"\u{2026}\",\"x\":8,\"y\":0}}  \
             {{\"op\":\"label\",\"target\":\"\u{2026}\",\"text\":\"\u{2026}\"}}\n\
             - {{\"op\":\"defchip\",\"name\":\"FullAdder\",\"inputs\":[\"a\",\"b\",\"cin\"],\
             \"outputs\":[\"sum\",\"cout\"],\"commands\":[ \u{2026}add/connect ops\u{2026} ]}}\n\
             \x20 Pin blocks are created for you — inside the body just wire \"from\":\"a\" or \
             \"to\":\"sum\" by pin name. Wire EVERY output pin. Bodies may instance previously \
             defined chips (that's how you build hierarchy). Redefining a name updates every \
             placed instance.\n\
             - {{\"op\":\"test\",\"steps\":[{{\"set\":{{\"WE\":1,\"D\":1}}}},\
             {{\"set\":{{\"WE\":0}}}},{{\"set\":{{\"D\":0}}}}]}}\n\
             \x20 Each step sets switches/buttons (by ref, #id, or label) and reports every LED \
             after settling — settings persist across steps. Use it to prove a register HOLDS \
             its value, a memory writes only when enabled, etc.\n\
             \n\
             HANDLES: \"from\"/\"to\"/\"target\" accept the \"ref\" you gave a new block, a \
             numeric id from the context below, or a block's label.\n\
             PORTS: input ports are 0,1,2\u{2026} top-to-bottom; outputs are usually just 0. On \
             chip instances use PIN NAMES: \"to_port\":\"cin\", \"from_port\":\"sum\". If you \
             OMIT \"to_port\", the first FREE input is picked automatically — so two plain \
             connects fill a 2-input gate's ports 0 then 1. An input may take several wires \
             (they OR together). LEDs have no outputs; switches/constants have no inputs.\n\
             \n\
             EXAMPLE — define a verified building block, then use it:\n\
             ```json\n\
             {{\"commands\":[\n\
             \x20{{\"op\":\"defchip\",\"name\":\"HalfAdder\",\"inputs\":[\"a\",\"b\"],\
             \"outputs\":[\"sum\",\"carry\"],\"commands\":[\n\
             \x20 {{\"op\":\"add\",\"ref\":\"x\",\"kind\":\"xor\",\"x\":8,\"y\":0}},\n\
             \x20 {{\"op\":\"add\",\"ref\":\"g\",\"kind\":\"and\",\"x\":8,\"y\":5}},\n\
             \x20 {{\"op\":\"connect\",\"from\":\"a\",\"to\":\"x\"}},\n\
             \x20 {{\"op\":\"connect\",\"from\":\"b\",\"to\":\"x\"}},\n\
             \x20 {{\"op\":\"connect\",\"from\":\"a\",\"to\":\"g\"}},\n\
             \x20 {{\"op\":\"connect\",\"from\":\"b\",\"to\":\"g\"}},\n\
             \x20 {{\"op\":\"connect\",\"from\":\"x\",\"to\":\"sum\"}},\n\
             \x20 {{\"op\":\"connect\",\"from\":\"g\",\"to\":\"carry\"}}]}},\n\
             \x20{{\"op\":\"add\",\"ref\":\"ha\",\"kind\":\"HalfAdder\",\"x\":8,\"y\":0}}\n\
             ]}}\n\
             ```\n\
             \n\
             LAYOUT: integer grid, x right, y down; a gate is ~2 wide and (inputs) tall, chips ~3 \
             wide. Never overlap blocks. One column per stage (x = 0, 8, 16, \u{2026}), ~4 rows \
             between blocks in a column. Position is cosmetic — correctness comes from wires.\n\
             \n\
             FINISHING: when the auto-check shows the circuit does what the user asked, reply \
             with a short summary and NO json block — that ends the run. Include a json block \
             ONLY to change or test the circuit; for plain questions, just answer.\n\
             \n\
             CURRENT STATE:\n{}",
            self.circuit_context()
        )
    }

    /// The automatic post-edit "auto-check" handed back to the agent: skipped-command errors,
    /// chip-verification and test results, unwired inputs, and the circuit's real behavior (a
    /// truth table when small enough, else a settled snapshot). This is what lets the assistant
    /// verify and fix its own work without the user prodding it.
    fn agent_observation(&self, report: &AiEditReport) -> String {
        use std::fmt::Write as _;
        let c = &self.manager.circuit;
        let inputs: Vec<&Block> = c
            .iter_blocks()
            .filter(|b| matches!(b.ty, BlockType::Switch | BlockType::Button))
            .collect();
        let outputs: Vec<&Block> = c.iter_blocks().filter(|b| b.ty == BlockType::Led).collect();
        let has_clock = c.iter_blocks().any(|b| b.ty == BlockType::Clock);

        let mut out = String::from("Auto-check after your edits.\n");

        if !report.errors.is_empty() {
            out.push_str("SKIPPED COMMANDS (these did NOT happen — fix and resend them):\n");
            for e in &report.errors {
                let _ = writeln!(out, "  - {e}");
            }
        }
        for obs in &report.observations {
            let _ = writeln!(out, "{obs}");
        }

        let name = |b: &Block| match &b.label {
            Some(l) if !l.trim().is_empty() => format!("#{}({})", b.id, l.trim()),
            _ => format!("#{}", b.id),
        };

        // Unwired inputs are the most common silent bug — list them explicitly.
        let unwired = self.unwired_inputs();
        if !unwired.is_empty() {
            let _ = writeln!(
                out,
                "UNWIRED INPUTS ({}): {}{}",
                unwired.len(),
                unwired
                    .iter()
                    .take(12)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", "),
                if unwired.len() > 12 { ", …" } else { "" }
            );
        }

        // A fresh sim so the user's live switch settings aren't disturbed.
        let mut sim = Simulation::build(&self.manager.circuit, &self.manager.chips);

        if inputs.is_empty() || outputs.is_empty() {
            let _ = writeln!(
                out,
                "Circuit has {} switch/button input(s) and {} LED output(s); add at least one of \
                 each to make behavior observable.",
                inputs.len(),
                outputs.len()
            );
        } else if inputs.len() > 6 {
            // Too many combinations for a full table — give a settled snapshot instead and point
            // at the `test` op for targeted verification.
            sim.step(0.0);
            let states: Vec<String> = inputs
                .iter()
                .map(|b| format!("{}={}", name(b), sim.switch_value(b.id) as u8))
                .collect();
            let reads: Vec<String> = outputs
                .iter()
                .map(|b| format!("{}={}", name(b), sim.led_value(b.id).unwrap_or(false) as u8))
                .collect();
            let _ = writeln!(
                out,
                "{} inputs — too many for a full truth table. Settled snapshot: inputs {}; \
                 LEDs {}. Use {{\"op\":\"test\",\"steps\":[…]}} to verify specific sequences.",
                inputs.len(),
                states.join(" "),
                reads.join(" ")
            );
        } else {
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
        }
        out.push_str(
            "If this matches the request, reply with a brief summary and NO commands. Otherwise \
             fix it with more commands.",
        );
        out
    }

    /// Input ports (on gates, LEDs, chip instances) with no incoming wire, as short descriptions.
    fn unwired_inputs(&self) -> Vec<String> {
        let c = &self.manager.circuit;
        let mut out = Vec::new();
        for b in c.iter_blocks() {
            let n = b.input_count(&self.manager.chips);
            for i in 0..n as u16 {
                if !c.has_connection_into(Port::input(b.id, i)) {
                    let kind = match b.ty {
                        BlockType::Chip(cid) => self
                            .manager
                            .chips
                            .get(cid)
                            .map(|d| d.name.clone())
                            .unwrap_or_else(|| "chip".to_string()),
                        ty => palette_name(ty).to_string(),
                    };
                    let label = b
                        .label
                        .as_deref()
                        .map(|l| format!("({l})"))
                        .unwrap_or_default();
                    out.push(format!("#{}{label} {kind} in{i}", b.id));
                }
            }
        }
        out
    }

    /// A compact text view of the chip library + circuit for the assistant: blocks, wires, and
    /// each chip's pin names. Far fewer tokens than raw JSON, and easier for small models to read.
    fn circuit_context(&self) -> String {
        use std::fmt::Write as _;
        let c = &self.manager.circuit;
        let mut out = String::new();

        if !self.manager.chips.is_empty() {
            out.push_str("Chip library (instance with \"kind\":\"<Name>\"):\n");
            for d in self.manager.chips.iter() {
                let pins = |ids: &[BlockId]| {
                    ids.iter()
                        .map(|id| {
                            d.circuit
                                .block(*id)
                                .and_then(|b| b.label.clone())
                                .unwrap_or_else(|| "?".to_string())
                        })
                        .collect::<Vec<_>>()
                        .join(", ")
                };
                let _ = writeln!(
                    out,
                    "- {}: inputs [{}] outputs [{}]",
                    d.name,
                    pins(&d.input_blocks),
                    pins(&d.output_blocks)
                );
            }
        }

        if c.blocks.is_empty() {
            out.push_str("The canvas is empty (no blocks placed yet).");
            return out;
        }

        out.push_str("Blocks (#id kind @(x,y) \"label\"):\n");
        for b in c.iter_blocks() {
            let kind = match b.ty {
                BlockType::Chip(cid) => self
                    .manager
                    .chips
                    .get(cid)
                    .map(|d| format!("CHIP:{}", d.name))
                    .unwrap_or_else(|| "CHIP:?".to_string()),
                ty if ty.variable_inputs() && b.gate_inputs() != 2 => {
                    format!("{}({}in)", palette_name(ty), b.gate_inputs())
                }
                ty => palette_name(ty).to_string(),
            };
            let label = b
                .label
                .as_deref()
                .map(|l| format!(" \"{l}\""))
                .unwrap_or_default();
            let state = match b.ty {
                BlockType::Switch | BlockType::Button => format!(" ={}", b.state as u8),
                _ => String::new(),
            };
            let _ = writeln!(
                out,
                "#{} {kind} @({},{}){label}{state}",
                b.id, b.pos.x, b.pos.y
            );
        }

        let wires: Vec<String> = c
            .iter_connections()
            .map(|w| {
                format!(
                    "#{}.{}\u{2192}#{}.{}",
                    w.from.block, w.from.index, w.to.block, w.to.index
                )
            })
            .collect();
        if wires.is_empty() {
            out.push_str("No wires yet.");
        } else {
            let _ = writeln!(
                out,
                "Wires (out\u{2192}in, block.port):\n{}",
                wires.join("  ")
            );
        }
        out
    }

    /// Apply a batch of AI-proposed edits as one undoable step. Handles are resolved (batch refs
    /// first, then numeric ids, then labels), ports are validated — with pin-name lookup on chip
    /// instances and auto-assignment of the first free input when the port is omitted — and any
    /// op that can't be applied is skipped and reported rather than aborting the whole batch.
    /// `defchip` ops extend the chip library (verified in isolation); `test` ops run against the
    /// finished circuit. Returns a summary for the chat "activity" view + agent feedback.
    fn apply_ai_ops(&mut self, ops: Vec<RawOp>) -> AiEditReport {
        let mut refs: HashMap<String, BlockId> = HashMap::new();
        let mut pending: BTreeMap<BlockId, Block> = BTreeMap::new();
        let mut used_inputs: BTreeSet<(BlockId, u16)> = BTreeSet::new();
        let mut batch: Vec<EditCommand> = Vec::new();
        let mut activity: Vec<String> = Vec::new();
        let mut errors: Vec<String> = Vec::new();
        let mut observations: Vec<String> = Vec::new();
        let mut new_ids: Vec<BlockId> = Vec::new();
        let mut tests: Vec<Vec<Vec<(String, bool)>>> = Vec::new();

        let mut chips_changed = false;
        for op in ops {
            match op {
                RawOp::DefChip {
                    name,
                    inputs,
                    outputs,
                    ops,
                } => match self.build_chip(&name, &inputs, &outputs, ops) {
                    Ok(obs) => {
                        activity.push(format!(
                            "Define chip {name} ({} in, {} out)",
                            inputs.len(),
                            outputs.len()
                        ));
                        observations.push(obs);
                        self.sim_dirty = true;
                        chips_changed = true;
                    }
                    Err(e) => errors.push(format!("defchip {name}: {e}")),
                },
                RawOp::Add {
                    r#ref,
                    kind,
                    x,
                    y,
                    inputs,
                    label,
                    state,
                } => {
                    let (ty, shown) = match &kind {
                        AddKind::Prim(p) => (*p, palette_name(*p).to_string()),
                        AddKind::Chip(n) => match self.find_chip(n) {
                            Some(cid) => (BlockType::Chip(cid), n.clone()),
                            None => {
                                errors.push(format!(
                                    "add: unknown chip \"{n}\" — define it with defchip first"
                                ));
                                continue;
                            }
                        },
                    };
                    let id = self.manager.circuit.allocate_block_id();
                    let mut b = Block::new(id, ty, Pos::new(x, y));
                    if ty.variable_inputs() {
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
                    activity.push(format!("Add {shown} at ({x}, {y})"));
                    batch.push(EditCommand::AddBlock { block: b });
                }
                RawOp::Connect {
                    from,
                    from_port,
                    to,
                    to_port,
                } => {
                    let (Some(fb), Some(tb)) = (
                        resolve_handle(&refs, &pending, &self.manager.circuit, &from),
                        resolve_handle(&refs, &pending, &self.manager.circuit, &to),
                    ) else {
                        errors.push(format!("connect: unknown block ({from} \u{2192} {to})"));
                        continue;
                    };
                    let fp = match port_out(
                        &pending,
                        &self.manager.circuit,
                        &self.manager.chips,
                        fb,
                        from_port.as_ref(),
                    ) {
                        Ok(p) => p,
                        Err(e) => {
                            errors.push(format!("connect {from}\u{2192}{to}: {e}"));
                            continue;
                        }
                    };
                    let tp = match port_in(
                        &pending,
                        &self.manager.circuit,
                        &self.manager.chips,
                        &used_inputs,
                        tb,
                        to_port.as_ref(),
                    ) {
                        Ok(p) => p,
                        Err(e) => {
                            errors.push(format!("connect {from}\u{2192}{to}: {e}"));
                            continue;
                        }
                    };
                    used_inputs.insert((tb, tp));
                    let cid = self.manager.circuit.allocate_conn_id();
                    activity.push(format!("Wire {from}.{fp} \u{2192} {to}.{tp}"));
                    batch.push(EditCommand::AddConnection {
                        conn: Connection {
                            id: cid,
                            from: Port::output(fb, fp),
                            to: Port::input(tb, tp),
                        },
                    });
                }
                RawOp::Remove { target } => {
                    match resolve_handle(&refs, &pending, &self.manager.circuit, &target) {
                        Some(id) if self.manager.circuit.block(id).is_some() => {
                            activity.push(format!("Remove block {target}"));
                            batch.push(EditCommand::RemoveBlock { id });
                        }
                        _ => errors.push(format!("remove: unknown block {target}")),
                    }
                }
                RawOp::Move { target, x, y } => {
                    match resolve_handle(&refs, &pending, &self.manager.circuit, &target) {
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
                    match resolve_handle(&refs, &pending, &self.manager.circuit, &target) {
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
                RawOp::Test { steps } => tests.push(steps),
            }
        }

        let applied = !batch.is_empty();
        if applied {
            self.manager.apply(batch);
            self.after_structural_edit();
            // Select what the AI just created, so it's easy to see/move/delete.
            self.selection = new_ids
                .into_iter()
                .filter(|id| self.manager.circuit.block(*id).is_some())
                .collect();
            self.selected_conns.clear();
        }
        // Scripted tests run against the finished circuit (they see this batch's edits).
        for steps in &tests {
            activity.push(format!("Ran test ({} steps)", steps.len()));
            observations.push(self.run_ai_test(&refs, steps));
        }
        AiEditReport {
            activity,
            errors,
            // Chip-library changes also count: they're saved with the project (dirty tracking).
            changed: applied || chips_changed,
            observations,
        }
    }

    /// Find a chip in the library by name (case-insensitive).
    fn find_chip(&self, name: &str) -> Option<ChipId> {
        let want = name.trim().to_lowercase();
        self.manager
            .chips
            .iter()
            .find(|d| d.name.trim().to_lowercase() == want)
            .map(|d| d.id)
    }

    /// Build and register a reusable chip from declared pins + inner ops run against a scratch
    /// circuit. Pin blocks (Switch = input, Led = output) are created automatically, labeled and
    /// ref'd by their pin names. Redefining an existing name replaces the chip in place, so all
    /// instances update. Returns the observation shown to the model (pins + isolated truth
    /// table when feasible).
    fn build_chip(
        &mut self,
        name: &str,
        inputs: &[String],
        outputs: &[String],
        ops: Vec<RawOp>,
    ) -> Result<String, String> {
        if name.trim().is_empty() {
            return Err("chip name is empty".to_string());
        }
        if outputs.is_empty() {
            return Err("a chip needs at least one output pin".to_string());
        }
        if inputs.len() > 16 || outputs.len() > 16 {
            return Err("too many pins (max 16 inputs and 16 outputs)".to_string());
        }

        let mut c = Circuit::new(name);
        let mut refs: HashMap<String, BlockId> = HashMap::new();
        let mut used_inputs: BTreeSet<(BlockId, u16)> = BTreeSet::new();
        let mut errs: Vec<String> = Vec::new();
        let empty_pending: BTreeMap<BlockId, Block> = BTreeMap::new();

        let mut in_ids = Vec::with_capacity(inputs.len());
        for (i, pin) in inputs.iter().enumerate() {
            let id = c.allocate_block_id();
            let mut b = Block::new(id, BlockType::Switch, Pos::new(0, (i as i32) * 4));
            b.label = Some(pin.clone());
            c.insert_block(b);
            refs.insert(pin.clone(), id);
            in_ids.push(id);
        }
        let mut out_ids = Vec::with_capacity(outputs.len());
        for (i, pin) in outputs.iter().enumerate() {
            let id = c.allocate_block_id();
            let mut b = Block::new(id, BlockType::Led, Pos::new(28, (i as i32) * 4));
            b.label = Some(pin.clone());
            c.insert_block(b);
            refs.insert(pin.clone(), id);
            out_ids.push(id);
        }

        for op in ops {
            match op {
                RawOp::Add {
                    r#ref,
                    kind,
                    x,
                    y,
                    inputs: n_in,
                    label,
                    state,
                } => {
                    let ty = match &kind {
                        AddKind::Prim(p) => *p,
                        AddKind::Chip(n) => match self.find_chip(n) {
                            Some(cid) => BlockType::Chip(cid),
                            None => {
                                errs.push(format!("unknown chip \"{n}\" inside body"));
                                continue;
                            }
                        },
                    };
                    let id = c.allocate_block_id();
                    let mut b = Block::new(id, ty, Pos::new(x, y));
                    if ty.variable_inputs() {
                        if let Some(n) = n_in {
                            b.inputs = Some(n.clamp(2, MAX_GATE_INPUTS as u16));
                        }
                    }
                    if let Some(l) = label {
                        if !l.trim().is_empty() {
                            b.label = Some(l);
                        }
                    }
                    b.state = state;
                    c.insert_block(b);
                    if let Some(r) = r#ref {
                        refs.insert(r, id);
                    }
                }
                RawOp::Connect {
                    from,
                    from_port,
                    to,
                    to_port,
                } => {
                    let (Some(fb), Some(tb)) = (
                        resolve_handle(&refs, &empty_pending, &c, &from),
                        resolve_handle(&refs, &empty_pending, &c, &to),
                    ) else {
                        errs.push(format!("connect: unknown block ({from} \u{2192} {to})"));
                        continue;
                    };
                    let fp = match port_out(
                        &empty_pending,
                        &c,
                        &self.manager.chips,
                        fb,
                        from_port.as_ref(),
                    ) {
                        Ok(p) => p,
                        Err(e) => {
                            errs.push(format!("connect {from}\u{2192}{to}: {e}"));
                            continue;
                        }
                    };
                    let tp = match port_in(
                        &empty_pending,
                        &c,
                        &self.manager.chips,
                        &used_inputs,
                        tb,
                        to_port.as_ref(),
                    ) {
                        Ok(p) => p,
                        Err(e) => {
                            errs.push(format!("connect {from}\u{2192}{to}: {e}"));
                            continue;
                        }
                    };
                    used_inputs.insert((tb, tp));
                    let cid = c.allocate_conn_id();
                    c.insert_connection(Connection {
                        id: cid,
                        from: Port::output(fb, fp),
                        to: Port::input(tb, tp),
                    });
                }
                RawOp::DefChip {
                    name: n,
                    inputs: i,
                    outputs: o,
                    ops: body,
                } => {
                    // Nested definition: register it globally, then the outer body can use it.
                    if let Err(e) = self.build_chip(&n, &i, &o, body) {
                        errs.push(format!("nested defchip {n}: {e}"));
                    }
                }
                RawOp::Label { target, text } => {
                    match resolve_handle(&refs, &empty_pending, &c, &target) {
                        Some(id) => {
                            if let Some(b) = c.block_mut(id) {
                                b.label = text;
                            }
                        }
                        None => errs.push(format!("label: unknown block {target}")),
                    }
                }
                RawOp::Move { target, x, y } => {
                    match resolve_handle(&refs, &empty_pending, &c, &target) {
                        Some(id) => {
                            if let Some(b) = c.block_mut(id) {
                                b.pos = Pos::new(x, y);
                            }
                        }
                        None => errs.push(format!("move: unknown block {target}")),
                    }
                }
                RawOp::Remove { target } => {
                    match resolve_handle(&refs, &empty_pending, &c, &target) {
                        Some(id) if !in_ids.contains(&id) && !out_ids.contains(&id) => {
                            c.remove_block(id);
                        }
                        _ => errs.push(format!("remove: unknown block or pin {target}")),
                    }
                }
                RawOp::Test { .. } => {
                    errs.push("test inside a defchip body is ignored".to_string());
                }
            }
        }

        // Register (reusing the id when redefining, so every instance updates).
        let id = self
            .find_chip(name)
            .unwrap_or_else(|| self.manager.chips.allocate_id());
        let def = ChipDef {
            id,
            name: name.trim().to_string(),
            circuit: c,
            input_blocks: in_ids,
            output_blocks: out_ids,
        };
        self.manager.chips.insert(def);

        // Verify the chip in isolation and report what it actually does.
        let mut obs = self.chip_observation(id, name, inputs, outputs);
        for e in errs {
            obs.push_str(&format!("\n  skipped: {e}"));
        }
        Ok(obs)
    }

    /// Describe a freshly (re)defined chip to the model: pins, unwired-output warnings, and an
    /// isolated truth table when the pin count allows one.
    fn chip_observation(
        &self,
        id: ChipId,
        name: &str,
        inputs: &[String],
        outputs: &[String],
    ) -> String {
        use std::fmt::Write as _;
        let mut out = format!(
            "Defined chip {name}: inputs [{}] outputs [{}].",
            inputs.join(", "),
            outputs.join(", ")
        );
        let Some(def) = self.manager.chips.get(id) else {
            return out;
        };
        // An output pin whose Led marker has no incoming wire is always a bug.
        for (i, ob) in def.output_blocks.iter().enumerate() {
            if !def.circuit.has_connection_into(Port::input(*ob, 0)) {
                let _ = write!(
                    out,
                    "\n  WARNING: output pin \"{}\" is not wired to anything.",
                    outputs.get(i).map(String::as_str).unwrap_or("?")
                );
            }
        }
        if inputs.len() <= 6 && !outputs.is_empty() {
            let mut sim = Simulation::build(&def.circuit, &self.manager.chips);
            let _ = write!(
                out,
                "\n  Verified truth table [{}] \u{2192} [{}]:",
                inputs.join(" "),
                outputs.join(" ")
            );
            let n = def.input_blocks.len();
            for combo in 0..(1u32 << n) {
                for (i, bid) in def.input_blocks.iter().enumerate() {
                    sim.set_switch(*bid, (combo >> (n - 1 - i)) & 1 == 1);
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
                let outs: Vec<&str> = def
                    .output_blocks
                    .iter()
                    .map(|bid| {
                        if sim.led_value(*bid).unwrap_or(false) {
                            "1"
                        } else {
                            "0"
                        }
                    })
                    .collect();
                let _ = write!(out, "\n    {} | {}", ins.join(" "), outs.join(" "));
            }
        }
        out
    }

    /// Run a scripted test on a scratch simulation of the current circuit (the user's live
    /// switch states are untouched). Each step sets switches/buttons then reads every LED.
    fn run_ai_test(
        &self,
        refs: &HashMap<String, BlockId>,
        steps: &[Vec<(String, bool)>],
    ) -> String {
        use std::fmt::Write as _;
        let empty: BTreeMap<BlockId, Block> = BTreeMap::new();
        let mut sim = Simulation::build(&self.manager.circuit, &self.manager.chips);
        sim.step(0.0);
        let name = |b: &Block| match &b.label {
            Some(l) if !l.trim().is_empty() => l.trim().to_string(),
            _ => format!("#{}", b.id),
        };
        let leds: Vec<&Block> = self
            .manager
            .circuit
            .iter_blocks()
            .filter(|b| b.ty == BlockType::Led)
            .collect();
        let mut out = String::from("Test results:");
        if leds.is_empty() {
            out.push_str("\n  (no LEDs to observe — add LEDs to make outputs visible)");
        }
        for (i, step) in steps.iter().enumerate() {
            let mut sets: Vec<String> = Vec::new();
            for (handle, val) in step {
                match resolve_handle(refs, &empty, &self.manager.circuit, handle) {
                    Some(id)
                        if self.manager.circuit.block(id).is_some_and(|b| {
                            matches!(b.ty, BlockType::Switch | BlockType::Button)
                        }) =>
                    {
                        sim.set_switch(id, *val);
                        sets.push(format!("{handle}={}", *val as u8));
                    }
                    Some(_) => sets.push(format!("{handle}\u{2260}switch(ignored)")),
                    None => sets.push(format!("{handle}=unknown(ignored)")),
                }
            }
            sim.step(0.0);
            let reads: Vec<String> = leds
                .iter()
                .map(|b| format!("{}={}", name(b), sim.led_value(b.id).unwrap_or(false) as u8))
                .collect();
            let _ = write!(
                out,
                "\n  step {}: set {} \u{2192} {}",
                i + 1,
                if sets.is_empty() {
                    "(nothing)".to_string()
                } else {
                    sets.join(", ")
                },
                reads.join(", ")
            );
        }
        out
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

        // Momentary buttons: a Button reads high only while the primary mouse button is held on
        // it, and returns to low on release (unlike a Switch, which latches). This drives the
        // live simulation state without touching the block's authored default.
        let (primary_pressed, primary_released) =
            ctx.input(|i| (i.pointer.primary_pressed(), i.pointer.primary_released()));
        if primary_pressed {
            if let Some(p) = pointer.or_else(|| response.hover_pos()) {
                if let Some(bid) = self.hit_block(p, origin) {
                    if self.manager.circuit.block(bid).map(|b| b.ty) == Some(BlockType::Button) {
                        self.pressed_button = Some(bid);
                        self.sim.set_switch(bid, true);
                        self.sim.step(0.0);
                    }
                }
            }
        }
        if primary_released {
            if let Some(bid) = self.pressed_button.take() {
                self.sim.set_switch(bid, false);
                self.sim.step(0.0);
            }
        }

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
            // A Switch latches its state on click (Buttons are momentary — see handle_canvas_input).
            let switch = self
                .manager
                .circuit
                .block(bid)
                .map(|b| (b.ty == BlockType::Switch, b.state));
            if let Some((is_switch, state)) = switch {
                if is_switch {
                    let ns = !state;
                    if let Some(bm) = self.manager.circuit.block_mut(bid) {
                        bm.state = ns;
                    }
                    self.sim.set_switch(bid, ns);
                    // Settle immediately so LEDs/wires update even while paused.
                    self.sim.step(0.0);
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
            let selected = self.selected_conns.contains(&conn.id);
            let color = if selected {
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
                // Build the agent feedback from the full report (skips, chip tables, test
                // results) BEFORE the activity list is handed to the chat view.
                let feedback = doc.agent_observation(&report);
                doc.ai.attach_edit_result(report.activity, report.errors);
                if doc.ai.should_continue_or_note() {
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

/// The outcome of applying a batch of AI-proposed edits, surfaced in the chat "activity" view
/// and fed back to the agent as its auto-check.
struct AiEditReport {
    /// One human-readable line per edit that was applied.
    activity: Vec<String>,
    /// One line per edit that was skipped (bad handle / port), so the user can see what didn't
    /// happen — and the model can fix it.
    errors: Vec<String>,
    /// Whether anything actually changed (empty batches touch nothing).
    changed: bool,
    /// Rich feedback for the agent loop: chip truth tables, scripted-test results, warnings.
    observations: Vec<String>,
}

/// Resolve an AI block handle to a real block id: a batch ref first, then a numeric id (with a
/// leading `#` tolerated), then a block label (case-insensitive) — so a model can say
/// `"target":"EN"` about a labeled switch.
fn resolve_handle(
    refs: &HashMap<String, BlockId>,
    pending: &BTreeMap<BlockId, Block>,
    circuit: &Circuit,
    handle: &str,
) -> Option<BlockId> {
    if let Some(id) = refs.get(handle) {
        return Some(*id);
    }
    let trimmed = handle.trim().trim_start_matches('#');
    if let Ok(n) = trimmed.parse::<u32>() {
        let id = BlockId(n);
        if circuit.block(id).is_some() || pending.contains_key(&id) {
            return Some(id);
        }
    }
    let want = trimmed.to_lowercase();
    let by_label = |b: &Block| {
        b.label
            .as_ref()
            .is_some_and(|l| l.trim().to_lowercase() == want)
    };
    pending
        .values()
        .find(|b| by_label(b))
        .or_else(|| circuit.iter_blocks().find(|b| by_label(b)))
        .map(|b| b.id)
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

/// Pin index of `name` on chip `cid` (input or output side), matched against the pin blocks'
/// labels case-insensitively.
fn chip_pin_by_name(
    chips: &llmc::model::ChipLibrary,
    cid: ChipId,
    name: &str,
    input: bool,
) -> Option<u16> {
    let def = chips.get(cid)?;
    let list = if input {
        &def.input_blocks
    } else {
        &def.output_blocks
    };
    let want = name.trim().to_lowercase();
    list.iter()
        .position(|bid| {
            def.circuit
                .block(*bid)
                .and_then(|b| b.label.as_ref())
                .is_some_and(|l| l.trim().to_lowercase() == want)
        })
        .map(|i| i as u16)
}

/// Resolve an output-port selector on `id`: omitted → 0, an index is validated, a name is looked
/// up among a chip's output pins.
fn port_out(
    pending: &BTreeMap<BlockId, Block>,
    circuit: &Circuit,
    chips: &llmc::model::ChipLibrary,
    id: BlockId,
    sel: Option<&PortSel>,
) -> Result<u16, String> {
    let Some(b) = block_ref(pending, circuit, id) else {
        return Err("unknown source block".to_string());
    };
    let count = b.output_count(chips);
    if count == 0 {
        return Err("the source block has no outputs (LEDs can't drive wires)".to_string());
    }
    match sel {
        None => Ok(0),
        Some(PortSel::Index(i)) if (*i as usize) < count => Ok(*i),
        Some(PortSel::Index(i)) => Err(format!("output port {i} out of range (block has {count})")),
        Some(PortSel::Name(n)) => match b.ty {
            BlockType::Chip(cid) => chip_pin_by_name(chips, cid, n, false)
                .ok_or_else(|| format!("no output pin named \"{n}\" on that chip")),
            _ => Err(format!(
                "\"{n}\": named ports only work on chip instances; use a number"
            )),
        },
    }
}

/// Resolve an input-port selector on `id`: omitted → the first FREE input (not wired in the
/// circuit and not used earlier in this batch; falls back to 0, since inputs OR-combine), an
/// index is validated, a name is looked up among a chip's input pins.
fn port_in(
    pending: &BTreeMap<BlockId, Block>,
    circuit: &Circuit,
    chips: &llmc::model::ChipLibrary,
    used: &BTreeSet<(BlockId, u16)>,
    id: BlockId,
    sel: Option<&PortSel>,
) -> Result<u16, String> {
    let Some(b) = block_ref(pending, circuit, id) else {
        return Err("unknown destination block".to_string());
    };
    let count = b.input_count(chips);
    if count == 0 {
        return Err(
            "the destination block has no inputs (switches/constants can't be driven)".to_string(),
        );
    }
    match sel {
        None => {
            for i in 0..count as u16 {
                if !used.contains(&(id, i)) && !circuit.has_connection_into(Port::input(id, i)) {
                    return Ok(i);
                }
            }
            Ok(0)
        }
        Some(PortSel::Index(i)) if (*i as usize) < count => Ok(*i),
        Some(PortSel::Index(i)) => Err(format!("input port {i} out of range (block has {count})")),
        Some(PortSel::Name(n)) => match b.ty {
            BlockType::Chip(cid) => chip_pin_by_name(chips, cid, n, true)
                .ok_or_else(|| format!("no input pin named \"{n}\" on that chip")),
            _ => Err(format!(
                "\"{n}\": named ports only work on chip instances; use a number"
            )),
        },
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

    fn add_op(r: &str, kind: BlockType, x: i32, y: i32) -> RawOp {
        RawOp::Add {
            r#ref: Some(r.to_string()),
            kind: AddKind::Prim(kind),
            x,
            y,
            inputs: None,
            label: None,
            state: false,
        }
    }

    /// A connect with auto-assigned ports (both selectors omitted).
    fn wire_op(from: &str, to: &str) -> RawOp {
        RawOp::Connect {
            from: from.to_string(),
            from_port: None,
            to: to.to_string(),
            to_port: None,
        }
    }

    #[test]
    fn apply_ai_ops_builds_wires_and_reports_bad_ops() {
        let mut d = doc();
        let ops = vec![
            add_op("a", BlockType::Switch, 0, 0),
            add_op("g", BlockType::And, 6, 1),
            add_op("led", BlockType::Led, 12, 2),
            wire_op("a", "g"),
            wire_op("g", "led"),
            // A switch has no input port, so this must be skipped and reported, not applied.
            wire_op("g", "a"),
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
    fn omitted_to_port_auto_assigns_free_inputs() {
        let mut d = doc();
        let ops = vec![
            add_op("a", BlockType::Switch, 0, 0),
            add_op("b", BlockType::Switch, 0, 4),
            add_op("g", BlockType::And, 6, 1),
            // Neither connect names a port — they must land on inputs 0 and 1, not both on 0.
            wire_op("a", "g"),
            wire_op("b", "g"),
        ];
        let report = d.apply_ai_ops(ops);
        assert!(report.errors.is_empty(), "errors: {:?}", report.errors);
        let mut to_ports: Vec<u16> = d
            .manager
            .circuit
            .iter_connections()
            .map(|c| c.to.index)
            .collect();
        to_ports.sort_unstable();
        assert_eq!(to_ports, vec![0, 1], "both gate inputs used");
    }

    #[test]
    fn defchip_registers_verifies_and_instances_work() {
        let mut d = doc();
        let ops = vec![
            RawOp::DefChip {
                name: "And2".to_string(),
                inputs: vec!["a".to_string(), "b".to_string()],
                outputs: vec!["y".to_string()],
                ops: vec![
                    add_op("g", BlockType::And, 8, 0),
                    wire_op("a", "g"),
                    wire_op("b", "g"),
                    wire_op("g", "y"),
                ],
            },
            add_op("s1", BlockType::Switch, 0, 0),
            add_op("s2", BlockType::Switch, 0, 4),
            RawOp::Add {
                r#ref: Some("u1".to_string()),
                kind: AddKind::Chip("And2".to_string()),
                x: 8,
                y: 0,
                inputs: None,
                label: None,
                state: false,
            },
            add_op("out", BlockType::Led, 16, 1),
            // Chip pins addressed by NAME.
            RawOp::Connect {
                from: "s1".to_string(),
                from_port: None,
                to: "u1".to_string(),
                to_port: Some(PortSel::Name("a".to_string())),
            },
            RawOp::Connect {
                from: "s2".to_string(),
                from_port: None,
                to: "u1".to_string(),
                to_port: Some(PortSel::Name("b".to_string())),
            },
            RawOp::Connect {
                from: "u1".to_string(),
                from_port: Some(PortSel::Name("y".to_string())),
                to: "out".to_string(),
                to_port: None,
            },
        ];
        let report = d.apply_ai_ops(ops);
        assert!(report.errors.is_empty(), "errors: {:?}", report.errors);
        // The chip was verified in isolation: its truth table is in the observations.
        let obs = report.observations.join("\n");
        assert!(obs.contains("Defined chip And2"), "obs: {obs}");
        assert!(obs.contains("1 1 | 1"), "verified AND table: {obs}");

        // The instanced chip actually computes AND on the canvas.
        d.rebuild_sim();
        let s1 = d
            .manager
            .circuit
            .iter_blocks()
            .find(|b| b.ty == BlockType::Switch)
            .unwrap()
            .id;
        let led = d
            .manager
            .circuit
            .iter_blocks()
            .find(|b| b.ty == BlockType::Led)
            .unwrap()
            .id;
        let s2 = d
            .manager
            .circuit
            .iter_blocks()
            .filter(|b| b.ty == BlockType::Switch)
            .nth(1)
            .unwrap()
            .id;
        d.sim.set_switch(s1, true);
        d.sim.set_switch(s2, true);
        d.sim.step(0.0);
        assert_eq!(
            d.sim.led_value(led),
            Some(true),
            "1 AND 1 = 1 through the chip"
        );
        d.sim.set_switch(s2, false);
        d.sim.step(0.0);
        assert_eq!(
            d.sim.led_value(led),
            Some(false),
            "1 AND 0 = 0 through the chip"
        );
    }

    #[test]
    fn scripted_test_reports_outputs_per_step() {
        let mut d = doc();
        let ops = vec![
            RawOp::Add {
                r#ref: Some("d0".to_string()),
                kind: AddKind::Prim(BlockType::Switch),
                x: 0,
                y: 0,
                inputs: None,
                label: Some("D".to_string()),
                state: false,
            },
            RawOp::Add {
                r#ref: Some("q0".to_string()),
                kind: AddKind::Prim(BlockType::Led),
                x: 8,
                y: 0,
                inputs: None,
                label: Some("Q".to_string()),
                state: false,
            },
            wire_op("d0", "q0"),
            RawOp::Test {
                steps: vec![
                    vec![("D".to_string(), true)],
                    vec![("D".to_string(), false)],
                ],
            },
        ];
        let report = d.apply_ai_ops(ops);
        assert!(report.errors.is_empty(), "errors: {:?}", report.errors);
        let obs = report.observations.join("\n");
        assert!(obs.contains("step 1: set D=1 \u{2192} Q=1"), "obs: {obs}");
        assert!(obs.contains("step 2: set D=0 \u{2192} Q=0"), "obs: {obs}");
    }

    #[test]
    fn observation_reports_skips_and_unwired_inputs() {
        let mut d = doc();
        let ops = vec![
            add_op("g", BlockType::And, 6, 1),
            // Bad connect: unknown block — must be skipped AND fed back to the model.
            wire_op("ghost", "g"),
        ];
        let report = d.apply_ai_ops(ops);
        assert_eq!(report.errors.len(), 1);
        let obs = d.agent_observation(&report);
        assert!(obs.contains("SKIPPED COMMANDS"), "skips fed back: {obs}");
        assert!(
            obs.contains("UNWIRED INPUTS"),
            "unwired gate inputs listed: {obs}"
        );
    }

    #[test]
    fn multi_driver_input_is_kept_and_or_combined() {
        let mut d = doc();
        let a = d.manager.add_block(BlockType::Switch, Pos::new(0, 0));
        let b = d.manager.add_block(BlockType::Switch, Pos::new(0, 4));
        let led = d.manager.add_block(BlockType::Led, Pos::new(6, 1));
        // Two switches driving the SAME LED input — allowed now (no replace-on-connect).
        d.manager.connect(Port::output(a, 0), Port::input(led, 0));
        d.manager.connect(Port::output(b, 0), Port::input(led, 0));
        assert_eq!(
            d.manager.circuit.connections.len(),
            2,
            "both wires are kept, not replaced"
        );

        d.rebuild_sim();
        d.sim.set_switch(a, true);
        d.sim.set_switch(b, false);
        d.sim.step(0.0);
        assert_eq!(
            d.sim.led_value(led),
            Some(true),
            "input is high if either driver is high"
        );
        d.sim.set_switch(a, false);
        d.sim.step(0.0);
        assert_eq!(
            d.sim.led_value(led),
            Some(false),
            "input is low only when all drivers are low"
        );
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

        let report = AiEditReport {
            activity: Vec::new(),
            errors: Vec::new(),
            changed: true,
            observations: Vec::new(),
        };
        let obs = d.agent_observation(&report);
        assert!(obs.contains("Truth table"), "reports a truth table");
        // AND: high only when both inputs are high.
        assert!(obs.contains("0 0 | 0"));
        assert!(obs.contains("0 1 | 0"));
        assert!(obs.contains("1 0 | 0"));
        assert!(obs.contains("1 1 | 1"));
    }
}
