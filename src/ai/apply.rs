//! Applying parsed assistant commands to a project.
//!
//! Every command in one reply is applied immediately (so a later command can wire a block — or
//! instance a chip — created by an earlier one) inside a single transaction, so the whole reply
//! is ONE undo step, chip definitions included. Commands that can't be applied are skipped and
//! reported back (to the user's activity view and to the model's auto-check) instead of
//! aborting the batch.

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;

use crate::backend::{CircuitManager, EditCommand, Simulation, Txn};
use crate::model::{
    Block, BlockId, BlockType, ChipDef, ChipId, ChipLibrary, Circuit, Connection, PinMap, Port,
    Pos, MAX_GATE_INPUTS,
};

use super::parse::{AddKind, PortSel, RawOp, TestStep};

/// Outcome of applying one reply's commands.
#[derive(Debug, Default, Clone)]
pub struct ApplyReport {
    /// One human-readable line per edit that was applied.
    pub activity: Vec<String>,
    /// One line per command that was skipped, with the reason (so the model can fix it).
    pub errors: Vec<String>,
    /// Whether the project changed at all.
    pub changed: bool,
    /// Rich feedback for the agent loop: chip truth tables, scripted-test results, warnings.
    pub observations: Vec<String>,
    /// Blocks created in the working circuit (the UI highlights and frames them).
    pub new_blocks: Vec<BlockId>,
    /// Blocks the model pointed at with `highlight` (no edit), plus its note.
    pub highlight: Vec<BlockId>,
    pub highlight_note: Option<String>,
    /// Scripted tests run, and their expectation results.
    pub tests_run: usize,
    pub checks_passed: usize,
    pub checks_failed: usize,
}

/// Guard rails for one reply.
#[derive(Debug, Default, Clone)]
pub struct ApplyOptions {
    /// Blocks the user placed that the model must not delete (or cut apart) in this reply —
    /// set when the user's request didn't ask to change the existing circuit. Refused commands
    /// are reported back so the model builds beside the user's work or asks first.
    pub protected: HashSet<BlockId>,
}

/// Did the user's message ask to change or remove what's already there? (If not, the model
/// may only add to the circuit — never delete the user's blocks.)
pub fn user_allows_removal(message: &str) -> bool {
    let m = message.to_lowercase();
    [
        "remove",
        "delete",
        "clear",
        "erase",
        "replace",
        "rebuild",
        "redo",
        "start over",
        "from scratch",
        "get rid",
        "fix",
        "repair",
        "debug",
        "change",
        "modify",
        "rewire",
        "swap",
        "convert",
        "simplify",
        "optimi",
        "clean",
        "refactor",
        "reorganize",
        "tidy",
        "instead",
        "wrong",
        "broken",
        "doesn't work",
        "does not work",
        "not working",
        "edit",
        "update",
        "improve",
        "correct",
    ]
    .iter()
    .any(|w| m.contains(w))
}

/// A block as the user knows it: its label, else its kind (chip name for chip instances).
pub fn block_display_name(m: &CircuitManager, id: BlockId) -> String {
    let Some(b) = m.circuit.block(id) else {
        return format!("#{id}");
    };
    if let Some(l) = b.label.as_deref().map(str::trim).filter(|l| !l.is_empty()) {
        return l.to_string();
    }
    match b.ty {
        BlockType::Chip(cid) => m
            .chips
            .get(cid)
            .map(|d| d.name.clone())
            .unwrap_or_else(|| "chip".to_string()),
        ty => ty.display_name().to_string(),
    }
}

/// `name`, `name.pin` (chips), or `name in 2` (multi-input gates) for one end of a wire.
fn port_display(m: &CircuitManager, port: Port) -> String {
    let name = block_display_name(m, port.block);
    let Some(b) = m.circuit.block(port.block) else {
        return name;
    };
    let input = port.kind == crate::model::PortKind::Input;
    if let BlockType::Chip(cid) = b.ty {
        if let Some(d) = m.chips.get(cid) {
            let pin = if input {
                d.input_name(port.index as usize)
            } else {
                d.output_name(port.index as usize)
            };
            if !pin.is_empty() {
                return format!("{name}.{pin}");
            }
        }
    }
    let count = if input {
        b.input_count(&m.chips)
    } else {
        b.output_count(&m.chips)
    };
    if count > 1 {
        format!(
            "{name} {}{}",
            if input { "in " } else { "out " },
            port.index + 1
        )
    } else {
        name
    }
}

/// Apply a reply's commands to `m` as a single undo step.
pub fn apply_ops(m: &mut CircuitManager, ops: Vec<RawOp>) -> ApplyReport {
    apply_ops_with(m, ops, &ApplyOptions::default())
}

/// [`apply_ops`] with guard rails (see [`ApplyOptions`]).
pub fn apply_ops_with(m: &mut CircuitManager, ops: Vec<RawOp>, opts: &ApplyOptions) -> ApplyReport {
    let mut r = ApplyReport::default();
    let mut txn = Txn::default();
    let mut refs: HashMap<String, BlockId> = HashMap::new();
    let mut wired = wired_inputs(&m.circuit);
    let mut tests: Vec<Vec<TestStep>> = Vec::new();
    let mut inspects: Vec<InspectReq> = Vec::new();

    for op in ops {
        match op {
            RawOp::DefChip {
                name,
                inputs,
                outputs,
                ops,
            } => match build_chip(m, &mut txn, &name, &inputs, &outputs, ops) {
                Ok(obs) => {
                    r.activity.push(format!(
                        "Define chip {name} ({} in, {} out)",
                        inputs.len(),
                        outputs.len()
                    ));
                    r.observations.push(obs);
                }
                Err(e) => r.errors.push(format!("defchip {name}: {e}")),
            },
            RawOp::Add {
                r#ref,
                kind,
                x,
                y,
                inputs,
                label,
                state,
                hz,
                color,
            } => {
                let (ty, shown) = match resolve_kind(&m.chips, &kind) {
                    Ok(v) => v,
                    Err(e) => {
                        r.errors.push(format!("add: {e}"));
                        continue;
                    }
                };
                let id = m.circuit.allocate_block_id();
                let mut b = Block::new(id, ty, Pos::new(x, y));
                configure(&mut b, inputs, label, Some(state), hz, color.map(Some));
                m.apply_in(&mut txn, vec![EditCommand::AddBlock { block: b }]);
                if let Some(rf) = r#ref {
                    refs.insert(rf, id);
                }
                r.new_blocks.push(id);
                r.activity
                    .push(match m.circuit.block(id).and_then(|b| b.label.clone()) {
                        Some(l) => format!("Add {shown} \u{201c}{l}\u{201d} at ({x}, {y})"),
                        None => format!("Add {shown} at ({x}, {y})"),
                    });
            }
            RawOp::Connect {
                from,
                from_port,
                to,
                to_port,
            } => {
                let (Some(fb), Some(tb)) = (
                    resolve_handle(&refs, &m.circuit, &from),
                    resolve_handle(&refs, &m.circuit, &to),
                ) else {
                    r.errors
                        .push(format!("connect: unknown block ({from} \u{2192} {to})"));
                    continue;
                };
                let ports = port_out(&m.circuit, &m.chips, fb, from_port.as_ref()).and_then(|fp| {
                    port_in(&m.circuit, &m.chips, &wired, tb, to_port.as_ref()).map(|tp| (fp, tp))
                });
                let (fp, tp) = match ports {
                    Ok(p) => p,
                    Err(e) => {
                        r.errors.push(format!("connect {from}\u{2192}{to}: {e}"));
                        continue;
                    }
                };
                let (fport, tport) = (Port::output(fb, fp), Port::input(tb, tp));
                if m.find_connection(fport, tport).is_some() {
                    r.observations.push(format!(
                        "note: {from}.{fp} \u{2192} {to}.{tp} was already wired"
                    ));
                    continue;
                }
                let cid = m.circuit.allocate_conn_id();
                m.apply_in(
                    &mut txn,
                    vec![EditCommand::AddConnection {
                        conn: Connection {
                            id: cid,
                            from: fport,
                            to: tport,
                        },
                    }],
                );
                wired.insert((tb, tp));
                r.activity.push(format!(
                    "Wire {} \u{2192} {}",
                    port_display(m, fport),
                    port_display(m, tport)
                ));
            }
            RawOp::Disconnect {
                from,
                from_port,
                to,
                to_port,
            } => {
                let (Some(fb), Some(tb)) = (
                    resolve_handle(&refs, &m.circuit, &from),
                    resolve_handle(&refs, &m.circuit, &to),
                ) else {
                    r.errors
                        .push(format!("disconnect: unknown block ({from} \u{2192} {to})"));
                    continue;
                };
                if opts.protected.contains(&fb) && opts.protected.contains(&tb) {
                    r.errors.push(format!(
                        "disconnect {from}\u{2192}{to}: those blocks were placed by the user \
                         \u{2014} don't rewire the user's circuit unless they ask; ask first"
                    ));
                    continue;
                }
                // Ports given → that exact pair; omitted → every wire between the two blocks.
                let fp = match from_port.as_ref() {
                    Some(sel) => match port_out(&m.circuit, &m.chips, fb, Some(sel)) {
                        Ok(p) => Some(p),
                        Err(e) => {
                            r.errors.push(format!("disconnect {from}\u{2192}{to}: {e}"));
                            continue;
                        }
                    },
                    None => None,
                };
                let tp = match to_port.as_ref() {
                    Some(sel) => {
                        match port_in(&m.circuit, &m.chips, &HashSet::new(), tb, Some(sel)) {
                            Ok(p) => Some(p),
                            Err(e) => {
                                r.errors.push(format!("disconnect {from}\u{2192}{to}: {e}"));
                                continue;
                            }
                        }
                    }
                    None => None,
                };
                let doomed: Vec<Connection> = m
                    .circuit
                    .iter_connections()
                    .filter(|w| {
                        w.from.block == fb
                            && w.to.block == tb
                            && fp.is_none_or(|p| w.from.index == p)
                            && tp.is_none_or(|p| w.to.index == p)
                    })
                    .copied()
                    .collect();
                if doomed.is_empty() {
                    r.errors.push(format!(
                        "disconnect: there is no wire from {from} to {to}{}",
                        if fp.is_some() || tp.is_some() {
                            " on those ports"
                        } else {
                            ""
                        }
                    ));
                    continue;
                }
                for w in &doomed {
                    r.activity.push(format!(
                        "Unwire {} \u{2192} {}",
                        port_display(m, w.from),
                        port_display(m, w.to)
                    ));
                }
                m.apply_in(
                    &mut txn,
                    doomed
                        .iter()
                        .map(|w| EditCommand::RemoveConnection { id: w.id })
                        .collect(),
                );
                wired = wired_inputs(&m.circuit);
            }
            RawOp::Remove { target } => match resolve_handle(&refs, &m.circuit, &target) {
                Some(id) if opts.protected.contains(&id) => r.errors.push(format!(
                    "remove {target}: {} was placed by the user \u{2014} never delete the \
                     user's blocks unless they ask; build beside them, or ask first",
                    block_display_name(m, id)
                )),
                Some(id) => {
                    let shown = block_display_name(m, id);
                    m.apply_in(&mut txn, vec![EditCommand::RemoveBlock { id }]);
                    r.new_blocks.retain(|b| *b != id);
                    wired = wired_inputs(&m.circuit);
                    r.activity.push(format!("Remove {shown}"));
                }
                None => r.errors.push(format!("remove: unknown block {target}")),
            },
            RawOp::DeleteChip { name } => {
                let Some(cid) = m.chips.find_by_name(&name) else {
                    r.errors
                        .push(format!("delete_chip: there is no chip named \"{name}\""));
                    continue;
                };
                let users: Vec<String> = m
                    .chips
                    .users_of(cid)
                    .into_iter()
                    .filter_map(|u| m.chips.get(u).map(|d| d.name.clone()))
                    .collect();
                if !users.is_empty() {
                    r.errors.push(format!(
                        "delete_chip {name}: it is used inside {} \u{2014} redefine or delete \
                         those first",
                        users.join(", ")
                    ));
                    continue;
                }
                let instances: Vec<BlockId> = m
                    .circuit
                    .iter_blocks()
                    .filter(|b| b.ty == BlockType::Chip(cid))
                    .map(|b| b.id)
                    .collect();
                if instances.iter().any(|id| opts.protected.contains(id)) {
                    r.errors.push(format!(
                        "delete_chip {name}: the user placed copies of it \u{2014} ask first"
                    ));
                    continue;
                }
                let mut batch: Vec<EditCommand> = instances
                    .iter()
                    .map(|id| EditCommand::RemoveBlock { id: *id })
                    .collect();
                batch.push(EditCommand::RemoveChip { id: cid });
                m.apply_in(&mut txn, batch);
                r.new_blocks.retain(|b| !instances.contains(b));
                wired = wired_inputs(&m.circuit);
                r.activity.push(format!(
                    "Delete chip {name}{}",
                    match instances.len() {
                        0 => String::new(),
                        1 => " (and its 1 copy)".to_string(),
                        n => format!(" (and its {n} copies)"),
                    }
                ));
            }
            RawOp::Inspect { targets } => {
                let ids: Vec<BlockId> = targets
                    .iter()
                    .filter_map(|t| {
                        let id = resolve_handle(&refs, &m.circuit, t);
                        if id.is_none() {
                            r.errors.push(format!("inspect: unknown block {t}"));
                        }
                        id
                    })
                    .collect();
                if !ids.is_empty() {
                    inspects.push(InspectReq::Blocks(ids));
                }
            }
            RawOp::Trace { target } => match resolve_handle(&refs, &m.circuit, &target) {
                Some(id) => inspects.push(InspectReq::Trace(id)),
                None => r.errors.push(format!("trace: unknown block {target}")),
            },
            RawOp::Highlight { targets, note } => {
                for t in &targets {
                    match resolve_handle(&refs, &m.circuit, t) {
                        Some(id) => r.highlight.push(id),
                        None => r.errors.push(format!("highlight: unknown block {t}")),
                    }
                }
                if note.is_some() {
                    r.highlight_note = note;
                }
            }
            RawOp::Move { target, x, y } => match resolve_handle(&refs, &m.circuit, &target) {
                Some(id) => {
                    m.apply_in(
                        &mut txn,
                        vec![EditCommand::MoveBlock {
                            id,
                            to: Pos::new(x, y),
                        }],
                    );
                    r.activity
                        .push(format!("Move {} to ({x}, {y})", block_display_name(m, id)));
                }
                None => r.errors.push(format!("move: unknown block {target}")),
            },
            RawOp::Label { target, text } => match resolve_handle(&refs, &m.circuit, &target) {
                Some(id) => {
                    let shown = block_display_name(m, id);
                    r.activity.push(match &text {
                        Some(t) => format!("Name {shown} \u{201c}{t}\u{201d}"),
                        None => format!("Clear the name of {shown}"),
                    });
                    m.apply_in(&mut txn, vec![EditCommand::SetLabel { id, label: text }]);
                }
                None => r.errors.push(format!("label: unknown block {target}")),
            },
            RawOp::Set {
                target,
                inputs,
                hz,
                color,
                state,
                label,
            } => {
                let Some(id) = resolve_handle(&refs, &m.circuit, &target) else {
                    r.errors.push(format!("set: unknown block {target}"));
                    continue;
                };
                let Some(mut b) = m.circuit.block(id).cloned() else {
                    continue;
                };
                if inputs.is_some() && !b.ty.variable_inputs() {
                    r.errors.push(format!(
                        "set {target}: only AND/OR/NAND/NOR/XOR/XNOR take an input count"
                    ));
                }
                if hz.is_some() && b.ty != BlockType::Clock {
                    r.errors
                        .push(format!("set {target}: only clocks have a frequency"));
                }
                configure(&mut b, inputs, label, state, hz, color);
                let batch = m.update_block_batch(b);
                if !batch.is_empty() {
                    m.apply_in(&mut txn, batch);
                    wired = wired_inputs(&m.circuit);
                    r.activity
                        .push(format!("Update {}", block_display_name(m, id)));
                }
            }
            RawOp::Test { steps } => tests.push(steps),
        }
    }

    r.changed = !txn.is_empty();
    if r.changed {
        m.commit(txn);
    }
    // Scripted tests and read-only queries run against the finished circuit (they see this
    // reply's edits).
    for steps in &tests {
        let t = run_test(m, &refs, steps);
        r.tests_run += 1;
        r.checks_passed += t.passed;
        r.checks_failed += t.failed;
        r.activity.push(if t.passed + t.failed > 0 {
            format!(
                "Ran a {}-step test: {}/{} checks passed",
                steps.len(),
                t.passed,
                t.passed + t.failed
            )
        } else {
            format!("Ran a {}-step test", steps.len())
        });
        r.observations.push(t.text);
    }
    for q in inspects {
        r.observations.push(match q {
            InspectReq::Blocks(ids) => inspect_blocks(m, &ids),
            InspectReq::Trace(id) => trace_block(m, id),
        });
    }
    if !r.highlight.is_empty() {
        r.activity.push(format!(
            "Pointed at {}",
            r.highlight
                .iter()
                .map(|id| block_display_name(m, *id))
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    r.new_blocks.retain(|id| m.circuit.block(*id).is_some());
    r.highlight.retain(|id| m.circuit.block(*id).is_some());
    r
}

/// A read-only query collected during [`apply_ops_with`], answered after the edits.
enum InspectReq {
    Blocks(Vec<BlockId>),
    Trace(BlockId),
}

/// A settled simulation of the circuit with switches at their current levels.
fn settled(m: &CircuitManager) -> Simulation {
    let mut sim = Simulation::build(&m.circuit, &m.chips);
    sim.step(0.0);
    sim
}

/// Level seen at an input port (the OR of its drivers), and who drives it.
fn input_level(m: &CircuitManager, sim: &Simulation, port: Port) -> (bool, Vec<Port>) {
    let drivers: Vec<Port> = m.circuit.connections_into(port).map(|w| w.from).collect();
    let level = drivers
        .iter()
        .any(|p| sim.output_value(*p).unwrap_or(false));
    (level, drivers)
}

/// One line per block: inputs (value ← drivers) and outputs (value → loads).
fn inspect_blocks(m: &CircuitManager, ids: &[BlockId]) -> String {
    let sim = settled(m);
    let mut out = String::from("Inspect (settled, switches at their current levels):");
    for &id in ids {
        let Some(b) = m.circuit.block(id) else {
            continue;
        };
        let _ = write!(out, "\n  #{} {}", id, block_display_name(m, id));
        if b.label.is_some() {
            let _ = write!(out, " ({})", b.ty.display_name());
        }
        let ins: Vec<String> = (0..b.input_count(&m.chips) as u16)
            .map(|i| {
                let port = Port::input(id, i);
                let (level, drivers) = input_level(m, &sim, port);
                let from = if drivers.is_empty() {
                    "unwired".to_string()
                } else {
                    drivers
                        .iter()
                        .map(|d| format!("#{}", d.block))
                        .collect::<Vec<_>>()
                        .join("|")
                };
                format!("{}={} \u{2190} {from}", port_display(m, port), level as u8)
            })
            .collect();
        if !ins.is_empty() {
            let _ = write!(out, "; inputs: {}", ins.join(", "));
        }
        let outs: Vec<String> = (0..b.output_count(&m.chips) as u16)
            .map(|i| {
                let port = Port::output(id, i);
                let level = match b.ty {
                    BlockType::Switch | BlockType::Button => sim.switch_value(id),
                    _ => sim.output_value(port).unwrap_or(false),
                };
                let loads = m
                    .circuit
                    .iter_connections()
                    .filter(|w| w.from == port)
                    .count();
                format!(
                    "{}={} \u{2192} {loads} load(s)",
                    port_display(m, port),
                    level as u8
                )
            })
            .collect();
        if !outs.is_empty() {
            let _ = write!(out, "; outputs: {}", outs.join(", "));
        }
        if b.ty == BlockType::Led {
            let _ = write!(out, "; lit={}", sim.led_value(id).unwrap_or(false) as u8);
        }
    }
    out
}

/// Walk back from `id` through everything that feeds it (breadth-first, up to 24 blocks), with
/// each block's live output — the chain of values behind one result.
fn trace_block(m: &CircuitManager, id: BlockId) -> String {
    const MAX: usize = 24;
    let sim = settled(m);
    let mut out = format!(
        "Trace of {} (settled; each line: block = output \u{2190} its inputs):",
        block_display_name(m, id)
    );
    let mut queue = std::collections::VecDeque::from([id]);
    let mut seen: HashSet<BlockId> = HashSet::from([id]);
    let mut shown = 0;
    while let Some(cur) = queue.pop_front() {
        if shown == MAX {
            out.push_str("\n  \u{2026} (stopped after 24 blocks)");
            break;
        }
        shown += 1;
        let Some(b) = m.circuit.block(cur) else {
            continue;
        };
        let value = match b.ty {
            BlockType::Led => sim.led_value(cur).unwrap_or(false),
            BlockType::Switch | BlockType::Button => sim.switch_value(cur),
            _ => sim.output_value(Port::output(cur, 0)).unwrap_or(false),
        };
        let ins: Vec<String> = (0..b.input_count(&m.chips) as u16)
            .map(|i| {
                let (level, drivers) = input_level(m, &sim, Port::input(cur, i));
                for d in &drivers {
                    if seen.insert(d.block) {
                        queue.push_back(d.block);
                    }
                }
                if drivers.is_empty() {
                    format!("{}:unwired", i + 1)
                } else {
                    let names: Vec<String> = drivers
                        .iter()
                        .map(|d| format!("#{} {}", d.block, block_display_name(m, d.block)))
                        .collect();
                    format!("{}:{}({})", i + 1, level as u8, names.join("|"))
                }
            })
            .collect();
        let _ = write!(
            out,
            "\n  #{cur} {} = {}",
            block_display_name(m, cur),
            value as u8
        );
        if !ins.is_empty() {
            let _ = write!(out, " \u{2190} {}", ins.join(", "));
        }
    }
    out
}

/// Apply optional property settings to a block (only fields that make sense for its kind).
fn configure(
    b: &mut Block,
    inputs: Option<u16>,
    label: Option<String>,
    state: Option<bool>,
    hz: Option<f32>,
    color: Option<Option<[u8; 3]>>,
) {
    if let Some(n) = inputs.filter(|_| b.ty.variable_inputs()) {
        b.inputs = Some(n.clamp(2, MAX_GATE_INPUTS as u16));
    }
    if let Some(l) = label {
        let l = l.trim();
        b.label = (!l.is_empty()).then(|| l.to_string());
    }
    if let Some(s) = state.filter(|_| matches!(b.ty, BlockType::Switch | BlockType::Button)) {
        b.state = s;
    }
    if let Some(h) = hz.filter(|_| b.ty == BlockType::Clock) {
        b.freq_hz = Some(h.clamp(0.01, 1000.0));
    }
    if let Some(c) = color {
        b.color = c;
    }
}

/// A primitive kind, or a library chip by name.
fn resolve_kind(chips: &ChipLibrary, kind: &AddKind) -> Result<(BlockType, String), String> {
    match kind {
        AddKind::Prim(p) => Ok((*p, p.display_name().to_string())),
        AddKind::Chip(n) => match chips.find_by_name(n) {
            Some(cid) => Ok((BlockType::Chip(cid), n.clone())),
            None => Err(format!(
                "unknown chip \"{n}\" \u{2014} define it with defchip first"
            )),
        },
    }
}

/// Every input port that already has a wire into it.
fn wired_inputs(c: &Circuit) -> HashSet<(BlockId, u16)> {
    c.iter_connections()
        .map(|w| (w.to.block, w.to.index))
        .collect()
}

/// Resolve a block handle: a ref given earlier in this reply, a numeric id (`7` or `#7`), or a
/// block's label (case-insensitive).
pub fn resolve_handle(
    refs: &HashMap<String, BlockId>,
    circuit: &Circuit,
    handle: &str,
) -> Option<BlockId> {
    if let Some(id) = refs.get(handle).or_else(|| refs.get(handle.trim())) {
        if circuit.block(*id).is_some() {
            return Some(*id);
        }
    }
    let trimmed = handle.trim().trim_start_matches('#');
    if let Ok(n) = trimmed.parse::<u32>() {
        let id = BlockId(n);
        if circuit.block(id).is_some() {
            return Some(id);
        }
    }
    let want = trimmed.to_lowercase();
    circuit
        .iter_blocks()
        .find(|b| {
            b.label
                .as_ref()
                .is_some_and(|l| l.trim().to_lowercase() == want)
        })
        .map(|b| b.id)
}

/// Pin index of `name` on chip `cid` (input or output side), matched case-insensitively.
fn chip_pin_by_name(chips: &ChipLibrary, cid: ChipId, name: &str, input: bool) -> Option<u16> {
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

/// Pin names of a chip side, for error messages.
fn pin_names(chips: &ChipLibrary, cid: ChipId, input: bool) -> String {
    let Some(def) = chips.get(cid) else {
        return String::new();
    };
    let n = if input {
        def.input_blocks.len()
    } else {
        def.output_blocks.len()
    };
    (0..n)
        .map(|i| {
            let s = if input {
                def.input_name(i)
            } else {
                def.output_name(i)
            };
            if s.is_empty() {
                i.to_string()
            } else {
                s.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// Resolve an output-port selector on `id`: omitted → 0, an index is validated, a name is looked
/// up among a chip's output pins.
fn port_out(
    circuit: &Circuit,
    chips: &ChipLibrary,
    id: BlockId,
    sel: Option<&PortSel>,
) -> Result<u16, String> {
    let Some(b) = circuit.block(id) else {
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
            BlockType::Chip(cid) => chip_pin_by_name(chips, cid, n, false).ok_or_else(|| {
                format!(
                    "no output pin named \"{n}\" (outputs are: {})",
                    pin_names(chips, cid, false)
                )
            }),
            _ => Err(format!(
                "\"{n}\": named ports only work on chip instances; use a number"
            )),
        },
    }
}

/// Resolve an input-port selector on `id`: omitted → the first FREE input (falls back to 0,
/// since inputs OR-combine), an index is validated, a name is looked up among a chip's pins.
fn port_in(
    circuit: &Circuit,
    chips: &ChipLibrary,
    wired: &HashSet<(BlockId, u16)>,
    id: BlockId,
    sel: Option<&PortSel>,
) -> Result<u16, String> {
    let Some(b) = circuit.block(id) else {
        return Err("unknown destination block".to_string());
    };
    let count = b.input_count(chips);
    if count == 0 {
        return Err(
            "the destination block has no inputs (switches/constants can't be driven)".to_string(),
        );
    }
    match sel {
        None => Ok((0..count as u16)
            .find(|i| !wired.contains(&(id, *i)))
            .unwrap_or(0)),
        Some(PortSel::Index(i)) if (*i as usize) < count => Ok(*i),
        Some(PortSel::Index(i)) => Err(format!("input port {i} out of range (block has {count})")),
        Some(PortSel::Name(n)) => match b.ty {
            BlockType::Chip(cid) => chip_pin_by_name(chips, cid, n, true).ok_or_else(|| {
                format!(
                    "no input pin named \"{n}\" (inputs are: {})",
                    pin_names(chips, cid, true)
                )
            }),
            _ => Err(format!(
                "\"{n}\": named ports only work on chip instances; use a number"
            )),
        },
    }
}

/// Build and register a reusable chip from declared pins + inner ops run against a scratch
/// circuit. Pin blocks (Switch = input, Led = output) are created automatically, labeled and
/// ref'd by their pin names. Redefining an existing name replaces the chip in place, so every
/// instance updates (wires follow pins by name). Returns the observation for the model.
fn build_chip(
    m: &mut CircuitManager,
    txn: &mut Txn,
    name: &str,
    inputs: &[String],
    outputs: &[String],
    ops: Vec<RawOp>,
) -> Result<String, String> {
    let name = name.trim();
    if name.is_empty() {
        return Err("chip name is empty".to_string());
    }
    if BlockType::primitives()
        .iter()
        .any(|p| super::parse::kind_from_str(name) == Some(*p))
    {
        return Err(format!(
            "\"{name}\" is a built-in block kind; pick another chip name"
        ));
    }
    if outputs.is_empty() {
        return Err("a chip needs at least one output pin".to_string());
    }
    if inputs.len() > 32 || outputs.len() > 32 {
        return Err("too many pins (max 32 inputs and 32 outputs)".to_string());
    }

    let mut c = Circuit::new(name);
    let mut refs: HashMap<String, BlockId> = HashMap::new();
    let mut errs: Vec<String> = Vec::new();

    let mut pin = |c: &mut Circuit, ty: BlockType, x: i32, i: usize, label: &str| {
        let id = c.allocate_block_id();
        let mut b = Block::new(id, ty, Pos::new(x, i as i32 * 4));
        b.label = Some(label.to_string());
        c.insert_block(b);
        refs.insert(label.to_string(), id);
        id
    };
    let in_ids: Vec<BlockId> = inputs
        .iter()
        .enumerate()
        .map(|(i, p)| pin(&mut c, BlockType::Switch, 0, i, p))
        .collect();
    let out_ids: Vec<BlockId> = outputs
        .iter()
        .enumerate()
        .map(|(i, p)| pin(&mut c, BlockType::Led, 28, i, p))
        .collect();
    let mut wired: HashSet<(BlockId, u16)> = HashSet::new();

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
                hz,
                color,
            } => {
                let ty = match resolve_kind(&m.chips, &kind) {
                    Ok((t, _)) => t,
                    Err(e) => {
                        errs.push(e);
                        continue;
                    }
                };
                let id = c.allocate_block_id();
                let mut b = Block::new(id, ty, Pos::new(x, y));
                configure(&mut b, n_in, label, Some(state), hz, color.map(Some));
                c.insert_block(b);
                if let Some(rf) = r#ref {
                    refs.insert(rf, id);
                }
            }
            RawOp::Connect {
                from,
                from_port,
                to,
                to_port,
            } => {
                let (Some(fb), Some(tb)) = (
                    resolve_handle(&refs, &c, &from),
                    resolve_handle(&refs, &c, &to),
                ) else {
                    errs.push(format!("connect: unknown block ({from} \u{2192} {to})"));
                    continue;
                };
                let ports = port_out(&c, &m.chips, fb, from_port.as_ref()).and_then(|fp| {
                    port_in(&c, &m.chips, &wired, tb, to_port.as_ref()).map(|tp| (fp, tp))
                });
                match ports {
                    Ok((fp, tp)) => {
                        let (fport, tport) = (Port::output(fb, fp), Port::input(tb, tp));
                        if c.iter_connections()
                            .any(|w| w.from == fport && w.to == tport)
                        {
                            continue;
                        }
                        let cid = c.allocate_conn_id();
                        c.insert_connection(Connection {
                            id: cid,
                            from: fport,
                            to: tport,
                        });
                        wired.insert((tb, tp));
                    }
                    Err(e) => errs.push(format!("connect {from}\u{2192}{to}: {e}")),
                }
            }
            RawOp::DefChip {
                name: n,
                inputs: i,
                outputs: o,
                ops: body,
            } => {
                // Nested definition: register it globally, then this body can use it.
                if let Err(e) = build_chip(m, txn, &n, &i, &o, body) {
                    errs.push(format!("nested defchip {n}: {e}"));
                }
            }
            RawOp::Label { target, text } => match resolve_handle(&refs, &c, &target) {
                Some(id) => {
                    if let Some(b) = c.block_mut(id) {
                        b.label = text;
                    }
                }
                None => errs.push(format!("label: unknown block {target}")),
            },
            RawOp::Move { target, x, y } => match resolve_handle(&refs, &c, &target) {
                Some(id) => {
                    if let Some(b) = c.block_mut(id) {
                        b.pos = Pos::new(x, y);
                    }
                }
                None => errs.push(format!("move: unknown block {target}")),
            },
            RawOp::Set {
                target,
                inputs: n,
                hz,
                color,
                state,
                label,
            } => match resolve_handle(&refs, &c, &target) {
                Some(id) => {
                    if let Some(b) = c.block_mut(id) {
                        configure(b, n, label, state, hz, color);
                    }
                }
                None => errs.push(format!("set: unknown block {target}")),
            },
            RawOp::Remove { target } => match resolve_handle(&refs, &c, &target) {
                Some(id) if !in_ids.contains(&id) && !out_ids.contains(&id) => {
                    c.remove_block(id);
                    wired = c
                        .iter_connections()
                        .map(|w| (w.to.block, w.to.index))
                        .collect();
                }
                _ => errs.push(format!("remove: unknown block or pin {target}")),
            },
            RawOp::Disconnect { from, to, .. } => {
                match (
                    resolve_handle(&refs, &c, &from),
                    resolve_handle(&refs, &c, &to),
                ) {
                    (Some(fb), Some(tb)) => {
                        let doomed: Vec<_> = c
                            .iter_connections()
                            .filter(|w| w.from.block == fb && w.to.block == tb)
                            .map(|w| w.id)
                            .collect();
                        for id in doomed {
                            c.remove_connection(id);
                        }
                        wired = c
                            .iter_connections()
                            .map(|w| (w.to.block, w.to.index))
                            .collect();
                    }
                    _ => errs.push(format!("disconnect: unknown block ({from} \u{2192} {to})")),
                }
            }
            RawOp::Test { .. }
            | RawOp::Inspect { .. }
            | RawOp::Trace { .. }
            | RawOp::Highlight { .. }
            | RawOp::DeleteChip { .. } => {
                errs.push(
                    "test/inspect/trace/highlight/delete_chip inside a defchip body are ignored \
                     (use them at the top level)"
                        .to_string(),
                );
            }
        }
    }

    // Register (reusing the id when redefining, so every instance updates).
    let existing = m.chips.find_by_name(name);
    let id = existing.unwrap_or_else(|| m.chips.allocate_id());
    if m.chips.would_cycle(id, &c) {
        return Err(format!(
            "{name} can't contain itself (directly or through another chip)"
        ));
    }
    let def = ChipDef {
        id,
        name: name.to_string(),
        circuit: c,
        input_blocks: in_ids,
        output_blocks: out_ids,
    };
    let removed_pins: Vec<String> = match existing.and_then(|e| m.chips.get(e)) {
        Some(old) => {
            let map = PinMap::between(old, &def);
            let mut gone: Vec<String> = Vec::new();
            for (i, slot) in map.inputs.iter().enumerate() {
                if slot.is_none() {
                    gone.push(format!("input {}", old.input_name(i)));
                }
            }
            for (i, slot) in map.outputs.iter().enumerate() {
                if slot.is_none() {
                    gone.push(format!("output {}", old.output_name(i)));
                }
            }
            gone
        }
        None => Vec::new(),
    };
    let report = m.put_chip_in(txn, def);

    // Verify the chip in isolation and report what it actually does.
    let mut obs = chip_observation(m, id, inputs, outputs);
    if existing.is_some() {
        let _ = write!(obs, "\n  (Redefined: every placed {name} updated.");
        if report.rewired > 0 {
            let _ = write!(
                obs,
                " {} wire(s) followed their pin to its new position.",
                report.rewired
            );
        }
        if report.removed > 0 {
            let _ = write!(
                obs,
                " {} wire(s) to removed pins [{}] were deleted \u{2014} reconnect them if needed.",
                report.removed,
                removed_pins.join(", ")
            );
        }
        obs.push(')');
    }
    for e in errs {
        let _ = write!(obs, "\n  skipped: {e}");
    }
    Ok(obs)
}

/// Describe a freshly (re)defined chip to the model: pins, unwired-output warnings, and an
/// isolated truth table when the pin count allows one.
fn chip_observation(
    m: &CircuitManager,
    id: ChipId,
    inputs: &[String],
    outputs: &[String],
) -> String {
    let Some(def) = m.chips.get(id) else {
        return String::new();
    };
    let mut out = format!(
        "Defined chip {}: inputs [{}] outputs [{}].",
        def.name,
        inputs.join(", "),
        outputs.join(", ")
    );
    let wired = wired_inputs(&def.circuit);
    for (i, ob) in def.output_blocks.iter().enumerate() {
        if !wired.contains(&(*ob, 0)) {
            let _ = write!(
                out,
                "\n  WARNING: output pin \"{}\" is not wired to anything.",
                outputs.get(i).map(String::as_str).unwrap_or("?")
            );
        }
    }
    if inputs.len() <= 6 && !outputs.is_empty() {
        let mut sim = Simulation::build(&def.circuit, &m.chips);
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
            let ins: String = (0..n)
                .map(|i| {
                    if (combo >> (n - 1 - i)) & 1 == 1 {
                        "1 "
                    } else {
                        "0 "
                    }
                })
                .collect();
            let outs: String = def
                .output_blocks
                .iter()
                .map(|bid| {
                    if sim.led_value(*bid).unwrap_or(false) {
                        "1 "
                    } else {
                        "0 "
                    }
                })
                .collect();
            let _ = write!(out, "\n    {}| {}", ins, outs.trim_end());
        }
        if sim.is_unstable() {
            out.push_str("\n  WARNING: the chip oscillates (never settles) for some inputs.");
        }
    }
    out
}

/// Run a scripted test on a scratch simulation of the current circuit (the user's live switch
/// states are untouched). Each step sets switches/buttons, pulses some, lets time pass, then
/// reads every LED.
/// The outcome of one scripted test.
struct TestOutcome {
    text: String,
    passed: usize,
    failed: usize,
}

fn run_test(
    m: &CircuitManager,
    refs: &HashMap<String, BlockId>,
    steps: &[TestStep],
) -> TestOutcome {
    let (mut passed, mut failed) = (0usize, 0usize);
    let circuit = &m.circuit;
    let mut sim = Simulation::build(circuit, &m.chips);
    sim.step(0.0);
    let name = |b: &Block| match &b.label {
        Some(l) if !l.trim().is_empty() => l.trim().to_string(),
        _ => format!("#{}", b.id),
    };
    let leds: Vec<&Block> = circuit
        .iter_blocks()
        .filter(|b| b.ty == BlockType::Led)
        .collect();
    let mut out = String::from("Test results:");
    if leds.is_empty() {
        out.push_str("\n  (no LEDs to observe \u{2014} add LEDs to make outputs visible)");
    }
    let is_input = |id: BlockId| {
        circuit
            .block(id)
            .is_some_and(|b| matches!(b.ty, BlockType::Switch | BlockType::Button))
    };
    for (i, step) in steps.iter().enumerate() {
        let mut did: Vec<String> = Vec::new();
        for (handle, val) in &step.set {
            match resolve_handle(refs, circuit, handle) {
                Some(id) if is_input(id) => {
                    sim.set_switch(id, *val);
                    did.push(format!("{handle}={}", *val as u8));
                }
                Some(_) => did.push(format!("{handle}\u{2260}switch(ignored)")),
                None => did.push(format!("{handle}=unknown(ignored)")),
            }
        }
        sim.step(0.0);
        for handle in &step.pulse {
            match resolve_handle(refs, circuit, handle) {
                Some(id) if is_input(id) => {
                    sim.set_switch(id, true);
                    sim.step(0.0);
                    sim.set_switch(id, false);
                    sim.step(0.0);
                    did.push(format!("pulse {handle}"));
                }
                _ => did.push(format!("pulse {handle}(not a switch/button, ignored)")),
            }
        }
        if let Some(w) = step.wait.filter(|w| *w > 0.0) {
            sim.step(w);
            did.push(format!("wait {w}s"));
        }
        if let Some(t) = step.ticks.filter(|t| *t > 0) {
            let mut done = 0;
            for _ in 0..t.min(10_000) {
                match sim.time_to_next_edge() {
                    Some(dt) => {
                        sim.step(dt + 1e-9);
                        done += 1;
                    }
                    None => break,
                }
            }
            if done == 0 {
                did.push("ticks ignored (no clock in the circuit)".to_string());
            } else {
                did.push(format!("{done} clock edge(s)"));
            }
        }
        let reads: Vec<String> = leds
            .iter()
            .map(|b| format!("{}={}", name(b), sim.led_value(b.id).unwrap_or(false) as u8))
            .collect();
        let mut verdicts: Vec<String> = Vec::new();
        for (handle, want) in &step.expect {
            let got = resolve_handle(refs, circuit, handle).and_then(|id| {
                let b = circuit.block(id)?;
                Some(match b.ty {
                    BlockType::Led => sim.led_value(id).unwrap_or(false),
                    BlockType::Switch | BlockType::Button => sim.switch_value(id),
                    _ => sim.output_value(Port::output(id, 0)).unwrap_or(false),
                })
            });
            match got {
                Some(v) if v == *want => passed += 1,
                Some(v) => {
                    failed += 1;
                    verdicts.push(format!(
                        "FAIL {handle}: expected {} got {}",
                        *want as u8, v as u8
                    ));
                }
                None => {
                    failed += 1;
                    verdicts.push(format!("FAIL {handle}: unknown block"));
                }
            }
        }
        if !step.expect.is_empty() && verdicts.is_empty() {
            verdicts.push("ok".to_string());
        }
        let _ = write!(
            out,
            "\n  step {}: {} \u{2192} {}{}{}",
            i + 1,
            if did.is_empty() {
                "(nothing)".to_string()
            } else {
                did.join(", ")
            },
            reads.join(", "),
            if sim.is_unstable() {
                "  [OSCILLATING]"
            } else {
                ""
            },
            if verdicts.is_empty() {
                String::new()
            } else {
                format!("  [{}]", verdicts.join("; "))
            }
        );
    }
    if passed + failed > 0 {
        let _ = write!(
            out,
            "\n  CHECKS: {passed}/{} passed{}",
            passed + failed,
            if failed > 0 {
                " \u{2014} fix the circuit until every check passes"
            } else {
                ""
            }
        );
    }
    TestOutcome {
        text: out,
        passed,
        failed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::parse::parse_reply;

    fn run(m: &mut CircuitManager, json: &str) -> ApplyReport {
        apply_ops(m, parse_reply(json).ops)
    }

    #[test]
    fn a_whole_reply_is_one_undo_step_including_chip_definitions() {
        let mut m = CircuitManager::new();
        let r = run(
            &mut m,
            r#"{"commands":[
              {"op":"defchip","name":"Flipper","inputs":["a"],"outputs":["y"],"commands":[
                {"op":"add","ref":"n","kind":"not","x":8,"y":0},
                {"op":"connect","from":"a","to":"n"},{"op":"connect","from":"n","to":"y"}]},
              {"op":"add","ref":"s","kind":"switch","x":0,"y":0},
              {"op":"add","ref":"i","kind":"Flipper","x":6,"y":0},
              {"op":"add","ref":"l","kind":"led","x":12,"y":0},
              {"op":"connect","from":"s","to":"i","to_port":"a"},
              {"op":"connect","from":"i","to":"l","from_port":"y"}
            ]}"#,
        );
        assert!(r.errors.is_empty(), "{:?}", r.errors);
        assert_eq!(m.chips.len(), 1);
        assert_eq!(m.circuit.blocks.len(), 3);
        assert!(m.undo());
        assert!(m.chips.is_empty(), "chip definition undone with the rest");
        assert!(m.circuit.blocks.is_empty());
    }

    #[test]
    fn a_chip_cannot_be_redefined_to_contain_itself() {
        let mut m = CircuitManager::new();
        run(
            &mut m,
            r#"{"commands":[{"op":"defchip","name":"Loop","inputs":["a"],"outputs":["y"],"commands":[
                {"op":"connect","from":"a","to":"y"}]}]}"#,
        );
        let r = run(
            &mut m,
            r#"{"commands":[{"op":"defchip","name":"Loop","inputs":["a"],"outputs":["y"],"commands":[
                {"op":"add","ref":"me","kind":"Loop","x":4,"y":0},
                {"op":"connect","from":"a","to":"me"},{"op":"connect","from":"me","to":"y"}]}]}"#,
        );
        assert!(
            r.errors.iter().any(|e| e.contains("can't contain itself")),
            "{:?}",
            r.errors
        );
        // And the simulator is still fine with the library.
        let mut sim = Simulation::build(&m.circuit, &m.chips);
        sim.step(0.0);
    }

    #[test]
    fn remove_works_on_a_block_added_in_the_same_reply() {
        let mut m = CircuitManager::new();
        let r = run(
            &mut m,
            r#"{"commands":[{"op":"add","ref":"g","kind":"and","x":0,"y":0},
                {"op":"remove","target":"g"}]}"#,
        );
        assert!(r.errors.is_empty(), "{:?}", r.errors);
        assert!(m.circuit.blocks.is_empty());
    }

    #[test]
    fn set_changes_properties_and_duplicate_wires_are_skipped() {
        let mut m = CircuitManager::new();
        let r = run(
            &mut m,
            r##"{"commands":[
              {"op":"add","ref":"c","kind":"clock","x":0,"y":0},
              {"op":"add","ref":"g","kind":"and","x":4,"y":0},
              {"op":"set","target":"c","hz":8,"color":"#00ff00","label":"CLK"},
              {"op":"set","target":"g","inputs":3},
              {"op":"connect","from":"c","to":"g","to_port":2},
              {"op":"connect","from":"c","to":"g","to_port":2}
            ]}"##,
        );
        assert!(r.errors.is_empty(), "{:?}", r.errors);
        let clk = m
            .circuit
            .iter_blocks()
            .find(|b| b.ty == BlockType::Clock)
            .unwrap();
        assert_eq!(clk.freq_hz, Some(8.0));
        assert_eq!(clk.color, Some([0, 255, 0]));
        assert_eq!(clk.label.as_deref(), Some("CLK"));
        assert_eq!(m.circuit.connections.len(), 1);
        assert!(r.observations.iter().any(|o| o.contains("already wired")));
    }

    #[test]
    fn tests_can_pulse_a_clock_input_to_load_a_register() {
        // D latch from gates: q = (d AND en) OR (q AND NOT en); verify with a pulse on EN.
        let mut m = CircuitManager::new();
        let r = run(
            &mut m,
            r#"{"commands":[
              {"op":"add","ref":"d","kind":"switch","label":"D","x":0,"y":0},
              {"op":"add","ref":"en","kind":"button","label":"EN","x":0,"y":4},
              {"op":"add","ref":"ne","kind":"not","x":4,"y":4},
              {"op":"add","ref":"a1","kind":"and","x":8,"y":0},
              {"op":"add","ref":"a2","kind":"and","x":8,"y":4},
              {"op":"add","ref":"o","kind":"or","x":12,"y":2},
              {"op":"add","ref":"q","kind":"led","label":"Q","x":16,"y":2},
              {"op":"connect","from":"d","to":"a1"},{"op":"connect","from":"en","to":"a1"},
              {"op":"connect","from":"en","to":"ne"},
              {"op":"connect","from":"o","to":"a2"},{"op":"connect","from":"ne","to":"a2"},
              {"op":"connect","from":"a1","to":"o"},{"op":"connect","from":"a2","to":"o"},
              {"op":"connect","from":"o","to":"q"},
              {"op":"test","steps":[{"set":{"D":1}},{"pulse":"EN"},{"set":{"D":0}},{"pulse":"EN"}]}
            ]}"#,
        );
        assert!(r.errors.is_empty(), "{:?}", r.errors);
        let t = r.observations.last().unwrap();
        assert!(t.contains("step 1: D=1 \u{2192} Q=0"), "{t}");
        assert!(t.contains("step 2: pulse EN \u{2192} Q=1"), "{t}");
        assert!(t.contains("step 3: D=0 \u{2192} Q=1"), "holds: {t}");
        assert!(t.contains("step 4: pulse EN \u{2192} Q=0"), "{t}");
    }

    #[test]
    fn redefining_a_chip_reports_rewired_and_removed_pins() {
        let mut m = CircuitManager::new();
        run(
            &mut m,
            r#"{"commands":[{"op":"defchip","name":"P","inputs":["a","b"],"outputs":["y"],"commands":[
                {"op":"add","ref":"g","kind":"and","x":8,"y":0},
                {"op":"connect","from":"a","to":"g"},{"op":"connect","from":"b","to":"g"},
                {"op":"connect","from":"g","to":"y"}]},
              {"op":"add","ref":"s","kind":"switch","x":0,"y":0},
              {"op":"add","ref":"p","kind":"P","x":6,"y":0},
              {"op":"connect","from":"s","to":"p","to_port":"b"}]}"#,
        );
        let r = run(
            &mut m,
            r#"{"commands":[{"op":"defchip","name":"P","inputs":["b"],"outputs":["y"],"commands":[
                {"op":"connect","from":"b","to":"y"}]}]}"#,
        );
        let obs = r.observations.join("\n");
        assert!(obs.contains("followed their pin"), "{obs}");
        let w = m.circuit.iter_connections().next().unwrap();
        assert_eq!(w.to.index, 0, "wire moved from pin 1 (b) to pin 0 (b)");
    }

    #[test]
    fn disconnect_removes_one_wire_without_deleting_blocks() {
        let mut m = CircuitManager::new();
        run(
            &mut m,
            r#"{"commands":[
              {"op":"add","ref":"a","kind":"switch","label":"A","x":0,"y":0},
              {"op":"add","ref":"g","kind":"xor","x":6,"y":0},
              {"op":"connect","from":"a","to":"g","to_port":0},
              {"op":"connect","from":"a","to":"g","to_port":1}]}"#,
        );
        let r = run(
            &mut m,
            r#"{"commands":[{"op":"disconnect","from":"A","to":"1","to_port":1}]}"#,
        );
        assert!(r.errors.is_empty(), "{:?}", r.errors);
        assert_eq!(m.circuit.connections.len(), 1);
        assert_eq!(m.circuit.blocks.len(), 2);
        assert_eq!(r.activity, vec!["Unwire A \u{2192} XOR in 2"]);
        let r = run(
            &mut m,
            r#"{"commands":[{"op":"disconnect","from":"A","to":"1","to_port":1}]}"#,
        );
        assert!(r.errors[0].contains("no wire"), "{:?}", r.errors);
    }

    #[test]
    fn protected_user_blocks_are_never_removed() {
        let mut m = CircuitManager::new();
        let mine = m.add_block(BlockType::Not, Pos::new(0, 0));
        let opts = ApplyOptions {
            protected: [mine].into_iter().collect(),
        };
        let r = apply_ops_with(
            &mut m,
            parse_reply(r#"{"commands":[{"op":"remove","target":"0"},{"op":"add","kind":"and","x":9,"y":0}]}"#).ops,
            &opts,
        );
        assert!(m.circuit.block(mine).is_some(), "the user's block survives");
        assert_eq!(
            m.circuit.blocks.len(),
            2,
            "the rest of the reply still applies"
        );
        assert!(r.errors[0].contains("placed by the user"), "{:?}", r.errors);
        assert!(!user_allows_removal("build a 2-bit counter"));
        assert!(user_allows_removal("Replace the NOT loop with a counter"));
        assert!(user_allows_removal("fix this, it doesn't work"));
    }

    #[test]
    fn activity_uses_names_not_ids() {
        let mut m = CircuitManager::new();
        run(
            &mut m,
            r#"{"commands":[{"op":"add","kind":"switch","label":"A","x":0,"y":0},
               {"op":"add","kind":"xor","x":6,"y":0}]}"#,
        );
        let r = run(&mut m, r#"{"commands":[{"op":"connect","from":0,"to":1}]}"#);
        assert_eq!(r.activity, vec!["Wire A \u{2192} XOR in 1"]);
    }

    #[test]
    fn test_expectations_are_checked_and_counted() {
        let mut m = CircuitManager::new();
        let r = run(
            &mut m,
            r#"{"commands":[
              {"op":"add","ref":"a","kind":"switch","label":"A","x":0,"y":0},
              {"op":"add","ref":"n","kind":"not","x":4,"y":0},
              {"op":"add","ref":"y","kind":"led","label":"Y","x":8,"y":0},
              {"op":"connect","from":"a","to":"n"},{"op":"connect","from":"n","to":"y"},
              {"op":"test","steps":[{"set":{"A":0},"expect":{"Y":1}},
                                    {"set":{"A":1},"expect":{"Y":1}}]}]}"#,
        );
        assert_eq!((r.checks_passed, r.checks_failed), (1, 1));
        let t = r.observations.last().unwrap();
        assert!(t.contains("FAIL Y: expected 1 got 0"), "{t}");
        assert!(t.contains("CHECKS: 1/2 passed"), "{t}");
    }

    #[test]
    fn inspect_and_trace_report_values_and_drivers() {
        let mut m = CircuitManager::new();
        let r = run(
            &mut m,
            r#"{"commands":[
              {"op":"add","ref":"a","kind":"constant1","x":0,"y":0},
              {"op":"add","ref":"g","kind":"and","label":"G","x":4,"y":0},
              {"op":"add","ref":"y","kind":"led","label":"CARRY","x":8,"y":0},
              {"op":"connect","from":"a","to":"g"},{"op":"connect","from":"g","to":"y"},
              {"op":"inspect","targets":["G"]},{"op":"trace","target":"CARRY"},
              {"op":"highlight","targets":["G"],"note":"this gate"}]}"#,
        );
        let obs = r.observations.join("\n");
        assert!(obs.contains("G in 2=0 \u{2190} unwired"), "{obs}");
        assert!(obs.contains("CARRY = 0 \u{2190} 1:0(#1 G)"), "{obs}");
        assert_eq!(r.highlight, vec![BlockId(1)]);
        assert_eq!(r.highlight_note.as_deref(), Some("this gate"));
    }

    #[test]
    fn delete_chip_removes_definition_and_copies() {
        let mut m = CircuitManager::new();
        run(
            &mut m,
            r#"{"commands":[{"op":"defchip","name":"Buf2","inputs":["a"],"outputs":["y"],
                "commands":[{"op":"connect","from":"a","to":"y"}]},
              {"op":"add","kind":"Buf2","x":0,"y":0}]}"#,
        );
        let r = run(
            &mut m,
            r#"{"commands":[{"op":"delete_chip","name":"buf2"}]}"#,
        );
        assert!(r.errors.is_empty(), "{:?}", r.errors);
        assert!(m.chips.is_empty() && m.circuit.blocks.is_empty());
        assert!(m.undo());
        assert_eq!(m.chips.len(), 1, "one undo restores it");
    }
}
