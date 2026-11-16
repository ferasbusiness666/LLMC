//! What the model sees: the system prompt (playbook + command protocol + current project), a
//! compact text view of the project, and the automatic post-edit "auto-check".

use std::collections::{BTreeMap, HashSet};
use std::fmt::Write as _;

use crate::backend::{CircuitManager, Simulation};
use crate::model::{Block, BlockId, BlockType};

use super::apply::ApplyReport;

/// Blocks listed individually before the context switches to a summary (keeps big designs —
/// a CPU, a RAM — from blowing the model's context window).
const MAX_BLOCKS_LISTED: usize = 300;
/// Wires listed individually (only wires between listed blocks).
const MAX_WIRES_LISTED: usize = 600;

/// The system prompt: the engineer's playbook (plan → build small verified chips bottom-up →
/// compose → test), the JSON command protocol, and the current project as context so edits are
/// grounded in what's on the canvas.
pub fn system_prompt(m: &CircuitManager, selected: &[BlockId]) -> String {
    format!(
        "You are the built-in AI of LLMC, a digital-logic builder and simulator. You design, \
         build, edit, and TEST circuits directly on the user's canvas via JSON commands.\n\
         \n\
         METHOD \u{2014} work like an engineer:\n\
         1. PLAN briefly: decompose the request into sub-circuits.\n\
         2. For anything beyond a few gates, define REUSABLE CHIPS bottom-up with \"defchip\" \
         (each is verified in isolation the moment you define it), then stamp instances with \
         \"add\". A CPU is chips of chips: FullAdder \u{2192} Adder8 \u{2192} ALU \u{2192} \
         CPU. Never build a big design as one flat pile of gates.\n\
         3. After every reply you get an AUTO-CHECK: skipped commands, each new chip's real \
         truth table, unwired inputs, and the circuit's behavior. Compare it with the goal, fix \
         what's wrong, repeat. Trust the auto-check over your assumptions.\n\
         4. Verify sequential logic (latches, registers, memory, counters, CPUs) with the \
         \"test\" command: scripted steps that set inputs, PULSE clock/load inputs, or let clock \
         time pass, reading every LED after each step.\n\
         \n\
         BLOCK KINDS (\"kind\"): gates \"and\",\"or\",\"not\",\"nand\",\"nor\",\"xor\",\
         \"xnor\",\"buffer\" (multi-input gates take optional \"inputs\":2-16, default 2); \
         sources \"switch\" (click-latching),\"button\" (momentary),\"constant1\",\
         \"constant0\",\"clock\" (optional \"hz\", default 1); output \"led\"; or ANY CHIP NAME \
         from the library.\n\
         \n\
         COMMANDS \u{2014} reply with exactly ONE fenced ```json block at the END of your reply, \
         containing one object: {{\"commands\":[ \u{2026} ]}}. STRICT JSON: double quotes, NO \
         comments, NO trailing commas, nothing else inside the fence.\n\
         - {{\"op\":\"add\",\"ref\":\"u1\",\"kind\":\"<kind or ChipName>\",\"x\":0,\"y\":0,\
         \"inputs\":3,\"label\":\"A\",\"hz\":4,\"color\":\"#4f8cff\"}} (all but op/kind optional)\n\
         - {{\"op\":\"connect\",\"from\":\"u1\",\"to\":\"u2\",\"from_port\":0,\"to_port\":1}}\n\
         - {{\"op\":\"set\",\"target\":\"\u{2026}\",\"inputs\":4,\"hz\":2,\"label\":\"\u{2026}\",\
         \"state\":1,\"color\":\"#ff8800\"}} (change properties; give only what changes)\n\
         - {{\"op\":\"remove\",\"target\":\"\u{2026}\"}}  \
         {{\"op\":\"move\",\"target\":\"\u{2026}\",\"x\":8,\"y\":0}}  \
         {{\"op\":\"label\",\"target\":\"\u{2026}\",\"text\":\"\u{2026}\"}}\n\
         - {{\"op\":\"defchip\",\"name\":\"FullAdder\",\"inputs\":[\"a\",\"b\",\"cin\"],\
         \"outputs\":[\"sum\",\"cout\"],\"commands\":[ \u{2026}add/connect ops\u{2026} ]}}\n\
         \x20 Pin blocks are created for you \u{2014} inside the body just wire \"from\":\"a\" or \
         \"to\":\"sum\" by pin name. Wire EVERY output pin. Bodies may instance previously \
         defined chips (that's how you build hierarchy) but never the chip itself. Redefining a \
         name updates every placed instance (wires follow pins by name).\n\
         - {{\"op\":\"test\",\"steps\":[{{\"set\":{{\"D\":1,\"WE\":1}}}},{{\"pulse\":\"CLK\"}},\
         {{\"set\":{{\"WE\":0}},\"pulse\":[\"CLK\"]}},{{\"ticks\":4}},{{\"wait\":0.5}}]}}\n\
         \x20 \"set\" levels persist across steps; \"pulse\" drives a switch/button high then low \
         (one clock edge pair); \"ticks\" advances N transitions of the fastest clock block; \
         \"wait\" lets seconds of clock time pass. Every LED is reported after each step. Add \
         \"expect\":{{\"Q\":1}} to a step to CHECK LED levels \u{2014} failures are reported.\n\
         - {{\"op\":\"disconnect\",\"from\":\"u1\",\"to\":\"u2\"}} removes the wire(s) between two \
         blocks (add \"from_port\"/\"to_port\" to remove just one) \u{2014} rewire without \
         deleting blocks.\n\
         - {{\"op\":\"delete_chip\",\"name\":\"Old\"}} deletes a chip definition and its copies.\n\
         - Read-only (no edit, results come back in the auto-check): \
         {{\"op\":\"inspect\",\"targets\":[\"A\",\"u2\"]}} (pin values and drivers), \
         {{\"op\":\"trace\",\"target\":\"CARRY\"}} (the chain of values feeding a block), \
         {{\"op\":\"highlight\",\"targets\":[\"u1\",\"u2\"],\"note\":\"\u{2026}\"}} (point at \
         blocks on the user's canvas while explaining).\n\
         \n\
         HANDLES: \"from\"/\"to\"/\"target\" accept the \"ref\" you gave a new block, a numeric \
         id from the context below, or a block's label.\n\
         PORTS: input ports are 0,1,2\u{2026} top-to-bottom; outputs are usually just 0. On chip \
         instances use PIN NAMES: \"to_port\":\"cin\", \"from_port\":\"sum\". If you OMIT \
         \"to_port\", the first FREE input is picked automatically \u{2014} so two plain \
         connects fill a 2-input gate's ports 0 then 1. An input may take several wires (they \
         OR together). LEDs have no outputs; switches/constants have no inputs.\n\
         \n\
         EXAMPLE \u{2014} define a verified building block, then use it:\n\
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
         wide and (pins+1) tall. Never overlap blocks. Inputs on the left, outputs on the right, \
         one column per stage (x = 0, 8, 16, \u{2026}), ~4 rows between blocks in a column. \
         Position is cosmetic \u{2014} correctness comes from wires.\n\
         \n\
         RULES:\n\
         1. KEEP THE USER'S WORK. Build new parts beside the existing circuit (see FREE SPACE). \
         Never remove, rewire or rebuild blocks the user placed unless they asked you to change \
         them; if something must go, ASK first (reply without a json block).\n\
         2. If the request is ambiguous, ask one short question instead of guessing.\n\
         3. When you say you will change the circuit, DO IT in the same reply: end with the json \
         block. Never describe changes you don't send.\n\
         4. In prose, refer to blocks by their labels (A, SUM, CLK), never by #id. Label every \
         switch, button, clock and LED you add.\n\
         5. \"this\"/\"these\"/\"the selected\u{2026}\" means the SELECTED blocks listed below.\n\
         \n\
         FINISHING: when the auto-check shows the circuit does what the user asked, reply with \
         a short summary and NO json block \u{2014} that ends the run. Include a json block ONLY \
         to change or test the circuit; for plain questions, just answer.\n\
         \n\
         CURRENT STATE:\n{}{}{}",
        circuit_context(m),
        free_space(m),
        selection_context(m, selected)
    )
}

/// Where new parts can go without landing on the user's blocks.
fn free_space(m: &CircuitManager) -> String {
    let mut max = (i32::MIN, i32::MIN);
    let mut min_y = i32::MAX;
    for b in m.circuit.iter_blocks() {
        let size = b.footprint(&m.chips);
        max.0 = max.0.max(b.pos.x + size.x.ceil() as i32);
        max.1 = max.1.max(b.pos.y + size.y.ceil() as i32);
        min_y = min_y.min(b.pos.y);
    }
    if max.0 == i32::MIN {
        return String::new();
    }
    format!(
        "\nFREE SPACE: the existing blocks end at x={} and y={}. Put new parts at x \u{2265} {} \
         (starting at y={min_y}), or at y \u{2265} {}.",
        max.0,
        max.1,
        max.0 + 6,
        max.1 + 4
    )
}

/// The user's current selection, so "fix this" has a referent.
fn selection_context(m: &CircuitManager, selected: &[BlockId]) -> String {
    let names: Vec<String> = selected
        .iter()
        .filter(|id| m.circuit.block(**id).is_some())
        .take(40)
        .map(|id| format!("#{id} {}", super::apply::block_display_name(m, *id)))
        .collect();
    if names.is_empty() {
        String::new()
    } else {
        format!(
            "\nSELECTED by the user ({}): {}",
            selected.len(),
            names.join(", ")
        )
    }
}

fn kind_name(m: &CircuitManager, b: &Block) -> String {
    match b.ty {
        BlockType::Chip(cid) => m
            .chips
            .get(cid)
            .map(|d| format!("CHIP:{}", d.name))
            .unwrap_or_else(|| "CHIP:?".to_string()),
        ty if ty.variable_inputs() && b.gate_inputs() != 2 => {
            format!("{}({}in)", ty.display_name(), b.gate_inputs())
        }
        BlockType::Clock => format!("Clock({}Hz)", b.freq_hz.unwrap_or(1.0)),
        ty => ty.display_name().to_string(),
    }
}

/// A compact text view of the chip library + circuit for the model: blocks, wires, and each
/// chip's pin names. Far fewer tokens than raw JSON and easier for small models to read. Large
/// designs are summarized: inputs, outputs, chips and labeled blocks are always listed; plain
/// gates beyond the budget are counted instead.
pub fn circuit_context(m: &CircuitManager) -> String {
    let c = &m.circuit;
    let mut out = String::new();

    if !m.chips.is_empty() {
        out.push_str("Chip library (instance with \"kind\":\"<Name>\"):\n");
        for d in m.chips.iter() {
            let ins: Vec<&str> = (0..d.input_blocks.len()).map(|i| d.input_name(i)).collect();
            let outs: Vec<&str> = (0..d.output_blocks.len())
                .map(|i| d.output_name(i))
                .collect();
            let _ = writeln!(
                out,
                "- {}: inputs [{}] outputs [{}]",
                d.name,
                ins.join(", "),
                outs.join(", ")
            );
        }
    }

    if c.blocks.is_empty() {
        out.push_str("The canvas is empty (no blocks placed yet).");
        return out;
    }

    // Priority order: I/O, chips and labeled blocks first, then everything else by id.
    let important =
        |b: &Block| !b.ty.is_gate() || b.label.as_deref().is_some_and(|l| !l.trim().is_empty());
    let mut ordered: Vec<&Block> = c.iter_blocks().filter(|b| important(b)).collect();
    ordered.extend(c.iter_blocks().filter(|b| !important(b)));
    let listed: Vec<&Block> = ordered.iter().take(MAX_BLOCKS_LISTED).copied().collect();
    let listed_ids: HashSet<BlockId> = listed.iter().map(|b| b.id).collect();
    let mut by_id: Vec<&Block> = listed.clone();
    by_id.sort_by_key(|b| b.id);

    let _ = writeln!(
        out,
        "Blocks ({} total; #id kind @(x,y) \"label\"):",
        c.blocks.len()
    );
    for b in &by_id {
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
            "#{} {} @({},{}){label}{state}",
            b.id,
            kind_name(m, b),
            b.pos.x,
            b.pos.y
        );
    }
    let omitted = c.blocks.len() - listed.len();
    if omitted > 0 {
        let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
        for b in ordered.iter().skip(MAX_BLOCKS_LISTED) {
            *counts.entry(b.ty.display_name()).or_default() += 1;
        }
        let summary: Vec<String> = counts
            .iter()
            .map(|(k, n)| format!("{k}\u{d7}{n}"))
            .collect();
        let _ = writeln!(
            out,
            "(+{omitted} more unlabeled gates not listed: {}. Wrap large logic in chips so the \
             top level stays small.)",
            summary.join(", ")
        );
    }

    let wires: Vec<String> = c
        .iter_connections()
        .filter(|w| listed_ids.contains(&w.from.block) && listed_ids.contains(&w.to.block))
        .take(MAX_WIRES_LISTED)
        .map(|w| {
            format!(
                "#{}.{}\u{2192}#{}.{}",
                w.from.block, w.from.index, w.to.block, w.to.index
            )
        })
        .collect();
    if c.connections.is_empty() {
        out.push_str("No wires yet.");
    } else {
        let _ = writeln!(
            out,
            "Wires ({} total, out\u{2192}in, block.port):\n{}",
            c.connections.len(),
            wires.join("  ")
        );
        if wires.len() < c.connections.len() {
            let _ = writeln!(
                out,
                "(+{} more wires not listed)",
                c.connections.len() - wires.len()
            );
        }
    }
    out
}

/// Input ports (on gates, LEDs, chip instances) with no incoming wire, as short descriptions.
fn unwired_inputs(m: &CircuitManager) -> Vec<String> {
    let c = &m.circuit;
    let wired: HashSet<(BlockId, u16)> = c
        .iter_connections()
        .map(|w| (w.to.block, w.to.index))
        .collect();
    let mut out = Vec::new();
    for b in c.iter_blocks() {
        let n = b.input_count(&m.chips);
        for i in 0..n as u16 {
            if !wired.contains(&(b.id, i)) {
                let kind = match b.ty {
                    BlockType::Chip(cid) => {
                        let pin = m
                            .chips
                            .get(cid)
                            .map(|d| d.input_name(i as usize).to_string())
                            .unwrap_or_default();
                        let name = m
                            .chips
                            .get(cid)
                            .map(|d| d.name.clone())
                            .unwrap_or_else(|| "chip".to_string());
                        if pin.is_empty() {
                            format!("{name} in{i}")
                        } else {
                            format!("{name} pin {pin}")
                        }
                    }
                    ty => format!("{} in{i}", ty.display_name()),
                };
                let label = b
                    .label
                    .as_deref()
                    .map(|l| format!("({l})"))
                    .unwrap_or_default();
                out.push(format!("#{}{label} {kind}", b.id));
            }
        }
    }
    out
}

/// The automatic post-edit "auto-check": the full text handed back to the model, plus a short
/// user-facing version for the chat (the model's instructions never leak into the UI).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Observation {
    /// Sent to the model as the next user turn.
    pub text: String,
    /// One line for the user, e.g. "16/16 input cases checked · 1 unwired input".
    pub summary: String,
    /// The facts (truth table, timeline, test results) without instructions to the model, shown
    /// when the user expands the note.
    pub details: String,
}

/// Rows of a clock timeline in the auto-check.
const TIMELINE_EDGES: usize = 8;

/// Does the circuit contain feedback (a loop at the top level, or a stateful chip)? Its outputs
/// then depend on history, so a truth table can't judge it.
pub fn has_feedback(m: &CircuitManager) -> bool {
    fn cyclic(c: &crate::model::Circuit) -> bool {
        let mut adj: BTreeMap<BlockId, Vec<BlockId>> = BTreeMap::new();
        for w in c.iter_connections() {
            adj.entry(w.from.block).or_default().push(w.to.block);
        }
        // Iterative three-color DFS.
        let mut state: BTreeMap<BlockId, u8> = BTreeMap::new(); // 1 = on stack, 2 = done
        for &start in adj.keys() {
            if state.contains_key(&start) {
                continue;
            }
            let mut stack: Vec<(BlockId, usize)> = vec![(start, 0)];
            state.insert(start, 1);
            while let Some((node, i)) = stack.pop() {
                let next = adj.get(&node).and_then(|v| v.get(i)).copied();
                match next {
                    Some(n) => {
                        stack.push((node, i + 1));
                        match state.get(&n) {
                            Some(1) => return true,
                            Some(_) => {}
                            None => {
                                state.insert(n, 1);
                                stack.push((n, 0));
                            }
                        }
                    }
                    None => {
                        state.insert(node, 2);
                    }
                }
            }
        }
        false
    }
    let mut stateful_chips = HashSet::new();
    for d in m.chips.iter() {
        if cyclic(&d.circuit) {
            stateful_chips.insert(d.id);
        }
    }
    // A chip that contains a stateful chip is stateful too.
    let mut changed = true;
    while changed {
        changed = false;
        for d in m.chips.iter() {
            if !stateful_chips.contains(&d.id)
                && d.direct_children()
                    .iter()
                    .any(|c| stateful_chips.contains(c))
            {
                stateful_chips.insert(d.id);
                changed = true;
            }
        }
    }
    cyclic(&m.circuit)
        || m.circuit
            .iter_blocks()
            .any(|b| matches!(b.ty, BlockType::Chip(c) if stateful_chips.contains(&c)))
}

fn plural(n: usize, one: &str, many: &str) -> String {
    format!("{n} {}", if n == 1 { one } else { many })
}

/// The automatic post-edit "auto-check" handed back to the agent: skipped-command errors,
/// chip-verification and test results, unwired inputs, and the circuit's real behavior — a
/// truth table for combinational logic, a clock timeline for clocked logic, a settled snapshot
/// when there are too many inputs. This is what lets the assistant verify and fix its own work
/// without the user prodding it.
pub fn agent_observation(m: &CircuitManager, report: &ApplyReport) -> Observation {
    let c = &m.circuit;
    let inputs: Vec<&Block> = c
        .iter_blocks()
        .filter(|b| matches!(b.ty, BlockType::Switch | BlockType::Button))
        .collect();
    let outputs: Vec<&Block> = c.iter_blocks().filter(|b| b.ty == BlockType::Led).collect();
    let has_clock = c.iter_blocks().any(|b| b.ty == BlockType::Clock);
    let feedback = has_feedback(m);

    let mut facts = String::new();
    let mut summary: Vec<String> = Vec::new();

    if !report.errors.is_empty() {
        facts.push_str("SKIPPED COMMANDS (these did NOT happen \u{2014} fix and resend them):\n");
        for e in &report.errors {
            let _ = writeln!(facts, "  - {e}");
        }
        summary.push(format!(
            "{} skipped",
            plural(report.errors.len(), "command", "commands")
        ));
    }
    for obs in &report.observations {
        let _ = writeln!(facts, "{obs}");
    }
    let chips_checked = report
        .observations
        .iter()
        .filter(|o| o.starts_with("Defined chip"))
        .count();
    if chips_checked > 0 {
        summary.push(format!(
            "{} verified",
            plural(chips_checked, "chip", "chips")
        ));
    }
    if report.checks_passed + report.checks_failed > 0 {
        summary.push(format!(
            "test checks {}/{} passed",
            report.checks_passed,
            report.checks_passed + report.checks_failed
        ));
    } else if report.tests_run > 0 {
        summary.push(plural(report.tests_run, "test run", "test runs"));
    }

    let name = |b: &Block| match &b.label {
        Some(l) if !l.trim().is_empty() => format!("#{}({})", b.id, l.trim()),
        _ => format!("#{}", b.id),
    };

    // Unwired inputs are the most common silent bug — list them explicitly.
    let unwired = unwired_inputs(m);
    if !unwired.is_empty() {
        let _ = writeln!(
            facts,
            "UNWIRED INPUTS ({}): {}{}",
            unwired.len(),
            unwired
                .iter()
                .take(12)
                .cloned()
                .collect::<Vec<_>>()
                .join(", "),
            if unwired.len() > 12 { ", \u{2026}" } else { "" }
        );
        summary.push(plural(unwired.len(), "unwired input", "unwired inputs"));
    }

    // A fresh sim so the user's live switch settings aren't disturbed.
    let mut sim = Simulation::build(&m.circuit, &m.chips);
    if sim.has_recursive_chips() {
        facts.push_str("WARNING: a chip contains itself; those instances are inert.\n");
    }

    let mut unstable = false;
    if outputs.is_empty() {
        let _ = writeln!(
            facts,
            "The circuit has no LED outputs, so its behavior can't be observed \u{2014} add LEDs \
             for the results."
        );
    } else if has_clock {
        // Clocked logic: a truth table at time 0 says nothing. Run the clocks instead and show
        // how the LEDs evolve, with the inputs at their current levels.
        sim.step(0.0);
        let leds: Vec<String> = outputs.iter().map(|b| name(b)).collect();
        let ins: Vec<String> = inputs
            .iter()
            .map(|b| format!("{}={}", name(b), sim.switch_value(b.id) as u8))
            .collect();
        let _ = writeln!(
            facts,
            "Clock timeline (inputs held: {}) \u{2014} LEDs [{}] after each clock edge:",
            if ins.is_empty() {
                "none".to_string()
            } else {
                ins.join(" ")
            },
            leds.join(", ")
        );
        let read = |sim: &Simulation| -> String {
            outputs
                .iter()
                .map(|b| {
                    if sim.led_value(b.id).unwrap_or(false) {
                        "1"
                    } else {
                        "0"
                    }
                })
                .collect::<Vec<_>>()
                .join(" ")
        };
        let _ = writeln!(facts, "  t=0.000s | {}", read(&sim));
        let mut edges = 0;
        for _ in 0..TIMELINE_EDGES {
            let Some(dt) = sim.time_to_next_edge() else {
                break;
            };
            sim.step(dt + 1e-9);
            edges += 1;
            unstable |= sim.is_unstable();
            let _ = writeln!(facts, "  t={:.3}s | {}", sim.time(), read(&sim));
        }
        facts.push_str(
            "(Sequential logic: verify specific sequences with \"test\" \u{2014} set inputs, \
             pulse or tick the clock, and give \"expect\" levels.)\n",
        );
        summary.push(format!(
            "ran the clock for {}",
            plural(edges, "edge", "edges")
        ));
    } else if inputs.is_empty() {
        sim.step(0.0);
        let reads: Vec<String> = outputs
            .iter()
            .map(|b| format!("{}={}", name(b), sim.led_value(b.id).unwrap_or(false) as u8))
            .collect();
        let _ = writeln!(
            facts,
            "No switch/button inputs; settled LEDs: {}",
            reads.join(" ")
        );
        summary.push("read the LEDs".to_string());
    } else if inputs.len() > 6 {
        // Too many combinations for a full table — give a settled snapshot instead and point at
        // the `test` op for targeted verification.
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
            facts,
            "{} inputs \u{2014} too many for a full truth table. Settled snapshot: inputs {}; \
             LEDs {}. Use {{\"op\":\"test\",\"steps\":[\u{2026}]}} with \"expect\" to verify \
             specific cases.",
            inputs.len(),
            states.join(" "),
            reads.join(" ")
        );
        summary.push(format!("snapshot of {} inputs", inputs.len()));
    } else {
        if feedback {
            facts.push_str(
                "(The circuit has FEEDBACK (latch/memory): rows below were applied in this order \
                 and each depends on the previous state. Verify behavior with \"test\" steps and \
                 \"expect\" instead of trusting this table.)\n",
            );
        }
        let _ = writeln!(
            facts,
            "Truth table \u{2014} inputs [{}] \u{2192} outputs [{}]:",
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
                sim.set_switch(b.id, (combo >> (n - 1 - i)) & 1 == 1);
            }
            sim.step(0.0);
            unstable |= sim.is_unstable();
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
            let _ = writeln!(facts, "  {} | {}", ins.join(" "), outs.join(" "));
        }
        summary.push(format!(
            "{} input cases checked{}",
            1u32 << n,
            if feedback { " (stateful)" } else { "" }
        ));
    }
    if unstable {
        facts.push_str(
            "WARNING: the circuit OSCILLATES (never settles) \u{2014} check for unintended \
             feedback loops (an odd number of inverters in a loop).\n",
        );
        summary.push("oscillates!".to_string());
    }

    let details = facts.trim_end().to_string();
    let text = format!(
        "Auto-check after your edits.\n{facts}If this matches the request, reply with a brief \
         summary and NO commands. Otherwise fix it with more commands."
    );
    let summary = if summary.is_empty() {
        "Checked the circuit".to_string()
    } else {
        let mut s = summary.join(" \u{b7} ");
        if let Some(first) = s.get(..1) {
            s = format!("{}{}", first.to_uppercase(), &s[1..]);
        }
        s
    };
    Observation {
        text,
        summary,
        details,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::{apply_ops, parse_reply};
    use crate::model::Pos;

    #[test]
    fn observation_reports_skips_unwired_inputs_and_truth_table() {
        let mut m = CircuitManager::new();
        let r = apply_ops(
            &mut m,
            parse_reply(
                r#"{"commands":[
                {"op":"add","ref":"a","kind":"switch","label":"A","x":0,"y":0},
                {"op":"add","ref":"g","kind":"and","x":4,"y":0},
                {"op":"add","ref":"l","kind":"led","label":"Y","x":8,"y":0},
                {"op":"connect","from":"a","to":"g"},
                {"op":"connect","from":"g","to":"l"},
                {"op":"connect","from":"nope","to":"g"}]}"#,
            )
            .ops,
        );
        let obs = agent_observation(&m, &r).text;
        assert!(obs.contains("SKIPPED COMMANDS"), "{obs}");
        assert!(obs.contains("UNWIRED INPUTS (1)"), "{obs}");
        assert!(obs.contains("Truth table"), "{obs}");
        assert!(
            obs.contains("  1 | 0"),
            "AND with a floating input is 0: {obs}"
        );
    }

    #[test]
    fn big_circuits_are_summarized_in_the_context() {
        let mut m = CircuitManager::new();
        m.add_block(BlockType::Switch, Pos::new(0, 0));
        for i in 0..(MAX_BLOCKS_LISTED as i32 + 50) {
            m.add_block(BlockType::And, Pos::new(4, i * 3));
        }
        let ctx = circuit_context(&m);
        assert!(ctx.contains("#0 Switch"), "I/O always listed");
        assert!(
            ctx.contains("more unlabeled gates not listed"),
            "{}",
            &ctx[ctx.len() - 300..]
        );
        assert!(ctx.lines().count() < MAX_BLOCKS_LISTED + 20);
    }

    #[test]
    fn system_prompt_mentions_every_command() {
        let p = system_prompt(&CircuitManager::new(), &[]);
        for op in [
            "\"add\"",
            "\"connect\"",
            "\"set\"",
            "\"remove\"",
            "\"defchip\"",
            "\"test\"",
            "pulse",
            "ticks",
            "\"disconnect\"",
            "\"inspect\"",
            "\"trace\"",
            "\"highlight\"",
            "\"delete_chip\"",
            "expect",
        ] {
            assert!(p.contains(op), "missing {op}");
        }
    }

    #[test]
    fn clocked_circuits_get_a_timeline_not_a_truth_table() {
        let mut m = CircuitManager::new();
        let r = apply_ops(
            &mut m,
            parse_reply(
                r#"{"commands":[
                {"op":"add","ref":"c","kind":"clock","label":"CLK","hz":2,"x":0,"y":0},
                {"op":"add","ref":"l","kind":"led","label":"Q","x":6,"y":0},
                {"op":"connect","from":"c","to":"l"}]}"#,
            )
            .ops,
        );
        let o = agent_observation(&m, &r);
        assert!(o.text.contains("Clock timeline"), "{}", o.text);
        assert!(!o.text.contains("add at least one"), "{}", o.text);
        assert!(o.details.contains("t=0.250s | 1"), "{}", o.details);
        assert!(
            o.summary.starts_with("Ran the clock for 8 edges"),
            "{}",
            o.summary
        );
        assert!(
            !o.details.contains("If this matches"),
            "instructions stay out of the UI"
        );
    }

    #[test]
    fn feedback_is_detected_and_flagged() {
        let mut m = CircuitManager::new();
        let r = apply_ops(
            &mut m,
            parse_reply(
                r#"{"commands":[
                {"op":"add","ref":"s","kind":"switch","label":"S","x":0,"y":0},
                {"op":"add","ref":"r","kind":"switch","label":"R","x":0,"y":4},
                {"op":"add","ref":"n1","kind":"nor","x":6,"y":0},
                {"op":"add","ref":"n2","kind":"nor","x":6,"y":4},
                {"op":"add","ref":"q","kind":"led","label":"Q","x":12,"y":0},
                {"op":"connect","from":"r","to":"n1"},{"op":"connect","from":"n2","to":"n1"},
                {"op":"connect","from":"s","to":"n2"},{"op":"connect","from":"n1","to":"n2"},
                {"op":"connect","from":"n1","to":"q"}]}"#,
            )
            .ops,
        );
        assert!(has_feedback(&m));
        let o = agent_observation(&m, &r);
        assert!(o.text.contains("FEEDBACK"), "{}", o.text);
        assert!(o.summary.contains("(stateful)"), "{}", o.summary);
    }

    #[test]
    fn prompt_lists_selection_and_free_space() {
        let mut m = CircuitManager::new();
        let id = m.add_block(BlockType::Not, Pos::new(2, 3));
        let p = system_prompt(&m, &[id]);
        assert!(p.contains("SELECTED by the user (1): #0 NOT"), "{p}");
        assert!(p.contains("FREE SPACE"), "{p}");
        assert!(p.contains("KEEP THE USER'S WORK"));
    }
}
