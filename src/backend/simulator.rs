//! The logic simulator.
//!
//! A [`Circuit`] (possibly containing chip instances) is flattened into a list of primitive gates
//! wired together by integer *nets*. Chip instances are expanded recursively; the boundary
//! between an instance's external ports and its internal nets is stitched together with a
//! union-find so ordering and feedback don't matter.
//!
//! Evaluation is **event-driven**: only gates whose inputs changed are re-evaluated, processed in
//! FIFO order (so a change ripples outward one gate "delay" at a time). Combinational logic
//! settles within a step; feedback loops (latches, flip-flops, memory) hold their state. A circuit
//! that never settles (e.g. an odd ring of inverters) hits an evaluation budget and is flagged
//! [`Simulation::is_unstable`] instead of hanging the app.
//!
//! Robustness: a chip that (directly or indirectly) contains itself is never expanded twice on
//! the same path — the recursive instance is skipped and reported via
//! [`Simulation::has_recursive_chips`] — so a malformed library can't overflow the stack.
//!
//! State survives rebuilds: every net carries a stable key (instance path + block + port), so
//! [`Simulation::carry_state_from`] can move register/latch/RAM contents across an edit.

use std::collections::{BTreeMap, HashMap, VecDeque};

use crate::model::{Block, BlockId, BlockType, ChipDef, ChipId, ChipLibrary, Circuit, Port};

type NetId = u32;

/// Net 0 is a permanent ground (logic low). Unconnected inputs read from it.
const GROUND: NetId = 0;

/// Chips nested deeper than this are not expanded (a guard on top of the cycle check).
const MAX_DEPTH: usize = 64;

/// Upper bound on substeps per [`Simulation::step`] when clocks are faster than the frame rate.
const MAX_SUBSTEPS: usize = 4096;

#[derive(Clone, Copy, Debug)]
enum PrimKind {
    And,
    Or,
    Not,
    Nand,
    Nor,
    Xor,
    Xnor,
    Buffer,
    /// A switch/button; the bool is the authored default level.
    Switch(bool),
    Const(bool),
    /// A clock; the f64 is the half-period in seconds.
    Clock(f64),
    /// An output sink (LED / chip output marker); produces no net.
    Sink,
}

struct FlatGate {
    kind: PrimKind,
    /// One entry per input port; each is the list of nets driving it (OR-combined — normally
    /// exactly one).
    inputs: Vec<Vec<NetId>>,
    output: Option<NetId>,
    /// Set only for gates that correspond to a *top-level* block, so the UI can toggle switches
    /// and read LEDs.
    top_block: Option<BlockId>,
}

/// Stable identity of a net across rebuilds: which instance path it lives in, and which block
/// output port drives it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct NetKey {
    path: u64,
    block: BlockId,
    port: u16,
}

#[derive(Default)]
struct UnionFind {
    parent: Vec<NetId>,
}

impl UnionFind {
    fn make_set(&mut self) -> NetId {
        let id = self.parent.len() as NetId;
        self.parent.push(id);
        id
    }

    fn find(&mut self, x: NetId) -> NetId {
        let mut root = x;
        while self.parent[root as usize] != root {
            root = self.parent[root as usize];
        }
        // Path compression.
        let mut cur = x;
        while self.parent[cur as usize] != root {
            let next = self.parent[cur as usize];
            self.parent[cur as usize] = root;
            cur = next;
        }
        root
    }

    fn union(&mut self, a: NetId, b: NetId) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra != rb {
            self.parent[rb as usize] = ra;
        }
    }
}

/// Mix an instance block id into a path hash (FNV-1a style, deterministic across runs).
fn child_path(parent: u64, instance: BlockId) -> u64 {
    let mut h = parent ^ 0xcbf2_9ce4_8422_2325;
    for b in instance.0.to_le_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

struct Builder<'a> {
    chips: &'a ChipLibrary,
    uf: UnionFind,
    gates: Vec<FlatGate>,
    /// Output-port nets for top-level blocks, so wires can be colored by signal.
    top_out_net: BTreeMap<Port, NetId>,
    /// Every allocated net's stable key (for carrying state across rebuilds).
    keys: Vec<(NetKey, NetId)>,
    /// Chips currently being expanded on this path (cycle guard).
    stack: Vec<ChipId>,
    recursive: bool,
}

fn prim_kind(block: &Block) -> PrimKind {
    match block.ty {
        BlockType::And => PrimKind::And,
        BlockType::Or => PrimKind::Or,
        BlockType::Not => PrimKind::Not,
        BlockType::Nand => PrimKind::Nand,
        BlockType::Nor => PrimKind::Nor,
        BlockType::Xor => PrimKind::Xor,
        BlockType::Xnor => PrimKind::Xnor,
        BlockType::Buffer => PrimKind::Buffer,
        BlockType::Switch | BlockType::Button => PrimKind::Switch(block.state),
        BlockType::ConstantHigh => PrimKind::Const(true),
        BlockType::ConstantLow => PrimKind::Const(false),
        BlockType::Clock => PrimKind::Clock(clock_half_period(block)),
        BlockType::Led => PrimKind::Sink,
        BlockType::Chip(_) => unreachable!("chips are expanded, not turned into gates"),
    }
}

/// A clock block's half-period in seconds (frequency defaults to 1 Hz).
pub fn clock_half_period(block: &Block) -> f64 {
    let hz = block.freq_hz.unwrap_or(1.0).max(0.01) as f64;
    1.0 / (2.0 * hz)
}

impl<'a> Builder<'a> {
    /// Nets driving an input port, resolved through the current scope's `out_net` map.
    fn drivers_of(
        incoming: &HashMap<(BlockId, u16), Vec<(BlockId, u16)>>,
        out_net: &HashMap<(BlockId, u16), NetId>,
        block: BlockId,
        index: u16,
    ) -> Vec<NetId> {
        incoming
            .get(&(block, index))
            .map(|srcs| {
                srcs.iter()
                    .filter_map(|s| out_net.get(s).copied())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// One net standing for the OR of `drivers`: ground for none, the driver itself for one,
    /// and a fresh buffer-combined net for several (so a pin fed by two wires behaves exactly
    /// like a gate input fed by two wires).
    fn merged(&mut self, drivers: Vec<NetId>) -> NetId {
        match drivers.len() {
            0 => GROUND,
            1 => drivers[0],
            _ => {
                let net = self.uf.make_set();
                self.gates.push(FlatGate {
                    kind: PrimKind::Buffer,
                    inputs: vec![drivers],
                    output: Some(net),
                    top_block: None,
                });
                net
            }
        }
    }

    /// Expand one circuit scope. For a chip instance, `input_nets` supplies the external net
    /// feeding each input pin; the return value is the net for each output pin. For the top
    /// level, the pin slices are empty and the return is empty.
    fn expand(
        &mut self,
        circuit: &'a Circuit,
        input_pin_blocks: &[BlockId],
        output_pin_blocks: &[BlockId],
        input_nets: &[NetId],
        path: u64,
    ) -> Vec<NetId> {
        let is_top = self.stack.is_empty();
        let in_index: HashMap<BlockId, usize> = input_pin_blocks
            .iter()
            .enumerate()
            .map(|(i, b)| (*b, i))
            .collect();
        let out_index: HashMap<BlockId, usize> = output_pin_blocks
            .iter()
            .enumerate()
            .map(|(i, b)| (*b, i))
            .collect();

        // Index each input port's incoming wires once (a per-port scan would be quadratic).
        let mut incoming: HashMap<(BlockId, u16), Vec<(BlockId, u16)>> = HashMap::new();
        for c in circuit.iter_connections() {
            incoming
                .entry((c.to.block, c.to.index))
                .or_default()
                .push((c.from.block, c.from.index));
        }

        // 1. Allocate a net for every output port. Input-pin markers alias to the external
        //    driver instead of getting a fresh net.
        let mut out_net: HashMap<(BlockId, u16), NetId> = HashMap::new();
        for block in circuit.iter_blocks() {
            if let Some(&pin) = in_index.get(&block.id) {
                out_net.insert(
                    (block.id, 0),
                    input_nets.get(pin).copied().unwrap_or(GROUND),
                );
            } else {
                for i in 0..block.output_count(self.chips) {
                    let net = self.uf.make_set();
                    out_net.insert((block.id, i as u16), net);
                    self.keys.push((
                        NetKey {
                            path,
                            block: block.id,
                            port: i as u16,
                        },
                        net,
                    ));
                    if is_top {
                        self.top_out_net
                            .insert(Port::output(block.id, i as u16), net);
                    }
                }
            }
        }

        // 2. Build gates (and recurse into chip instances).
        let mut output_nets = vec![GROUND; output_pin_blocks.len()];
        for block in circuit.iter_blocks() {
            if in_index.contains_key(&block.id) {
                continue; // input markers contribute no gate
            }
            match block.ty {
                BlockType::Chip(cid) => {
                    let chips = self.chips;
                    let Some(chip): Option<&'a ChipDef> = chips.get(cid) else {
                        continue;
                    };
                    // A chip containing itself (directly or through others) is never expanded
                    // twice on one path — that would recurse forever.
                    if self.stack.contains(&cid) || self.stack.len() >= MAX_DEPTH {
                        self.recursive = true;
                        continue;
                    }
                    let mut in_nets = Vec::with_capacity(chip.input_blocks.len());
                    for i in 0..chip.input_blocks.len() {
                        let d = Self::drivers_of(&incoming, &out_net, block.id, i as u16);
                        in_nets.push(self.merged(d));
                    }
                    self.stack.push(cid);
                    let inner_out = self.expand(
                        &chip.circuit,
                        &chip.input_blocks,
                        &chip.output_blocks,
                        &in_nets,
                        child_path(path, block.id),
                    );
                    self.stack.pop();
                    for (j, inner) in inner_out.iter().enumerate() {
                        if let Some(&outer) = out_net.get(&(block.id, j as u16)) {
                            self.uf.union(outer, *inner);
                        }
                    }
                }
                _ => {
                    let ninputs = block.input_count(self.chips);
                    let mut inputs = Vec::with_capacity(ninputs);
                    for i in 0..ninputs {
                        inputs.push(Self::drivers_of(&incoming, &out_net, block.id, i as u16));
                    }

                    if let Some(&pin) = out_index.get(&block.id) {
                        // Output-pin marker: its (OR-combined) input becomes the chip's output.
                        let d = inputs.first().cloned().unwrap_or_default();
                        output_nets[pin] = self.merged(d);
                    }

                    let output = if block.output_count(self.chips) > 0 {
                        out_net.get(&(block.id, 0)).copied()
                    } else {
                        None
                    };

                    self.gates.push(FlatGate {
                        kind: prim_kind(block),
                        inputs,
                        output,
                        top_block: if is_top { Some(block.id) } else { None },
                    });
                }
            }
        }

        output_nets
    }
}

/// A compiled, runnable simulation of a circuit.
pub struct Simulation {
    gates: Vec<FlatGate>,
    values: Vec<bool>,
    /// Live top-level switch/button levels.
    switch_states: BTreeMap<BlockId, bool>,
    time: f64,
    /// Canonical net for each top-level output port (for wire coloring).
    top_out_net: BTreeMap<Port, NetId>,
    /// Gate index for each top-level block (for reading LEDs / toggling switches).
    top_gate: BTreeMap<BlockId, usize>,
    /// For each net, the gates that read it.
    fanout: Vec<Vec<u32>>,
    /// Indices of clock gates (re-evaluated whenever time advances).
    clocks: Vec<u32>,
    /// Smallest clock half-period, if the circuit has clocks.
    min_half_period: Option<f64>,
    queue: VecDeque<u32>,
    queued: Vec<bool>,
    /// Stable key → canonical net, for carrying state across rebuilds.
    net_keys: HashMap<NetKey, NetId>,
    unstable: bool,
    recursive: bool,
}

impl Simulation {
    /// Flatten and compile a circuit into a runnable simulation. Nothing is evaluated until the
    /// first [`Simulation::step`] (every gate starts queued).
    pub fn build(circuit: &Circuit, chips: &ChipLibrary) -> Self {
        let mut builder = Builder {
            chips,
            uf: UnionFind::default(),
            gates: Vec::new(),
            top_out_net: BTreeMap::new(),
            keys: Vec::new(),
            stack: Vec::new(),
            recursive: false,
        };
        // Reserve GROUND as net 0.
        let ground = builder.uf.make_set();
        debug_assert_eq!(ground, GROUND);

        builder.expand(circuit, &[], &[], &[], 0);

        let Builder {
            mut uf,
            mut gates,
            top_out_net,
            keys,
            recursive,
            ..
        } = builder;

        // Canonicalize nets into a compact [0, k) range (ground is always canonical 0).
        let mut canon: HashMap<NetId, u32> = HashMap::new();
        let mut next = 0u32;
        let mut resolve = |uf: &mut UnionFind, net: NetId| -> u32 {
            let root = uf.find(net);
            *canon.entry(root).or_insert_with(|| {
                let id = next;
                next += 1;
                id
            })
        };
        let ground = resolve(&mut uf, GROUND);
        debug_assert_eq!(ground, 0);

        for gate in &mut gates {
            for drivers in &mut gate.inputs {
                for n in drivers.iter_mut() {
                    *n = resolve(&mut uf, *n);
                }
            }
            if let Some(o) = &mut gate.output {
                *o = resolve(&mut uf, *o);
            }
        }
        let top_out_net: BTreeMap<Port, NetId> = top_out_net
            .into_iter()
            .map(|(p, n)| (p, resolve(&mut uf, n)))
            .collect();
        let net_keys: HashMap<NetKey, NetId> = keys
            .into_iter()
            .map(|(k, n)| (k, resolve(&mut uf, n)))
            .collect();

        let nets = next as usize;
        let mut fanout: Vec<Vec<u32>> = vec![Vec::new(); nets];
        let mut top_gate = BTreeMap::new();
        let mut switch_states = BTreeMap::new();
        let mut clocks = Vec::new();
        let mut min_half_period: Option<f64> = None;
        for (idx, gate) in gates.iter().enumerate() {
            for drivers in &gate.inputs {
                for &n in drivers {
                    let list = &mut fanout[n as usize];
                    if list.last() != Some(&(idx as u32)) {
                        list.push(idx as u32);
                    }
                }
            }
            if let Some(id) = gate.top_block {
                top_gate.insert(id, idx);
                if let PrimKind::Switch(default) = gate.kind {
                    switch_states.insert(id, default);
                }
            }
            if let PrimKind::Clock(h) = gate.kind {
                clocks.push(idx as u32);
                min_half_period = Some(min_half_period.map_or(h, |m: f64| m.min(h)));
            }
        }

        let n_gates = gates.len();
        Self {
            gates,
            values: vec![false; nets],
            switch_states,
            time: 0.0,
            top_out_net,
            top_gate,
            fanout,
            clocks,
            min_half_period,
            queue: (0..n_gates as u32).collect(),
            queued: vec![true; n_gates],
            net_keys,
            unstable: false,
            recursive,
        }
    }

    /// Copy register/latch/memory contents (and the clock phase) from a previous build of the
    /// same project, matching nets by their stable keys. Gates are re-evaluated on the next step,
    /// so combinational logic recomputes while feedback loops keep what they held.
    pub fn carry_state_from(&mut self, old: &Simulation) {
        for (key, &net) in &self.net_keys {
            if let Some(&old_net) = old.net_keys.get(key) {
                if let Some(&v) = old.values.get(old_net as usize) {
                    self.values[net as usize] = v;
                }
            }
        }
        self.time = old.time;
    }

    fn enqueue(&mut self, gate: u32) {
        let g = gate as usize;
        if !self.queued[g] {
            self.queued[g] = true;
            self.queue.push_back(gate);
        }
    }

    pub fn set_switch(&mut self, id: BlockId, state: bool) {
        let prev = self.switch_states.insert(id, state);
        if prev != Some(state) {
            if let Some(&g) = self.top_gate.get(&id) {
                self.enqueue(g as u32);
            }
        }
    }

    pub fn toggle_switch(&mut self, id: BlockId) {
        let v = self.switch_value(id);
        self.set_switch(id, !v);
    }

    /// Advance time by `dt` seconds and settle the network. When clocks run faster than `dt`,
    /// time is advanced in substeps so no clock edge is skipped (bounded per call).
    pub fn step(&mut self, dt: f64) {
        let dt = dt.max(0.0);
        let n = match self.min_half_period {
            Some(h) if dt > 0.0 => ((dt / (h * 0.5)).ceil() as usize).clamp(1, MAX_SUBSTEPS),
            _ => 1,
        };
        let sub = dt / n as f64;
        self.unstable = false;
        for _ in 0..n {
            self.time += sub;
            for i in 0..self.clocks.len() {
                let g = self.clocks[i];
                self.enqueue(g);
            }
            self.settle();
        }
    }

    /// Seconds until the next clock edge (or `None` without clocks). Used by "Step" so one
    /// press advances exactly one clock transition.
    pub fn time_to_next_edge(&self) -> Option<f64> {
        let mut best: Option<f64> = None;
        for &g in &self.clocks {
            if let PrimKind::Clock(h) = self.gates[g as usize].kind {
                let k = (self.time / h + 1e-9).floor() + 1.0;
                let dt = (k * h - self.time).max(0.0);
                best = Some(best.map_or(dt, |b: f64| b.min(dt)));
            }
        }
        best
    }

    /// Process queued gates until nothing changes, or the budget runs out (an oscillating
    /// circuit). Pending work is kept so an oscillation keeps animating on the next step.
    fn settle(&mut self) {
        let budget = 20_000 + self.gates.len() * 32;
        let mut evals = 0usize;
        while let Some(g) = self.queue.pop_front() {
            self.queued[g as usize] = false;
            evals += 1;
            let gate = &self.gates[g as usize];
            let val = eval_gate(gate, &self.values, self.time, &self.switch_states);
            if let Some(net) = gate.output {
                let net = net as usize;
                if self.values[net] != val {
                    self.values[net] = val;
                    for i in 0..self.fanout[net].len() {
                        let f = self.fanout[net][i];
                        self.enqueue(f);
                    }
                }
            }
            if evals >= budget && !self.queue.is_empty() {
                self.unstable = true;
                break;
            }
        }
    }

    /// True when the last step hit the evaluation budget: something in the circuit oscillates
    /// (it never settles), e.g. a NOT gate feeding itself.
    pub fn is_unstable(&self) -> bool {
        self.unstable
    }

    /// True when a chip contains itself (directly or via other chips); such instances are left
    /// unexpanded.
    pub fn has_recursive_chips(&self) -> bool {
        self.recursive
    }

    /// Simulated seconds elapsed.
    pub fn time(&self) -> f64 {
        self.time
    }

    /// Value of a top-level output port (for coloring wires leaving it).
    pub fn output_value(&self, port: Port) -> Option<bool> {
        self.top_out_net
            .get(&port)
            .map(|&n| self.values[n as usize])
    }

    /// Value observed at a top-level LED (OR of its drivers).
    pub fn led_value(&self, block: BlockId) -> Option<bool> {
        let &idx = self.top_gate.get(&block)?;
        let gate = &self.gates[idx];
        Some(input_value(gate, 0, &self.values))
    }

    /// Live level of a top-level switch/button.
    pub fn switch_value(&self, block: BlockId) -> bool {
        self.switch_states.get(&block).copied().unwrap_or(false)
    }

    /// Number of primitive gates after flattening (for diagnostics / status).
    pub fn gate_count(&self) -> usize {
        self.gates.len()
    }
}

fn input_value(gate: &FlatGate, i: usize, values: &[bool]) -> bool {
    gate.inputs
        .get(i)
        .is_some_and(|drivers| drivers.iter().any(|&n| values[n as usize]))
}

fn eval_gate(
    gate: &FlatGate,
    values: &[bool],
    time: f64,
    switches: &BTreeMap<BlockId, bool>,
) -> bool {
    let n = gate.inputs.len();
    let count_true = || (0..n).filter(|&i| input_value(gate, i, values)).count();
    match gate.kind {
        PrimKind::And => (0..n).all(|i| input_value(gate, i, values)),
        PrimKind::Or => (0..n).any(|i| input_value(gate, i, values)),
        PrimKind::Not => !input_value(gate, 0, values),
        PrimKind::Nand => !(0..n).all(|i| input_value(gate, i, values)),
        PrimKind::Nor => !(0..n).any(|i| input_value(gate, i, values)),
        PrimKind::Xor => count_true() % 2 == 1,
        PrimKind::Xnor => count_true() % 2 == 0,
        PrimKind::Buffer => input_value(gate, 0, values),
        PrimKind::Switch(default) => match gate.top_block {
            Some(id) => switches.get(&id).copied().unwrap_or(default),
            None => default,
        },
        PrimKind::Const(b) => b,
        PrimKind::Clock(half_period) => (time / half_period + 1e-9).floor() as i64 % 2 != 0,
        PrimKind::Sink => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CircuitManager;
    use crate::model::{Block, ChipDef, Pos};

    fn buffer_chip(m: &mut CircuitManager, name: &str) -> ChipId {
        let mut inner = CircuitManager::new();
        let i = inner.add_block(BlockType::Switch, Pos::new(0, 0));
        let o = inner.add_block(BlockType::Led, Pos::new(6, 0));
        inner.connect(Port::output(i, 0), Port::input(o, 0));
        let cid = m.chips.allocate_id();
        m.chips
            .insert(ChipDef::from_circuit(cid, name, inner.circuit.clone()));
        cid
    }

    #[test]
    fn chip_input_pin_ors_multiple_wires_like_a_gate() {
        let mut m = CircuitManager::new();
        let cid = buffer_chip(&mut m, "Buf");
        let a = m.add_block(BlockType::Switch, Pos::new(0, 0));
        let b = m.add_block(BlockType::Switch, Pos::new(0, 4));
        let chip = m.add_block(BlockType::Chip(cid), Pos::new(6, 0));
        let led = m.add_block(BlockType::Led, Pos::new(12, 0));
        m.connect(Port::output(a, 0), Port::input(chip, 0));
        m.connect(Port::output(b, 0), Port::input(chip, 0));
        m.connect(Port::output(chip, 0), Port::input(led, 0));
        let mut sim = Simulation::build(&m.circuit, &m.chips);
        for (av, bv) in [(false, false), (false, true), (true, false), (true, true)] {
            sim.set_switch(a, av);
            sim.set_switch(b, bv);
            sim.step(0.0);
            assert_eq!(sim.led_value(led), Some(av || bv), "a={av} b={bv}");
        }
    }

    #[test]
    fn self_containing_chip_does_not_crash() {
        let mut m = CircuitManager::new();
        let cid = buffer_chip(&mut m, "Loop");
        // Put an instance of the chip inside its own definition.
        let mut def = m.chips.get(cid).unwrap().clone();
        let id = def.circuit.allocate_block_id();
        def.circuit
            .insert_block(Block::new(id, BlockType::Chip(cid), Pos::new(3, 3)));
        m.chips.insert(def);
        m.add_block(BlockType::Chip(cid), Pos::new(0, 0));
        let mut sim = Simulation::build(&m.circuit, &m.chips);
        sim.step(0.0);
        assert!(sim.has_recursive_chips());
    }

    #[test]
    fn ring_oscillator_is_flagged_unstable_not_hung() {
        let mut m = CircuitManager::new();
        let n = m.add_block(BlockType::Not, Pos::new(0, 0));
        m.connect(Port::output(n, 0), Port::input(n, 0));
        let mut sim = Simulation::build(&m.circuit, &m.chips);
        sim.step(0.0);
        assert!(sim.is_unstable());
    }

    #[test]
    fn latch_state_survives_a_rebuild() {
        // Cross-coupled NOR latch, set, then an unrelated edit (adding a block) rebuilds.
        let mut m = CircuitManager::new();
        let s = m.add_block(BlockType::Switch, Pos::new(0, 0));
        let r = m.add_block(BlockType::Switch, Pos::new(0, 4));
        let q = m.add_block(BlockType::Nor, Pos::new(4, 0));
        let qn = m.add_block(BlockType::Nor, Pos::new(4, 4));
        let led = m.add_block(BlockType::Led, Pos::new(8, 0));
        m.connect(Port::output(r, 0), Port::input(q, 0));
        m.connect(Port::output(qn, 0), Port::input(q, 1));
        m.connect(Port::output(s, 0), Port::input(qn, 0));
        m.connect(Port::output(q, 0), Port::input(qn, 1));
        m.connect(Port::output(q, 0), Port::input(led, 0));
        let mut sim = Simulation::build(&m.circuit, &m.chips);
        sim.set_switch(s, true);
        sim.step(0.0);
        sim.set_switch(s, false);
        sim.step(0.0);
        assert_eq!(sim.led_value(led), Some(true));

        m.add_block(BlockType::And, Pos::new(20, 20));
        let mut fresh = Simulation::build(&m.circuit, &m.chips);
        fresh.carry_state_from(&sim);
        fresh.step(0.0);
        assert_eq!(fresh.led_value(led), Some(true), "latch kept its 1");
    }

    #[test]
    fn fast_clock_edges_are_not_skipped() {
        // "Step" advances exactly to the next edge of a fast clock, and coarse steps are split
        // into substeps so the clock is sampled on both sides of every edge.
        let mut m = CircuitManager::new();
        let clk = m.add_block(BlockType::Clock, Pos::new(0, 0));
        m.circuit.block_mut(clk).unwrap().freq_hz = Some(40.0);
        let led = m.add_block(BlockType::Led, Pos::new(4, 0));
        m.connect(Port::output(clk, 0), Port::input(led, 0));
        let mut sim = Simulation::build(&m.circuit, &m.chips);
        let edge = sim.time_to_next_edge().unwrap();
        assert!((edge - 0.0125).abs() < 1e-6, "40 Hz half-period is 12.5 ms");
        sim.step(edge + 1e-6);
        assert_eq!(sim.led_value(led), Some(true), "first edge reached");
    }
}
