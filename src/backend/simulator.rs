//! The logic simulator.
//!
//! A [`Circuit`] (possibly containing chip instances) is flattened into a list of
//! primitive gates wired together by integer *nets*. Chip instances are expanded
//! recursively; the boundary between an instance's external ports and its internal
//! nets is stitched together with a union-find so ordering and feedback don't matter.
//! Evaluation then iterates the gates to a fixpoint each step, which settles
//! combinational logic immediately and propagates sequential logic one step at a time.

use std::collections::{BTreeMap, BTreeSet};

use crate::model::{Block, BlockId, BlockType, ChipLibrary, Circuit, Port};

type NetId = u32;

/// Net 0 is a permanent ground (logic low). Unconnected inputs read from it.
const GROUND: NetId = 0;

const MAX_ITERATIONS: usize = 200;

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
    Clock,
    /// An output sink (LED / chip output marker); produces no net.
    Sink,
}

struct FlatGate {
    kind: PrimKind,
    /// One entry per input port; each is the list of nets driving it (OR-combined —
    /// normally exactly one).
    inputs: Vec<Vec<NetId>>,
    output: Option<NetId>,
    /// Set only for gates that correspond to a *top-level* block, so the UI can toggle
    /// switches and read LEDs.
    top_block: Option<BlockId>,
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

struct Builder<'a> {
    chips: &'a ChipLibrary,
    uf: UnionFind,
    gates: Vec<FlatGate>,
    /// Output-port nets for top-level blocks, so wires can be colored by signal.
    top_out_net: BTreeMap<Port, NetId>,
    /// Top-level input ports driven by more than one wire (multi-driver conflicts).
    conflicts: BTreeSet<Port>,
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
        BlockType::Clock => PrimKind::Clock,
        BlockType::Led => PrimKind::Sink,
        BlockType::Chip(_) => unreachable!("chips are expanded, not turned into gates"),
    }
}

impl<'a> Builder<'a> {
    /// Nets driving an input port, resolved through the current scope's `out_net` map.
    fn drivers_of(
        circuit: &Circuit,
        out_net: &BTreeMap<(BlockId, u16), NetId>,
        port: Port,
    ) -> Vec<NetId> {
        circuit
            .connections_into(port)
            .filter_map(|c| out_net.get(&(c.from.block, c.from.index)).copied())
            .collect()
    }

    /// Expand one circuit scope. For a chip instance, `input_nets` supplies the external
    /// net feeding each input pin; the return value is the net for each output pin. For
    /// the top level, the pin slices are empty and the return is empty.
    fn expand(
        &mut self,
        circuit: &Circuit,
        input_pin_blocks: &[BlockId],
        output_pin_blocks: &[BlockId],
        input_nets: &[NetId],
        is_top: bool,
    ) -> Vec<NetId> {
        let in_index: BTreeMap<BlockId, usize> = input_pin_blocks
            .iter()
            .enumerate()
            .map(|(i, b)| (*b, i))
            .collect();
        let out_index: BTreeMap<BlockId, usize> = output_pin_blocks
            .iter()
            .enumerate()
            .map(|(i, b)| (*b, i))
            .collect();

        // 1. Allocate a net for every output port. Input-pin markers alias to the
        //    external driver instead of getting a fresh net.
        let mut out_net: BTreeMap<(BlockId, u16), NetId> = BTreeMap::new();
        for block in circuit.iter_blocks() {
            if let Some(&pin) = in_index.get(&block.id) {
                out_net.insert((block.id, 0), input_nets[pin]);
            } else {
                for i in 0..block.output_count(self.chips) {
                    let net = self.uf.make_set();
                    out_net.insert((block.id, i as u16), net);
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
                    let Some(chip) = self.chips.get(cid) else {
                        continue;
                    };
                    let chip_circuit = chip.circuit.clone();
                    let chip_inputs = chip.input_blocks.clone();
                    let chip_outputs = chip.output_blocks.clone();

                    let mut in_nets = Vec::with_capacity(chip_inputs.len());
                    for i in 0..chip_inputs.len() {
                        let d =
                            Self::drivers_of(circuit, &out_net, Port::input(block.id, i as u16));
                        if d.len() > 1 && is_top {
                            self.conflicts.insert(Port::input(block.id, i as u16));
                        }
                        in_nets.push(d.first().copied().unwrap_or(GROUND));
                    }

                    let inner_out =
                        self.expand(&chip_circuit, &chip_inputs, &chip_outputs, &in_nets, false);
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
                        let d =
                            Self::drivers_of(circuit, &out_net, Port::input(block.id, i as u16));
                        if d.len() > 1 && is_top {
                            self.conflicts.insert(Port::input(block.id, i as u16));
                        }
                        inputs.push(d);
                    }

                    if let Some(&pin) = out_index.get(&block.id) {
                        // Output-pin marker: its input net becomes the chip's output net.
                        output_nets[pin] = inputs
                            .first()
                            .and_then(|d| d.first().copied())
                            .unwrap_or(GROUND);
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
    /// Live top-level switch/button overrides (persist across rebuilds via the app).
    pub switch_states: BTreeMap<BlockId, bool>,
    time: f64,
    clock_period: f64,
    /// Canonical net for each top-level output port (for wire coloring).
    top_out_net: BTreeMap<Port, NetId>,
    /// Gate index for each top-level block (for reading LEDs).
    top_gate: BTreeMap<BlockId, usize>,
    conflicts: BTreeSet<Port>,
}

impl Simulation {
    /// Flatten and compile a circuit into a runnable simulation.
    pub fn build(circuit: &Circuit, chips: &ChipLibrary) -> Self {
        let mut builder = Builder {
            chips,
            uf: UnionFind::default(),
            gates: Vec::new(),
            top_out_net: BTreeMap::new(),
            conflicts: BTreeSet::new(),
        };
        // Reserve GROUND as net 0.
        let ground = builder.uf.make_set();
        debug_assert_eq!(ground, GROUND);

        builder.expand(circuit, &[], &[], &[], true);

        // Canonicalize nets into a compact [0, k) range.
        let mut canon: BTreeMap<NetId, u32> = BTreeMap::new();
        let mut next = 0u32;
        let mut resolve = |uf: &mut UnionFind, net: NetId| -> u32 {
            let root = uf.find(net);
            *canon.entry(root).or_insert_with(|| {
                let id = next;
                next += 1;
                id
            })
        };

        let Builder {
            mut uf,
            mut gates,
            top_out_net,
            conflicts,
            ..
        } = builder;

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

        let mut top_gate = BTreeMap::new();
        let mut switch_states = BTreeMap::new();
        for (idx, gate) in gates.iter().enumerate() {
            if let Some(id) = gate.top_block {
                top_gate.insert(id, idx);
                if let PrimKind::Switch(default) = gate.kind {
                    switch_states.insert(id, default);
                }
            }
        }

        Self {
            gates,
            values: vec![false; next as usize],
            switch_states,
            time: 0.0,
            clock_period: 1.0,
            top_out_net,
            top_gate,
            conflicts,
        }
    }

    /// Seconds per half-period for `Clock` blocks (default 1.0s).
    pub fn set_clock_period(&mut self, period: f64) {
        self.clock_period = period.max(0.01);
    }

    pub fn set_switch(&mut self, id: BlockId, state: bool) {
        self.switch_states.insert(id, state);
    }

    pub fn toggle_switch(&mut self, id: BlockId) {
        let v = self.switch_states.entry(id).or_insert(false);
        *v = !*v;
    }

    /// Advance time and settle the network.
    pub fn step(&mut self, dt: f64) {
        self.time += dt;
        for _ in 0..MAX_ITERATIONS {
            let mut changed = false;
            for gate in &self.gates {
                let val = eval_gate(
                    gate,
                    &self.values,
                    self.time,
                    self.clock_period,
                    &self.switch_states,
                );
                if let Some(net) = gate.output {
                    let net = net as usize;
                    if self.values[net] != val {
                        self.values[net] = val;
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
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

    pub fn switch_value(&self, block: BlockId) -> bool {
        self.switch_states.get(&block).copied().unwrap_or(false)
    }

    /// Whether an input port is driven by more than one wire.
    pub fn is_conflict(&self, port: Port) -> bool {
        self.conflicts.contains(&port)
    }

    pub fn has_conflicts(&self) -> bool {
        !self.conflicts.is_empty()
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
    clock_period: f64,
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
        PrimKind::Clock => (time / clock_period).floor() as i64 % 2 != 0,
        PrimKind::Sink => false,
    }
}
