//! Load-time validation: make a project read from disk internally consistent before anything
//! touches it.
//!
//! Files can be damaged (a crash in another tool, a hand edit, an AI-written file, a chip that
//! was deleted in a newer build), and the editor's invariants — every wire joins an existing
//! output to an existing input, every chip instance has a definition, id counters never hand out
//! an id that is already taken — must hold or later edits misbehave in confusing ways. Repairs
//! are reported as short human-readable lines so the user is told what was fixed.

use std::collections::{BTreeMap, HashSet};

use super::block::{BlockType, MAX_GATE_INPUTS};
use super::chip::ChipLibrary;
use super::circuit::Circuit;
use super::connection::PortKind;
use super::ids::{BlockId, ConnId};

/// Repair `circuit` and `chips` in place. Returns one line per kind of problem fixed (empty when
/// the project was already consistent).
pub fn repair_project(circuit: &mut Circuit, chips: &mut ChipLibrary) -> Vec<String> {
    let mut notes = Vec::new();

    // 1. The chip library: map keys must match the ids inside, and pin lists must name exactly
    //    the chip body's switches (inputs) and LEDs (outputs).
    let rekeyed: BTreeMap<_, _> = std::mem::take(&mut chips.chips)
        .into_values()
        .map(|d| (d.id, d))
        .collect();
    chips.chips = rekeyed;
    chips.fix_counter();
    let mut pin_fixes = 0usize;
    for def in chips.chips.values_mut() {
        let before = (def.input_blocks.clone(), def.output_blocks.clone());
        fix_pins(&def.circuit, &mut def.input_blocks, BlockType::Switch);
        fix_pins(&def.circuit, &mut def.output_blocks, BlockType::Led);
        if before != (def.input_blocks.clone(), def.output_blocks.clone()) {
            pin_fixes += 1;
        }
    }
    if pin_fixes > 0 {
        notes.push(format!(
            "rebuilt the pin list of {pin_fixes} chip{}",
            plural(pin_fixes)
        ));
    }

    // 2. Every chip body, then the main circuit, against the (now pin-consistent) library.
    let snapshot = chips.clone();
    let mut body = Tally::default();
    for def in chips.chips.values_mut() {
        body.add(repair_circuit(&mut def.circuit, &snapshot));
    }
    body.describe("inside chips", &mut notes);
    let top = repair_circuit(circuit, &snapshot);
    top.describe("", &mut notes);
    notes
}

fn plural(n: usize) -> &'static str {
    if n == 1 {
        ""
    } else {
        "s"
    }
}

/// Keep the listed pins that still exist with the right type (in their saved order, without
/// duplicates), then append any unlisted ones in position order.
fn fix_pins(c: &Circuit, pins: &mut Vec<BlockId>, ty: BlockType) {
    let mut seen = HashSet::new();
    pins.retain(|id| c.block(*id).is_some_and(|b| b.ty == ty) && seen.insert(*id));
    let mut extra: Vec<_> = c
        .iter_blocks()
        .filter(|b| b.ty == ty && !seen.contains(&b.id))
        .collect();
    extra.sort_by_key(|b| (b.pos.y, b.pos.x));
    pins.extend(extra.into_iter().map(|b| b.id));
}

/// What one circuit's repair changed.
#[derive(Default)]
struct Tally {
    unknown_chips: usize,
    bad_wires: usize,
    duplicate_wires: usize,
    fixed_props: usize,
}

impl Tally {
    fn add(&mut self, o: Tally) {
        self.unknown_chips += o.unknown_chips;
        self.bad_wires += o.bad_wires;
        self.duplicate_wires += o.duplicate_wires;
        self.fixed_props += o.fixed_props;
    }

    fn describe(&self, place: &str, notes: &mut Vec<String>) {
        let at = if place.is_empty() {
            String::new()
        } else {
            format!(" {place}")
        };
        if self.unknown_chips > 0 {
            notes.push(format!(
                "removed {} instance{} of a chip that no longer exists{at}",
                self.unknown_chips,
                plural(self.unknown_chips)
            ));
        }
        if self.bad_wires > 0 {
            notes.push(format!(
                "removed {} broken wire{}{at} (pointing at missing blocks or pins)",
                self.bad_wires,
                plural(self.bad_wires)
            ));
        }
        if self.duplicate_wires > 0 {
            notes.push(format!(
                "removed {} duplicate wire{}{at}",
                self.duplicate_wires,
                plural(self.duplicate_wires)
            ));
        }
        if self.fixed_props > 0 {
            notes.push(format!(
                "fixed {} invalid block setting{}{at}",
                self.fixed_props,
                plural(self.fixed_props)
            ));
        }
    }
}

fn repair_circuit(c: &mut Circuit, chips: &ChipLibrary) -> Tally {
    let mut t = Tally::default();

    // Keys must match ids (a map key and the id inside can disagree in a hand-edited file).
    let blocks = std::mem::take(&mut c.blocks);
    c.blocks = blocks.into_values().map(|b| (b.id, b)).collect();
    let conns = std::mem::take(&mut c.connections);
    c.connections = conns.into_values().map(|w| (w.id, w)).collect();

    // Instances of chips that aren't in the library can't be simulated or drawn: drop them.
    let orphans: Vec<BlockId> = c
        .iter_blocks()
        .filter(|b| matches!(b.ty, BlockType::Chip(cid) if chips.get(cid).is_none()))
        .map(|b| b.id)
        .collect();
    for id in orphans {
        c.blocks.remove(&id);
        t.unknown_chips += 1;
    }

    // Out-of-range settings.
    for b in c.blocks.values_mut() {
        if let Some(n) = b.inputs {
            let ok = b.ty.variable_inputs() && (2..=MAX_GATE_INPUTS as u16).contains(&n);
            if !ok {
                b.inputs =
                    b.ty.variable_inputs()
                        .then(|| n.clamp(2, MAX_GATE_INPUTS as u16));
                t.fixed_props += 1;
            }
        }
        if let Some(hz) = b.freq_hz {
            if !hz.is_finite() || hz <= 0.0 || b.ty != BlockType::Clock {
                b.freq_hz = None;
                t.fixed_props += 1;
            } else if hz > 1000.0 {
                b.freq_hz = Some(1000.0);
                t.fixed_props += 1;
            }
        }
        let clamp = |v: i32| v.clamp(-1_000_000, 1_000_000);
        if b.pos.x != clamp(b.pos.x) || b.pos.y != clamp(b.pos.y) {
            b.pos.x = clamp(b.pos.x);
            b.pos.y = clamp(b.pos.y);
            t.fixed_props += 1;
        }
    }

    // Wires: both ends must exist, run output → input, and name a real pin; no duplicates.
    let mut seen = HashSet::new();
    let mut drop: Vec<ConnId> = Vec::new();
    for w in c.connections.values() {
        let valid = w.from.kind == PortKind::Output
            && w.to.kind == PortKind::Input
            && c.block(w.from.block)
                .is_some_and(|b| (w.from.index as usize) < b.output_count(chips))
            && c.block(w.to.block)
                .is_some_and(|b| (w.to.index as usize) < b.input_count(chips));
        if !valid {
            t.bad_wires += 1;
            drop.push(w.id);
        } else if !seen.insert((w.from, w.to)) {
            t.duplicate_wires += 1;
            drop.push(w.id);
        }
    }
    for id in drop {
        c.connections.remove(&id);
    }

    c.fix_counters();
    t
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Block, ChipDef, Connection, Port, Pos};

    fn wire(c: &mut Circuit, from: Port, to: Port) {
        let id = c.allocate_conn_id();
        c.insert_connection(Connection { id, from, to });
    }

    #[test]
    fn consistent_project_is_untouched() {
        let mut c = Circuit::new("ok");
        let a = c.allocate_block_id();
        c.insert_block(Block::new(a, BlockType::Switch, Pos::new(0, 0)));
        let l = c.allocate_block_id();
        c.insert_block(Block::new(l, BlockType::Led, Pos::new(4, 0)));
        wire(&mut c, Port::output(a, 0), Port::input(l, 0));
        let before = c.clone();
        let mut chips = ChipLibrary::default();
        assert!(repair_project(&mut c, &mut chips).is_empty());
        assert_eq!(c, before);
    }

    #[test]
    fn broken_wires_unknown_chips_and_stale_counters_are_fixed() {
        // Simulate a damaged file: JSON with dangling wires, a bad port, a duplicate, an unknown
        // chip, and no id counters.
        let json = r#"{
            "name": "damaged",
            "blocks": {
                "0": {"id": 0, "ty": "Switch", "pos": {"x": 0, "y": 0}},
                "1": {"id": 1, "ty": "Led", "pos": {"x": 6, "y": 0}},
                "2": {"id": 2, "ty": {"Chip": 7}, "pos": {"x": 3, "y": 3}},
                "3": {"id": 3, "ty": "And", "pos": {"x": 3, "y": 0}, "inputs": 99}
            },
            "connections": {
                "0": {"id": 0, "from": {"block": 0, "kind": "Output", "index": 0},
                               "to": {"block": 1, "kind": "Input", "index": 0}},
                "1": {"id": 1, "from": {"block": 0, "kind": "Output", "index": 0},
                               "to": {"block": 1, "kind": "Input", "index": 0}},
                "2": {"id": 2, "from": {"block": 9, "kind": "Output", "index": 0},
                               "to": {"block": 1, "kind": "Input", "index": 0}},
                "3": {"id": 3, "from": {"block": 0, "kind": "Output", "index": 0},
                               "to": {"block": 1, "kind": "Input", "index": 5}},
                "4": {"id": 4, "from": {"block": 0, "kind": "Output", "index": 0},
                               "to": {"block": 2, "kind": "Input", "index": 0}}
            }
        }"#;
        let mut c: Circuit = serde_json::from_str(json).unwrap();
        let mut chips = ChipLibrary::default();
        let notes = repair_project(&mut c, &mut chips);
        assert!(notes
            .iter()
            .any(|n| n.contains("chip that no longer exists")));
        assert!(
            notes.iter().any(|n| n.contains("3 broken wires")),
            "{notes:?}"
        );
        assert!(
            notes.iter().any(|n| n.contains("1 duplicate wire")),
            "{notes:?}"
        );
        assert!(notes.iter().any(|n| n.contains("invalid block setting")));
        assert!(c.block(BlockId(2)).is_none());
        assert_eq!(c.connections.len(), 1);
        assert_eq!(c.block(BlockId(3)).unwrap().inputs, Some(16));
        // The counters were 0 in the file: the next allocations must not collide.
        assert_eq!(c.allocate_block_id(), BlockId(4));
        assert_eq!(c.allocate_conn_id(), ConnId(1));
        // A second pass finds nothing left to fix.
        assert!(repair_project(&mut c, &mut chips).is_empty());
    }

    #[test]
    fn chip_pin_lists_are_rebuilt_from_the_body() {
        let mut body = Circuit::new("body");
        let a = body.allocate_block_id();
        body.insert_block(Block::new(a, BlockType::Switch, Pos::new(0, 0)));
        let y = body.allocate_block_id();
        body.insert_block(Block::new(y, BlockType::Led, Pos::new(4, 0)));
        let mut chips = ChipLibrary::default();
        let id = chips.allocate_id();
        let mut def = ChipDef::from_circuit(id, "Thru", body);
        def.input_blocks = vec![BlockId(42), a, a];
        def.output_blocks.clear();
        chips.insert(def);
        let mut top = Circuit::new("top");
        let notes = repair_project(&mut top, &mut chips);
        assert_eq!(notes.len(), 1, "{notes:?}");
        let d = chips.get(id).unwrap();
        assert_eq!(d.input_blocks, vec![a]);
        assert_eq!(d.output_blocks, vec![y]);
    }
}
