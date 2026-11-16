//! Reusable sub-circuits ("chips" / ICs).
//!
//! A chip is authored as an ordinary [`Circuit`]. `Switch` blocks become its external
//! input pins and `Led` blocks become its external output pins (so inside a chip you
//! use `ConstantHigh`/`ConstantLow` for fixed levels, not switches). A freshly created chip
//! orders its pins top-to-bottom then left-to-right; a *re*-defined chip keeps the order of the
//! pins it already had (see [`ChipDef::rederive`]) so placed instances stay wired correctly.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use super::block::{Block, BlockType};
use super::circuit::Circuit;
use super::ids::{BlockId, ChipId};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChipDef {
    pub id: ChipId,
    pub name: String,
    pub circuit: Circuit,
    /// Marker blocks acting as external input pins, in pin order.
    pub input_blocks: Vec<BlockId>,
    /// Marker blocks acting as external output pins, in pin order.
    pub output_blocks: Vec<BlockId>,
}

/// Position-based pin order: top-to-bottom, then left-to-right.
fn pins_by_position(circuit: &Circuit, ty: BlockType) -> Vec<BlockId> {
    let mut pins: Vec<&Block> = circuit.iter_blocks().filter(|b| b.ty == ty).collect();
    pins.sort_by_key(|b| (b.pos.y, b.pos.x));
    pins.iter().map(|b| b.id).collect()
}

impl ChipDef {
    pub fn from_circuit(id: ChipId, name: impl Into<String>, circuit: Circuit) -> Self {
        let input_blocks = pins_by_position(&circuit, BlockType::Switch);
        let output_blocks = pins_by_position(&circuit, BlockType::Led);
        Self {
            id,
            name: name.into(),
            circuit,
            input_blocks,
            output_blocks,
        }
    }

    /// Re-derive this chip from an edited circuit, keeping the existing pin order: pins that
    /// still exist keep their index (even if they were moved), new pins are appended in position
    /// order, and deleted pins drop out. This is what keeps placed instances wired to the right
    /// pins after "Update chip".
    pub fn rederive(&self, circuit: Circuit) -> ChipDef {
        let keep = |old: &[BlockId], ty: BlockType| -> Vec<BlockId> {
            let mut out: Vec<BlockId> = old
                .iter()
                .copied()
                .filter(|id| circuit.block(*id).is_some_and(|b| b.ty == ty))
                .collect();
            for id in pins_by_position(&circuit, ty) {
                if !out.contains(&id) {
                    out.push(id);
                }
            }
            out
        };
        let input_blocks = keep(&self.input_blocks, BlockType::Switch);
        let output_blocks = keep(&self.output_blocks, BlockType::Led);
        ChipDef {
            id: self.id,
            name: self.name.clone(),
            circuit,
            input_blocks,
            output_blocks,
        }
    }

    /// The label of input pin `i` (empty if unlabeled).
    pub fn input_name(&self, i: usize) -> &str {
        self.pin_name(&self.input_blocks, i)
    }

    /// The label of output pin `i` (empty if unlabeled).
    pub fn output_name(&self, i: usize) -> &str {
        self.pin_name(&self.output_blocks, i)
    }

    fn pin_name(&self, pins: &[BlockId], i: usize) -> &str {
        pins.get(i)
            .and_then(|id| self.circuit.block(*id))
            .and_then(|b| b.label.as_deref())
            .unwrap_or("")
    }

    /// Chips instanced directly inside this chip's circuit.
    pub fn direct_children(&self) -> BTreeSet<ChipId> {
        self.circuit
            .iter_blocks()
            .filter_map(|b| match b.ty {
                BlockType::Chip(c) => Some(c),
                _ => None,
            })
            .collect()
    }
}

/// How the pins of a chip line up across a redefinition: for each OLD pin index, its NEW index
/// (or `None` when the pin was removed).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PinMap {
    pub inputs: Vec<Option<usize>>,
    pub outputs: Vec<Option<usize>>,
}

impl PinMap {
    /// Match old pins to new ones by marker block id first (an edited chip keeps its ids), then
    /// by a unique, non-empty label (an AI redefinition rebuilds the circuit with new ids but the
    /// same pin names).
    pub fn between(old: &ChipDef, new: &ChipDef) -> PinMap {
        fn side(
            old: &ChipDef,
            new: &ChipDef,
            old_pins: &[BlockId],
            new_pins: &[BlockId],
        ) -> Vec<Option<usize>> {
            let label = |c: &ChipDef, id: BlockId| -> Option<String> {
                c.circuit
                    .block(id)
                    .and_then(|b| b.label.as_deref())
                    .map(|l| l.trim().to_lowercase())
                    .filter(|l| !l.is_empty())
            };
            let unique = |c: &ChipDef, pins: &[BlockId], l: &str| {
                pins.iter()
                    .filter(|id| label(c, **id).as_deref() == Some(l))
                    .count()
                    == 1
            };
            old_pins
                .iter()
                .map(|&oid| {
                    if let Some(j) = new_pins.iter().position(|&nid| nid == oid) {
                        // Same marker block — only trust it if it's still the same pin kind.
                        if new.circuit.block(oid).map(|b| b.ty)
                            == old.circuit.block(oid).map(|b| b.ty)
                        {
                            return Some(j);
                        }
                    }
                    let l = label(old, oid)?;
                    if !unique(old, old_pins, &l) || !unique(new, new_pins, &l) {
                        return None;
                    }
                    new_pins
                        .iter()
                        .position(|&nid| label(new, nid).as_deref() == Some(l.as_str()))
                })
                .collect()
        }
        PinMap {
            inputs: side(old, new, &old.input_blocks, &new.input_blocks),
            outputs: side(old, new, &old.output_blocks, &new.output_blocks),
        }
    }

    /// True when every old pin keeps its index (nothing needs rewiring).
    pub fn is_identity(&self) -> bool {
        self.inputs.iter().enumerate().all(|(i, m)| *m == Some(i))
            && self.outputs.iter().enumerate().all(|(i, m)| *m == Some(i))
    }
}

/// The set of chip definitions available for instancing.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ChipLibrary {
    pub chips: BTreeMap<ChipId, ChipDef>,
    #[serde(default)]
    next_id: u32,
}

impl ChipLibrary {
    pub fn allocate_id(&mut self) -> ChipId {
        let id = ChipId(self.next_id);
        self.next_id += 1;
        id
    }

    pub fn insert(&mut self, def: ChipDef) {
        self.next_id = self.next_id.max(def.id.0 + 1);
        self.chips.insert(def.id, def);
    }

    pub fn get(&self, id: ChipId) -> Option<&ChipDef> {
        self.chips.get(&id)
    }

    pub fn get_mut(&mut self, id: ChipId) -> Option<&mut ChipDef> {
        self.chips.get_mut(&id)
    }

    pub fn remove(&mut self, id: ChipId) -> Option<ChipDef> {
        self.chips.remove(&id)
    }

    pub fn iter(&self) -> impl Iterator<Item = &ChipDef> {
        self.chips.values()
    }

    pub fn is_empty(&self) -> bool {
        self.chips.is_empty()
    }

    pub fn len(&self) -> usize {
        self.chips.len()
    }

    /// `(input_count, output_count)` for a chip, if it exists.
    pub fn io_counts(&self, id: ChipId) -> Option<(usize, usize)> {
        self.chips
            .get(&id)
            .map(|c| (c.input_blocks.len(), c.output_blocks.len()))
    }

    /// Find a chip by name (case-insensitive, trimmed).
    pub fn find_by_name(&self, name: &str) -> Option<ChipId> {
        let want = name.trim().to_lowercase();
        self.iter()
            .find(|d| d.name.trim().to_lowercase() == want)
            .map(|d| d.id)
    }

    /// Does chip `outer` contain `target` anywhere inside it (at any depth)? A chip does not
    /// count as containing itself unless it literally instances itself.
    pub fn contains(&self, outer: ChipId, target: ChipId) -> bool {
        let mut stack: Vec<ChipId> = match self.get(outer) {
            Some(d) => d.direct_children().into_iter().collect(),
            None => return false,
        };
        let mut seen = BTreeSet::new();
        while let Some(c) = stack.pop() {
            if c == target {
                return true;
            }
            if seen.insert(c) {
                if let Some(d) = self.get(c) {
                    stack.extend(d.direct_children());
                }
            }
        }
        false
    }

    /// Would using `circuit` as the body of chip `chip` make it contain itself (directly or via
    /// other chips)? Such a definition can never be simulated, so callers reject it.
    pub fn would_cycle(&self, chip: ChipId, circuit: &Circuit) -> bool {
        circuit.iter_blocks().any(|b| match b.ty {
            BlockType::Chip(c) => c == chip || self.contains(c, chip),
            _ => false,
        })
    }

    /// Chips (other than `id`) whose bodies instance `id` directly.
    pub fn users_of(&self, id: ChipId) -> Vec<ChipId> {
        self.iter()
            .filter(|d| d.id != id && d.direct_children().contains(&id))
            .map(|d| d.id)
            .collect()
    }

    /// `id` plus every chip it depends on, dependencies first (so they can be copied into
    /// another library in a valid order).
    pub fn closure(&self, roots: &[ChipId]) -> Vec<ChipId> {
        fn visit(
            lib: &ChipLibrary,
            id: ChipId,
            seen: &mut BTreeSet<ChipId>,
            out: &mut Vec<ChipId>,
        ) {
            if !seen.insert(id) {
                return;
            }
            if let Some(d) = lib.get(id) {
                for c in d.direct_children() {
                    visit(lib, c, seen, out);
                }
                out.push(id);
            }
        }
        let mut seen = BTreeSet::new();
        let mut out = Vec::new();
        for &r in roots {
            visit(self, r, &mut seen, &mut out);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{BlockType, Pos};

    fn pin(c: &mut Circuit, ty: BlockType, x: i32, y: i32, label: &str) -> BlockId {
        let id = c.allocate_block_id();
        let mut b = Block::new(id, ty, Pos::new(x, y));
        b.label = Some(label.to_string());
        c.insert_block(b);
        id
    }

    #[test]
    fn rederive_keeps_pin_order_when_pins_move() {
        let mut c = Circuit::new("c");
        let a = pin(&mut c, BlockType::Switch, 0, 0, "a");
        let b = pin(&mut c, BlockType::Switch, 0, 4, "b");
        pin(&mut c, BlockType::Led, 8, 0, "y");
        let def = ChipDef::from_circuit(ChipId(0), "C", c.clone());
        assert_eq!(def.input_blocks, vec![a, b]);
        // Move `a` below `b`: position order would swap them, but the chip keeps a=0, b=1.
        c.block_mut(a).unwrap().pos = Pos::new(0, 9);
        let new = def.rederive(c);
        assert_eq!(new.input_blocks, vec![a, b]);
        assert!(PinMap::between(&def, &new).is_identity());
    }

    #[test]
    fn pin_map_matches_by_label_and_drops_removed_pins() {
        let mut c1 = Circuit::new("c");
        pin(&mut c1, BlockType::Switch, 0, 0, "a");
        pin(&mut c1, BlockType::Switch, 0, 4, "b");
        pin(&mut c1, BlockType::Led, 8, 0, "y");
        let old = ChipDef::from_circuit(ChipId(0), "C", c1);
        // New definition (fresh ids): b first, a removed.
        let mut c2 = Circuit::new("c");
        pin(&mut c2, BlockType::Switch, 0, 0, "b");
        pin(&mut c2, BlockType::Led, 8, 0, "y");
        // Make ids differ from the old ones.
        let mut c2b = Circuit::new("c");
        c2b.allocate_block_id();
        c2b.allocate_block_id();
        c2b.allocate_block_id();
        for blk in c2.iter_blocks() {
            let id = c2b.allocate_block_id();
            let mut nb = blk.clone();
            nb.id = id;
            c2b.insert_block(nb);
        }
        let new = ChipDef::from_circuit(ChipId(0), "C", c2b);
        let map = PinMap::between(&old, &new);
        assert_eq!(map.inputs, vec![None, Some(0)]);
        assert_eq!(map.outputs, vec![Some(0)]);
    }

    #[test]
    fn cycle_detection_sees_indirect_self_reference() {
        let mut lib = ChipLibrary::default();
        let a = lib.allocate_id();
        let b = lib.allocate_id();
        let mut ca = Circuit::new("A");
        let id = ca.allocate_block_id();
        ca.insert_block(Block::new(id, BlockType::Chip(b), Pos::new(0, 0)));
        lib.insert(ChipDef::from_circuit(a, "A", ca));
        lib.insert(ChipDef::from_circuit(b, "B", Circuit::new("B")));
        // Giving B a body that contains A would make B ⊃ A ⊃ B.
        let mut cb = Circuit::new("B");
        let id = cb.allocate_block_id();
        cb.insert_block(Block::new(id, BlockType::Chip(a), Pos::new(0, 0)));
        assert!(lib.would_cycle(b, &cb));
        assert!(!lib.would_cycle(a, &Circuit::new("x")));
        assert_eq!(lib.closure(&[a]), vec![b, a]);
        assert_eq!(lib.users_of(b), vec![a]);
    }
}
