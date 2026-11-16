//! Owns the working circuit, the chip library, and the undo history, and exposes the
//! high-level editing operations the UI and the AI drive. Every operation here is one undo step.

use std::collections::{BTreeMap, BTreeSet};

use crate::model::{
    Block, BlockId, BlockType, ChipDef, ChipId, ChipLibrary, Circuit, ConnId, Connection,
    Orientation, PinMap, Port, PortKind, Pos,
};

use super::commands::{apply_batch, EditCommand};
use super::undo::UndoSystem;

#[derive(Default)]
pub struct CircuitManager {
    pub circuit: Circuit,
    pub chips: ChipLibrary,
    pub undo: UndoSystem,
    /// Bumped on every change (edit, undo, redo, or [`CircuitManager::touch`]), so views can
    /// cache derived data — and the dirty check can skip re-hashing — until something changes.
    revision: u64,
}

/// A group of edits that are applied immediately (so later edits can build on earlier ones) but
/// recorded as ONE undo step by [`CircuitManager::commit`]. Used for AI replies, which may define
/// chips and then place them in the same breath.
#[derive(Default)]
pub struct Txn {
    inverses: Vec<Vec<EditCommand>>,
}

impl Txn {
    pub fn is_empty(&self) -> bool {
        self.inverses.iter().all(|i| i.is_empty())
    }
}

/// What a chip redefinition did to existing instances.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct RedefineReport {
    /// Wires that were moved to a pin's new index.
    pub rewired: usize,
    /// Wires removed because the pin they used no longer exists.
    pub removed: usize,
}

impl CircuitManager {
    pub fn new() -> Self {
        Self {
            circuit: Circuit::new("untitled"),
            ..Default::default()
        }
    }

    pub fn from_parts(circuit: Circuit, chips: ChipLibrary) -> Self {
        Self {
            circuit,
            chips,
            ..Default::default()
        }
    }

    /// Monotonic change counter (see the field docs).
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// Note a change made outside the command path (e.g. toggling a switch's live level).
    pub fn touch(&mut self) {
        self.revision += 1;
    }

    /// Apply an edit batch as one undoable step.
    pub fn apply(&mut self, batch: Vec<EditCommand>) {
        if batch.is_empty() {
            return;
        }
        let inverse = apply_batch(&mut self.circuit, &mut self.chips, batch);
        self.undo.record(inverse);
        self.revision += 1;
    }

    /// Apply an edit batch that belongs to a continuous series (a slider drag, typing a name):
    /// consecutive calls with the same `key` collapse into a single undo step.
    pub fn apply_coalesced(&mut self, batch: Vec<EditCommand>, key: u64) {
        if batch.is_empty() {
            return;
        }
        let inverse = apply_batch(&mut self.circuit, &mut self.chips, batch);
        self.undo.record_coalesced(inverse, key);
        self.revision += 1;
    }

    /// Apply a batch as part of a transaction (see [`Txn`]).
    pub fn apply_in(&mut self, txn: &mut Txn, batch: Vec<EditCommand>) {
        if batch.is_empty() {
            return;
        }
        let inverse = apply_batch(&mut self.circuit, &mut self.chips, batch);
        txn.inverses.push(inverse);
        self.revision += 1;
    }

    /// Record a finished transaction as a single undo step.
    pub fn commit(&mut self, txn: Txn) {
        let mut inverses = txn.inverses;
        inverses.reverse();
        self.undo.record(inverses.concat());
    }

    pub fn undo(&mut self) -> bool {
        let done = self.undo.undo(&mut self.circuit, &mut self.chips);
        if done {
            self.revision += 1;
        }
        done
    }

    pub fn redo(&mut self) -> bool {
        let done = self.undo.redo(&mut self.circuit, &mut self.chips);
        if done {
            self.revision += 1;
        }
        done
    }

    // --- convenience operations (each is a single undo step) ---

    pub fn add_block(&mut self, ty: BlockType, pos: Pos) -> BlockId {
        let id = self.circuit.allocate_block_id();
        self.apply(vec![EditCommand::AddBlock {
            block: Block::new(id, ty, pos),
        }]);
        id
    }

    /// An existing wire with exactly this source and target, if any.
    pub fn find_connection(&self, from: Port, to: Port) -> Option<ConnId> {
        self.circuit
            .iter_connections()
            .find(|c| c.from == from && c.to == to)
            .map(|c| c.id)
    }

    /// Wire `from` (an output) to `to` (an input). An input may be fed by more than one wire;
    /// the simulator OR-combines the drivers, so wires are added rather than replaced. Drawing
    /// a wire that already exists is a no-op that returns the existing wire.
    pub fn connect(&mut self, from: Port, to: Port) -> ConnId {
        if let Some(existing) = self.find_connection(from, to) {
            return existing;
        }
        let id = self.circuit.allocate_conn_id();
        self.apply(vec![EditCommand::AddConnection {
            conn: Connection { id, from, to },
        }]);
        id
    }

    /// Delete blocks (and their wires) plus loose wires in one step.
    pub fn delete(&mut self, blocks: &[BlockId], conns: &[ConnId]) {
        let mut batch = Vec::new();
        for &c in conns {
            batch.push(EditCommand::RemoveConnection { id: c });
        }
        for &b in blocks {
            batch.push(EditCommand::RemoveBlock { id: b });
        }
        self.apply(batch);
    }

    /// Commit a drag: all moves in a single undo step (never per-frame).
    pub fn move_blocks(&mut self, moves: &[(BlockId, Pos)]) {
        let batch = moves
            .iter()
            .map(|&(id, to)| EditCommand::MoveBlock { id, to })
            .collect();
        self.apply(batch);
    }

    pub fn set_orientations(&mut self, items: &[(BlockId, Orientation)]) {
        let batch = items
            .iter()
            .map(|&(id, orientation)| EditCommand::SetOrientation { id, orientation })
            .collect();
        self.apply(batch);
    }

    /// Rotate (or flip) a group of blocks *as a group* around its center, so a selected
    /// sub-circuit turns as one piece instead of each block spinning in place.
    pub fn transform_group(
        &mut self,
        ids: &[BlockId],
        f: impl Fn(Orientation) -> Orientation,
        kind: GroupTransform,
    ) {
        let blocks: Vec<&Block> = ids
            .iter()
            .filter_map(|id| self.circuit.block(*id))
            .collect();
        if blocks.is_empty() {
            return;
        }
        // Group bounds in cells (using each block's oriented footprint).
        let mut min = (f32::MAX, f32::MAX);
        let mut max = (f32::MIN, f32::MIN);
        for b in &blocks {
            let sz = b.footprint(&self.chips);
            min.0 = min.0.min(b.pos.x as f32);
            min.1 = min.1.min(b.pos.y as f32);
            max.0 = max.0.max(b.pos.x as f32 + sz.x);
            max.1 = max.1.max(b.pos.y as f32 + sz.y);
        }
        let (cx, cy) = ((min.0 + max.0) * 0.5, (min.1 + max.1) * 0.5);
        let mut batch = Vec::new();
        for b in blocks {
            let mut nb = b.clone();
            nb.orientation = f(b.orientation);
            let old_sz = b.footprint(&self.chips);
            let new_sz = nb.footprint(&self.chips);
            // Block center relative to the group center, transformed.
            let (bx, by) = (
                b.pos.x as f32 + old_sz.x * 0.5 - cx,
                b.pos.y as f32 + old_sz.y * 0.5 - cy,
            );
            let (nx, ny) = match kind {
                GroupTransform::RotateCw => (-by, bx),
                GroupTransform::RotateCcw => (by, -bx),
                GroupTransform::Flip => (-bx, by),
            };
            nb.pos = Pos::new(
                (cx + nx - new_sz.x * 0.5).round() as i32,
                (cy + ny - new_sz.y * 0.5).round() as i32,
            );
            if nb != *b {
                batch.push(EditCommand::UpdateBlock { block: nb });
            }
        }
        self.apply(batch);
    }

    pub fn set_state(&mut self, id: BlockId, state: bool) {
        self.apply(vec![EditCommand::SetState { id, state }]);
    }

    /// Commit a property edit for one block (name, color, frequency, input count, …). Wires on
    /// input pins that no longer exist are removed in the same step. With `series`, consecutive
    /// edits of the same property collapse into one undo step.
    pub fn update_block(&mut self, block: Block, series: Option<u64>) {
        let batch = self.update_block_batch(block);
        match series {
            Some(key) => self.apply_coalesced(batch, key),
            None => self.apply(batch),
        }
    }

    /// The commands for a property edit: the block replacement plus removal of wires on input
    /// pins that no longer exist. Empty if nothing changes.
    pub fn update_block_batch(&self, block: Block) -> Vec<EditCommand> {
        let Some(old) = self.circuit.block(block.id) else {
            return Vec::new();
        };
        if *old == block {
            return Vec::new();
        }
        let n_inputs = block.input_count(&self.chips) as u16;
        let id = block.id;
        let mut batch = vec![EditCommand::UpdateBlock { block }];
        for c in self.circuit.iter_connections() {
            if c.to.block == id && c.to.kind == PortKind::Input && c.to.index >= n_inputs {
                batch.push(EditCommand::RemoveConnection { id: c.id });
            }
        }
        batch
    }

    /// Duplicate blocks and any wires wholly inside the set, offset by `offset`.
    /// Returns the ids of the new blocks (the new selection).
    pub fn duplicate(&mut self, ids: &[BlockId], offset: Pos) -> Vec<BlockId> {
        let set: BTreeSet<BlockId> = ids.iter().copied().collect();
        let blocks: Vec<Block> = ids
            .iter()
            .filter_map(|id| self.circuit.block(*id).cloned())
            .collect();
        let conns: Vec<Connection> = self
            .circuit
            .iter_connections()
            .filter(|c| set.contains(&c.from.block) && set.contains(&c.to.block))
            .cloned()
            .collect();
        self.insert_group(&blocks, &conns, offset, &[])
    }

    /// Insert a group of blocks (and the wires among them) with fresh ids, offset by `offset`.
    /// Used by duplicate and paste. `chip_defs` carries the chip definitions the blocks refer to
    /// (from the clipboard's source project): each is matched into this library by name, or
    /// copied in, and the pasted instances are re-pointed accordingly — all in the same undo
    /// step. Returns the new block ids.
    pub fn insert_group(
        &mut self,
        blocks: &[Block],
        conns: &[Connection],
        offset: Pos,
        chip_defs: &[ChipDef],
    ) -> Vec<BlockId> {
        let mut batch = Vec::new();
        let chip_map = self.import_chips(chip_defs, &mut batch);

        let mut id_map: BTreeMap<BlockId, BlockId> = BTreeMap::new();
        let mut new_ids = Vec::new();
        for b in blocks {
            let new_id = self.circuit.allocate_block_id();
            let mut clone = b.clone();
            clone.id = new_id;
            clone.pos = clone.pos + offset;
            if let BlockType::Chip(cid) = clone.ty {
                if let Some(&mapped) = chip_map.get(&cid) {
                    clone.ty = BlockType::Chip(mapped);
                }
            }
            id_map.insert(b.id, new_id);
            new_ids.push(new_id);
            batch.push(EditCommand::AddBlock { block: clone });
        }
        for c in conns {
            let (Some(&nf), Some(&nt)) = (id_map.get(&c.from.block), id_map.get(&c.to.block))
            else {
                continue;
            };
            let id = self.circuit.allocate_conn_id();
            batch.push(EditCommand::AddConnection {
                conn: Connection {
                    id,
                    from: Port {
                        block: nf,
                        ..c.from
                    },
                    to: Port { block: nt, ..c.to },
                },
            });
        }

        self.apply(batch);
        new_ids
    }

    /// Plan bringing foreign chip definitions (dependencies first) into this library. A chip
    /// with the same id *and* name (same project) or the same name is reused; anything else is
    /// copied in under a fresh id, with its inner chip references re-pointed. Pushes the needed
    /// `PutChip` commands onto `batch` and returns the source-id → local-id map.
    fn import_chips(
        &mut self,
        defs: &[ChipDef],
        batch: &mut Vec<EditCommand>,
    ) -> BTreeMap<ChipId, ChipId> {
        let mut map: BTreeMap<ChipId, ChipId> = BTreeMap::new();
        let mut planned: Vec<ChipDef> = Vec::new();
        for def in defs {
            let same = self
                .chips
                .get(def.id)
                .is_some_and(|d| d.name.trim().eq_ignore_ascii_case(def.name.trim()));
            if same {
                map.insert(def.id, def.id);
                continue;
            }
            let by_name = self.chips.find_by_name(&def.name).or_else(|| {
                planned
                    .iter()
                    .find(|p| p.name.trim().eq_ignore_ascii_case(def.name.trim()))
                    .map(|p| p.id)
            });
            if let Some(local) = by_name {
                map.insert(def.id, local);
                continue;
            }
            let new_id = self.chips.allocate_id();
            let mut copy = def.clone();
            copy.id = new_id;
            for b in copy.circuit.blocks.values_mut() {
                if let BlockType::Chip(inner) = b.ty {
                    if let Some(&m) = map.get(&inner) {
                        b.ty = BlockType::Chip(m);
                    }
                }
            }
            map.insert(def.id, new_id);
            planned.push(copy.clone());
            batch.push(EditCommand::PutChip { def: copy });
        }
        map
    }

    /// Register the working circuit as a reusable chip (Switch pins → inputs, Led → outputs).
    pub fn create_chip_from_active(&mut self, name: impl Into<String>) -> ChipId {
        let id = self.chips.allocate_id();
        let def = ChipDef::from_circuit(id, name, self.circuit.clone());
        self.apply(vec![EditCommand::PutChip { def }]);
        id
    }

    /// Rename a chip (one undo step). Returns false if it doesn't exist.
    pub fn rename_chip(&mut self, id: ChipId, name: &str) -> bool {
        let Some(mut def) = self.chips.get(id).cloned() else {
            return false;
        };
        def.name = name.trim().to_string();
        self.apply(vec![EditCommand::PutChip { def }]);
        true
    }

    /// Delete a chip and every placed instance of it in the working circuit (one undo step).
    pub fn delete_chip(&mut self, id: ChipId) {
        let mut batch: Vec<EditCommand> = self
            .circuit
            .iter_blocks()
            .filter(|b| b.ty == BlockType::Chip(id))
            .map(|b| EditCommand::RemoveBlock { id: b.id })
            .collect();
        batch.push(EditCommand::RemoveChip { id });
        self.apply(batch);
    }

    /// Define a new chip or redefine an existing one (one undo step). When redefining, wires on
    /// placed instances — in the working circuit and inside other chips — follow their pins to
    /// the new indices, and wires to pins that no longer exist are removed, so nothing ends up
    /// silently connected to the wrong pin.
    pub fn put_chip(&mut self, def: ChipDef) -> RedefineReport {
        let (batch, report) = self.put_chip_batch(def);
        self.apply(batch);
        report
    }

    /// [`CircuitManager::put_chip`] inside a transaction.
    pub fn put_chip_in(&mut self, txn: &mut Txn, def: ChipDef) -> RedefineReport {
        let (batch, report) = self.put_chip_batch(def);
        self.apply_in(txn, batch);
        report
    }

    fn put_chip_batch(&mut self, def: ChipDef) -> (Vec<EditCommand>, RedefineReport) {
        let mut report = RedefineReport::default();
        let map = self.chips.get(def.id).map(|old| PinMap::between(old, &def));
        let mut batch = vec![EditCommand::PutChip { def: def.clone() }];
        if let Some(map) = map.filter(|m| !m.is_identity()) {
            let id = def.id;
            // Working circuit.
            let (cmds, r) = remap_instances(&mut self.circuit, id, &map);
            batch.extend(cmds);
            report.rewired += r.rewired;
            report.removed += r.removed;
            // Other chips that instance it.
            for user in self.chips.users_of(id) {
                if let Some(udef) = self.chips.get(user) {
                    let mut ucirc = udef.circuit.clone();
                    let (cmds, r) = remap_instances(&mut ucirc, id, &map);
                    if !cmds.is_empty() {
                        let mut tmp = ChipLibrary::default();
                        apply_batch(&mut ucirc, &mut tmp, cmds);
                        let mut nd = udef.clone();
                        nd.circuit = ucirc;
                        batch.push(EditCommand::PutChip { def: nd });
                        report.rewired += r.rewired;
                        report.removed += r.removed;
                    }
                }
            }
        }
        (batch, report)
    }
}

/// How [`CircuitManager::transform_group`] moves blocks around the group center.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GroupTransform {
    RotateCw,
    RotateCcw,
    Flip,
}

/// Commands that move wires on instances of `chip` in `circuit` to their pins' new indices
/// (allocating ids from `circuit`, which is why it's `&mut`).
fn remap_instances(
    circuit: &mut Circuit,
    chip: ChipId,
    map: &PinMap,
) -> (Vec<EditCommand>, RedefineReport) {
    let instances: BTreeSet<BlockId> = circuit
        .iter_blocks()
        .filter(|b| b.ty == BlockType::Chip(chip))
        .map(|b| b.id)
        .collect();
    let mut report = RedefineReport::default();
    let mut cmds = Vec::new();
    let conns: Vec<Connection> = circuit
        .iter_connections()
        .filter(|c| instances.contains(&c.from.block) || instances.contains(&c.to.block))
        .copied()
        .collect();
    for c in conns {
        let mut nc = c;
        let mut dropped = false;
        if instances.contains(&c.to.block) {
            match map.inputs.get(c.to.index as usize).copied().flatten() {
                Some(j) => nc.to.index = j as u16,
                None => dropped = true,
            }
        }
        if instances.contains(&c.from.block) {
            match map.outputs.get(c.from.index as usize).copied().flatten() {
                Some(j) => nc.from.index = j as u16,
                None => dropped = true,
            }
        }
        if dropped {
            cmds.push(EditCommand::RemoveConnection { id: c.id });
            report.removed += 1;
        } else if nc != c {
            cmds.push(EditCommand::RemoveConnection { id: c.id });
            nc.id = circuit.allocate_conn_id();
            cmds.push(EditCommand::AddConnection { conn: nc });
            report.rewired += 1;
        }
    }
    (cmds, report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Pos;

    #[test]
    fn property_edits_are_one_undo_step_per_series() {
        let mut m = CircuitManager::new();
        let g = m.add_block(BlockType::And, Pos::new(0, 0));
        let original = m.circuit.block(g).unwrap().clone();
        for n in 3..=6u16 {
            let mut b = m.circuit.block(g).unwrap().clone();
            b.inputs = Some(n);
            m.update_block(b, Some(42));
        }
        assert_eq!(m.circuit.block(g).unwrap().inputs, Some(6));
        assert!(m.undo());
        assert_eq!(
            *m.circuit.block(g).unwrap(),
            original,
            "whole drag undone at once"
        );
        assert!(m.undo(), "the add is the previous step");
        assert!(m.circuit.block(g).is_none());
    }

    #[test]
    fn shrinking_inputs_removes_orphan_wires_and_undo_restores_them() {
        let mut m = CircuitManager::new();
        let s = m.add_block(BlockType::Switch, Pos::new(0, 0));
        let g = m.add_block(BlockType::And, Pos::new(4, 0));
        let mut b = m.circuit.block(g).unwrap().clone();
        b.inputs = Some(4);
        m.update_block(b, None);
        m.connect(Port::output(s, 0), Port::input(g, 3));
        let mut b = m.circuit.block(g).unwrap().clone();
        b.inputs = Some(2);
        m.update_block(b, None);
        assert_eq!(m.circuit.connections.len(), 0);
        m.undo();
        assert_eq!(m.circuit.connections.len(), 1);
        assert_eq!(m.circuit.block(g).unwrap().inputs, Some(4));
    }

    #[test]
    fn chip_delete_and_rename_are_undoable() {
        let mut m = CircuitManager::new();
        let id = m.create_chip_from_active("Thing");
        let inst = m.add_block(BlockType::Chip(id), Pos::new(0, 0));
        assert!(m.rename_chip(id, "Gizmo"));
        m.delete_chip(id);
        assert!(m.chips.get(id).is_none());
        assert!(m.circuit.block(inst).is_none());
        m.undo();
        assert_eq!(m.chips.get(id).unwrap().name, "Gizmo");
        assert!(m.circuit.block(inst).is_some());
        m.undo();
        assert_eq!(m.chips.get(id).unwrap().name, "Thing");
    }

    #[test]
    fn connect_ignores_an_identical_existing_wire() {
        let mut m = CircuitManager::new();
        let s = m.add_block(BlockType::Switch, Pos::new(0, 0));
        let l = m.add_block(BlockType::Led, Pos::new(4, 0));
        let a = m.connect(Port::output(s, 0), Port::input(l, 0));
        let b = m.connect(Port::output(s, 0), Port::input(l, 0));
        assert_eq!(a, b);
        assert_eq!(m.circuit.connections.len(), 1);
    }

    fn two_pin_chip(m: &mut CircuitManager) -> (ChipId, Vec<BlockId>) {
        let mut c = Circuit::new("c");
        let mut ids = Vec::new();
        for (i, l) in ["a", "b"].iter().enumerate() {
            let id = c.allocate_block_id();
            let mut b = Block::new(id, BlockType::Switch, Pos::new(0, i as i32 * 4));
            b.label = Some(l.to_string());
            c.insert_block(b);
            ids.push(id);
        }
        let y = c.allocate_block_id();
        c.insert_block(Block::new(y, BlockType::Led, Pos::new(8, 0)));
        let id = m.chips.allocate_id();
        let def = ChipDef::from_circuit(id, "Two", c);
        m.put_chip(def);
        (id, ids)
    }

    #[test]
    fn redefining_a_chip_rewires_instances_to_moved_pins() {
        let mut m = CircuitManager::new();
        let (cid, pins) = two_pin_chip(&mut m);
        let s = m.add_block(BlockType::Switch, Pos::new(-6, 0));
        let inst = m.add_block(BlockType::Chip(cid), Pos::new(0, 0));
        m.connect(Port::output(s, 0), Port::input(inst, 1)); // wired to pin "b"
                                                             // Redefine with pins reversed (b first) and "a" removed.
        let old = m.chips.get(cid).unwrap().clone();
        let mut new = old.clone();
        new.circuit.remove_block(pins[0]);
        new.input_blocks = vec![pins[1]];
        let r = m.put_chip(new);
        assert_eq!(r.rewired, 1);
        let c = m.circuit.iter_connections().next().unwrap();
        assert_eq!(c.to, Port::input(inst, 0), "wire followed pin b to index 0");
        m.undo();
        let c = m.circuit.iter_connections().next().unwrap();
        assert_eq!(c.to, Port::input(inst, 1));
    }

    #[test]
    fn pasting_chip_instances_into_another_project_brings_the_chip_along() {
        let mut src = CircuitManager::new();
        let (cid, _) = two_pin_chip(&mut src);
        let inst = src.add_block(BlockType::Chip(cid), Pos::new(0, 0));
        let blocks = vec![src.circuit.block(inst).unwrap().clone()];
        let defs: Vec<ChipDef> = src
            .chips
            .closure(&[cid])
            .into_iter()
            .filter_map(|id| src.chips.get(id).cloned())
            .collect();

        // A target project that already has an unrelated chip at the same id.
        let mut dst = CircuitManager::new();
        dst.create_chip_from_active("Other");
        let new = dst.insert_group(&blocks, &[], Pos::new(0, 0), &defs);
        let b = dst.circuit.block(new[0]).unwrap();
        let BlockType::Chip(local) = b.ty else {
            panic!("not a chip")
        };
        assert_eq!(dst.chips.get(local).unwrap().name, "Two");
        // Pasting again reuses the copy instead of duplicating the chip.
        dst.insert_group(&blocks, &[], Pos::new(5, 0), &defs);
        assert_eq!(dst.chips.len(), 2);
        // One undo removes the second paste; another removes the first paste AND the chip copy.
        dst.undo();
        dst.undo();
        assert_eq!(dst.chips.len(), 1);
    }

    #[test]
    fn group_rotation_turns_the_selection_as_one_piece() {
        let mut m = CircuitManager::new();
        let a = m.add_block(BlockType::Not, Pos::new(0, 0));
        let b = m.add_block(BlockType::Not, Pos::new(4, 0));
        m.transform_group(&[a, b], |o| o.cw(), GroupTransform::RotateCw);
        let pa = m.circuit.block(a).unwrap().pos;
        let pb = m.circuit.block(b).unwrap().pos;
        assert_eq!(pa.x, pb.x, "a horizontal pair becomes a vertical pair");
        assert!(pb.y > pa.y);
    }
}
