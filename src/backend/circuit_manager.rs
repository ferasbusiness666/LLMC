//! Owns the working circuit, the chip library, and the undo history, and exposes the
//! high-level editing operations the UI and (later) the AI drive.

use std::collections::{BTreeMap, BTreeSet};

use crate::model::{
    Block, BlockId, BlockType, ChipDef, ChipId, ChipLibrary, Circuit, ConnId, Connection,
    Orientation, Port, Pos,
};

use super::commands::{apply_batch, EditCommand};
use super::undo::UndoSystem;

#[derive(Default)]
pub struct CircuitManager {
    pub circuit: Circuit,
    pub chips: ChipLibrary,
    pub undo: UndoSystem,
}

impl CircuitManager {
    pub fn new() -> Self {
        Self {
            circuit: Circuit::new("untitled"),
            chips: ChipLibrary::default(),
            undo: UndoSystem::default(),
        }
    }

    pub fn from_parts(circuit: Circuit, chips: ChipLibrary) -> Self {
        Self {
            circuit,
            chips,
            undo: UndoSystem::default(),
        }
    }

    /// Apply an edit batch as one undoable step.
    pub fn apply(&mut self, batch: Vec<EditCommand>) {
        let inverse = apply_batch(&mut self.circuit, batch);
        self.undo.record(inverse);
    }

    pub fn undo(&mut self) -> bool {
        self.undo.undo(&mut self.circuit)
    }

    pub fn redo(&mut self) -> bool {
        self.undo.redo(&mut self.circuit)
    }

    // --- convenience operations (each is a single undo step) ---

    pub fn add_block(&mut self, ty: BlockType, pos: Pos) -> BlockId {
        let id = self.circuit.allocate_block_id();
        self.apply(vec![EditCommand::AddBlock {
            block: Block::new(id, ty, pos),
        }]);
        id
    }

    /// Wire `from` (an output) to `to` (an input). An input may be fed by more than one wire;
    /// the simulator OR-combines the drivers, so wires are added rather than replaced.
    pub fn connect(&mut self, from: Port, to: Port) -> ConnId {
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

    pub fn set_state(&mut self, id: BlockId, state: bool) {
        self.apply(vec![EditCommand::SetState { id, state }]);
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
        self.insert_group(&blocks, &conns, offset)
    }

    /// Insert a group of blocks (and the wires among them) with fresh ids, offset by
    /// `offset`. Used by duplicate and paste. Returns the new block ids.
    pub fn insert_group(
        &mut self,
        blocks: &[Block],
        conns: &[Connection],
        offset: Pos,
    ) -> Vec<BlockId> {
        let mut id_map: BTreeMap<BlockId, BlockId> = BTreeMap::new();
        let mut new_ids = Vec::new();
        let mut batch = Vec::new();

        for b in blocks {
            let new_id = self.circuit.allocate_block_id();
            let mut clone = b.clone();
            clone.id = new_id;
            clone.pos = clone.pos + offset;
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

    /// Register the working circuit as a reusable chip (Switch pins → inputs, Led → outputs).
    pub fn create_chip_from_active(&mut self, name: impl Into<String>) -> ChipId {
        let id = self.chips.allocate_id();
        let def = ChipDef::from_circuit(id, name, self.circuit.clone());
        self.chips.insert(def);
        id
    }
}
