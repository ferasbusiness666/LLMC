//! Reusable sub-circuits ("chips" / ICs).
//!
//! A chip is authored as an ordinary [`Circuit`]. `Switch` blocks become its external
//! input pins and `Led` blocks become its external output pins (so inside a chip you
//! use `ConstantHigh`/`ConstantLow` for fixed levels, not switches). Pins are ordered
//! top-to-bottom then left-to-right for a stable, predictable layout.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::block::{Block, BlockType};
use super::circuit::Circuit;
use super::ids::{BlockId, ChipId};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChipDef {
    pub id: ChipId,
    pub name: String,
    pub circuit: Circuit,
    /// Marker blocks acting as external input pins, in pin order.
    pub input_blocks: Vec<BlockId>,
    /// Marker blocks acting as external output pins, in pin order.
    pub output_blocks: Vec<BlockId>,
}

impl ChipDef {
    pub fn from_circuit(id: ChipId, name: impl Into<String>, circuit: Circuit) -> Self {
        let key = |b: &&Block| (b.pos.y, b.pos.x);

        let mut inputs: Vec<&Block> = circuit
            .iter_blocks()
            .filter(|b| matches!(b.ty, BlockType::Switch))
            .collect();
        inputs.sort_by_key(key);

        let mut outputs: Vec<&Block> = circuit
            .iter_blocks()
            .filter(|b| matches!(b.ty, BlockType::Led))
            .collect();
        outputs.sort_by_key(key);

        let input_blocks = inputs.iter().map(|b| b.id).collect();
        let output_blocks = outputs.iter().map(|b| b.id).collect();

        Self {
            id,
            name: name.into(),
            circuit,
            input_blocks,
            output_blocks,
        }
    }
}

/// The set of chip definitions available for instancing.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
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

    pub fn remove(&mut self, id: ChipId) -> Option<ChipDef> {
        self.chips.remove(&id)
    }

    pub fn iter(&self) -> impl Iterator<Item = &ChipDef> {
        self.chips.values()
    }

    pub fn is_empty(&self) -> bool {
        self.chips.is_empty()
    }

    /// `(input_count, output_count)` for a chip, if it exists.
    pub fn io_counts(&self, id: ChipId) -> Option<(usize, usize)> {
        self.chips
            .get(&id)
            .map(|c| (c.input_blocks.len(), c.output_blocks.len()))
    }
}
