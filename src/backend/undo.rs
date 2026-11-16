//! Undo/redo built directly on [`EditCommand`] batches.

use crate::model::{ChipLibrary, Circuit};

use super::commands::{apply_batch, EditCommand};

const DEFAULT_LIMIT: usize = 500;

/// A stack of reversible edits. Each stored batch, when applied, performs the undo (or redo)
/// and yields the batch for the opposite direction.
#[derive(Debug)]
pub struct UndoSystem {
    undo: Vec<Vec<EditCommand>>,
    redo: Vec<Vec<EditCommand>>,
    limit: usize,
    /// Key of the edit series the top undo entry belongs to (see [`UndoSystem::record_coalesced`]).
    series: Option<u64>,
}

impl Default for UndoSystem {
    fn default() -> Self {
        Self {
            undo: Vec::new(),
            redo: Vec::new(),
            limit: DEFAULT_LIMIT,
            series: None,
        }
    }
}

impl UndoSystem {
    /// Record the inverse of an edit that was just applied. Clears the redo stack.
    pub fn record(&mut self, inverse_batch: Vec<EditCommand>) {
        self.series = None;
        self.push(inverse_batch);
    }

    fn push(&mut self, inverse_batch: Vec<EditCommand>) {
        if inverse_batch.is_empty() {
            return;
        }
        self.undo.push(inverse_batch);
        if self.undo.len() > self.limit {
            self.undo.remove(0);
        }
        self.redo.clear();
    }

    /// Record an edit that is part of a continuous series (dragging a slider, typing a name): the
    /// whole series collapses into ONE undo step that restores the state before it began.
    /// A different `key`, or any other edit/undo/redo in between, starts a new step.
    pub fn record_coalesced(&mut self, inverse_batch: Vec<EditCommand>, key: u64) {
        if inverse_batch.is_empty() {
            return;
        }
        if self.series == Some(key) && self.redo.is_empty() {
            if let Some(top) = self.undo.pop() {
                // Undo applies the newest inverse first, then the series' original inverse. Any
                // `UpdateBlock` in the newest inverse is redundant (the original restores that
                // block anyway), so drop those to keep long drags cheap.
                let restored: Vec<_> = top
                    .iter()
                    .filter_map(|c| match c {
                        EditCommand::UpdateBlock { block } => Some(block.id),
                        _ => None,
                    })
                    .collect();
                let mut merged: Vec<EditCommand> = inverse_batch
                    .into_iter()
                    .filter(|c| {
                        !matches!(c, EditCommand::UpdateBlock { block } if restored.contains(&block.id))
                    })
                    .collect();
                merged.extend(top);
                self.undo.push(merged);
                return;
            }
        }
        self.push(inverse_batch);
        self.series = Some(key);
    }

    /// End any coalescing series (the next edit becomes its own undo step).
    pub fn break_series(&mut self) {
        self.series = None;
    }

    pub fn undo(&mut self, circuit: &mut Circuit, chips: &mut ChipLibrary) -> bool {
        self.series = None;
        match self.undo.pop() {
            Some(batch) => {
                let redo = apply_batch(circuit, chips, batch);
                self.redo.push(redo);
                true
            }
            None => false,
        }
    }

    pub fn redo(&mut self, circuit: &mut Circuit, chips: &mut ChipLibrary) -> bool {
        self.series = None;
        match self.redo.pop() {
            Some(batch) => {
                let undo = apply_batch(circuit, chips, batch);
                self.undo.push(undo);
                true
            }
            None => false,
        }
    }

    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    pub fn clear(&mut self) {
        self.undo.clear();
        self.redo.clear();
        self.series = None;
    }
}
