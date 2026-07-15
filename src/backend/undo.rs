//! Undo/redo built directly on [`EditCommand`] batches.

use crate::model::Circuit;

use super::commands::{apply_batch, EditCommand};

const DEFAULT_LIMIT: usize = 256;

/// A stack of reversible edits. Each stored batch, when applied, performs the
/// undo (or redo) and yields the batch for the opposite direction.
#[derive(Debug)]
pub struct UndoSystem {
    undo: Vec<Vec<EditCommand>>,
    redo: Vec<Vec<EditCommand>>,
    limit: usize,
}

impl Default for UndoSystem {
    fn default() -> Self {
        Self {
            undo: Vec::new(),
            redo: Vec::new(),
            limit: DEFAULT_LIMIT,
        }
    }
}

impl UndoSystem {
    /// Record the inverse of an edit that was just applied. Clears the redo stack.
    pub fn record(&mut self, inverse_batch: Vec<EditCommand>) {
        if inverse_batch.is_empty() {
            return;
        }
        self.undo.push(inverse_batch);
        if self.undo.len() > self.limit {
            self.undo.remove(0);
        }
        self.redo.clear();
    }

    pub fn undo(&mut self, circuit: &mut Circuit) -> bool {
        match self.undo.pop() {
            Some(batch) => {
                let redo = apply_batch(circuit, batch);
                self.redo.push(redo);
                true
            }
            None => false,
        }
    }

    pub fn redo(&mut self, circuit: &mut Circuit) -> bool {
        match self.redo.pop() {
            Some(batch) => {
                let undo = apply_batch(circuit, batch);
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
    }
}
