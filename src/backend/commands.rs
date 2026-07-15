//! The single mutation path for circuits.
//!
//! Every edit — placing, moving, deleting, wiring, rotating, pasting, and (later)
//! anything the AI does — is expressed as an [`EditCommand`]. Applying a command
//! returns the commands needed to undo it, which is what powers undo/redo and lets an
//! AI edit be committed as one atomic, reversible step.

use serde::{Deserialize, Serialize};

use crate::model::{Block, BlockId, Circuit, ConnId, Connection, Orientation, Pos};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EditCommand {
    /// Insert a block (id pre-allocated by the caller via `Circuit::allocate_block_id`).
    AddBlock {
        block: Block,
    },
    RemoveBlock {
        id: BlockId,
    },
    MoveBlock {
        id: BlockId,
        to: Pos,
    },
    SetOrientation {
        id: BlockId,
        orientation: Orientation,
    },
    /// Set the authored default level of a Switch/Button.
    SetState {
        id: BlockId,
        state: bool,
    },
    SetLabel {
        id: BlockId,
        label: Option<String>,
    },
    /// Insert a wire (id pre-allocated via `Circuit::allocate_conn_id`).
    AddConnection {
        conn: Connection,
    },
    RemoveConnection {
        id: ConnId,
    },
}

impl EditCommand {
    /// Apply this command to `circuit`, returning the commands that undo it. A missing
    /// target is treated as a no-op (empty inverse) so replaying stale batches is safe.
    pub fn apply(self, circuit: &mut Circuit) -> Vec<EditCommand> {
        match self {
            EditCommand::AddBlock { block } => {
                let id = block.id;
                circuit.insert_block(block);
                vec![EditCommand::RemoveBlock { id }]
            }
            EditCommand::RemoveBlock { id } => match circuit.remove_block(id) {
                Some((block, conns)) => {
                    let mut inv = Vec::with_capacity(conns.len() + 1);
                    inv.push(EditCommand::AddBlock { block });
                    inv.extend(
                        conns
                            .into_iter()
                            .map(|conn| EditCommand::AddConnection { conn }),
                    );
                    inv
                }
                None => vec![],
            },
            EditCommand::MoveBlock { id, to } => match circuit.block_mut(id) {
                Some(block) => {
                    let from = block.pos;
                    block.pos = to;
                    vec![EditCommand::MoveBlock { id, to: from }]
                }
                None => vec![],
            },
            EditCommand::SetOrientation { id, orientation } => match circuit.block_mut(id) {
                Some(block) => {
                    let prev = block.orientation;
                    block.orientation = orientation;
                    vec![EditCommand::SetOrientation {
                        id,
                        orientation: prev,
                    }]
                }
                None => vec![],
            },
            EditCommand::SetState { id, state } => match circuit.block_mut(id) {
                Some(block) => {
                    let prev = block.state;
                    block.state = state;
                    vec![EditCommand::SetState { id, state: prev }]
                }
                None => vec![],
            },
            EditCommand::SetLabel { id, label } => match circuit.block_mut(id) {
                Some(block) => {
                    let prev = block.label.take();
                    block.label = label;
                    vec![EditCommand::SetLabel { id, label: prev }]
                }
                None => vec![],
            },
            EditCommand::AddConnection { conn } => {
                let id = conn.id;
                circuit.insert_connection(conn);
                vec![EditCommand::RemoveConnection { id }]
            }
            EditCommand::RemoveConnection { id } => match circuit.remove_connection(id) {
                Some(conn) => vec![EditCommand::AddConnection { conn }],
                None => vec![],
            },
        }
    }
}

/// Apply a batch of commands in order, returning the batch that undoes the whole thing.
///
/// The undo batch is the per-command inverses concatenated in reverse command order, so
/// applying it left-to-right reverses the original effects exactly. Applying the returned
/// batch again yields the redo batch, so undo/redo is just repeated application.
pub fn apply_batch(circuit: &mut Circuit, batch: Vec<EditCommand>) -> Vec<EditCommand> {
    let mut inverses: Vec<Vec<EditCommand>> = batch.into_iter().map(|c| c.apply(circuit)).collect();
    inverses.reverse();
    inverses.concat()
}
