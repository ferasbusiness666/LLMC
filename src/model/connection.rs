//! Ports and connections.
//!
//! The model is *directed*: a wire runs from an output port to an input port. An output
//! may fan out to many inputs, and an input may be fed by more than one wire (the
//! simulator OR-combines those drivers). There is no bidirectional wire node — branching
//! is achieved by drawing multiple wires.

use serde::{Deserialize, Serialize};

use super::ids::{BlockId, ConnId};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum PortKind {
    Input,
    Output,
}

/// Identifies a specific port on a specific block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Port {
    pub block: BlockId,
    pub kind: PortKind,
    pub index: u16,
}

impl Port {
    pub fn input(block: BlockId, index: u16) -> Self {
        Self {
            block,
            kind: PortKind::Input,
            index,
        }
    }

    pub fn output(block: BlockId, index: u16) -> Self {
        Self {
            block,
            kind: PortKind::Output,
            index,
        }
    }
}

/// A wire from an output port (`from`) to an input port (`to`).
///
/// Invariant maintained by the editing layer: `from.kind == Output` and
/// `to.kind == Input`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Connection {
    pub id: ConnId,
    pub from: Port,
    pub to: Port,
}
