//! A single circuit: the container of blocks and connections (the "BlockContainer").
//!
//! `BTreeMap`s are used deliberately so iteration order is deterministic — this keeps
//! simulation, serialization, and (later) AI-facing diffs stable across runs.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::block::Block;
use super::connection::{Connection, Port};
use super::ids::{BlockId, ConnId};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Circuit {
    pub name: String,
    pub blocks: BTreeMap<BlockId, Block>,
    pub connections: BTreeMap<ConnId, Connection>,
    #[serde(default)]
    next_block: u32,
    #[serde(default)]
    next_conn: u32,
}

impl Circuit {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Default::default()
        }
    }

    pub fn allocate_block_id(&mut self) -> BlockId {
        let id = BlockId(self.next_block);
        self.next_block += 1;
        id
    }

    pub fn allocate_conn_id(&mut self) -> ConnId {
        let id = ConnId(self.next_conn);
        self.next_conn += 1;
        id
    }

    /// Insert a block verbatim (keeping its id). Bumps the id counter so future
    /// allocations never collide, which is what makes undo/redo re-insertion safe.
    pub fn insert_block(&mut self, block: Block) {
        self.next_block = self.next_block.max(block.id.0 + 1);
        self.blocks.insert(block.id, block);
    }

    pub fn insert_connection(&mut self, conn: Connection) {
        self.next_conn = self.next_conn.max(conn.id.0 + 1);
        self.connections.insert(conn.id, conn);
    }

    /// Remove a block and every connection touching it. Returns the removed block and
    /// its connections so the operation can be undone.
    pub fn remove_block(&mut self, id: BlockId) -> Option<(Block, Vec<Connection>)> {
        let block = self.blocks.remove(&id)?;
        let touching: Vec<Connection> = self
            .connections
            .values()
            .filter(|c| c.from.block == id || c.to.block == id)
            .cloned()
            .collect();
        for c in &touching {
            self.connections.remove(&c.id);
        }
        Some((block, touching))
    }

    pub fn remove_connection(&mut self, id: ConnId) -> Option<Connection> {
        self.connections.remove(&id)
    }

    pub fn block(&self, id: BlockId) -> Option<&Block> {
        self.blocks.get(&id)
    }

    pub fn block_mut(&mut self, id: BlockId) -> Option<&mut Block> {
        self.blocks.get_mut(&id)
    }

    pub fn connection(&self, id: ConnId) -> Option<&Connection> {
        self.connections.get(&id)
    }

    pub fn iter_blocks(&self) -> impl Iterator<Item = &Block> {
        self.blocks.values()
    }

    pub fn iter_connections(&self) -> impl Iterator<Item = &Connection> {
        self.connections.values()
    }

    /// All connections feeding an input port. Normally zero or one; more than one is a
    /// (flagged) multi-driver conflict.
    pub fn connections_into(&self, port: Port) -> impl Iterator<Item = &Connection> {
        self.connections.values().filter(move |c| c.to == port)
    }

    pub fn has_connection_into(&self, port: Port) -> bool {
        self.connections.values().any(|c| c.to == port)
    }

    /// The id of an existing connection into `port`, if any (used to replace on rewire).
    pub fn connection_id_into(&self, port: Port) -> Option<ConnId> {
        self.connections
            .values()
            .find(|c| c.to == port)
            .map(|c| c.id)
    }
}
