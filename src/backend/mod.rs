//! Core logic and circuit management (the "Backend").

pub mod circuit_manager;
pub mod commands;
pub mod simulator;
pub mod undo;

pub use circuit_manager::{CircuitManager, GroupTransform, RedefineReport, Txn};
pub use commands::{apply_batch, EditCommand};
pub use simulator::{clock_half_period, Simulation};
pub use undo::UndoSystem;
