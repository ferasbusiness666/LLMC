//! The circuit data model — pure data, no GUI or simulation logic.

pub mod block;
pub mod chip;
pub mod circuit;
pub mod connection;
pub mod geometry;
pub mod ids;

pub use block::{Block, BlockType, PortLayout};
pub use chip::{ChipDef, ChipLibrary};
pub use circuit::Circuit;
pub use connection::{Connection, Port, PortKind};
pub use geometry::{Orientation, Pos, Rotation, Vec2f};
pub use ids::{BlockId, ChipId, ConnId};
