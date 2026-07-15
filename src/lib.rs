//! LLMC core logic library — free of any GUI dependency so it can be unit-tested
//! headlessly (`cargo test --lib`). The eframe/egui GUI lives in the binary crate.
//!
//! Layering (inspired by the Connection Machine): the binary owns the environment,
//! which holds a [`backend::CircuitManager`] over the [`model`], and drives a
//! [`backend::Simulation`]. Every mutation flows through [`backend::EditCommand`].

pub mod backend;
pub mod io;
pub mod model;
