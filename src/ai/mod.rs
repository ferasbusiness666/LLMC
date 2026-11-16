//! The assistant's "hands": parsing a model's reply into commands ([`parse`]), applying them to
//! a project as one undoable step ([`apply`]), and describing the project back to the model —
//! the system prompt, the compact circuit context, and the post-edit auto-check ([`prompt`]).
//!
//! All of it is GUI-free so it runs (and is tested) headlessly; the app only shuttles text to
//! and from the provider.

pub mod apply;
pub mod parse;
pub mod prompt;

pub use apply::{
    apply_ops, apply_ops_with, block_display_name, user_allows_removal, ApplyOptions, ApplyReport,
};
pub use parse::{
    live_split, parse_reply, AddKind, LiveView, ParsedReply, PortSel, RawOp, TestStep,
};
pub use prompt::{agent_observation, circuit_context, system_prompt, Observation};
