//! Persistence: project files and application settings.

pub mod config;
pub mod fs;
pub mod project;

pub use config::{AiProviderConfig, AppConfig, GateStyle, SavedModel, WireStyle};
pub use fs::write_atomic;
pub use project::{CameraState, ChatRecord, Project, ProjectError, SCHEMA_VERSION};
