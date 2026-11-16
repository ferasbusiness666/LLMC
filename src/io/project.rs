//! On-disk project format (`.llmc`) — a versioned JSON document.
//!
//! `schema_version` is written from day one so that when a field is later added to
//! `BlockType`, `Connection`, etc., old files can still be recognized and migrated
//! instead of silently failing to load.

use std::fmt;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::model::{ChipLibrary, Circuit};

/// Bump when the format changes in a way that needs migration.
pub const SCHEMA_VERSION: u32 = 1;

fn default_schema_version() -> u32 {
    SCHEMA_VERSION
}

/// Persisted camera so a file reopens where the user left off.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CameraState {
    pub pan_x: f32,
    pub pan_y: f32,
    /// Pixels per grid cell.
    pub zoom: f32,
}

impl Default for CameraState {
    fn default() -> Self {
        Self {
            pan_x: 0.0,
            pan_y: 0.0,
            zoom: 24.0,
        }
    }
}

/// One saved assistant chat message (so a project reopens with its conversation).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ChatRecord {
    /// "user" or "assistant".
    pub role: String,
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub activity: Vec<String>,
    /// "message", "auto" (agent auto-check), "error", or "notice".
    #[serde(default)]
    pub kind: String,
    /// For an auto-check: the one-line result shown to the user, and the facts behind it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Project {
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,
    pub circuit: Circuit,
    #[serde(default)]
    pub chips: ChipLibrary,
    #[serde(default)]
    pub camera: CameraState,
    /// The project's assistant conversation (optional; see the "save chat" setting).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub chat: Vec<ChatRecord>,
}

impl Project {
    pub fn new(circuit: Circuit, chips: ChipLibrary, camera: CameraState) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            circuit,
            chips,
            camera,
            chat: Vec::new(),
        }
    }

    pub fn to_json(&self) -> Result<String, ProjectError> {
        serde_json::to_string_pretty(self).map_err(ProjectError::Json)
    }

    pub fn from_json(text: &str) -> Result<Self, ProjectError> {
        // Peek at the version before fully deserializing so a newer file fails loudly
        // rather than dropping data.
        let value: serde_json::Value = serde_json::from_str(text).map_err(ProjectError::Json)?;
        let version = value
            .get("schema_version")
            .and_then(|v| v.as_u64())
            .unwrap_or(SCHEMA_VERSION as u64) as u32;
        if version > SCHEMA_VERSION {
            return Err(ProjectError::UnsupportedVersion(version));
        }
        serde_json::from_value(value).map_err(ProjectError::Json)
    }

    /// Save crash-safely (temp file + rename), so a failed save never corrupts the project.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), ProjectError> {
        super::fs::write_atomic(path.as_ref(), self.to_json()?.as_bytes()).map_err(ProjectError::Io)
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, ProjectError> {
        let text = std::fs::read_to_string(path).map_err(ProjectError::Io)?;
        Self::from_json(&text)
    }
}

#[derive(Debug)]
pub enum ProjectError {
    Io(std::io::Error),
    Json(serde_json::Error),
    UnsupportedVersion(u32),
}

impl fmt::Display for ProjectError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProjectError::Io(e) => write!(f, "file error: {e}"),
            ProjectError::Json(e) => write!(f, "invalid project data: {e}"),
            ProjectError::UnsupportedVersion(v) => write!(
                f,
                "project was saved by a newer version (format v{v}, this build supports v{SCHEMA_VERSION})"
            ),
        }
    }
}

impl std::error::Error for ProjectError {}
