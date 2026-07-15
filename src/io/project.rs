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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Project {
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,
    pub circuit: Circuit,
    #[serde(default)]
    pub chips: ChipLibrary,
    #[serde(default)]
    pub camera: CameraState,
}

impl Project {
    pub fn new(circuit: Circuit, chips: ChipLibrary, camera: CameraState) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            circuit,
            chips,
            camera,
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

    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), ProjectError> {
        std::fs::write(path, self.to_json()?).map_err(ProjectError::Io)
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
