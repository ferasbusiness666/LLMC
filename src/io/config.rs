//! Persistent application settings, stored in the OS config directory.
//!
//! Phase 2 will extend this with AI provider entries; API *keys* will live in the OS
//! keyring rather than this file.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    #[serde(default = "default_true")]
    pub dark_mode: bool,
    #[serde(default)]
    pub last_dir: Option<PathBuf>,
    /// Clock half-period in seconds for simulation.
    #[serde(default = "default_clock_period")]
    pub clock_period: f64,
    /// Saved AI provider setup (enabled/base-url/name). API keys are NOT stored here — they
    /// live in the OS keyring / a separate local file (see the UI's keystore).
    #[serde(default)]
    pub ai_providers: Vec<AiProviderConfig>,
}

/// Persisted, non-secret configuration for one AI provider.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AiProviderConfig {
    /// Stable id matching the UI provider (e.g. "groq", "custom").
    pub id: String,
    pub name: String,
    pub base_url: String,
    pub enabled: bool,
}

fn default_true() -> bool {
    true
}

fn default_clock_period() -> f64 {
    1.0
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            dark_mode: true,
            last_dir: None,
            clock_period: 1.0,
            ai_providers: Vec::new(),
        }
    }
}

impl AppConfig {
    fn path() -> Option<PathBuf> {
        directories::ProjectDirs::from("dev", "llmc", "LLMC")
            .map(|dirs| dirs.config_dir().join("config.json"))
    }

    /// Load config, falling back to defaults on any error (missing file, bad JSON).
    pub fn load() -> Self {
        Self::path()
            .and_then(|p| std::fs::read_to_string(p).ok())
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    /// Best-effort save; errors are ignored (settings are non-critical).
    pub fn save(&self) {
        let Some(path) = Self::path() else {
            return;
        };
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(text) = serde_json::to_string_pretty(self) {
            let _ = std::fs::write(path, text);
        }
    }
}
