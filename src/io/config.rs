//! Persistent application settings, stored in the OS config directory. API *keys* never live
//! here — they're in the OS keyring (see the UI's keystore).

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// How logic gates are drawn on the canvas.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum GateStyle {
    /// Standard distinctive-shape symbols (AND's D, OR's shield, …).
    #[default]
    Shapes,
    /// Labeled rectangles.
    Boxes,
}

/// How wires are routed on the canvas.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum WireStyle {
    /// Smooth curves (easy to tell apart when they cross).
    #[default]
    Curved,
    /// Right-angle routing, like a schematic.
    Orthogonal,
}

/// The assistant model picked last time, so it's preselected on the next launch.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SavedModel {
    pub provider: String,
    pub model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    #[serde(default = "default_true")]
    pub dark_mode: bool,
    #[serde(default)]
    pub last_dir: Option<PathBuf>,
    /// Most recently opened/saved projects, newest first.
    #[serde(default)]
    pub recent_files: Vec<PathBuf>,
    /// Saved AI provider setup (enabled/base-url/name). API keys are NOT stored here.
    #[serde(default)]
    pub ai_providers: Vec<AiProviderConfig>,
    #[serde(default)]
    pub last_model: Option<SavedModel>,
    #[serde(default)]
    pub gate_style: GateStyle,
    #[serde(default)]
    pub wire_style: WireStyle,
    /// Scroll wheel zooms (true) or pans (false; Ctrl+scroll / pinch zooms).
    #[serde(default = "default_true")]
    pub scroll_zooms: bool,
    #[serde(default = "default_true")]
    pub show_grid: bool,
    #[serde(default = "default_true")]
    pub show_minimap: bool,
    #[serde(default = "default_true")]
    pub show_palette: bool,
    /// Keep each project's assistant conversation inside its `.llmc` file.
    #[serde(default = "default_true")]
    pub save_chat: bool,
    /// Models the assistant switches to, in order, when the current one is rate-limited.
    #[serde(default)]
    pub fallback_models: Vec<SavedModel>,
    /// Write prompt and reply text to the local AI request log (metadata is always logged).
    #[serde(default = "default_true")]
    pub log_prompts: bool,
    #[serde(default)]
    pub window: Option<WindowState>,
    /// Project files that were open when the app last closed (reopened on launch).
    #[serde(default)]
    pub open_tabs: Vec<PathBuf>,
}

/// The main window's last size and position (logical points), restored on the next launch.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct WindowState {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    #[serde(default)]
    pub maximized: bool,
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

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            dark_mode: true,
            last_dir: None,
            recent_files: Vec::new(),
            ai_providers: Vec::new(),
            last_model: None,
            gate_style: GateStyle::default(),
            wire_style: WireStyle::default(),
            scroll_zooms: true,
            show_grid: true,
            show_minimap: true,
            show_palette: true,
            save_chat: true,
            fallback_models: Vec::new(),
            log_prompts: true,
            window: None,
            open_tabs: Vec::new(),
        }
    }
}

/// How many recent projects to remember.
const MAX_RECENT: usize = 10;

impl AppConfig {
    fn path() -> Option<PathBuf> {
        Self::dir().map(|d| d.join("config.json"))
    }

    /// The app's config directory (also holds the single-instance lock, crash log and
    /// autosaves).
    pub fn dir() -> Option<PathBuf> {
        directories::ProjectDirs::from("dev", "llmc", "LLMC").map(|d| d.config_dir().to_path_buf())
    }

    /// Load config, falling back to defaults on any error (missing file, bad JSON).
    pub fn load() -> Self {
        Self::path()
            .and_then(|p| std::fs::read_to_string(p).ok())
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    /// Best-effort, crash-safe save; errors are ignored (settings are non-critical).
    pub fn save(&self) {
        let Some(path) = Self::path() else {
            return;
        };
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(text) = serde_json::to_string_pretty(self) {
            let _ = super::fs::write_atomic(&path, text.as_bytes());
        }
    }

    /// Move `path` to the front of the recent-files list.
    pub fn push_recent(&mut self, path: &Path) {
        self.recent_files.retain(|p| p != path);
        self.recent_files.insert(0, path.to_path_buf());
        self.recent_files.truncate(MAX_RECENT);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn old_config_files_still_load() {
        // A config written by an older build (with a field that no longer exists).
        let old = r#"{"dark_mode":false,"last_dir":null,"clock_period":1.0,"ai_providers":[]}"#;
        let c: AppConfig = serde_json::from_str(old).unwrap();
        assert!(!c.dark_mode);
        assert!(c.scroll_zooms && c.show_grid && c.save_chat);
        assert_eq!(c.gate_style, GateStyle::Shapes);
    }

    #[test]
    fn recent_files_are_deduplicated_newest_first() {
        let mut c = AppConfig::default();
        c.push_recent(Path::new("a.llmc"));
        c.push_recent(Path::new("b.llmc"));
        c.push_recent(Path::new("a.llmc"));
        assert_eq!(
            c.recent_files,
            vec![PathBuf::from("a.llmc"), PathBuf::from("b.llmc")]
        );
    }
}
