//! Secure, persistent storage for provider API keys.
//!
//! On Windows the key lives in the **Windows Credential Manager** via the `keyring` crate.
//! On other platforms (dev/CI) it falls back to a JSON file in the OS config directory — the
//! app targets Windows, where the secure store is used; the fallback just keeps things working
//! elsewhere. Keys never leave the machine either way.

#[cfg(windows)]
const SERVICE: &str = "LLMC";

#[cfg(windows)]
pub fn save(id: &str, key: &str) -> Result<(), String> {
    keyring::Entry::new(SERVICE, id)
        .and_then(|e| e.set_password(key))
        .map_err(|e| e.to_string())
}

#[cfg(windows)]
pub fn load(id: &str) -> Option<String> {
    keyring::Entry::new(SERVICE, id).ok()?.get_password().ok()
}

/// Remove a stored key (used when a provider's key is cleared/removed).
#[cfg(windows)]
#[allow(dead_code)]
pub fn delete(id: &str) {
    if let Ok(entry) = keyring::Entry::new(SERVICE, id) {
        let _ = entry.delete_credential();
    }
}

// ----- non-Windows fallback: a JSON map in the config dir -----

#[cfg(not(windows))]
mod fallback {
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    fn path() -> Option<PathBuf> {
        directories::ProjectDirs::from("dev", "llmc", "LLMC")
            .map(|d| d.config_dir().join("keys.json"))
    }

    fn read() -> BTreeMap<String, String> {
        path()
            .and_then(|p| std::fs::read_to_string(p).ok())
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    fn write(map: &BTreeMap<String, String>) -> Result<(), String> {
        let path = path().ok_or_else(|| "no config directory".to_string())?;
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let text = serde_json::to_string_pretty(map).map_err(|e| e.to_string())?;
        std::fs::write(path, text).map_err(|e| e.to_string())
    }

    pub fn save(id: &str, key: &str) -> Result<(), String> {
        let mut map = read();
        map.insert(id.to_string(), key.to_string());
        write(&map)
    }

    pub fn load(id: &str) -> Option<String> {
        read().get(id).cloned()
    }

    pub fn delete(id: &str) {
        let mut map = read();
        if map.remove(id).is_some() {
            let _ = write(&map);
        }
    }
}

#[cfg(not(windows))]
pub fn save(id: &str, key: &str) -> Result<(), String> {
    fallback::save(id, key)
}

#[cfg(not(windows))]
pub fn load(id: &str) -> Option<String> {
    fallback::load(id)
}

/// Remove a stored key (used when a provider's key is cleared/removed).
#[cfg(not(windows))]
#[allow(dead_code)]
pub fn delete(id: &str) {
    fallback::delete(id);
}
