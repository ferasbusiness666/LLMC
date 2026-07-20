//! A best-effort local debug log of AI requests and replies, appended to a file in the OS data
//! directory. Handy when a weak model thrashes in agent mode: you can see exactly what was sent
//! and what came back. Never logs the API key.

use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;

use super::ai_net::ChatRequest;

/// Serializes concurrent appends from multiple worker threads.
static LOCK: Mutex<()> = Mutex::new(());

/// Keep the log from growing without bound (rotated by simple truncation past this size).
const MAX_BYTES: u64 = 4_000_000;

fn path() -> Option<PathBuf> {
    directories::ProjectDirs::from("dev", "llmc", "LLMC").map(|d| d.data_dir().join("ai.log"))
}

/// Absolute path of the log file, for display in settings.
pub fn log_path_display() -> String {
    path()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| "(unavailable)".to_string())
}

fn timestamp() -> String {
    match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
        Ok(d) => format!("unix={}", d.as_secs()),
        Err(_) => "unix=?".to_string(),
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() > max {
        format!("{}…[truncated]", s.chars().take(max).collect::<String>())
    } else {
        s.to_string()
    }
}

fn append(section: &str, body: &str) {
    let Some(path) = path() else {
        return;
    };
    let Ok(_guard) = LOCK.lock() else {
        return;
    };
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if std::fs::metadata(&path)
        .map(|m| m.len() > MAX_BYTES)
        .unwrap_or(false)
    {
        let _ = std::fs::remove_file(&path);
    }
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    {
        let _ = writeln!(f, "===== {} {} =====", timestamp(), section);
        let _ = writeln!(f, "{body}\n");
    }
}

/// Log an outgoing request (model + the latest turn, which is the freshest context).
pub fn request(req: &ChatRequest) {
    let last = req
        .turns
        .last()
        .map(|t| format!("[{}] {}", t.role, t.content))
        .unwrap_or_default();
    append(
        &format!("REQUEST model={} turns={}", req.model, req.turns.len()),
        &truncate(&last, 4000),
    );
}

/// Log the reply text or the error.
pub fn reply(req: &ChatRequest, result: &Result<String, String>) {
    match result {
        Ok(text) => append(&format!("REPLY model={}", req.model), &truncate(text, 8000)),
        Err(e) => append(&format!("ERROR model={}", req.model), e),
    }
}
