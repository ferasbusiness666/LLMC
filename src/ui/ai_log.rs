//! A local debug log of AI requests and replies: one JSON object per line in `ai.log` in the OS
//! data directory, so it can be read by people and tools alike. Every request logs its metadata
//! (model, provider host, timing, HTTP status, sizes, token usage when the provider reports it);
//! the prompt and reply TEXT is included only while "Log prompts and replies" is on. The API key
//! is never logged. The file is rotated to `ai.log.1` when it passes a size cap.

use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;

use super::ai_net::ChatRequest;

/// Serializes concurrent appends from multiple worker threads.
static LOCK: Mutex<()> = Mutex::new(());

/// Rotate past this size (the previous log is kept as `ai.log.1`).
const MAX_BYTES: u64 = 2_000_000;

fn path() -> Option<PathBuf> {
    directories::ProjectDirs::from("dev", "llmc", "LLMC").map(|d| d.data_dir().join("ai.log"))
}

/// Absolute path of the log file, for display in settings.
pub fn log_path_display() -> String {
    path()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| "(unavailable)".to_string())
}

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() > max {
        format!("{}…[truncated]", s.chars().take(max).collect::<String>())
    } else {
        s.to_string()
    }
}

fn append(entry: serde_json::Value) {
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
        let _ = std::fs::rename(&path, path.with_extension("log.1"));
    }
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    {
        let _ = writeln!(f, "{entry}");
    }
}

/// The provider's host (never the full URL, which some setups put a key into).
fn host(base_url: &str) -> String {
    let s = base_url.trim();
    let s = s.split_once("://").map(|(_, r)| r).unwrap_or(s);
    s.split(['/', '?']).next().unwrap_or("").to_string()
}

/// How one request ended, for the log.
pub struct Outcome<'a> {
    pub result: &'a Result<String, String>,
    pub elapsed_ms: u128,
    /// HTTP status of a failed request, if it got that far.
    pub status: Option<u16>,
    /// Raw provider error body (trimmed), when there was one.
    pub provider_body: Option<&'a str>,
    pub truncated: bool,
    pub usage: Option<&'a serde_json::Value>,
}

/// Log a finished request: one line with the metadata, plus the latest turn and the reply when
/// prompt logging is on.
pub fn finished(req: &ChatRequest, out: &Outcome) {
    let chars_sent: usize = req.turns.iter().map(|t| t.content.len()).sum();
    let mut entry = serde_json::json!({
        "ts": unix_now(),
        "model": req.model,
        "host": host(&req.base_url),
        "turns": req.turns.len(),
        "chars_sent": chars_sent,
        "elapsed_ms": out.elapsed_ms as u64,
        "ok": out.result.is_ok(),
    });
    let obj = entry.as_object_mut().expect("object");
    if let Some(s) = out.status {
        obj.insert("status".into(), s.into());
    }
    if out.truncated {
        obj.insert("truncated".into(), true.into());
    }
    if let Some(u) = out.usage {
        obj.insert("usage".into(), u.clone());
    }
    match out.result {
        Ok(text) => {
            obj.insert("reply_chars".into(), text.len().into());
        }
        Err(e) => {
            obj.insert("error".into(), e.clone().into());
            if let Some(b) = out.provider_body.filter(|b| !b.trim().is_empty()) {
                obj.insert("provider_body".into(), truncate(b.trim(), 2000).into());
            }
        }
    }
    if req.log_text {
        if let Some(last) = req.turns.last() {
            obj.insert(
                "prompt".into(),
                serde_json::json!({ "role": last.role, "content": truncate(&last.content, 6000) }),
            );
        }
        if let Ok(text) = out.result {
            obj.insert("reply".into(), truncate(text, 12000).into());
        }
    }
    append(entry);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_never_includes_path_or_query() {
        assert_eq!(host("https://api.groq.com/openai/v1"), "api.groq.com");
        assert_eq!(host("http://localhost:11434/v1?key=x"), "localhost:11434");
    }
}
