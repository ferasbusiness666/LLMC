//! Networking for the AI providers: a tiny blocking HTTP client run on worker threads so the
//! UI thread never blocks. Results come back over a channel the panel drains each frame.
//!
//! Everything here is OpenAI-compatible (`GET {base}/models`, `Authorization: Bearer <key>`),
//! which every configured provider speaks; provider-specific quirks can be added later without
//! touching the UI.

use std::sync::mpsc::Sender;
use std::time::Duration;

/// A result handed back to the UI thread from a worker.
pub enum AiEvent {
    /// Outcome of a "Test connection" (or startup reconnect) for `provider` (an index into
    /// [`super::assistant::AiSettings::providers`]).
    Models {
        provider: usize,
        result: Result<Vec<String>, String>,
    },
}

/// Fetch a provider's model list on a background thread and send the result to `tx`.
pub fn spawn_fetch_models(tx: Sender<AiEvent>, provider: usize, base_url: String, api_key: String) {
    std::thread::spawn(move || {
        let result = fetch_models(&base_url, &api_key);
        let _ = tx.send(AiEvent::Models { provider, result });
    });
}

fn agent() -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(10))
        .timeout(Duration::from_secs(25))
        .build()
}

/// Blocking call: list the models available to `api_key` at `base_url`.
pub fn fetch_models(base_url: &str, api_key: &str) -> Result<Vec<String>, String> {
    let base = base_url.trim().trim_end_matches('/');
    if base.is_empty() {
        return Err("Set a base URL first.".to_string());
    }
    if api_key.trim().is_empty() {
        return Err("Paste an API key first.".to_string());
    }
    let url = format!("{base}/models");
    let resp = agent()
        .get(&url)
        .set("Authorization", &format!("Bearer {}", api_key.trim()))
        .call()
        .map_err(describe_error)?;
    let value: serde_json::Value = resp
        .into_json()
        .map_err(|e| format!("Unexpected response: {e}"))?;
    let models = parse_models(&value);
    if models.is_empty() {
        Err("Connected, but the provider returned no models.".to_string())
    } else {
        Ok(models)
    }
}

/// Extract model ids from a provider's `/models` payload, tolerating the common shapes
/// (`{"data":[{"id":…}]}`, `{"models":[{"id"|"name":…}]}`, or a bare array).
fn parse_models(v: &serde_json::Value) -> Vec<String> {
    let from_array = |arr: &Vec<serde_json::Value>| -> Vec<String> {
        arr.iter()
            .filter_map(|m| {
                m.get("id")
                    .or_else(|| m.get("name"))
                    .and_then(|i| i.as_str())
                    .or_else(|| m.as_str())
                    .map(str::to_string)
            })
            .collect()
    };
    let mut out = if let Some(arr) = v.get("data").and_then(|d| d.as_array()) {
        from_array(arr)
    } else if let Some(arr) = v.get("models").and_then(|d| d.as_array()) {
        from_array(arr)
    } else if let Some(arr) = v.as_array() {
        from_array(arr)
    } else {
        Vec::new()
    };
    // Gemini's OpenAI endpoint prefixes ids with "models/"; trim for readability.
    for m in &mut out {
        if let Some(stripped) = m.strip_prefix("models/") {
            *m = stripped.to_string();
        }
    }
    out.sort();
    out.dedup();
    out
}

fn describe_error(e: ureq::Error) -> String {
    match e {
        ureq::Error::Status(code, resp) => {
            let body = resp.into_string().unwrap_or_default();
            let detail = extract_message(&body);
            match code {
                401 | 403 => format!("Rejected (HTTP {code}) — check the API key."),
                404 => format!("Not found (HTTP {code}) — check the base URL."),
                429 => "Rate limited (HTTP 429) — try again shortly.".to_string(),
                _ if !detail.is_empty() => format!("HTTP {code}: {detail}"),
                _ => format!("HTTP {code}."),
            }
        }
        ureq::Error::Transport(t) => format!("Network error: {t}"),
    }
}

/// Pull a human-readable message out of a JSON error body, else the first line of text.
fn extract_message(body: &str) -> String {
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(body) {
        if let Some(msg) = v
            .get("error")
            .and_then(|e| e.get("message").or(Some(e)))
            .and_then(|m| m.as_str())
        {
            return truncate(msg);
        }
    }
    truncate(body.lines().next().unwrap_or("").trim())
}

fn truncate(s: &str) -> String {
    const MAX: usize = 140;
    if s.chars().count() > MAX {
        format!("{}…", s.chars().take(MAX).collect::<String>())
    } else {
        s.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_openai_data_shape() {
        let v = serde_json::json!({
            "data": [
                {"id": "llama-3.3-70b-versatile"},
                {"id": "llama-3.1-8b-instant"}
            ]
        });
        let m = parse_models(&v);
        assert_eq!(m, vec!["llama-3.1-8b-instant", "llama-3.3-70b-versatile"]);
    }

    #[test]
    fn parses_models_key_and_strips_gemini_prefix() {
        let v = serde_json::json!({ "models": [ {"name": "models/gemini-2.0-flash"} ] });
        assert_eq!(parse_models(&v), vec!["gemini-2.0-flash"]);
    }

    #[test]
    fn dedups_and_sorts() {
        let v = serde_json::json!({ "data": [ {"id": "b"}, {"id": "a"}, {"id": "b"} ] });
        assert_eq!(parse_models(&v), vec!["a", "b"]);
    }

    #[test]
    fn error_message_extracted_from_json() {
        let m = extract_message(r#"{"error":{"message":"Invalid API Key"}}"#);
        assert_eq!(m, "Invalid API Key");
    }
}
