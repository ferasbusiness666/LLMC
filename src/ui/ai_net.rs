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

// ===================== chat =====================

/// One turn in a conversation, in the OpenAI `messages` shape (`role` is `system`/`user`/
/// `assistant`).
pub struct ChatTurn {
    pub role: String,
    pub content: String,
}

/// Everything a worker needs to run one chat completion.
pub struct ChatRequest {
    pub base_url: String,
    pub api_key: String,
    pub model: String,
    pub turns: Vec<ChatTurn>,
}

/// A chat reply handed back to the UI thread, tagged with the request `token` so it lands in
/// the tab that started it — a request keeps running in the background when you switch tabs.
pub enum ChatEvent {
    Reply {
        token: u64,
        result: Result<String, String>,
    },
}

/// Run one chat completion on a background thread and send the reply to `tx`.
pub fn spawn_chat(tx: Sender<ChatEvent>, token: u64, req: ChatRequest) {
    std::thread::spawn(move || {
        let result = chat(&req);
        let _ = tx.send(ChatEvent::Reply { token, result });
    });
}

/// Chat completions can take a while to generate, so allow a longer read timeout than the
/// quick model-list fetch.
fn chat_agent() -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(10))
        .timeout(Duration::from_secs(120))
        .build()
}

/// Blocking call: POST an OpenAI-compatible chat completion and return the assistant's text.
fn chat(req: &ChatRequest) -> Result<String, String> {
    let base = req.base_url.trim().trim_end_matches('/');
    if base.is_empty() {
        return Err("No base URL for the selected provider.".to_string());
    }
    if req.api_key.trim().is_empty() {
        return Err("No API key for the selected provider.".to_string());
    }
    let messages: Vec<serde_json::Value> = req
        .turns
        .iter()
        .map(|t| serde_json::json!({ "role": t.role, "content": t.content }))
        .collect();
    let body = serde_json::json!({
        "model": req.model,
        "messages": messages,
        "stream": false,
    });
    let url = format!("{base}/chat/completions");
    let resp = chat_agent()
        .post(&url)
        .set("Authorization", &format!("Bearer {}", req.api_key.trim()))
        .send_json(body)
        .map_err(describe_error)?;
    let value: serde_json::Value = resp
        .into_json()
        .map_err(|e| format!("Unexpected response: {e}"))?;
    parse_chat_reply(&value)
}

/// Pull the assistant text out of an OpenAI-style chat-completion response
/// (`choices[0].message.content`), tolerating a plain string or an array of text parts.
fn parse_chat_reply(v: &serde_json::Value) -> Result<String, String> {
    if let Some(choice) = v
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|a| a.first())
    {
        if let Some(text) = choice
            .get("message")
            .and_then(|m| content_text(m.get("content")))
        {
            if !text.trim().is_empty() {
                return Ok(text);
            }
        }
    }
    // A few providers return an error object with an HTTP 200.
    if let Some(err) = v.get("error") {
        let msg = err
            .get("message")
            .and_then(|m| m.as_str())
            .unwrap_or("unknown error");
        return Err(truncate(msg));
    }
    Err("The provider returned an empty reply.".to_string())
}

/// Chat content is usually a string, but some providers return an array of
/// `{ "type": "text", "text": "…" }` parts — concatenate their text.
fn content_text(content: Option<&serde_json::Value>) -> Option<String> {
    match content {
        Some(serde_json::Value::String(s)) => Some(s.clone()),
        Some(serde_json::Value::Array(parts)) => {
            let mut out = String::new();
            for part in parts {
                if let Some(t) = part.get("text").and_then(|t| t.as_str()) {
                    out.push_str(t);
                }
            }
            (!out.is_empty()).then_some(out)
        }
        _ => None,
    }
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

    #[test]
    fn chat_reply_from_string_content() {
        let v = serde_json::json!({
            "choices": [ { "message": { "role": "assistant", "content": "Hello there" } } ]
        });
        assert_eq!(parse_chat_reply(&v).unwrap(), "Hello there");
    }

    #[test]
    fn chat_reply_from_array_content() {
        let v = serde_json::json!({
            "choices": [ { "message": { "content": [
                { "type": "text", "text": "Part 1. " },
                { "type": "text", "text": "Part 2." }
            ] } } ]
        });
        assert_eq!(parse_chat_reply(&v).unwrap(), "Part 1. Part 2.");
    }

    #[test]
    fn chat_reply_surfaces_200_error_body() {
        let v = serde_json::json!({ "error": { "message": "model not found" } });
        assert_eq!(parse_chat_reply(&v).unwrap_err(), "model not found");
    }

    #[test]
    fn chat_reply_empty_is_error() {
        let v = serde_json::json!({ "choices": [] });
        assert!(parse_chat_reply(&v).is_err());
    }
}
