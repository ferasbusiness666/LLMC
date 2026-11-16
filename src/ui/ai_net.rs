//! Networking for the AI providers: a tiny blocking HTTP client run on worker threads so the
//! UI thread never blocks. Results (and streaming tokens) come back over channels the panel
//! drains each frame.
//!
//! Everything here is OpenAI-compatible (`GET {base}/models`, `POST {base}/chat/completions`
//! with `Authorization: Bearer <key>`), which every configured provider speaks. Chat is
//! streamed via Server-Sent Events, with a fallback for providers that answer with a plain
//! JSON body.

use std::io::BufRead;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::Arc;
use std::time::Duration;

/// A result handed back to the UI thread from a worker.
pub enum AiEvent {
    /// Outcome of a "Test connection" (or startup reconnect) for `provider` (an index into
    /// [`super::assistant::AiSettings::providers`]).
    Models {
        provider: usize,
        result: Result<ModelList, String>,
        /// How long the provider took to answer.
        elapsed_ms: u128,
    },
}

/// A provider's models, split into the ones that can chat and a count of the rest (image,
/// video, audio, embedding models — useless to the assistant, so they're never offered).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ModelList {
    pub chat: Vec<String>,
    pub hidden: usize,
}

/// Fetch a provider's model list on a background thread and send the result to `tx`.
pub fn spawn_fetch_models(tx: Sender<AiEvent>, provider: usize, base_url: String, api_key: String) {
    std::thread::spawn(move || {
        let started = std::time::Instant::now();
        let result = fetch_models(&base_url, &api_key);
        let _ = tx.send(AiEvent::Models {
            provider,
            result,
            elapsed_ms: started.elapsed().as_millis(),
        });
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
    /// Raised by the UI's Stop: the worker abandons the stream at the next chunk, so a stopped
    /// run stops consuming tokens (and rate-limit budget) right away.
    pub cancel: Arc<AtomicBool>,
    /// Include prompt/reply text in the local request log (metadata is always logged).
    pub log_text: bool,
}

/// A chat event handed back to the UI thread, tagged with the request `token` so it lands in
/// the tab that started it — a request keeps running in the background when you switch tabs.
pub enum ChatEvent {
    /// A streaming update: the reasoning and answer accumulated so far (full snapshots, so the
    /// UI just stores the latest).
    Delta {
        token: u64,
        reasoning: String,
        content: String,
    },
    /// The stream finished (or a non-streaming reply / an error arrived). `result` is the full
    /// reply text (reasoning folded into `<think>` tags) or a short error message; `retryable`
    /// marks failures that may succeed if simply tried again later (rate limits, overloaded
    /// providers, dropped or stalled connections), which the agent retries on its own.
    Reply {
        token: u64,
        result: Result<String, String>,
        retryable: bool,
        /// The failure was a rate limit / usage cap on this model (another model may work).
        rate_limited: bool,
        /// The connection dropped mid-reply; `result` holds what arrived (offer "Continue").
        truncated: bool,
    },
}

/// Why a chat request failed, and whether trying again later could succeed.
struct ChatFailure {
    message: String,
    retryable: bool,
    rate_limited: bool,
    status: Option<u16>,
    body: Option<String>,
}

impl ChatFailure {
    fn new(message: impl Into<String>, retryable: bool) -> Self {
        Self {
            message: message.into(),
            retryable,
            rate_limited: false,
            status: None,
            body: None,
        }
    }

    fn fatal(message: impl Into<String>) -> Self {
        Self::new(message, false)
    }

    fn transient(message: impl Into<String>) -> Self {
        Self::new(message, true)
    }

    /// Classify a provider-supplied error message by its wording.
    fn from_message(message: String) -> Self {
        let mut f = Self::new(message, false);
        f.retryable = looks_transient(&f.message);
        f.rate_limited = looks_rate_limited(&f.message);
        f
    }
}

/// A finished stream.
struct StreamOk {
    text: String,
    truncated: bool,
    usage: Option<serde_json::Value>,
}

/// Run one chat completion on a background thread, streaming tokens to `tx` as they arrive and
/// sending a terminal [`ChatEvent::Reply`] when done. Logged to the local request log.
pub fn spawn_chat(tx: Sender<ChatEvent>, token: u64, req: ChatRequest) {
    std::thread::spawn(move || {
        let started = std::time::Instant::now();
        let outcome = stream_chat(&tx, token, &req);
        let elapsed_ms = started.elapsed().as_millis();
        let (result, retryable, rate_limited, truncated) = match &outcome {
            Ok(ok) => (Ok(ok.text.clone()), false, false, ok.truncated),
            Err(f) => (Err(f.message.clone()), f.retryable, f.rate_limited, false),
        };
        super::ai_log::finished(
            &req,
            &super::ai_log::Outcome {
                result: &result,
                elapsed_ms,
                status: outcome.as_ref().err().and_then(|f| f.status),
                provider_body: outcome.as_ref().err().and_then(|f| f.body.as_deref()),
                truncated,
                usage: outcome.as_ref().ok().and_then(|o| o.usage.as_ref()),
            },
        );
        let _ = tx.send(ChatEvent::Reply {
            token,
            result,
            retryable,
            rate_limited,
            truncated,
        });
    });
}

/// Seconds without a single byte from the provider before a stream counts as dead.
const IDLE_SECS: u64 = 300;

/// Chat agent: NO overall deadline — a generation may legitimately stream for as long as it
/// needs (thinking models reason for minutes; big circuits are hundreds of commands). The only
/// limit is an IDLE timeout per socket read: a provider that sends nothing at all — not even a
/// keep-alive — for [`IDLE_SECS`] is treated as dead (and, in a run, retried, keeping whatever
/// already arrived).
fn chat_agent() -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(15))
        .timeout_read(Duration::from_secs(IDLE_SECS))
        .build()
}

/// POST an OpenAI-compatible chat completion with `stream: true`, forwarding each token to `tx`
/// via [`ChatEvent::Delta`] and returning the full reply text. Providers that don't stream
/// (respond with a normal JSON body) are handled transparently.
fn stream_chat(
    tx: &Sender<ChatEvent>,
    token: u64,
    req: &ChatRequest,
) -> Result<StreamOk, ChatFailure> {
    let base = req.base_url.trim().trim_end_matches('/');
    if base.is_empty() {
        return Err(ChatFailure::fatal("No base URL for the selected provider."));
    }
    let messages: Vec<serde_json::Value> = req
        .turns
        .iter()
        .map(|t| serde_json::json!({ "role": t.role, "content": t.content }))
        .collect();
    let body = serde_json::json!({
        "model": req.model,
        "messages": messages,
        "stream": true,
        // Low temperature: circuit edits are a precise, structured task, so favour reliable,
        // deterministic output over creativity (helps weaker models emit valid commands).
        "temperature": 0.2,
    });
    let url = format!("{base}/chat/completions");
    let resp = with_auth(chat_agent().post(&url), &req.api_key)
        .set("Accept", "text/event-stream")
        .send_json(body)
        .map_err(describe_chat_error)?;

    // Some providers ignore `stream: true` (or don't support it) and return a normal JSON body.
    if !resp.content_type().contains("event-stream") {
        let value: serde_json::Value = resp
            .into_json()
            .map_err(|e| ChatFailure::transient(format!("Unexpected response: {e}")))?;
        let text = parse_chat_reply(&value).map_err(ChatFailure::from_message)?;
        return Ok(StreamOk {
            text,
            truncated: false,
            usage: value.get("usage").cloned(),
        });
    }

    // Server-Sent Events: `data: {json}` lines, terminated by `data: [DONE]`. Each chunk carries
    // a `choices[0].delta` with `content` and/or a reasoning field.
    let reader = std::io::BufReader::new(resp.into_reader());
    let mut content = String::new();
    let mut reasoning = String::new();
    let mut usage: Option<serde_json::Value> = None;
    let mut truncated = false;
    for line in reader.lines() {
        // Stop pressed: abandon the stream now (the UI already discarded this request).
        if req.cancel.load(Ordering::Relaxed) {
            return Err(ChatFailure::fatal("Stopped."));
        }
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                // The stream broke (stalled or dropped). If real content already arrived,
                // SALVAGE it — the tolerant parser applies every complete command, and the
                // agent's auto-check lets the model pick up from there — instead of throwing
                // the whole reply away.
                if !content.trim().is_empty() {
                    truncated = true;
                    break;
                }
                let stalled = matches!(
                    e.kind(),
                    std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                );
                let mut f = ChatFailure::transient(if stalled {
                    format!(
                        "The provider stopped responding (nothing for {} minutes).",
                        IDLE_SECS / 60
                    )
                } else {
                    "Connection lost while the reply was arriving \u{2014} check your internet \
                     connection."
                        .to_string()
                });
                f.body = Some(e.to_string());
                return Err(f);
            }
        };
        let Some(data) = line.trim().strip_prefix("data:") else {
            continue;
        };
        let data = data.trim();
        if data == "[DONE]" {
            break;
        }
        if data.is_empty() {
            continue;
        }
        let Ok(v) = serde_json::from_str::<serde_json::Value>(data) else {
            continue;
        };
        if let Some(u) = v.get("usage").filter(|u| !u.is_null()) {
            usage = Some(u.clone());
        }
        if let Some(err) = v.get("error") {
            let msg = err
                .get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("stream error");
            return Err(ChatFailure::from_message(truncate(msg)));
        }
        let Some(delta) = v
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|a| a.first())
            .and_then(|c| c.get("delta"))
        else {
            continue;
        };
        let mut changed = false;
        if let Some(t) = delta.get("content").and_then(|c| c.as_str()) {
            if !t.is_empty() {
                content.push_str(t);
                changed = true;
            }
        }
        // OpenRouter streams `reasoning`; DeepSeek streams `reasoning_content`.
        if let Some(t) = delta
            .get("reasoning")
            .or_else(|| delta.get("reasoning_content"))
            .and_then(|c| c.as_str())
        {
            if !t.is_empty() {
                reasoning.push_str(t);
                changed = true;
            }
        }
        if changed {
            let _ = tx.send(ChatEvent::Delta {
                token,
                reasoning: reasoning.clone(),
                content: content.clone(),
            });
        }
    }

    if content.trim().is_empty() && reasoning.trim().is_empty() {
        return Err(ChatFailure::transient(
            "The provider returned an empty reply.",
        ));
    }
    // Fold a separate reasoning stream into `<think>` tags so the reply parser handles both
    // streaming styles uniformly.
    let text = if reasoning.trim().is_empty() {
        content
    } else {
        format!("<think>{reasoning}</think>{content}")
    };
    Ok(StreamOk {
        text,
        truncated,
        usage,
    })
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
pub fn fetch_models(base_url: &str, api_key: &str) -> Result<ModelList, String> {
    let base = base_url.trim().trim_end_matches('/');
    if base.is_empty() {
        return Err("Set a base URL first.".to_string());
    }
    let url = format!("{base}/models");
    let resp = with_auth(agent().get(&url), api_key)
        .call()
        .map_err(describe_error)?;
    let value: serde_json::Value = resp
        .into_json()
        .map_err(|e| format!("Unexpected response: {e}"))?;
    let models = parse_models(&value);
    if models.chat.is_empty() {
        Err(if models.hidden > 0 {
            format!(
                "Connected, but none of its {} models can chat (they're image/audio/video models).",
                models.hidden
            )
        } else {
            "Connected, but the provider returned no models.".to_string()
        })
    } else {
        Ok(models)
    }
}

/// Attach `Authorization: Bearer <key>` — unless there is no key, so keyless local servers
/// (Ollama, LM Studio, llama.cpp) work through the Custom provider.
fn with_auth(req: ureq::Request, api_key: &str) -> ureq::Request {
    let key = api_key.trim();
    if key.is_empty() {
        req
    } else {
        req.set("Authorization", &format!("Bearer {key}"))
    }
}

/// Extract model ids from a provider's `/models` payload, tolerating the common shapes
/// (`{"data":[{"id":…}]}`, `{"models":[{"id"|"name":…}]}`, or a bare array), and set aside the
/// models that can't chat.
fn parse_models(v: &serde_json::Value) -> ModelList {
    let arr: &[serde_json::Value] = v
        .get("data")
        .and_then(|d| d.as_array())
        .or_else(|| v.get("models").and_then(|d| d.as_array()))
        .or_else(|| v.as_array())
        .map(|a| a.as_slice())
        .unwrap_or(&[]);
    let mut chat: Vec<String> = Vec::new();
    let mut hidden = 0;
    for m in arr {
        let Some(id) = m
            .get("id")
            .or_else(|| m.get("name"))
            .and_then(|i| i.as_str())
            .or_else(|| m.as_str())
        else {
            continue;
        };
        // Gemini's OpenAI endpoint prefixes ids with "models/"; trim for readability.
        let id = id.strip_prefix("models/").unwrap_or(id).to_string();
        if is_chat_model(&id, m) {
            chat.push(id);
        } else {
            hidden += 1;
        }
    }
    chat.sort();
    chat.dedup();
    ModelList { chat, hidden }
}

/// Can this model hold a text chat? Uses the provider's metadata when it says (OpenRouter's
/// `architecture.output_modalities`, a `type`/`task` field, …), else the model's name: image,
/// video, speech, transcription, embedding and moderation models are filtered out.
fn is_chat_model(id: &str, meta: &serde_json::Value) -> bool {
    let field = |k: &str| {
        meta.get(k)
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_lowercase()
    };
    let kind = format!(
        "{} {} {}",
        field("type"),
        field("task"),
        field("model_type")
    );
    const NON_CHAT_KINDS: [&str; 12] = [
        "image",
        "video",
        "audio",
        "embedding",
        "rerank",
        "tts",
        "stt",
        "speech",
        "transcription",
        "moderation",
        "text-to-image",
        "diffusion",
    ];
    if NON_CHAT_KINDS.iter().any(|k| kind.contains(k)) {
        return false;
    }
    if let Some(outs) = meta
        .get("architecture")
        .and_then(|a| a.get("output_modalities"))
        .and_then(|o| o.as_array())
    {
        if !outs.iter().any(|o| o.as_str() == Some("text")) {
            return false;
        }
    }
    let name = id.to_lowercase();
    const NON_CHAT_NAMES: [&str; 47] = [
        "embed",
        "rerank",
        "moderation",
        "whisper",
        "tts",
        "transcribe",
        "speech",
        "audio",
        "dall-e",
        "dalle",
        "gpt-image",
        "imagen",
        "image-",
        "-image",
        "flux",
        "sdxl",
        "stable-diffusion",
        "stablediffusion",
        "sd3",
        "playground-v",
        "kandinsky",
        "midjourney",
        "ideogram",
        "recraft",
        "phoenix",
        "lucid-origin",
        "aleph",
        "veo-",
        "sora",
        "kling",
        "hailuo",
        "seedance",
        "seedream",
        "wan-",
        "luma",
        "runway",
        "gen-3",
        "pika",
        "video",
        "aura-",
        "melotts",
        "nova-3",
        "nova-2",
        "deepgram",
        "moondream",
        "musicgen",
        "bark",
    ];
    !NON_CHAT_NAMES.iter().any(|w| name.contains(w))
}

/// The model-list fetch's error text (the chat path uses the classified [`describe_chat_error`]).
fn describe_error(e: ureq::Error) -> String {
    describe_chat_error(e).message
}

/// Turn an HTTP/transport failure into a plain-language message plus a retry verdict. Rate
/// limits, timeouts, server-side (5xx) failures and dropped connections are worth trying again
/// later; client errors (bad key, a model that can't chat, oversized request) are not. Raw OS
/// error text never reaches the user (it goes to the request log instead).
fn describe_chat_error(e: ureq::Error) -> ChatFailure {
    match e {
        ureq::Error::Status(code, resp) => {
            let body = resp.into_string().unwrap_or_default();
            let detail = extract_message(&body);
            let limited = code == 429 || looks_rate_limited(&detail);
            let message = match code {
                401 | 403 if !limited => {
                    format!("Rejected (HTTP {code}) \u{2014} check the API key.")
                }
                404 => format!(
                    "This model can't be used for chat (HTTP 404) \u{2014} pick another model{}",
                    if detail.is_empty() {
                        ".".to_string()
                    } else {
                        format!(". Provider said: {detail}")
                    }
                ),
                _ if limited => format!(
                    "This model is rate-limited (HTTP {code}){}",
                    if detail.is_empty() {
                        ".".to_string()
                    } else {
                        format!(": {detail}")
                    }
                ),
                _ if !detail.is_empty() => format!("HTTP {code}: {detail}"),
                _ => format!("HTTP {code}."),
            };
            let retryable = limited
                || matches!(code, 408 | 425)
                || code >= 500
                || (code >= 400 && looks_transient(&detail));
            ChatFailure {
                message,
                retryable,
                rate_limited: limited,
                status: Some(code),
                body: Some(body),
            }
        }
        ureq::Error::Transport(t) => {
            use ureq::ErrorKind as K;
            let message = match t.kind() {
                K::Dns => "Couldn't find the provider's server \u{2014} check your internet \
                           connection and the base URL."
                    .to_string(),
                K::ConnectionFailed => "Couldn't connect to the provider \u{2014} check your \
                                        internet connection."
                    .to_string(),
                K::InvalidUrl | K::UnknownScheme => {
                    "The provider's base URL isn't valid \u{2014} check it in AI providers."
                        .to_string()
                }
                K::TooManyRedirects | K::BadStatus | K::BadHeader => {
                    "The provider sent a response LLMC couldn't read.".to_string()
                }
                _ => "Connection lost \u{2014} check your internet connection.".to_string(),
            };
            let mut f = ChatFailure::transient(message);
            f.body = Some(t.to_string());
            f
        }
    }
}

/// Does a provider's error wording describe a temporary condition (overload, rate limit,
/// timeout) rather than a problem with the request itself?
fn looks_transient(message: &str) -> bool {
    let m = message.to_lowercase();
    looks_rate_limited(&m)
        || [
            "overload",
            "capacity",
            "temporar",
            "unavailable",
            "try again",
            "timeout",
            "timed out",
            "busy",
            "server error",
            "internal error",
            "upstream",
            "empty reply",
        ]
        .iter()
        .any(|w| m.contains(w))
}

/// Is this a per-model rate limit or usage cap (so a different model may well work now)?
fn looks_rate_limited(message: &str) -> bool {
    let m = message.to_lowercase();
    [
        "rate limit",
        "rate-limit",
        "ratelimit",
        "too many requests",
        "too much usage",
        "usage limit",
        "quota",
        "requests per",
        "tokens per",
    ]
    .iter()
    .any(|w| m.contains(w))
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
        assert_eq!(
            m.chat,
            vec!["llama-3.1-8b-instant", "llama-3.3-70b-versatile"]
        );
    }

    #[test]
    fn parses_models_key_and_strips_gemini_prefix() {
        let v = serde_json::json!({ "models": [ {"name": "models/gemini-2.0-flash"} ] });
        assert_eq!(parse_models(&v).chat, vec!["gemini-2.0-flash"]);
    }

    #[test]
    fn dedups_and_sorts() {
        let v = serde_json::json!({ "data": [ {"id": "b"}, {"id": "a"}, {"id": "b"} ] });
        assert_eq!(parse_models(&v).chat, vec!["a", "b"]);
    }

    #[test]
    fn image_video_and_audio_models_are_never_offered_for_chat() {
        let ids = [
            "aleph-2",
            "aura-2-en",
            "flux-1-schnell",
            "flux-2-pro",
            "sdxl-lightning",
            "phoenix-1.0",
            "lucid-origin",
            "mai-image-2.6-flash",
            "veo-3.1",
            "sora-2-pro",
            "kling-v3.0-pro",
            "hailuo-3-max",
            "seedance-2.0-fast",
            "wan-3.0-prime",
            "grok-imagine-video",
            "whisper-large-v3-turbo",
            "nova-3",
            "melotts",
            "moondream3.1",
            "text-embedding-3-small",
            "grok-4.6",
            "kimi-k2.6",
            "glm-5",
            "logfare/auto",
            "qwen-3.8-27b",
            "step-3.7-flash",
            "deepseek-v3.2",
            "llama-3.3-70b-versatile",
        ];
        let v = serde_json::json!({ "data": ids.iter().map(|i| serde_json::json!({"id": i})).collect::<Vec<_>>() });
        let m = parse_models(&v);
        assert_eq!(
            m.chat,
            vec![
                "deepseek-v3.2",
                "glm-5",
                "grok-4.6",
                "kimi-k2.6",
                "llama-3.3-70b-versatile",
                "logfare/auto",
                "qwen-3.8-27b",
                "step-3.7-flash",
            ]
        );
        assert_eq!(m.hidden, 20);
        // Metadata wins when present.
        let v = serde_json::json!({ "data": [
            {"id": "x-chat", "architecture": {"output_modalities": ["text"]}},
            {"id": "x-pix", "architecture": {"output_modalities": ["image"]}},
            {"id": "x-emb", "type": "embedding"}
        ]});
        assert_eq!(parse_models(&v).chat, vec!["x-chat"]);
    }

    #[test]
    fn rate_limits_and_usage_caps_are_recognized() {
        for m in [
            "Rate limit exceeded for this API key",
            "Too much usage, slow down",
            "You exceeded your current quota",
        ] {
            assert!(looks_rate_limited(m), "{m}");
            assert!(looks_transient(m), "{m}");
        }
        assert!(!looks_rate_limited("model not found"));
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

    #[test]
    fn transient_errors_are_recognized() {
        for m in [
            "Rate limit reached for model",
            "The server is overloaded, try again later",
            "Service Unavailable",
            "upstream connect error",
        ] {
            assert!(looks_transient(m), "{m}");
        }
        for m in [
            "Invalid API Key",
            "model not found",
            "context length exceeded",
        ] {
            assert!(!looks_transient(m), "{m}");
        }
    }
}
