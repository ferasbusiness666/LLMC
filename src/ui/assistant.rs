//! The AI assistant: a calm right-side panel plus a provider settings modal.
//!
//! This module owns the assistant's UI and state (providers, per-project chat session,
//! selected model) and drives provider connections through `ai_net` (off-thread HTTP) and
//! `keystore` (secure key storage). Replies are parsed by `ai_edit` into reasoning + prose +
//! circuit-edit ops; the ops are handed to the app (which owns the circuit) to apply as one
//! undoable step, and the result is shown back here as an "activity" list under the reply.
//!
//! Design goals: the base app stays calm (just one toolbar toggle + this panel), and the
//! panel itself reads like a real product — a clear header, roomy chat, and a composer with
//! the model switcher right where you type.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Arc;
use std::time::{Duration, Instant};

use eframe::egui::{
    self, Align, Align2, Color32, CornerRadius, Frame, Layout, Margin, Pos2, RichText, Sense,
    Stroke, StrokeKind,
};

use llmc::ai::{self as ai_edit, RawOp};
use llmc::io::{AiProviderConfig, ChatRecord, SavedModel};

use super::ai_net::{self, AiEvent, ChatEvent, ChatRequest, ChatTurn};
use super::design::{self, Kind, Theme};
use super::icons;
use super::keystore;

/// A provider the settings can offer. Most are OpenAI-compatible, so they share one client;
/// Google AI Studio uses its own adapter (or its OpenAI-compatible endpoint).
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ProviderKind {
    Groq,
    OpenRouter,
    GoogleAiStudio,
    Zen,
    Cerebras,
    Mistral,
    OllamaCloud,
    Custom,
}

impl ProviderKind {
    /// Stable id used to persist config and key it in the keystore.
    fn id(self) -> &'static str {
        match self {
            ProviderKind::Groq => "groq",
            ProviderKind::OpenRouter => "openrouter",
            ProviderKind::GoogleAiStudio => "google-ai-studio",
            ProviderKind::Zen => "zen",
            ProviderKind::Cerebras => "cerebras",
            ProviderKind::Mistral => "mistral",
            ProviderKind::OllamaCloud => "ollama-cloud",
            ProviderKind::Custom => "custom",
        }
    }

    fn label(self) -> &'static str {
        match self {
            ProviderKind::Groq => "Groq",
            ProviderKind::OpenRouter => "OpenRouter",
            ProviderKind::GoogleAiStudio => "Google AI Studio",
            ProviderKind::Zen => "Zen (opencode)",
            ProviderKind::Cerebras => "Cerebras",
            ProviderKind::Mistral => "Mistral",
            ProviderKind::OllamaCloud => "Ollama Cloud",
            ProviderKind::Custom => "Custom",
        }
    }

    /// One-line "what is this" shown in settings.
    fn blurb(self) -> &'static str {
        match self {
            ProviderKind::Groq => "Free tier · very fast · open models",
            ProviderKind::OpenRouter => "Aggregator · many models, some free",
            ProviderKind::GoogleAiStudio => "Free tier · Gemini models",
            ProviderKind::Zen => "opencode Zen · open models",
            ProviderKind::Cerebras => "Free tier · fastest inference",
            ProviderKind::Mistral => "Free tier · Mistral models",
            ProviderKind::OllamaCloud => "Cloud models · hosted Ollama",
            ProviderKind::Custom => "Any OpenAI-compatible endpoint",
        }
    }

    /// Default OpenAI-compatible base URL. Verified against each provider's docs when the
    /// networking layer lands; a couple are best-effort until then.
    fn default_base_url(self) -> &'static str {
        match self {
            ProviderKind::Groq => "https://api.groq.com/openai/v1",
            ProviderKind::OpenRouter => "https://openrouter.ai/api/v1",
            ProviderKind::GoogleAiStudio => {
                "https://generativelanguage.googleapis.com/v1beta/openai"
            }
            ProviderKind::Zen => "https://opencode.ai/zen/v1",
            ProviderKind::Cerebras => "https://api.cerebras.ai/v1",
            ProviderKind::Mistral => "https://api.mistral.ai/v1",
            ProviderKind::OllamaCloud => "https://ollama.com/v1",
            ProviderKind::Custom => "",
        }
    }

    /// Where to create an API key.
    fn key_url(self) -> Option<&'static str> {
        match self {
            ProviderKind::Groq => Some("https://console.groq.com/keys"),
            ProviderKind::OpenRouter => Some("https://openrouter.ai/keys"),
            ProviderKind::GoogleAiStudio => Some("https://aistudio.google.com/apikey"),
            ProviderKind::Zen => Some("https://opencode.ai/auth"),
            ProviderKind::Cerebras => Some("https://cloud.cerebras.ai"),
            ProviderKind::Mistral => Some("https://console.mistral.ai/api-keys"),
            ProviderKind::OllamaCloud => Some("https://ollama.com/settings/keys"),
            ProviderKind::Custom => None,
        }
    }
}

/// Connection state for a provider.
#[derive(Clone, PartialEq)]
pub enum ConnStatus {
    Idle,
    Testing,
    Connected,
    /// A test/reconnect failed; the string is a short human-readable reason.
    Failed(String),
}

/// A configured provider (global — shared across all project tabs).
pub struct Provider {
    pub kind: ProviderKind,
    /// Display name (editable, mainly for Custom entries).
    pub name: String,
    pub base_url: String,
    /// Held in memory while running; a working key is persisted to the OS keyring (see keystore).
    pub api_key: String,
    pub enabled: bool,
    /// Chat-capable models fetched from the provider by "Test connection" / startup reconnect.
    pub models: Vec<String>,
    /// How many of the provider's models were hidden because they can't chat.
    pub hidden_models: usize,
    /// How long the last successful connection test took.
    pub latency_ms: Option<u128>,
    pub status: ConnStatus,
}

impl Provider {
    fn preset(kind: ProviderKind) -> Self {
        Self {
            kind,
            name: kind.label().to_string(),
            base_url: kind.default_base_url().to_string(),
            api_key: String::new(),
            enabled: false,
            models: Vec::new(),
            hidden_models: 0,
            latency_ms: None,
            status: ConnStatus::Idle,
        }
    }

    fn is_live(&self) -> bool {
        self.enabled && self.status == ConnStatus::Connected && !self.models.is_empty()
    }

    /// Local/self-hosted endpoints (Ollama, LM Studio, llama.cpp) often need no key.
    fn key_optional(&self) -> bool {
        self.kind == ProviderKind::Custom
    }
}

/// Global assistant settings: the provider list plus the worker-thread channel that carries
/// connection results back to the UI.
pub struct AiSettings {
    pub providers: Vec<Provider>,
    tx: Sender<AiEvent>,
    rx: Receiver<AiEvent>,
    /// Set when the persisted config changed (enabled/base-url/name/model); the app writes it out.
    dirty: bool,
    /// The model picked last (preselected in new tabs and after a restart).
    pub preferred: Option<SavedModel>,
    /// Models to switch to, in order, when the current one is rate-limited.
    pub fallbacks: Vec<SavedModel>,
    /// Rate-limited models and when they may be tried again: (provider id, model, until).
    cooldowns: Vec<(String, String, Instant)>,
    /// Include prompt/reply text in the local request log.
    pub log_prompts: bool,
}

/// How long a rate-limited model is skipped by the fallback chain.
const RATE_LIMIT_COOLDOWN: Duration = Duration::from_secs(90);

impl Default for AiSettings {
    fn default() -> Self {
        let providers = [
            ProviderKind::Groq,
            ProviderKind::OpenRouter,
            ProviderKind::GoogleAiStudio,
            ProviderKind::Zen,
            ProviderKind::Cerebras,
            ProviderKind::Mistral,
            ProviderKind::OllamaCloud,
            ProviderKind::Custom,
        ]
        .into_iter()
        .map(Provider::preset)
        .collect();
        let (tx, rx) = std::sync::mpsc::channel();
        Self {
            providers,
            tx,
            rx,
            dirty: false,
            preferred: None,
            fallbacks: Vec::new(),
            cooldowns: Vec::new(),
            log_prompts: true,
        }
    }
}

impl AiSettings {
    /// Every (provider index, model) that can currently be picked.
    fn live_models(&self) -> Vec<(usize, &str)> {
        let mut out = Vec::new();
        for (i, p) in self.providers.iter().enumerate() {
            if p.is_live() {
                for m in &p.models {
                    out.push((i, m.as_str()));
                }
            }
        }
        out
    }

    fn any_live(&self) -> bool {
        self.providers.iter().any(Provider::is_live)
    }

    /// Is a connection test in flight (used to keep repainting so results land promptly)?
    pub fn any_testing(&self) -> bool {
        self.providers
            .iter()
            .any(|p| p.status == ConnStatus::Testing)
    }

    /// Drain any results that arrived from worker threads and update provider state.
    pub fn poll(&mut self) {
        while let Ok(event) = self.rx.try_recv() {
            match event {
                AiEvent::Models {
                    provider,
                    result,
                    elapsed_ms,
                } => {
                    if let Some(p) = self.providers.get_mut(provider) {
                        match result {
                            Ok(list) => {
                                p.models = list.chat;
                                p.hidden_models = list.hidden;
                                p.latency_ms = Some(elapsed_ms);
                                p.status = ConnStatus::Connected;
                                // A working key is worth remembering.
                                if !p.api_key.trim().is_empty() {
                                    let _ = keystore::save(p.kind.id(), p.api_key.trim());
                                }
                            }
                            Err(e) => {
                                p.models.clear();
                                p.status = ConnStatus::Failed(e);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Start a "Test connection" for provider `i` on a worker thread.
    fn start_test(&mut self, i: usize) {
        let Some(p) = self.providers.get_mut(i) else {
            return;
        };
        if p.base_url.trim().is_empty() || (p.api_key.trim().is_empty() && !p.key_optional()) {
            p.status = ConnStatus::Failed("Enter a base URL and API key first.".to_string());
            return;
        }
        p.status = ConnStatus::Testing;
        let (base, key) = (p.base_url.clone(), p.api_key.clone());
        ai_net::spawn_fetch_models(self.tx.clone(), i, base, key);
        self.dirty = true;
    }

    /// Restore saved provider config + keys, and reconnect any that were enabled.
    pub fn apply_config(&mut self, saved: &[AiProviderConfig]) {
        for i in 0..self.providers.len() {
            let id = self.providers[i].kind.id();
            if let Some(cfg) = saved.iter().find(|c| c.id == id) {
                let p = &mut self.providers[i];
                p.enabled = cfg.enabled;
                if !cfg.name.trim().is_empty() {
                    p.name = cfg.name.clone();
                }
                if !cfg.base_url.trim().is_empty() {
                    p.base_url = cfg.base_url.clone();
                }
            }
            if let Some(key) = keystore::load(id) {
                self.providers[i].api_key = key;
            }
            // Reconnect providers that were on and have a key (or don't need one).
            let ready = {
                let p = &self.providers[i];
                p.enabled
                    && (!p.api_key.trim().is_empty() || p.key_optional())
                    && !p.base_url.trim().is_empty()
            };
            if ready {
                self.providers[i].status = ConnStatus::Testing;
                let (base, key) = (
                    self.providers[i].base_url.clone(),
                    self.providers[i].api_key.clone(),
                );
                ai_net::spawn_fetch_models(self.tx.clone(), i, base, key);
            }
        }
    }

    /// The persistable (non-secret) view of the providers.
    pub fn config_snapshot(&self) -> Vec<AiProviderConfig> {
        self.providers
            .iter()
            .map(|p| AiProviderConfig {
                id: p.kind.id().to_string(),
                name: p.name.clone(),
                base_url: p.base_url.clone(),
                enabled: p.enabled,
            })
            .collect()
    }

    /// Forget a provider's stored key (removes it from the OS keyring too) and disconnect it.
    fn forget_key(&mut self, i: usize) {
        if let Some(p) = self.providers.get_mut(i) {
            keystore::delete(p.kind.id());
            p.api_key.clear();
            p.models.clear();
            p.status = ConnStatus::Idle;
            self.dirty = true;
        }
    }

    /// Remember the picked model (persisted by the app).
    fn set_preferred(&mut self, provider: usize, model: &str) {
        if let Some(p) = self.providers.get(provider) {
            self.preferred = Some(SavedModel {
                provider: p.kind.id().to_string(),
                model: model.to_string(),
            });
            self.dirty = true;
        }
    }

    /// Time left before a rate-limited model may be used again by the fallback chain.
    pub fn cooldown_left(&self, provider: usize, model: &str) -> Option<Duration> {
        let id = self.providers.get(provider)?.kind.id();
        let now = Instant::now();
        self.cooldowns
            .iter()
            .find(|(p, m, until)| p == id && m == model && *until > now)
            .map(|(_, _, until)| *until - now)
    }

    fn mark_rate_limited(&mut self, provider: usize, model: &str) {
        let Some(id) = self
            .providers
            .get(provider)
            .map(|p| p.kind.id().to_string())
        else {
            return;
        };
        let now = Instant::now();
        self.cooldowns
            .retain(|(p, m, until)| *until > now && !(*p == id && m == model));
        self.cooldowns
            .push((id, model.to_string(), now + RATE_LIMIT_COOLDOWN));
    }

    /// Resolve a saved (provider id, model) to a live, pickable model.
    fn resolve(&self, saved: &SavedModel) -> Option<ModelRef> {
        let (i, p) = self
            .providers
            .iter()
            .enumerate()
            .find(|(_, p)| p.kind.id() == saved.provider)?;
        (p.is_live() && p.models.contains(&saved.model)).then(|| ModelRef {
            provider: i,
            model: saved.model.clone(),
        })
    }

    /// The next model in the fallback chain after `current` that is connected and not cooling
    /// down (entries after the current one first, then from the top).
    fn next_fallback(&self, current: &ModelRef) -> Option<ModelRef> {
        let cur_id = self.providers.get(current.provider)?.kind.id();
        let start = self
            .fallbacks
            .iter()
            .position(|f| f.provider == cur_id && f.model == current.model)
            .map(|i| i + 1)
            .unwrap_or(0);
        let n = self.fallbacks.len();
        (0..n)
            .map(|k| &self.fallbacks[(start + k) % n.max(1)])
            .filter_map(|f| self.resolve(f))
            .find(|r| {
                !(r.provider == current.provider && r.model == current.model)
                    && self.cooldown_left(r.provider, &r.model).is_none()
            })
    }

    fn model_label(&self, r: &ModelRef) -> String {
        r.model.clone()
    }

    /// Take (and clear) the "config changed, please persist" flag.
    pub fn take_dirty(&mut self) -> bool {
        std::mem::take(&mut self.dirty)
    }
}

/// Who sent a chat message.
#[derive(Clone, Copy, PartialEq)]
pub enum Role {
    User,
    Assistant,
}

pub struct ChatMessage {
    pub role: Role,
    pub text: String,
    /// Reasoning pulled from `<think>` tags, shown dimmed above the answer.
    pub thinking: Option<String>,
    /// A failed reply (network/provider error) — rendered in a distinct, muted-red style and
    /// never sent back to the model as conversation history.
    pub error: bool,
    /// Human-readable summary of edits this reply applied to the circuit (the "activity" view).
    pub activity: Vec<String>,
    /// An automatic agent-loop message (the "auto-check" fed back after edits). Sent to the model
    /// as a user turn, but rendered as a subtle step note rather than a chat bubble.
    pub auto: bool,
    /// A status line for the user only (e.g. "Stopped") — rendered as a quiet centered note and
    /// never sent to the model.
    pub notice: bool,
    /// For an auto-check: the one-line result and the facts behind it, shown to the user instead
    /// of the full text sent to the model (which includes instructions to it).
    pub summary: Option<String>,
    pub details: Option<String>,
    /// The connection dropped before this reply finished (offer "Continue").
    pub truncated: bool,
    /// When a failure happened, and which consecutive attempt it was.
    pub when: Option<Instant>,
    pub attempt: u32,
}

impl ChatMessage {
    fn base(role: Role, text: String) -> Self {
        Self {
            role,
            text,
            thinking: None,
            error: false,
            activity: Vec::new(),
            auto: false,
            notice: false,
            summary: None,
            details: None,
            truncated: false,
            when: None,
            attempt: 0,
        }
    }

    fn user(text: String) -> Self {
        Self::base(Role::User, text)
    }

    fn reply(text: String, thinking: Option<String>) -> Self {
        Self {
            thinking,
            ..Self::base(Role::Assistant, text)
        }
    }

    fn failure(text: String) -> Self {
        Self {
            error: true,
            when: Some(Instant::now()),
            attempt: 1,
            ..Self::base(Role::Assistant, text)
        }
    }

    /// An automatic agent-loop feedback turn (the post-edit "auto-check").
    fn auto(text: String) -> Self {
        Self {
            auto: true,
            ..Self::base(Role::User, text)
        }
    }

    fn notice(text: String) -> Self {
        Self {
            notice: true,
            ..Self::base(Role::Assistant, text)
        }
    }
}

/// The selected model: an index into `AiSettings::providers` plus the model id.
#[derive(Clone)]
pub struct ModelRef {
    pub provider: usize,
    pub model: String,
}

/// Per-project assistant state. Each tab keeps its own conversation *and* its own reply
/// channel, so a request started in one tab keeps running — and lands in that tab — even while
/// you work in another.
pub struct AiSession {
    pub messages: Vec<ChatMessage>,
    pub input: String,
    pub selected: Option<ModelRef>,
    /// Replies from chat workers come back here; drained by [`AiSession::poll_chat`].
    chat_tx: Sender<ChatEvent>,
    chat_rx: Receiver<ChatEvent>,
    /// Token of the in-flight request, if any. A reply is accepted only when its token matches
    /// (so a reply for a since-cleared conversation is quietly dropped).
    pending: Option<u64>,
    next_token: u64,
    /// Edit ops parsed from the latest reply, waiting for the app (which owns the circuit) to
    /// apply them. Picked up via [`AiSession::take_pending_ops`].
    pending_ops: Option<Vec<RawOp>>,
    /// Agent mode: when on, the assistant iterates on its own (build → auto-check → fix → …)
    /// until the circuit is right, instead of stopping after one reply. Persists per tab.
    pub agent_enabled: bool,
    /// Whether an autonomous run is currently in flight (drives the Stop button / status line).
    agent_running: bool,
    /// Auto-continue steps taken in the current run. Unbounded: a run lasts as long as the model
    /// keeps making progress — only the user's Stop (or finishing) ends it.
    agent_steps: u32,
    /// Set when the user presses Stop.
    agent_stop: bool,
    /// Cancel flag of the in-flight request (raised by Stop so the worker abandons the stream).
    cancel: Arc<AtomicBool>,
    /// When a transient failure (rate limit, overloaded provider, dropped connection) interrupts
    /// a run, the next attempt is scheduled here instead of ending the run.
    retry_at: Option<Instant>,
    /// Consecutive automatic retries (reset by any successful reply).
    retries: u32,
    /// Why the run is waiting to retry (shown in the status row).
    retry_reason: String,
    /// Live text filter for the model picker (providers can list hundreds of models).
    model_filter: String,
    /// Streaming buffers for the in-flight reply: reasoning and answer accumulated so far. Shown
    /// as a live bubble until the terminal reply lands and replaces them with a parsed message.
    stream_reason: String,
    stream_content: String,
    /// When the in-flight request and the current run started (for the elapsed-time display).
    request_started: Option<Instant>,
    run_started: Option<Instant>,
    /// Times this run asked the model to actually send the commands it described.
    nudges: u32,
    /// Commands in the latest reply that couldn't be read (fed back to the model).
    pending_problems: Vec<String>,
    /// The latest failure was a rate limit on the selected model (try the fallback chain).
    rate_limited: bool,
    /// Consecutive failed attempts at the current step (shown on the error card).
    failures: u32,
}

/// How many of the most recent conversation turns are sent verbatim each step. Older assistant
/// turns are trimmed (the user's own messages are always kept): the system prompt already
/// carries the complete *current* circuit, so a long run never outgrows the model's context
/// window and dies from it.
const HISTORY_TAIL: usize = 16;

/// Consecutive transient failures tolerated before a run gives up (≈ 18 minutes of an
/// unusable provider at the capped backoff). Any successful reply resets the count.
const MAX_CONSECUTIVE_RETRIES: u32 = 20;

/// Backoff before automatic retry `n` (1-based): 5s, 10s, 20s, 40s, then 60s.
fn retry_delay(n: u32) -> Duration {
    Duration::from_secs((5u64 << n.saturating_sub(1).min(4)).min(60))
}

impl Default for AiSession {
    fn default() -> Self {
        let (chat_tx, chat_rx) = std::sync::mpsc::channel();
        Self {
            messages: Vec::new(),
            input: String::new(),
            selected: None,
            chat_tx,
            chat_rx,
            pending: None,
            next_token: 0,
            pending_ops: None,
            agent_enabled: true,
            agent_running: false,
            agent_steps: 0,
            agent_stop: false,
            cancel: Arc::new(AtomicBool::new(false)),
            retry_at: None,
            retries: 0,
            retry_reason: String::new(),
            model_filter: String::new(),
            stream_reason: String::new(),
            stream_content: String::new(),
            request_started: None,
            run_started: None,
            nudges: 0,
            pending_problems: Vec::new(),
            rate_limited: false,
            failures: 0,
        }
    }
}

impl AiSession {
    /// Is a reply currently being awaited?
    pub fn is_pending(&self) -> bool {
        self.pending.is_some()
    }

    /// Drain any chat replies that arrived from worker threads. A successful reply is split into
    /// reasoning / prose / edit-ops; the ops (if any) are queued for the app to apply. Safe to
    /// call on every tab each frame — a request keeps running when its tab is in the background.
    /// Returns true if a reply (success or error) was received this call.
    pub fn poll_chat(&mut self) -> bool {
        let mut got = false;
        while let Ok(event) = self.chat_rx.try_recv() {
            match event {
                ChatEvent::Delta {
                    token,
                    reasoning,
                    content,
                } => {
                    if self.pending == Some(token) {
                        self.stream_reason = reasoning;
                        self.stream_content = content;
                    }
                }
                ChatEvent::Reply {
                    token,
                    result,
                    retryable,
                    rate_limited,
                    truncated,
                } => {
                    if self.pending != Some(token) {
                        continue; // stale reply (stopped, cleared, or superseded) — drop it.
                    }
                    self.pending = None;
                    self.request_started = None;
                    self.stream_reason.clear();
                    self.stream_content.clear();
                    got = true;
                    self.rate_limited = result.is_err() && rate_limited;
                    match result {
                        Ok(raw) => {
                            self.retries = 0;
                            self.failures = 0;
                            let parsed = ai_edit::parse_reply(&raw);
                            let mut msg = ChatMessage::reply(parsed.text, parsed.thinking);
                            msg.truncated = truncated;
                            self.messages.push(msg);
                            self.pending_problems = parsed.problems;
                            if !parsed.ops.is_empty() {
                                self.pending_ops = Some(parsed.ops);
                            }
                        }
                        // A transient failure mid-run (rate limit, overloaded provider, dropped
                        // connection) doesn't end the run: back off and try the same step again.
                        Err(e)
                            if retryable
                                && self.agent_running
                                && !self.agent_stop
                                && self.retries < MAX_CONSECUTIVE_RETRIES =>
                        {
                            self.retries += 1;
                            self.retry_at = Some(Instant::now() + retry_delay(self.retries));
                            self.retry_reason = e;
                        }
                        Err(e) if retryable && self.retries >= MAX_CONSECUTIVE_RETRIES => {
                            self.failures += 1;
                            let mut m = ChatMessage::failure(format!(
                                "{e} \u{2014} still failing after {MAX_CONSECUTIVE_RETRIES} \
                                 automatic retries. Press Retry to resume, or switch model."
                            ));
                            m.attempt = self.failures;
                            self.messages.push(m);
                        }
                        Err(e) => {
                            self.failures += 1;
                            let mut m = ChatMessage::failure(e);
                            m.attempt = self.failures;
                            self.messages.push(m);
                        }
                    }
                }
            }
        }
        got
    }

    /// Commands from the last reply that couldn't be read (reported as skipped).
    pub fn take_problems(&mut self) -> Vec<String> {
        std::mem::take(&mut self.pending_problems)
    }

    /// The latest message the user typed (the request the assistant is working on).
    pub fn last_user_text(&self) -> &str {
        self.messages
            .iter()
            .rev()
            .find(|m| m.role == Role::User && !m.auto)
            .map(|m| m.text.as_str())
            .unwrap_or("")
    }

    /// After a rate limit on the selected model: put it on cooldown and switch to the next model
    /// in the fallback chain, retrying right away. Returns false when no fallback is available
    /// (the run then waits out its normal backoff).
    pub fn try_fallback(&mut self, settings: &mut AiSettings) -> bool {
        if !std::mem::take(&mut self.rate_limited) || self.retry_at.is_none() {
            return false;
        }
        let Some(cur) = self.selected.clone() else {
            return false;
        };
        settings.mark_rate_limited(cur.provider, &cur.model);
        let Some(next) = settings.next_fallback(&cur) else {
            return false;
        };
        self.messages.push(ChatMessage::notice(format!(
            "{} is rate-limited \u{2014} switched to {} (fallback models are set in AI providers).",
            settings.model_label(&cur),
            settings.model_label(&next)
        )));
        self.selected = Some(next);
        self.retries = 0;
        self.retry_at = Some(Instant::now());
        true
    }

    /// The model replied without any commands. If it described changes (or sent commands that
    /// couldn't be read), ask it once or twice to actually send them — a model saying "I'll add
    /// a latch" and changing nothing must not end the run silently. Otherwise the run is done.
    pub fn handle_reply_without_edits(&mut self, settings: &AiSettings, system_prompt: String) {
        let problems = self.take_problems();
        let promised = self.messages.last().is_some_and(|m| {
            m.role == Role::Assistant && !m.error && !m.notice && promises_edit(&m.text)
        });
        if problems.is_empty() && !promised {
            self.end_agent();
            return;
        }
        if self.agent_enabled && !self.agent_stop && self.nudges < 2 {
            self.nudges += 1;
            let (text, summary) = if problems.is_empty() {
                (
                    "Your reply described changes to the circuit but contained no json command \
                     block, so NOTHING changed on the canvas. If you meant to change the circuit, \
                     send the commands now (one ```json block at the end). If no change is \
                     needed, say so in one sentence without commands."
                        .to_string(),
                    "No changes were made \u{2014} asked the assistant to send its commands",
                )
            } else {
                (
                    format!(
                        "Your commands were NOT applied:\n- {}\nNothing changed. Resend the \
                         corrected commands as strict JSON in one ```json block.",
                        problems.join("\n- ")
                    ),
                    "Its commands couldn't be read \u{2014} asked it to resend them",
                )
            };
            let mut m = ChatMessage::auto(text);
            m.summary = Some(summary.to_string());
            m.details = (!problems.is_empty()).then(|| problems.join("\n"));
            self.messages.push(m);
            self.agent_steps += 1;
            self.dispatch(settings, system_prompt);
        } else {
            let mut note = "No changes were made to the circuit.".to_string();
            if !problems.is_empty() {
                note.push_str(&format!(" Unreadable commands: {}", problems.join("; ")));
            }
            self.messages.push(ChatMessage::notice(note));
            self.end_agent();
        }
    }

    /// Elapsed time of the in-flight request and of the whole run.
    pub fn elapsed(&self) -> (Option<Duration>, Option<Duration>) {
        (
            self.request_started.map(|t| t.elapsed()),
            self.run_started
                .filter(|_| self.agent_running)
                .map(|t| t.elapsed()),
        )
    }

    /// Take the edit ops parsed from the last reply (if any) so the app can apply them.
    pub fn take_pending_ops(&mut self) -> Option<Vec<RawOp>> {
        self.pending_ops.take()
    }

    /// Attach the outcome of applying edits to the most recent assistant message (the "activity"
    /// view). Skipped ops are appended as muted notes.
    pub fn attach_edit_result(&mut self, mut activity: Vec<String>, errors: Vec<String>) {
        for e in errors {
            activity.push(format!("Skipped — {e}"));
        }
        if let Some(msg) = self.messages.last_mut() {
            if msg.role == Role::Assistant && !msg.error {
                msg.activity = activity;
            }
        }
    }

    /// Is an autonomous run in progress (for the Stop button / status line)?
    pub fn is_running(&self) -> bool {
        self.agent_running
    }

    /// The current step number in a run (1-based-ish, for the status line).
    pub fn agent_step(&self) -> u32 {
        self.agent_steps
    }

    /// Stop the assistant NOW: cancel the in-flight request (its worker abandons the stream, and
    /// any late reply is dropped as stale), drop a scheduled retry, and end the run. The user is
    /// never left waiting on a slow or stalled provider.
    pub fn request_stop(&mut self) {
        let was_busy = self.pending.is_some() || self.retry_at.is_some();
        self.cancel.store(true, Ordering::Relaxed);
        self.pending = None;
        self.pending_ops = None;
        self.pending_problems.clear();
        self.retry_at = None;
        self.request_started = None;
        self.stream_reason.clear();
        self.stream_content.clear();
        self.agent_stop = true;
        self.agent_running = false;
        if was_busy {
            self.messages.push(ChatMessage::notice(
                "Stopped. Send a message (e.g. \u{201c}continue\u{201d}) to pick up from here."
                    .to_string(),
            ));
        }
    }

    /// After a reply that made edits, should the agent keep iterating on its own? There is no
    /// step cap: a run continues until the model finishes (replies with no edits), the user
    /// presses Stop, or Agent mode is switched off.
    pub fn should_continue(&self) -> bool {
        self.agent_enabled && !self.agent_stop
    }

    /// Feed the post-edit auto-check back to the model and dispatch the next step of the run.
    pub fn continue_agent(
        &mut self,
        settings: &AiSettings,
        system_prompt: String,
        feedback: ai_edit::Observation,
    ) {
        self.agent_steps += 1;
        let mut m = ChatMessage::auto(feedback.text);
        m.summary = Some(feedback.summary);
        m.details = Some(feedback.details);
        self.messages.push(m);
        self.dispatch(settings, system_prompt);
    }

    /// End the current run (the model replied with no further edits, or a non-retryable error
    /// arrived). A run that is merely waiting to retry a transient failure keeps going.
    pub fn end_agent(&mut self) {
        if self.retry_at.is_none() {
            self.agent_running = false;
        }
    }

    /// Is an automatic retry scheduled (the run is backing off after a transient failure)?
    pub fn retry_scheduled(&self) -> bool {
        self.retry_at.is_some()
    }

    /// Has the scheduled retry's backoff elapsed (and nothing else is in flight)?
    pub fn retry_due(&self) -> bool {
        self.pending.is_none() && self.retry_at.is_some_and(|t| Instant::now() >= t)
    }

    /// Re-send the step that failed transiently. The conversation already ends with the turn that
    /// failed (the user's message or the latest auto-check), so it is simply dispatched again.
    pub fn retry_now(&mut self, settings: &AiSettings, system_prompt: String) {
        self.retry_at = None;
        self.dispatch(settings, system_prompt);
    }

    /// Status text while backing off, e.g. "Rate limited (HTTP 429) — retrying in 12s (retry 3)".
    pub fn retry_status(&self) -> Option<String> {
        let at = self.retry_at?;
        let secs = at.saturating_duration_since(Instant::now()).as_secs() + 1;
        Some(format!(
            "{} \u{2014} retrying in {secs}s (retry {})",
            self.retry_reason.trim_end_matches('.'),
            self.retries
        ))
    }

    /// If nothing valid is selected, fall back to the first available model (and recover from a
    /// selection whose provider has since disconnected).
    fn ensure_selection(&mut self, live: &[(usize, &str)], settings: &AiSettings) {
        let valid = self.selected.as_ref().is_some_and(|s| {
            live.iter()
                .any(|(pi, m)| *pi == s.provider && *m == s.model)
        });
        if !valid {
            // Prefer the model picked last time, else the first available one.
            let preferred = settings.preferred.as_ref().and_then(|pref| {
                live.iter().find(|(pi, m)| {
                    settings
                        .providers
                        .get(*pi)
                        .is_some_and(|p| p.kind.id() == pref.provider)
                        && *m == pref.model
                })
            });
            self.selected = preferred.or(live.first()).map(|(pi, m)| ModelRef {
                provider: *pi,
                model: m.to_string(),
            });
        }
    }

    /// Restore a conversation saved in a project file.
    pub fn restore(&mut self, records: &[ChatRecord]) {
        self.messages = records
            .iter()
            .map(|r| {
                let role = if r.role == "user" {
                    Role::User
                } else {
                    Role::Assistant
                };
                let mut m = ChatMessage::base(role, r.text.clone());
                m.thinking = r.thinking.clone();
                m.activity = r.activity.clone();
                m.summary = r.summary.clone();
                m.details = r.details.clone();
                match r.kind.as_str() {
                    "auto" => m.auto = true,
                    "error" => m.error = true,
                    "notice" => m.notice = true,
                    _ => {}
                }
                m
            })
            .collect();
    }

    /// The conversation as saveable records.
    pub fn records(&self) -> Vec<ChatRecord> {
        self.messages
            .iter()
            .map(|m| ChatRecord {
                role: match m.role {
                    Role::User => "user".to_string(),
                    Role::Assistant => "assistant".to_string(),
                },
                text: m.text.clone(),
                thinking: m.thinking.clone(),
                activity: m.activity.clone(),
                kind: if m.auto {
                    "auto"
                } else if m.error {
                    "error"
                } else if m.notice {
                    "notice"
                } else {
                    "message"
                }
                .to_string(),
                summary: m.summary.clone(),
                details: m.details.clone(),
            })
            .collect()
    }

    /// Send the current composer input as a new user turn. `system_prompt` carries the circuit
    /// context assembled by the caller (the circuit lives with the document). Runs off-thread;
    /// the reply arrives via [`AiSession::poll_chat`].
    pub fn send(&mut self, settings: &AiSettings, system_prompt: String) {
        let text = self.input.trim().to_string();
        if text.is_empty() || self.pending.is_some() {
            return;
        }
        self.input.clear();
        self.messages.push(ChatMessage::user(text));
        self.begin_run();
        self.dispatch(settings, system_prompt);
    }

    /// Re-run the conversation after a failed reply (the "Retry" affordance). Drops trailing
    /// error bubbles / notices and resends the existing turns; no new user message is added.
    pub fn resend(&mut self, settings: &AiSettings, system_prompt: String) {
        if self.pending.is_some() {
            return;
        }
        while self.messages.last().is_some_and(|m| m.error || m.notice) {
            self.messages.pop();
        }
        if !self.messages.iter().any(|m| m.role == Role::User) {
            return;
        }
        self.begin_run();
        self.dispatch(settings, system_prompt);
    }

    /// Reset the run state for a new run started by the user.
    fn begin_run(&mut self) {
        self.agent_running = true;
        self.agent_steps = 0;
        self.agent_stop = false;
        self.retries = 0;
        self.retry_at = None;
        self.nudges = 0;
        self.run_started = Some(Instant::now());
    }

    /// Build the request from the current turns + `system_prompt` and fire it off-thread.
    fn dispatch(&mut self, settings: &AiSettings, system_prompt: String) {
        let Some(sel) = self.selected.clone() else {
            self.messages
                .push(ChatMessage::failure("No model selected.".to_string()));
            self.agent_running = false;
            return;
        };
        let Some(provider) = settings.providers.get(sel.provider).filter(|p| p.is_live()) else {
            self.messages.push(ChatMessage::failure(
                "The selected provider isn't connected.".to_string(),
            ));
            self.agent_running = false;
            return;
        };

        let (history, trimmed) = history_turns(&self.messages);
        let mut system = system_prompt;
        if trimmed {
            system.push_str(
                "\n\nNOTE: this is a long run, so earlier steps were trimmed from the \
                 conversation. CURRENT STATE above reflects everything built so far \u{2014} \
                 continue from it.",
            );
        }
        let mut turns = vec![ChatTurn {
            role: "system".to_string(),
            content: system,
        }];
        turns.extend(history);

        let token = self.next_token;
        self.next_token += 1;
        self.pending = Some(token);
        self.request_started = Some(Instant::now());
        self.stream_reason.clear();
        self.stream_content.clear();
        // A fresh cancel flag per request, so Stop aborts exactly the one in flight.
        let cancel = Arc::new(AtomicBool::new(false));
        self.cancel = cancel.clone();
        let req = ChatRequest {
            base_url: provider.base_url.clone(),
            api_key: provider.api_key.clone(),
            model: sel.model.clone(),
            turns,
            cancel,
            log_text: settings.log_prompts,
        };
        ai_net::spawn_chat(self.chat_tx.clone(), token, req);
    }
}

/// Does a reply announce edits ("I'll add…", "Let me wire…")? Used to catch a model that
/// describes changes but forgets to send the commands.
fn promises_edit(text: &str) -> bool {
    let t = text.to_lowercase();
    const VERBS: [&str; 12] = [
        "add", "build", "create", "place", "wire", "connect", "fix", "change", "update", "remove",
        "replace", "rewire",
    ];
    const LEADS: [&str; 9] = [
        "i'll ",
        "i will ",
        "let me ",
        "i'm going to ",
        "i am going to ",
        "now i'll ",
        "let's ",
        "i'll now ",
        "next, i'll ",
    ];
    LEADS
        .iter()
        .any(|lead| VERBS.iter().any(|v| t.contains(&format!("{lead}{v}"))))
        || t.contains("here's the plan")
        || t.contains("here is the plan")
}

/// The conversation turns to send, trimmed so an arbitrarily long run never outgrows the model's
/// context window: every message the USER typed is kept (the goal and any follow-ups); the last
/// [`HISTORY_TAIL`] turns are kept verbatim; of the automatic auto-checks only the newest (older
/// ones describe superseded states). Errors, notices and empty turns are never sent. The system
/// prompt carries the complete *current* circuit, so trimming loses no work — only old chatter.
/// Returns the turns and whether anything was trimmed.
fn history_turns(messages: &[ChatMessage]) -> (Vec<ChatTurn>, bool) {
    let last_auto = messages.iter().rposition(|m| m.auto);
    let tail_start = messages.len().saturating_sub(HISTORY_TAIL);
    let mut trimmed = false;
    let mut history = Vec::new();
    for (i, m) in messages.iter().enumerate() {
        if m.error || m.notice || m.text.trim().is_empty() {
            continue;
        }
        if m.auto && Some(i) != last_auto {
            continue;
        }
        let user_typed = m.role == Role::User && !m.auto;
        if i < tail_start && !user_typed {
            trimmed = true;
            continue;
        }
        let role = match m.role {
            Role::User => "user",
            Role::Assistant => "assistant",
        };
        history.push(ChatTurn {
            role: role.to_string(),
            content: m.text.clone(),
        });
    }
    (history, trimmed)
}

// ===================== rendering =====================

/// What the panel needs the app to do after rendering.
#[derive(Default)]
pub struct PanelResponse {
    /// The user pressed Send on a ready model; dispatch a chat request.
    pub send: bool,
    /// The user pressed Retry on a failed reply; re-run the last request.
    pub retry: bool,
    /// Open the provider settings.
    pub open_settings: bool,
    /// The user pressed Continue on a reply cut off by a dropped connection.
    pub continue_reply: bool,
}

/// "12s", "3m 05s".
fn fmt_elapsed(d: Duration) -> String {
    let s = d.as_secs();
    if s < 60 {
        format!("{s}s")
    } else {
        format!("{}m {:02}s", s / 60, s % 60)
    }
}

/// "just now", "2 min ago", "1 h ago".
fn fmt_ago(t: Instant) -> String {
    let s = t.elapsed().as_secs();
    if s < 30 {
        "just now".to_string()
    } else if s < 3600 {
        format!("{} min ago", (s / 60).max(1))
    } else {
        format!("{} h ago", s / 3600)
    }
}

/// Starter prompts shown on an empty conversation.
const SUGGESTIONS: [(&str, &str); 4] = [
    ("Build a 4-bit adder", "Build a 4-bit ripple-carry adder from full-adder chips, with switches for A and B and LEDs for the sum and carry. Verify it."),
    ("D flip-flop + test", "Build an edge-triggered D flip-flop as a chip, then place one with a D switch, a CLK button and a Q LED, and test that it only changes on the clock."),
    ("2-bit counter", "Build a 2-bit binary counter driven by a 2 Hz clock, with LEDs for each bit."),
    ("Explain this circuit", "Explain what the circuit on the canvas does, block by block, and point out anything that looks wrong."),
];

/// Render the assistant inside the right sidebar.
pub fn panel(
    ui: &mut egui::Ui,
    t: &Theme,
    settings: &mut AiSettings,
    session: &mut AiSession,
) -> PanelResponse {
    let mut response = PanelResponse::default();

    // Header: agent mode + actions.
    ui.horizontal(|ui| {
        let running = session.is_running() || session.is_pending();
        let (color, label) = if session.retry_scheduled() {
            (t.warning, "Waiting to retry".to_string())
        } else if running {
            match session.elapsed() {
                (_, Some(run)) => (t.accent, format!("Working \u{b7} {}", fmt_elapsed(run))),
                (Some(req), None) => (t.accent, format!("Working \u{b7} {}", fmt_elapsed(req))),
                _ => (t.accent, "Working".to_string()),
            }
        } else if settings.any_live() {
            (t.success, "Ready".to_string())
        } else {
            (t.text_faint, "Not connected".to_string())
        };
        design::pill(ui, t, color, &label);
        if running {
            ui.ctx()
                .request_repaint_after(std::time::Duration::from_millis(500));
        }
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            if design::icon_button(ui, t, icons::SETTINGS_3, "Providers & API keys", false)
                .clicked()
            {
                response.open_settings = true;
            }
            let can_clear = !session.messages.is_empty() && !running && !session.retry_scheduled();
            if design::icon_button_sized(
                ui,
                t,
                icons::DELETE_BIN_6,
                "Clear conversation",
                false,
                30.0,
                can_clear,
            )
            .clicked()
                && can_clear
            {
                session.messages.clear();
            }
        });
    });
    ui.add_space(6.0);

    // Reserve room for the composer at the bottom, give the rest to the conversation.
    let composer_h = 128.0;
    let chat_h = (ui.available_height() - composer_h).max(140.0);
    let width = ui.available_width();
    ui.allocate_ui(egui::vec2(width, chat_h), |ui| {
        if !settings.any_live() {
            disconnected_state(ui, t, &mut response);
        } else if session.messages.is_empty() && !session.is_pending() {
            if let Some(prompt) = welcome(ui, t) {
                session.input = prompt.to_string();
                response.send = true;
            }
        } else {
            let (retry, cont) = conversation(ui, t, session);
            response.retry |= retry;
            response.continue_reply |= cont;
        }
    });
    ui.add_space(6.0);
    response.send |= composer(ui, t, settings, session);
    response
}

/// Shown before any provider is connected.
fn disconnected_state(ui: &mut egui::Ui, t: &Theme, response: &mut PanelResponse) {
    ui.vertical_centered(|ui| {
        ui.add_space(36.0);
        icon_tile(ui, t, icons::PLUG, 44.0);
        ui.add_space(12.0);
        ui.label(
            RichText::new("Connect a provider")
                .font(design::semibold(16.0))
                .color(t.text),
        );
        ui.add_space(4.0);
        ui.label(
            RichText::new(
                "Add a free provider (Groq, Cerebras, Gemini, …) or a local server, and the \
                 assistant can build, wire, and test circuits for you.",
            )
            .font(design::regular(12.5))
            .color(t.text_muted),
        );
        ui.add_space(14.0);
        if design::button(ui, t, Kind::Primary, Some(icons::KEY_2), "Set up providers").clicked() {
            response.open_settings = true;
        }
    });
}

/// Empty conversation: a greeting and starter prompts. Returns a picked prompt.
fn welcome(ui: &mut egui::Ui, t: &Theme) -> Option<&'static str> {
    let mut picked = None;
    egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
        ui.vertical_centered(|ui| {
            ui.add_space(24.0);
            icon_tile(ui, t, icons::SPARKLING_2_FILL, 44.0);
            ui.add_space(12.0);
            ui.label(RichText::new("What should we build?").font(design::semibold(16.0)).color(t.text));
            ui.add_space(4.0);
            ui.label(
                RichText::new("Describe a circuit. In Agent mode the assistant builds it, checks the real truth table, and fixes it until it works.")
                    .font(design::regular(12.5))
                    .color(t.text_muted),
            );
        });
        ui.add_space(14.0);
        for (title, prompt) in SUGGESTIONS {
            let w = ui.available_width();
            let (rect, resp) = ui.allocate_exact_size(egui::vec2(w, 34.0), Sense::click());
            let p = ui.painter();
            let hovered = resp.hovered();
            p.rect_filled(rect, CornerRadius::same(design::radius::MD), if hovered { t.surface_hover } else { t.surface });
            p.rect_stroke(rect, CornerRadius::same(design::radius::MD), Stroke::new(1.0, if hovered { t.border_strong } else { t.border }), StrokeKind::Inside);
            p.text(Pos2::new(rect.left() + 14.0, rect.center().y), Align2::CENTER_CENTER, icons::MAGIC, design::icon_font(14.0), t.accent);
            p.text(Pos2::new(rect.left() + 28.0, rect.center().y), Align2::LEFT_CENTER, title, design::medium(12.5), t.text);
            p.text(Pos2::new(rect.right() - 12.0, rect.center().y), Align2::RIGHT_CENTER, icons::ARROW_RIGHT_S, design::icon_font(14.0), t.text_faint);
            if hovered {
                ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
            }
            if resp.on_hover_text(prompt).clicked() {
                picked = Some(prompt);
            }
            ui.add_space(2.0);
        }
    });
    picked
}

fn icon_tile(ui: &mut egui::Ui, t: &Theme, glyph: &str, size: f32) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(size, size), Sense::hover());
    ui.painter()
        .rect_filled(rect, CornerRadius::same(design::radius::XL), t.accent_soft);
    ui.painter().rect_stroke(
        rect,
        CornerRadius::same(design::radius::XL),
        Stroke::new(1.0, t.accent.gamma_multiply(0.4)),
        StrokeKind::Inside,
    );
    design::paint_icon(ui, rect, glyph, size * 0.48, t.accent);
}

/// The scrolling message list (plus a live/status row while a reply is in flight). Returns
/// whether "Retry" and "Continue" were clicked. When the user has scrolled up and something new
/// arrives, a "New messages" pill offers a jump back down.
fn conversation(ui: &mut egui::Ui, t: &Theme, session: &AiSession) -> (bool, bool) {
    let row_width = ui.available_width().max(160.0);
    let mut retry = false;
    let mut cont = false;
    let jump_id = ui.id().with("chat-jump");
    let seen_id = ui.id().with("chat-seen");
    let jump = ui
        .ctx()
        .data_mut(|d| d.remove_temp::<bool>(jump_id).unwrap_or(false));
    // A cheap "how much content is there" signature: messages plus live stream length.
    let progress = session.messages.len() * 100_000
        + session.stream_content.len()
        + session.stream_reason.len();
    let mut area = egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .stick_to_bottom(true);
    if jump {
        area = area.vertical_scroll_offset(f32::MAX);
    }
    let out = area.show(ui, |ui| {
        ui.set_width(row_width - 6.0);
        ui.add_space(2.0);
        for (i, msg) in session.messages.iter().enumerate() {
            if msg.auto {
                auto_note(ui, t, msg, i);
            } else if msg.notice {
                notice(ui, t, &msg.text);
            } else if msg.role == Role::User {
                user_bubble(ui, t, &msg.text, row_width);
            } else {
                let (r, c) = assistant_message(ui, t, msg, i);
                retry |= r;
                cont |= c && i + 1 == session.messages.len() && !session.is_pending();
            }
            ui.add_space(10.0);
        }
        if let Some(status) = session.retry_status().filter(|_| !session.is_pending()) {
            status_row(ui, t, icons::TIMER, &status);
            ui.ctx()
                .request_repaint_after(std::time::Duration::from_millis(250));
        } else if session.is_pending() {
            if session.stream_reason.is_empty() && session.stream_content.is_empty() {
                thinking(ui, t, session);
            } else {
                stream_message(
                    ui,
                    t,
                    session,
                    &session.stream_reason,
                    &session.stream_content,
                );
            }
        }
        ui.add_space(4.0);
    });
    let max_offset = (out.content_size.y - out.inner_rect.height()).max(0.0);
    let at_bottom = out.state.offset.y >= max_offset - 24.0;
    let seen = ui.ctx().data_mut(|d| *d.get_temp_mut_or(seen_id, progress));
    if at_bottom || jump {
        ui.ctx().data_mut(|d| d.insert_temp(seen_id, progress));
    } else if progress != seen {
        // New content below the fold: offer to jump down.
        let pill = egui::Rect::from_center_size(
            Pos2::new(out.inner_rect.center().x, out.inner_rect.bottom() - 22.0),
            egui::vec2(132.0, 28.0),
        );
        let resp = ui.interact(pill, ui.id().with("chat-jump-pill"), Sense::click());
        let p = ui.painter();
        p.rect_filled(
            pill.translate(egui::vec2(0.0, 2.0)),
            CornerRadius::same(14),
            Color32::from_black_alpha(40),
        );
        design::vertical_gradient(p, pill, CornerRadius::same(14), t.accent, t.accent_strong);
        p.text(
            pill.center(),
            Align2::CENTER_CENTER,
            format!("{}  New messages", icons::ARROW_DOWN_S),
            design::medium(12.0),
            Color32::WHITE,
        );
        if resp
            .on_hover_cursor(egui::CursorIcon::PointingHand)
            .clicked()
        {
            ui.ctx().data_mut(|d| d.insert_temp(jump_id, true));
        }
    }
    (retry, cont)
}

fn avatar_row(ui: &mut egui::Ui, t: &Theme, live: bool) {
    ui.horizontal(|ui| {
        let (rect, _) = ui.allocate_exact_size(egui::vec2(20.0, 20.0), Sense::hover());
        design::vertical_gradient(
            ui.painter(),
            rect,
            CornerRadius::same(6),
            t.accent,
            t.accent_strong,
        );
        design::paint_icon(ui, rect, icons::SPARKLING_2_FILL, 11.0, Color32::WHITE);
        ui.label(
            RichText::new("Assistant")
                .font(design::semibold(12.5))
                .color(t.text),
        );
        if live {
            let tm = ui.input(|i| i.time) as f32;
            let (r, _) = ui.allocate_exact_size(egui::vec2(26.0, 12.0), Sense::hover());
            for k in 0..3 {
                let a = ((tm * 4.0 - k as f32 * 0.7).sin() * 0.5 + 0.5).clamp(0.15, 1.0);
                ui.painter().circle_filled(
                    Pos2::new(r.left() + 5.0 + k as f32 * 7.0, r.center().y),
                    2.3,
                    t.accent.gamma_multiply(a),
                );
            }
            ui.ctx()
                .request_repaint_after(std::time::Duration::from_millis(60));
        }
    });
}

fn user_bubble(ui: &mut egui::Ui, t: &Theme, text: &str, row_width: f32) {
    ui.with_layout(Layout::right_to_left(Align::TOP), |ui| {
        let max_w = (row_width * 0.85).max(140.0);
        Frame::NONE
            .fill(t.accent_soft)
            .stroke(Stroke::new(1.0, t.accent.gamma_multiply(0.35)))
            .corner_radius(CornerRadius {
                nw: 12,
                ne: 4,
                sw: 12,
                se: 12,
            })
            .inner_margin(Margin::symmetric(12, 8))
            .show(ui, |ui| {
                ui.set_max_width(max_w);
                ui.label(
                    RichText::new(text)
                        .font(design::regular(13.0))
                        .color(t.text),
                );
            });
    });
}

/// An assistant reply: avatar, optional reasoning, answer, applied-edits card, retry on error.
/// Returns whether "Retry" / "Continue" were clicked.
fn assistant_message(ui: &mut egui::Ui, t: &Theme, msg: &ChatMessage, idx: usize) -> (bool, bool) {
    let mut retry = false;
    let mut cont = false;
    if msg.error {
        Frame::NONE
            .fill(t.danger_soft)
            .stroke(Stroke::new(1.0, t.danger.gamma_multiply(0.45)))
            .corner_radius(CornerRadius::same(design::radius::LG))
            .inner_margin(Margin::same(10))
            .show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new(icons::ERROR_WARNING)
                            .font(design::icon_font(15.0))
                            .color(t.danger),
                    );
                    ui.label(
                        RichText::new("Request failed")
                            .font(design::semibold(12.5))
                            .color(t.text),
                    );
                    // When, and which attempt — so a repeated failure doesn't look identical.
                    let mut meta = Vec::new();
                    if msg.attempt > 1 {
                        meta.push(format!("attempt {}", msg.attempt));
                    }
                    if let Some(w) = msg.when {
                        meta.push(fmt_ago(w));
                    }
                    if !meta.is_empty() {
                        ui.label(
                            RichText::new(meta.join(" \u{b7} "))
                                .font(design::regular(11.5))
                                .color(t.text_faint),
                        );
                    }
                });
                ui.label(
                    RichText::new(&msg.text)
                        .font(design::regular(12.5))
                        .color(t.text),
                );
                ui.add_space(4.0);
                if design::button_ex(
                    ui,
                    t,
                    Kind::Secondary,
                    Some(icons::REFRESH),
                    "Retry",
                    true,
                    28.0,
                )
                .clicked()
                {
                    retry = true;
                }
            });
        return (retry, false);
    }
    avatar_row(ui, t, false);
    ui.indent(("ai-msg", idx), |ui| {
        if let Some(reasoning) = &msg.thinking {
            egui::CollapsingHeader::new(
                RichText::new(format!("{}  Reasoning", icons::BRAIN))
                    .font(design::medium(12.0))
                    .color(t.text_faint),
            )
            .id_salt(("ai-reasoning", idx))
            .default_open(false)
            .show(ui, |ui| {
                ui.label(
                    RichText::new(reasoning)
                        .font(design::regular(12.0))
                        .italics()
                        .color(t.text_faint),
                );
            });
        }
        if !msg.text.trim().is_empty() {
            markdown(ui, t, &msg.text);
        }
        activity_view(ui, t, &msg.activity, idx);
        if msg.truncated {
            ui.add_space(4.0);
            ui.horizontal_wrapped(|ui| {
                ui.label(
                    RichText::new(format!(
                        "{}  The connection dropped before this reply finished.",
                        icons::ERROR_WARNING
                    ))
                    .font(design::regular(12.0))
                    .color(t.warning),
                );
                if design::button_ex(
                    ui,
                    t,
                    Kind::Secondary,
                    Some(icons::PLAY_FILL),
                    "Continue",
                    true,
                    26.0,
                )
                .on_hover_text("Ask the assistant to pick up where it stopped")
                .clicked()
                {
                    cont = true;
                }
            });
        }
    });
    (retry, cont)
}

/// Render the small subset of Markdown models actually use: headings, bullet/numbered lists,
/// fenced code blocks, **bold**, and `inline code`. Everything else is plain text.
fn markdown(ui: &mut egui::Ui, t: &Theme, text: &str) {
    let mut in_code = false;
    let mut code = String::new();
    let mut table: Vec<Vec<String>> = Vec::new();
    ui.spacing_mut().item_spacing.y = 4.0;
    for line in text.lines() {
        let trimmed = line.trim_start();
        // Pipe tables: collect rows, render when the table ends.
        if !in_code && trimmed.starts_with('|') {
            let cells: Vec<String> = trimmed
                .trim()
                .trim_matches('|')
                .split('|')
                .map(|c| c.trim().to_string())
                .collect();
            let separator = cells
                .iter()
                .all(|c| !c.is_empty() && c.chars().all(|ch| matches!(ch, '-' | ':' | ' ')));
            if !separator {
                table.push(cells);
            }
            continue;
        }
        if !table.is_empty() {
            md_table(ui, t, &std::mem::take(&mut table));
        }
        if trimmed.starts_with("```") {
            if in_code {
                code_block(ui, t, code.trim_end());
                code.clear();
            }
            in_code = !in_code;
            continue;
        }
        if in_code {
            code.push_str(line);
            code.push('\n');
            continue;
        }
        if trimmed.is_empty() {
            ui.add_space(2.0);
            continue;
        }
        let (size, font_semibold, body, bullet) = if let Some(h) = trimmed.strip_prefix("### ") {
            (13.0, true, h, None)
        } else if let Some(h) = trimmed.strip_prefix("## ") {
            (14.0, true, h, None)
        } else if let Some(h) = trimmed.strip_prefix("# ") {
            (15.0, true, h, None)
        } else if let Some(b) = trimmed
            .strip_prefix("- ")
            .or_else(|| trimmed.strip_prefix("* "))
            .or_else(|| trimmed.strip_prefix("\u{2022} "))
        {
            (13.0, false, b, Some("\u{2022}".to_string()))
        } else if let Some((num, rest)) = trimmed
            .split_once(". ")
            .filter(|(n, _)| !n.is_empty() && n.len() <= 3 && n.chars().all(|c| c.is_ascii_digit()))
        {
            (13.0, false, rest, Some(format!("{num}.")))
        } else {
            (13.0, false, trimmed, None)
        };
        let job = inline_job(t, body, size, font_semibold);
        match bullet {
            Some(b) => {
                ui.horizontal_wrapped(|ui| {
                    ui.add_space(4.0);
                    ui.label(
                        RichText::new(b)
                            .font(design::medium(13.0))
                            .color(t.text_muted),
                    );
                    ui.add(egui::Label::new(job).wrap());
                });
            }
            None => {
                ui.add(egui::Label::new(job).wrap());
            }
        }
    }
    if !table.is_empty() {
        md_table(ui, t, &table);
    }
    if in_code && !code.is_empty() {
        code_block(ui, t, code.trim_end());
    }
}

/// A Markdown pipe table: bold header row, zebra rows, in a bordered card that scrolls
/// sideways when wide.
fn md_table(ui: &mut egui::Ui, t: &Theme, rows: &[Vec<String>]) {
    let cols = rows.iter().map(Vec::len).max().unwrap_or(0);
    if cols == 0 {
        return;
    }
    Frame::NONE
        .stroke(Stroke::new(1.0, t.border))
        .corner_radius(CornerRadius::same(design::radius::MD))
        .inner_margin(Margin::symmetric(8, 6))
        .show(ui, |ui| {
            egui::ScrollArea::horizontal()
                .id_salt(ui.next_auto_id())
                .auto_shrink([false, true])
                .show(ui, |ui| {
                    egui::Grid::new(ui.next_auto_id())
                        .striped(true)
                        .spacing(egui::vec2(14.0, 5.0))
                        .show(ui, |ui| {
                            for (r, row) in rows.iter().enumerate() {
                                for c in 0..cols {
                                    let cell = row.get(c).map(String::as_str).unwrap_or("");
                                    ui.add(egui::Label::new(inline_job(t, cell, 12.5, r == 0)));
                                }
                                ui.end_row();
                            }
                        });
                });
        });
}

fn code_block(ui: &mut egui::Ui, t: &Theme, code: &str) {
    Frame::NONE
        .fill(t.bg_canvas)
        .stroke(Stroke::new(1.0, t.border))
        .corner_radius(CornerRadius::same(design::radius::MD))
        .inner_margin(Margin::same(8))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.label(
                RichText::new(code)
                    .font(design::mono(11.5))
                    .color(t.text_muted),
            );
        });
}

/// One line with **bold** and `code` spans.
fn inline_job(t: &Theme, text: &str, size: f32, semibold: bool) -> egui::text::LayoutJob {
    use egui::text::{LayoutJob, TextFormat};
    let mut job = LayoutJob::default();
    let base = |bold: bool| TextFormat {
        font_id: if bold || semibold {
            design::semibold(size)
        } else {
            design::regular(size)
        },
        color: t.text,
        ..Default::default()
    };
    let mut bold = false;
    let mut rest = text;
    while !rest.is_empty() {
        let next_bold = rest.find("**");
        let next_code = rest.find('`');
        match (next_bold, next_code) {
            (Some(b), c) if c.is_none_or_else(|c| c > b) => {
                job.append(&rest[..b], 0.0, base(bold));
                bold = !bold;
                rest = &rest[b + 2..];
            }
            (_, Some(c)) => {
                job.append(&rest[..c], 0.0, base(bold));
                let after = &rest[c + 1..];
                match after.find('`') {
                    Some(end) => {
                        job.append(
                            &after[..end],
                            0.0,
                            TextFormat {
                                font_id: design::mono(size - 1.0),
                                color: t.accent,
                                background: t.surface_active,
                                ..Default::default()
                            },
                        );
                        rest = &after[end + 1..];
                    }
                    None => {
                        job.append(&rest[c..], 0.0, base(bold));
                        rest = "";
                    }
                }
            }
            _ => {
                job.append(rest, 0.0, base(bold));
                rest = "";
            }
        }
    }
    job
}

/// `opt.is_none_or_else(f)`: true when `None`, else `f(value)` (MSRV-friendly `is_none_or`).
trait NoneOrElse {
    fn is_none_or_else(&self, f: impl FnOnce(usize) -> bool) -> bool;
}

impl NoneOrElse for Option<usize> {
    fn is_none_or_else(&self, f: impl FnOnce(usize) -> bool) -> bool {
        match self {
            None => true,
            Some(v) => f(*v),
        }
    }
}

/// A live assistant message rendered from the streaming buffers.
fn stream_message(
    ui: &mut egui::Ui,
    t: &Theme,
    session: &AiSession,
    reason_field: &str,
    content_field: &str,
) {
    let live = ai_edit::live_split(reason_field, content_field);
    let elapsed = session
        .elapsed()
        .0
        .map(|d| format!(" {}", fmt_elapsed(d)))
        .unwrap_or_default();
    avatar_row(ui, t, true);
    ui.indent("ai-live", |ui| {
        if let Some(r) = &live.reasoning {
            let tail: String = {
                let chars: Vec<char> = r.chars().collect();
                let n = chars.len();
                chars[n.saturating_sub(600)..].iter().collect()
            };
            Frame::NONE
                .fill(t.surface)
                .corner_radius(CornerRadius::same(design::radius::MD))
                .inner_margin(Margin::same(8))
                .show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    ui.label(
                        RichText::new(format!("{}  Thinking\u{2026}{elapsed}", icons::BRAIN))
                            .font(design::medium(11.5))
                            .color(t.text_faint),
                    );
                    ui.label(
                        RichText::new(tail)
                            .font(design::regular(12.0))
                            .italics()
                            .color(t.text_faint),
                    );
                });
        }
        if !live.answer.is_empty() {
            markdown(ui, t, &live.answer);
        }
        if let Some(cmds) = &live.commands {
            let ops = cmds.matches("\"op\"").count();
            let title = if ops > 0 {
                format!(
                    "{}  Writing commands\u{2026} ({ops} so far)",
                    icons::TERMINAL_BOX
                )
            } else {
                format!("{}  Writing commands\u{2026}", icons::TERMINAL_BOX)
            };
            egui::CollapsingHeader::new(
                RichText::new(title)
                    .font(design::medium(12.0))
                    .color(t.accent),
            )
            .id_salt("ai-live-commands")
            .default_open(false)
            .show(ui, |ui| {
                ui.label(
                    RichText::new(cmds)
                        .font(design::mono(11.0))
                        .color(t.text_muted),
                );
            });
        }
    });
    ui.ctx()
        .request_repaint_after(std::time::Duration::from_millis(60));
}

/// A compact, foldable "auto-check" row for an agent feedback turn: its one-line result, and
/// the facts behind it when expanded (never the instructions sent to the model).
fn auto_note(ui: &mut egui::Ui, t: &Theme, msg: &ChatMessage, idx: usize) {
    let title = msg
        .summary
        .clone()
        .unwrap_or_else(|| "Checked the circuit".to_string());
    let warn = title.contains("skipped")
        || title.contains("unwired")
        || title.contains("oscillates")
        || title.starts_with("No changes")
        || title.contains("couldn't")
        || title
            .split("test checks ")
            .nth(1)
            .and_then(|r| r.split(' ').next())
            .and_then(|f| f.split_once('/'))
            .is_some_and(|(a, b)| a != b);
    let details = msg.details.as_deref().unwrap_or(&msg.text);
    egui::CollapsingHeader::new(
        RichText::new(format!(
            "{}  {title}",
            if warn {
                icons::ERROR_WARNING
            } else {
                icons::TEST_TUBE
            }
        ))
        .font(design::medium(12.0))
        .color(if warn { t.warning } else { t.text_faint }),
    )
    .id_salt(("ai-autocheck", idx))
    .default_open(false)
    .show(ui, |ui| {
        if details.trim().is_empty() {
            return;
        }
        Frame::NONE
            .fill(t.surface)
            .corner_radius(CornerRadius::same(design::radius::MD))
            .inner_margin(Margin::same(8))
            .show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.label(
                    RichText::new(details)
                        .font(design::mono(11.0))
                        .color(t.text_muted),
                );
            });
    });
}

/// "Thinking… 12s" / "Step 3 — reading the auto-check… 8s · 1m 20s total" while awaiting the
/// first token.
fn thinking(ui: &mut egui::Ui, t: &Theme, session: &AiSession) {
    avatar_row(ui, t, true);
    let (req, run) = session.elapsed();
    let secs = req
        .map(|d| format!(" {}", fmt_elapsed(d)))
        .unwrap_or_default();
    let label = if session.is_running() && session.agent_step() > 0 {
        let total = run
            .map(|d| format!(" \u{b7} {} total", fmt_elapsed(d)))
            .unwrap_or_default();
        format!(
            "Step {} \u{2014} reading the auto-check\u{2026}{secs}{total}",
            session.agent_step() + 1
        )
    } else {
        format!("Thinking\u{2026}{secs}")
    };
    ui.indent("ai-thinking", |ui| {
        ui.label(
            RichText::new(label)
                .font(design::regular(12.5))
                .italics()
                .color(t.text_faint),
        );
    });
}

fn status_row(ui: &mut egui::Ui, t: &Theme, glyph: &str, text: &str) {
    ui.horizontal_wrapped(|ui| {
        ui.label(
            RichText::new(glyph)
                .font(design::icon_font(14.0))
                .color(t.warning),
        );
        ui.label(
            RichText::new(text)
                .font(design::regular(12.5))
                .color(t.text_muted),
        );
    });
}

/// A system notice (e.g. "Stopped.") — a quiet centered line.
fn notice(ui: &mut egui::Ui, t: &Theme, text: &str) {
    ui.vertical_centered(|ui| {
        ui.label(
            RichText::new(text)
                .font(design::regular(11.5))
                .italics()
                .color(t.text_faint),
        );
    });
}

/// The edits a reply applied: a compact card, folded when long.
fn activity_view(ui: &mut egui::Ui, t: &Theme, activity: &[String], idx: usize) {
    if activity.is_empty() {
        return;
    }
    ui.add_space(4.0);
    let read_only = |a: &str| a.starts_with("Ran a ") || a.starts_with("Pointed at");
    let applied = activity
        .iter()
        .filter(|a| !a.starts_with("Skipped") && !read_only(a))
        .count();
    let skipped = activity.iter().filter(|a| a.starts_with("Skipped")).count();
    Frame::NONE
        .fill(t.surface)
        .stroke(Stroke::new(1.0, t.border))
        .corner_radius(CornerRadius::same(design::radius::MD))
        .inner_margin(Margin::symmetric(10, 6))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            let mut title = if applied == 0 {
                format!("{}  No changes", icons::CHECKBOX_CIRCLE_FILL)
            } else {
                format!(
                    "{}  Applied {applied} change{}",
                    icons::CHECKBOX_CIRCLE_FILL,
                    if applied == 1 { "" } else { "s" }
                )
            };
            if skipped > 0 {
                title.push_str(&format!(" \u{b7} {skipped} skipped"));
            }
            let open_default = activity.len() <= 6;
            egui::CollapsingHeader::new(
                RichText::new(title)
                    .font(design::medium(12.0))
                    .color(if skipped > 0 { t.warning } else { t.success }),
            )
            .id_salt(("ai-activity", idx))
            .default_open(open_default)
            .show(ui, |ui| {
                for line in activity {
                    let skipped = line.starts_with("Skipped");
                    ui.label(
                        RichText::new(line)
                            .font(design::regular(11.5))
                            .color(if skipped { t.danger } else { t.text_muted }),
                    );
                }
            });
            if applied > 0 {
                ui.label(
                    RichText::new("Ctrl+Z undoes this whole step")
                        .font(design::regular(11.0))
                        .color(t.text_faint),
                );
            }
        });
}

/// The composer: input card, model switcher, agent toggle, send/stop.
fn composer(
    ui: &mut egui::Ui,
    t: &Theme,
    settings: &mut AiSettings,
    session: &mut AiSession,
) -> bool {
    let live = settings.live_models();
    let ready = !live.is_empty();
    session.ensure_selection(&live, settings);
    let pending = session.is_pending();
    let mut send = false;

    let id = ui.id().with("composer-input");
    let focused = ui.ctx().memory(|m| m.has_focus(id));
    Frame::NONE
        .fill(t.surface)
        .stroke(Stroke::new(
            1.0,
            if focused { t.accent } else { t.border_strong },
        ))
        .corner_radius(CornerRadius::same(design::radius::XL))
        .inner_margin(Margin {
            left: 12,
            right: 8,
            top: 8,
            bottom: 6,
        })
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            let hint = if !ready {
                "Connect a provider to start\u{2026}"
            } else if pending {
                "Working\u{2026} you can type your next message"
            } else {
                "Describe a circuit to build or change\u{2026}"
            };
            // Enter sends; Shift+Enter inserts a newline. The box stays editable while a reply is
            // pending (disabling it would drop focus and send keystrokes to the canvas).
            let enter = ui.input(|i| i.key_pressed(egui::Key::Enter) && !i.modifiers.shift);
            let editor = ui.add_enabled(
                ready,
                egui::TextEdit::multiline(&mut session.input)
                    .id(id)
                    .frame(Frame::NONE)
                    .desired_rows(2)
                    .hint_text(RichText::new(hint).color(t.text_faint))
                    .desired_width(f32::INFINITY),
            );
            if enter && editor.has_focus() && !session.input.trim().is_empty() {
                send = true;
            }
            ui.horizontal(|ui| {
                model_switcher(ui, t, settings, session);
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    if session.is_running() || pending {
                        let r = send_button(ui, t, icons::STOP_FILL, true)
                            .on_hover_text("Stop the assistant now");
                        if r.clicked() {
                            session.request_stop();
                        }
                    } else {
                        let can_send = ready && !session.input.trim().is_empty();
                        if send_button(ui, t, icons::ARROW_UP, can_send)
                            .on_hover_text("Send (Enter)")
                            .clicked()
                            && can_send
                        {
                            send = true;
                        }
                    }
                    ui.add_space(4.0);
                    agent_toggle(ui, t, &mut session.agent_enabled);
                });
            });
        });
    if send {
        while session.input.ends_with('\n') {
            session.input.pop();
        }
    }
    send && ready && !pending && !session.input.trim().is_empty()
}

fn send_button(ui: &mut egui::Ui, t: &Theme, glyph: &str, enabled: bool) -> egui::Response {
    let (rect, resp) = ui.allocate_exact_size(
        egui::vec2(30.0, 30.0),
        if enabled {
            Sense::click()
        } else {
            Sense::hover()
        },
    );
    let p = ui.painter();
    if enabled {
        design::vertical_gradient(p, rect, CornerRadius::same(15), t.accent, t.accent_strong);
        if resp.hovered() {
            p.circle_filled(rect.center(), 15.0, Color32::from_white_alpha(18));
        }
    } else {
        p.circle_filled(rect.center(), 15.0, t.surface_active);
    }
    design::paint_icon(
        ui,
        rect,
        glyph,
        15.0,
        if enabled {
            Color32::WHITE
        } else {
            t.text_faint
        },
    );
    resp
}

/// A compact "Agent" toggle: when on, the assistant iterates (build → self-check → fix).
fn agent_toggle(ui: &mut egui::Ui, t: &Theme, on: &mut bool) {
    let text = format!("{}  Auto-check", icons::ROBOT_2);
    let font = design::medium(12.0);
    let galley = ui
        .painter()
        .layout_no_wrap(text.clone(), font.clone(), t.text);
    let (rect, resp) =
        ui.allocate_exact_size(egui::vec2(galley.size().x + 18.0, 26.0), Sense::click());
    let p = ui.painter();
    let rad = CornerRadius::same(13);
    if *on {
        p.rect_filled(rect, rad, t.accent_soft);
        p.rect_stroke(
            rect,
            rad,
            Stroke::new(1.0, t.accent.gamma_multiply(0.6)),
            StrokeKind::Inside,
        );
    } else {
        p.rect_stroke(
            rect,
            rad,
            Stroke::new(1.0, t.border_strong),
            StrokeKind::Inside,
        );
    }
    p.text(
        rect.center(),
        Align2::CENTER_CENTER,
        text,
        font,
        if *on { t.accent } else { t.text_muted },
    );
    if resp
        .on_hover_text(
            "On: after every change the assistant tests the real circuit and keeps fixing it until \
             it matches your request \u{2014} for as long as it needs (Stop ends it any time).\n\
             Off: one reply per message. It can still edit the circuit, but doesn't check or \
             iterate on its own.",
        )
        .clicked()
    {
        *on = !*on;
    }
}

/// The in-composer model picker.
fn model_switcher(
    ui: &mut egui::Ui,
    t: &Theme,
    settings: &mut AiSettings,
    session: &mut AiSession,
) {
    let live: Vec<(usize, String)> = settings
        .live_models()
        .into_iter()
        .map(|(i, m)| (i, m.to_string()))
        .collect();
    let multi_provider = {
        let mut seen: Vec<usize> = Vec::new();
        for (pi, _) in &live {
            if !seen.contains(pi) {
                seen.push(*pi);
            }
        }
        seen.len() > 1
    };
    let label = match &session.selected {
        Some(sel) => sel.model.clone(),
        None => "Choose a model".to_string(),
    };
    let font = design::medium(12.0);
    let max_w = (ui.available_width() - 120.0).max(80.0);
    let shown = super::glyphs::ellipsize(ui.painter(), &label, &font, max_w - 34.0);
    let galley = ui
        .painter()
        .layout_no_wrap(shown.clone(), font.clone(), t.text);
    let (rect, button) =
        ui.allocate_exact_size(egui::vec2(galley.size().x + 46.0, 26.0), Sense::click());
    let p = ui.painter();
    if button.hovered()
        || egui::Popup::is_id_open(ui.ctx(), egui::Popup::default_response_id(&button))
    {
        p.rect_filled(rect, CornerRadius::same(13), t.surface_hover);
    }
    p.text(
        Pos2::new(rect.left() + 12.0, rect.center().y),
        Align2::CENTER_CENTER,
        icons::CPU,
        design::icon_font(13.0),
        t.text_muted,
    );
    p.text(
        Pos2::new(rect.left() + 22.0, rect.center().y),
        Align2::LEFT_CENTER,
        shown,
        font,
        t.text_muted,
    );
    p.text(
        Pos2::new(rect.right() - 12.0, rect.center().y),
        Align2::CENTER_CENTER,
        icons::ARROW_DOWN_S,
        design::icon_font(13.0),
        t.text_faint,
    );
    let button = button.on_hover_text(label);
    if button.clicked() {
        session.model_filter.clear();
    }

    egui::Popup::menu(&button)
        .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
        .show(|ui| {
            ui.set_width(320.0);
            if live.len() > 8 {
                design::text_field(
                    ui,
                    t,
                    &mut session.model_filter,
                    "Filter models\u{2026}",
                    Some(icons::SEARCH),
                );
                ui.add_space(4.0);
            }
            let filter = session.model_filter.trim().to_lowercase();
            let mut picked: Option<(usize, String)> = None;
            egui::ScrollArea::vertical()
                .max_height(340.0)
                .auto_shrink([false, true])
                .show(ui, |ui| {
                    let mut shown = 0usize;
                    let mut last_provider = usize::MAX;
                    for (pi, model) in &live {
                        let provider = settings
                            .providers
                            .get(*pi)
                            .map(|p| p.name.as_str())
                            .unwrap_or("");
                        if !filter.is_empty()
                            && !format!("{model} {provider}")
                                .to_lowercase()
                                .contains(&filter)
                        {
                            continue;
                        }
                        if multi_provider && *pi != last_provider {
                            last_provider = *pi;
                            ui.add_space(4.0);
                            design::menu_section(ui, t, provider);
                        }
                        shown += 1;
                        let is_sel = session
                            .selected
                            .as_ref()
                            .is_some_and(|s| s.provider == *pi && s.model == *model);
                        let limited = settings
                            .cooldown_left(*pi, model)
                            .map(|d| format!("limited \u{b7} {}s", d.as_secs() + 1))
                            .unwrap_or_default();
                        if design::menu_item_ex(ui, t, "", model, &limited, true, is_sel) {
                            picked = Some((*pi, model.to_string()));
                        }
                    }
                    if shown == 0 {
                        design::hint(ui, t, "No matching models");
                    }
                });
            if let Some((provider, model)) = picked {
                settings.set_preferred(provider, &model);
                session.selected = Some(ModelRef { provider, model });
                session.model_filter.clear();
                egui::Popup::close_all(ui.ctx());
            }
        });
}

// ===================== settings dialog =====================

/// The provider settings dialog (modal). Set `open` false to close.
pub fn settings_dialog(ctx: &egui::Context, t: &Theme, settings: &mut AiSettings, open: &mut bool) {
    let modal = egui::Modal::new(egui::Id::new("ai-settings"))
        .frame(design::dialog_frame(t))
        .backdrop_color(Color32::from_black_alpha(if t.dark { 140 } else { 70 }))
        .show(ctx, |ui| {
            ui.set_width(520.0);
            design::dialog_header(
                ui,
                t,
                icons::KEY_2,
                t.accent,
                "AI providers",
                "Keys are stored in your OS keychain and never leave this machine.",
            );
            ui.add_space(14.0);
            let mut test_request: Option<usize> = None;
            let mut forget: Option<usize> = None;
            let mut changed = false;
            egui::ScrollArea::vertical()
                .auto_shrink([false, true])
                .max_height(ctx.content_rect().height() * 0.62)
                .show(ui, |ui| {
                    for i in 0..settings.providers.len() {
                        let act = provider_card(ui, t, &mut settings.providers[i]);
                        test_request = test_request.or(act.test.then_some(i));
                        forget = forget.or(act.forget.then_some(i));
                        changed |= act.changed;
                        ui.add_space(8.0);
                    }
                });
            ui.add_space(4.0);
            fallback_section(ui, t, settings);
            ui.add_space(8.0);
            let mut log = settings.log_prompts;
            if design::toggle_row(
                ui,
                t,
                "Log prompts and replies",
                "Keeps request text in the local request log for debugging (timing and errors \
                 are always logged; keys never are).",
                &mut log,
            ) {
                settings.log_prompts = log;
                changed = true;
            }
            if let Some(i) = test_request {
                settings.start_test(i);
            }
            if let Some(i) = forget {
                settings.forget_key(i);
            }
            if changed {
                settings.dirty = true;
            }
            ui.add_space(10.0);
            ui.horizontal(|ui| {
                let full = super::ai_log::log_path_display();
                let short = std::path::Path::new(&full)
                    .file_name()
                    .map(|f| f.to_string_lossy().into_owned())
                    .unwrap_or_else(|| full.clone());
                ui.label(
                    RichText::new(format!("{}  Request log: {short}", icons::FILE))
                        .font(design::regular(11.5))
                        .color(t.text_faint),
                )
                .on_hover_text(full);
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    if design::button(ui, t, Kind::Primary, None, "Done").clicked() {
                        ui.close();
                    }
                });
            });
        });
    if modal.should_close() {
        *open = false;
        settings.dirty = true;
    }
}

/// The ordered list of models to fall back to when one is rate-limited.
fn fallback_section(ui: &mut egui::Ui, t: &Theme, settings: &mut AiSettings) {
    design::section_label(ui, t, "Fallback models");
    design::hint(
        ui,
        t,
        "When a model hits a rate limit, the assistant switches to the next one here and \
         keeps going.",
    );
    ui.add_space(4.0);
    let mut remove: Option<usize> = None;
    let mut up: Option<usize> = None;
    for (i, f) in settings.fallbacks.iter().enumerate() {
        let live = settings.resolve(f).is_some();
        let provider = settings
            .providers
            .iter()
            .find(|p| p.kind.id() == f.provider)
            .map(|p| p.name.clone())
            .unwrap_or_else(|| f.provider.clone());
        ui.horizontal(|ui| {
            ui.label(
                RichText::new(format!("{}.", i + 1))
                    .font(design::medium(12.0))
                    .color(t.text_faint),
            );
            ui.label(
                RichText::new(&f.model)
                    .font(design::medium(12.5))
                    .color(if live { t.text } else { t.text_faint }),
            );
            ui.label(
                RichText::new(if live {
                    provider
                } else {
                    format!("{provider} \u{b7} not connected")
                })
                .font(design::regular(11.5))
                .color(t.text_faint),
            );
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                if design::icon_button(ui, t, icons::CLOSE, "Remove", false).clicked() {
                    remove = Some(i);
                }
                if i > 0 && design::icon_button(ui, t, icons::ARROW_UP, "Move up", false).clicked()
                {
                    up = Some(i);
                }
            });
        });
    }
    if let Some(i) = remove {
        settings.fallbacks.remove(i);
        settings.dirty = true;
    }
    if let Some(i) = up {
        settings.fallbacks.swap(i, i - 1);
        settings.dirty = true;
    }
    let candidates: Vec<(usize, String)> = settings
        .live_models()
        .into_iter()
        .filter(|(pi, m)| {
            let id = settings.providers[*pi].kind.id();
            !settings
                .fallbacks
                .iter()
                .any(|f| f.provider == id && f.model == *m)
        })
        .map(|(pi, m)| (pi, m.to_string()))
        .collect();
    if candidates.is_empty() {
        if settings.fallbacks.is_empty() {
            design::hint(ui, t, "Connect a provider to choose fallback models.");
        }
        return;
    }
    let resp = design::button_ex(
        ui,
        t,
        Kind::Secondary,
        Some(icons::ADD),
        "Add a fallback model",
        true,
        28.0,
    );
    let filter_id = ui.id().with("fallback-filter");
    if resp.clicked() {
        ui.ctx()
            .data_mut(|d| d.insert_temp(filter_id, String::new()));
    }
    egui::Popup::menu(&resp)
        .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
        .show(|ui| {
            ui.set_width(300.0);
            let mut filter: String = ui
                .ctx()
                .data_mut(|d| d.get_temp(filter_id).unwrap_or_default());
            if candidates.len() > 8 {
                design::text_field(
                    ui,
                    t,
                    &mut filter,
                    "Filter models\u{2026}",
                    Some(icons::SEARCH),
                );
                ui.ctx()
                    .data_mut(|d| d.insert_temp(filter_id, filter.clone()));
            }
            let f = filter.trim().to_lowercase();
            egui::ScrollArea::vertical()
                .max_height(280.0)
                .show(ui, |ui| {
                    for (pi, m) in &candidates {
                        if !f.is_empty() && !m.to_lowercase().contains(&f) {
                            continue;
                        }
                        let hint = settings.providers[*pi].name.clone();
                        if design::menu_item_ex(ui, t, "", m, &hint, true, false) {
                            let id = settings.providers[*pi].kind.id().to_string();
                            settings.fallbacks.push(SavedModel {
                                provider: id,
                                model: m.clone(),
                            });
                            settings.dirty = true;
                            egui::Popup::close_all(ui.ctx());
                        }
                    }
                });
        });
}

#[derive(Default)]
struct CardAction {
    test: bool,
    forget: bool,
    changed: bool,
}

/// One provider in settings: toggle, name, status, key field, actions.
fn provider_card(ui: &mut egui::Ui, t: &Theme, p: &mut Provider) -> CardAction {
    let mut action = CardAction::default();
    Frame::NONE
        .fill(t.surface)
        .stroke(Stroke::new(
            1.0,
            if p.enabled { t.border_strong } else { t.border },
        ))
        .corner_radius(CornerRadius::same(design::radius::LG))
        .inner_margin(Margin::same(12))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    ui.spacing_mut().item_spacing.y = 1.0;
                    ui.label(
                        RichText::new(&p.name)
                            .font(design::semibold(13.5))
                            .color(t.text),
                    );
                    ui.label(
                        RichText::new(p.kind.blurb())
                            .font(design::regular(11.5))
                            .color(t.text_faint),
                    );
                });
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    if design::toggle(ui, t, p.enabled).clicked() {
                        p.enabled = !p.enabled;
                        action.changed = true;
                    }
                    ui.add_space(6.0);
                    let (color, text) = match &p.status {
                        ConnStatus::Idle => (t.text_faint, "Not connected"),
                        ConnStatus::Testing => (t.accent, "Testing\u{2026}"),
                        ConnStatus::Connected => (t.success, "Connected"),
                        ConnStatus::Failed(_) => (t.danger, "Failed"),
                    };
                    design::pill(ui, t, color, text);
                });
            });

            if !p.enabled {
                return;
            }
            ui.add_space(10.0);
            if p.kind == ProviderKind::Custom {
                design::field_label(ui, t, "Name");
                if design::text_field(ui, t, &mut p.name, "My server", None).changed() {
                    action.changed = true;
                }
                ui.add_space(4.0);
                design::field_label(ui, t, "Base URL");
                if design::text_field(
                    ui,
                    t,
                    &mut p.base_url,
                    "http://localhost:11434/v1",
                    Some(icons::EXTERNAL_LINK),
                )
                .changed()
                {
                    action.changed = true;
                }
                ui.add_space(4.0);
            }
            design::field_label(
                ui,
                t,
                if p.key_optional() {
                    "API key (optional for local servers)"
                } else {
                    "API key"
                },
            );
            let show_id = ui.id().with(("show-key", p.kind.id()));
            let mut show = ui.ctx().data_mut(|d| *d.get_temp_mut_or(show_id, false));
            ui.horizontal(|ui| {
                let w = ui.available_width() - 36.0;
                let id = ui.id().with(("key", p.kind.id()));
                let focused = ui.ctx().memory(|m| m.has_focus(id));
                Frame::NONE
                    .fill(t.bg_panel)
                    .stroke(Stroke::new(
                        1.0,
                        if focused { t.accent } else { t.border_strong },
                    ))
                    .corner_radius(CornerRadius::same(design::radius::MD))
                    .inner_margin(Margin::symmetric(9, 6))
                    .show(ui, |ui| {
                        ui.add(
                            egui::TextEdit::singleline(&mut p.api_key)
                                .id(id)
                                .frame(Frame::NONE)
                                .password(!show)
                                .hint_text(RichText::new("Paste your key").color(t.text_faint))
                                .desired_width(w - 20.0),
                        );
                    });
                if design::icon_button(
                    ui,
                    t,
                    if show { icons::EYE_OFF } else { icons::EYE },
                    if show { "Hide key" } else { "Show key" },
                    false,
                )
                .clicked()
                {
                    show = !show;
                    ui.ctx().data_mut(|d| d.insert_temp(show_id, show));
                }
            });
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                let can_test = (!p.api_key.trim().is_empty() || p.key_optional())
                    && p.status != ConnStatus::Testing
                    && !p.base_url.trim().is_empty();
                if design::button_ex(
                    ui,
                    t,
                    Kind::Primary,
                    Some(icons::PLUG),
                    "Test connection",
                    can_test,
                    28.0,
                )
                .clicked()
                {
                    action.test = true;
                }
                if !p.api_key.trim().is_empty()
                    && design::button_ex(
                        ui,
                        t,
                        Kind::Ghost,
                        Some(icons::DELETE_BIN_6),
                        "Forget key",
                        true,
                        28.0,
                    )
                    .on_hover_text("Remove the saved key from this computer")
                    .clicked()
                {
                    action.forget = true;
                }
                if let Some(url) = p.kind.key_url() {
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        ui.hyperlink_to(
                            RichText::new(format!("Get a key {}", icons::EXTERNAL_LINK))
                                .font(design::medium(12.0))
                                .color(t.accent),
                            url,
                        );
                    });
                }
            });
            if p.base_url.to_lowercase().contains("logfare") {
                ui.add_space(6.0);
                ui.label(
                    RichText::new(format!(
                        "{}  Logfare stores all API traffic for research \u{2014} don't send \
                         anything private through it.",
                        icons::ERROR_WARNING
                    ))
                    .font(design::regular(11.5))
                    .color(t.warning),
                );
            }
            match &p.status {
                ConnStatus::Connected => {
                    ui.add_space(6.0);
                    let mut text = format!(
                        "{}  Connected{} \u{b7} {} chat model{}",
                        icons::CHECKBOX_CIRCLE_FILL,
                        p.latency_ms
                            .map(|ms| format!(" in {ms} ms"))
                            .unwrap_or_default(),
                        p.models.len(),
                        if p.models.len() == 1 { "" } else { "s" }
                    );
                    if p.hidden_models > 0 {
                        text.push_str(&format!(
                            " ({} image/audio/video model{} hidden)",
                            p.hidden_models,
                            if p.hidden_models == 1 { "" } else { "s" }
                        ));
                    }
                    ui.label(
                        RichText::new(text)
                            .font(design::regular(12.0))
                            .color(t.success),
                    );
                }
                ConnStatus::Failed(msg) => {
                    ui.add_space(6.0);
                    ui.label(
                        RichText::new(format!("{}  {msg}", icons::ERROR_WARNING))
                            .font(design::regular(12.0))
                            .color(t.danger),
                    );
                }
                _ => {}
            }
        });
    action
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reply_event(token: u64, result: Result<String, String>, retryable: bool) -> ChatEvent {
        ChatEvent::Reply {
            token,
            result,
            retryable,
            rate_limited: false,
            truncated: false,
        }
    }

    /// A session mid-run with request `0` in flight.
    fn running_session() -> AiSession {
        let mut s = AiSession::default();
        s.messages
            .push(ChatMessage::user("build a full adder".into()));
        s.begin_run();
        s.pending = Some(0);
        s.next_token = 1;
        s
    }

    #[test]
    fn retry_backoff_grows_then_caps_at_a_minute() {
        let secs: Vec<u64> = (1..=7).map(|n| retry_delay(n).as_secs()).collect();
        assert_eq!(secs, vec![5, 10, 20, 40, 60, 60, 60]);
    }

    #[test]
    fn transient_failure_mid_run_backs_off_instead_of_ending() {
        let mut s = running_session();
        s.chat_tx
            .send(reply_event(0, Err("Rate limited (HTTP 429).".into()), true))
            .unwrap();
        assert!(s.poll_chat());
        s.end_agent();
        assert!(s.is_running(), "a transient failure must not end the run");
        assert!(s.retry_scheduled());
        assert!(!s.retry_due(), "the retry waits out its backoff first");
        assert!(s.retry_status().unwrap().contains("retrying in"));
        assert!(
            !s.messages.iter().any(|m| m.error),
            "no error bubble while retrying"
        );
    }

    #[test]
    fn fatal_failure_ends_the_run_with_an_error() {
        let mut s = running_session();
        s.chat_tx
            .send(reply_event(
                0,
                Err("Rejected: check your API key.".into()),
                false,
            ))
            .unwrap();
        assert!(s.poll_chat());
        s.end_agent();
        assert!(!s.is_running());
        assert!(!s.retry_scheduled());
        assert!(s.messages.last().unwrap().error);
    }

    #[test]
    fn retries_give_up_after_the_consecutive_cap() {
        let mut s = running_session();
        s.retries = MAX_CONSECUTIVE_RETRIES;
        s.chat_tx
            .send(reply_event(0, Err("Overloaded.".into()), true))
            .unwrap();
        s.poll_chat();
        s.end_agent();
        assert!(!s.is_running());
        assert!(s.messages.last().unwrap().error);
    }

    #[test]
    fn success_resets_the_retry_count_and_never_caps_steps() {
        let mut s = running_session();
        s.retries = 3;
        s.agent_steps = 10_000;
        s.chat_tx
            .send(reply_event(0, Ok("Done.".into()), false))
            .unwrap();
        s.poll_chat();
        assert_eq!(s.retries, 0);
        assert!(
            s.should_continue(),
            "no step limit: only Stop or finishing ends a run"
        );
    }

    #[test]
    fn stop_cancels_the_request_and_drops_its_late_reply() {
        let mut s = running_session();
        let cancel = s.cancel.clone();
        s.request_stop();
        assert!(
            cancel.load(Ordering::Relaxed),
            "the worker is told to abandon the stream"
        );
        assert!(!s.is_pending());
        assert!(!s.is_running());
        assert!(!s.should_continue());
        assert!(s.messages.last().unwrap().notice);
        let before = s.messages.len();
        s.chat_tx
            .send(reply_event(0, Ok("late".into()), false))
            .unwrap();
        assert!(!s.poll_chat(), "a reply for a stopped request is stale");
        assert_eq!(s.messages.len(), before);
    }

    #[test]
    fn stop_also_cancels_a_scheduled_retry() {
        let mut s = running_session();
        s.pending = None;
        s.retry_at = Some(Instant::now());
        s.request_stop();
        assert!(!s.retry_scheduled());
        assert!(!s.retry_due());
    }

    #[test]
    fn long_runs_trim_old_turns_but_keep_what_the_user_typed() {
        let mut msgs = vec![ChatMessage::user("GOAL: build an 8-bit ALU".into())];
        for i in 0..50 {
            msgs.push(ChatMessage::reply(format!("step {i}"), None));
            msgs.push(ChatMessage::auto(format!("check {i}")));
        }
        msgs.push(ChatMessage::failure("boom".into()));
        msgs.push(ChatMessage::notice("Stopped.".into()));
        let (turns, trimmed) = history_turns(&msgs);
        assert!(trimmed);
        assert!(turns.len() <= HISTORY_TAIL + 1);
        assert_eq!(turns[0].content, "GOAL: build an 8-bit ALU");
        // Only the newest auto-check survives; errors and notices are never sent.
        let autos: Vec<_> = turns
            .iter()
            .filter(|t| t.content.starts_with("check"))
            .collect();
        assert_eq!(autos.len(), 1);
        assert_eq!(autos[0].content, "check 49");
        assert!(!turns
            .iter()
            .any(|t| t.content == "boom" || t.content == "Stopped."));
        assert_eq!(turns.last().unwrap().content, "check 49");
    }

    #[test]
    fn short_conversations_are_sent_whole() {
        let msgs = vec![
            ChatMessage::user("hi".into()),
            ChatMessage::reply("hello".into(), None),
        ];
        let (turns, trimmed) = history_turns(&msgs);
        assert!(!trimmed);
        assert_eq!(turns.len(), 2);
        assert_eq!(turns[1].role, "assistant");
    }
}
