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

use std::sync::mpsc::{Receiver, Sender};

use eframe::egui::{self, Align, Color32, CornerRadius, Frame, Layout, Margin, RichText, Stroke};

use llmc::io::AiProviderConfig;

use super::ai_edit::{self, RawOp};
use super::ai_net::{self, AiEvent, ChatEvent, ChatRequest, ChatTurn};
use super::keystore;
use super::theme::Theme;

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
    /// Models fetched from the provider by "Test connection" / startup reconnect.
    pub models: Vec<String>,
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
            status: ConnStatus::Idle,
        }
    }

    fn is_live(&self) -> bool {
        self.enabled && self.status == ConnStatus::Connected && !self.models.is_empty()
    }
}

/// Global assistant settings: the provider list plus the worker-thread channel that carries
/// connection results back to the UI.
pub struct AiSettings {
    pub providers: Vec<Provider>,
    tx: Sender<AiEvent>,
    rx: Receiver<AiEvent>,
    /// Set when the persisted config changed (enabled/base-url/name); the app writes it out.
    dirty: bool,
}

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
                AiEvent::Models { provider, result } => {
                    if let Some(p) = self.providers.get_mut(provider) {
                        match result {
                            Ok(models) => {
                                p.models = models;
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
        if p.api_key.trim().is_empty() || p.base_url.trim().is_empty() {
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
            // Reconnect providers that were on and have a key.
            let ready = {
                let p = &self.providers[i];
                p.enabled && !p.api_key.trim().is_empty() && !p.base_url.trim().is_empty()
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
}

impl ChatMessage {
    fn user(text: String) -> Self {
        Self {
            role: Role::User,
            text,
            thinking: None,
            error: false,
            activity: Vec::new(),
            auto: false,
        }
    }

    fn failure(text: String) -> Self {
        Self {
            role: Role::Assistant,
            text,
            thinking: None,
            error: true,
            activity: Vec::new(),
            auto: false,
        }
    }

    /// An automatic agent-loop feedback turn (the post-edit "auto-check").
    fn auto(text: String) -> Self {
        Self {
            role: Role::User,
            text,
            thinking: None,
            error: false,
            activity: Vec::new(),
            auto: true,
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
    /// Auto-continue steps taken in the current run (bounded by [`MAX_AGENT_STEPS`]).
    agent_steps: u32,
    /// Set when the user presses Stop; the loop ends after the in-flight reply lands.
    agent_stop: bool,
    /// Live text filter for the model picker (providers can list hundreds of models).
    model_filter: String,
    /// Streaming buffers for the in-flight reply: reasoning and answer accumulated so far. Shown
    /// as a live bubble until the terminal reply lands and replaces them with a parsed message.
    stream_reason: String,
    stream_content: String,
}

/// Safety cap on autonomous agent iterations (each is one provider call). The user can always
/// Stop sooner, or re-prompt to go further.
const MAX_AGENT_STEPS: u32 = 40;

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
            model_filter: String::new(),
            stream_reason: String::new(),
            stream_content: String::new(),
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
                ChatEvent::Reply { token, result } => {
                    if self.pending != Some(token) {
                        continue; // stale reply (conversation cleared, or superseded) — drop it.
                    }
                    self.pending = None;
                    self.stream_reason.clear();
                    self.stream_content.clear();
                    got = true;
                    match result {
                        Ok(raw) => {
                            let parsed = ai_edit::parse_reply(&raw);
                            self.messages.push(ChatMessage {
                                role: Role::Assistant,
                                text: parsed.text,
                                thinking: parsed.thinking,
                                error: false,
                                activity: Vec::new(),
                                auto: false,
                            });
                            if !parsed.ops.is_empty() {
                                self.pending_ops = Some(parsed.ops);
                            }
                        }
                        Err(e) => self.messages.push(ChatMessage::failure(e)),
                    }
                }
            }
        }
        got
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

    /// Ask the running agent to stop after the in-flight reply lands.
    pub fn request_stop(&mut self) {
        self.agent_stop = true;
        self.agent_running = false;
    }

    /// After a reply that made edits, decide whether the agent keeps iterating on its own. If it
    /// stops purely because it hit the step cap (not user Stop / agent off), leave a short note so
    /// the pause isn't silent.
    pub fn should_continue_or_note(&mut self) -> bool {
        if !self.agent_enabled || self.agent_stop {
            return false;
        }
        if self.agent_steps < MAX_AGENT_STEPS {
            return true;
        }
        self.messages.push(ChatMessage {
            role: Role::Assistant,
            text: format!(
                "Paused after {MAX_AGENT_STEPS} steps. Tell me to continue if it isn't finished."
            ),
            thinking: None,
            error: false,
            activity: Vec::new(),
            auto: false,
        });
        false
    }

    /// Feed the post-edit auto-check back to the model and dispatch the next step of the run.
    pub fn continue_agent(
        &mut self,
        settings: &AiSettings,
        system_prompt: String,
        feedback: String,
    ) {
        self.agent_steps += 1;
        self.messages.push(ChatMessage::auto(feedback));
        self.dispatch(settings, system_prompt);
    }

    /// End the current run (the model replied with no further edits, or hit an error / the cap).
    pub fn end_agent(&mut self) {
        self.agent_running = false;
    }

    /// If nothing valid is selected, fall back to the first available model (and recover from a
    /// selection whose provider has since disconnected).
    fn ensure_selection(&mut self, live: &[(usize, &str)]) {
        let valid = self.selected.as_ref().is_some_and(|s| {
            live.iter()
                .any(|(pi, m)| *pi == s.provider && *m == s.model)
        });
        if !valid {
            self.selected = live.first().map(|(pi, m)| ModelRef {
                provider: *pi,
                model: m.to_string(),
            });
        }
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
    /// error bubbles and resends the existing turns; no new user message is added.
    pub fn resend(&mut self, settings: &AiSettings, system_prompt: String) {
        if self.pending.is_some() {
            return;
        }
        while self.messages.last().is_some_and(|m| m.error) {
            self.messages.pop();
        }
        if !self.messages.iter().any(|m| m.role == Role::User) {
            return;
        }
        self.begin_run();
        self.dispatch(settings, system_prompt);
    }

    /// Reset the agent counters for a new run started by the user.
    fn begin_run(&mut self) {
        self.agent_running = true;
        self.agent_steps = 0;
        self.agent_stop = false;
    }

    /// Build the request from the current turns + `system_prompt` and fire it off-thread.
    fn dispatch(&mut self, settings: &AiSettings, system_prompt: String) {
        let Some(sel) = self.selected.clone() else {
            self.messages
                .push(ChatMessage::failure("No model selected.".to_string()));
            return;
        };
        let Some(provider) = settings.providers.get(sel.provider).filter(|p| p.is_live()) else {
            self.messages.push(ChatMessage::failure(
                "The selected provider isn't connected.".to_string(),
            ));
            return;
        };

        // system prompt + the visible conversation. Skip error bubbles and empty turns; the
        // system prompt already carries the *current* circuit, so include only the newest
        // auto-check turn (older ones describe superseded states and would just waste tokens).
        let last_auto = self.messages.iter().rposition(|m| m.auto);
        let mut turns = vec![ChatTurn {
            role: "system".to_string(),
            content: system_prompt,
        }];
        for (i, m) in self.messages.iter().enumerate() {
            if m.error || m.text.trim().is_empty() {
                continue;
            }
            if m.auto && Some(i) != last_auto {
                continue;
            }
            let role = match m.role {
                Role::User => "user",
                Role::Assistant => "assistant",
            };
            turns.push(ChatTurn {
                role: role.to_string(),
                content: m.text.clone(),
            });
        }

        let token = self.next_token;
        self.next_token += 1;
        self.pending = Some(token);
        self.stream_reason.clear();
        self.stream_content.clear();
        let req = ChatRequest {
            base_url: provider.base_url.clone(),
            api_key: provider.api_key.clone(),
            model: sel.model.clone(),
            turns,
        };
        ai_net::spawn_chat(self.chat_tx.clone(), token, req);
    }
}

// ===================== rendering =====================

/// What the panel needs the app to do after rendering. Kept tiny so the app can build the
/// circuit context (which lives with the document) and hand it to [`AiSession::send`].
#[derive(Default)]
pub struct PanelResponse {
    /// The user pressed Send on a ready model; dispatch a chat request.
    pub send: bool,
    /// The user pressed Retry on a failed reply; re-run the last request.
    pub retry: bool,
}

/// Render the assistant panel on the right. Mutates session/settings, may flip `open_settings`
/// when the gear is pressed, and reports (via the return value) when a chat send was requested.
pub fn panel(
    ui: &mut egui::Ui,
    theme: &Theme,
    settings: &mut AiSettings,
    session: &mut AiSession,
    open_settings: &mut bool,
) -> PanelResponse {
    let mut response = PanelResponse::default();
    egui::Panel::right("ai_panel")
        .resizable(true)
        .default_size(370.0)
        .min_size(300.0)
        .max_size(560.0)
        .show(ui, |ui| {
            // Header ------------------------------------------------------------
            ui.add_space(9.0);
            ui.horizontal(|ui| {
                ui.add_space(12.0);
                accent_dot(ui, theme, 5.0);
                ui.add_space(4.0);
                ui.label(RichText::new("Assistant").strong().size(15.0));
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    ui.add_space(10.0);
                    if text_button(ui, theme, "Settings", "Providers & API keys") {
                        *open_settings = true;
                    }
                    // Don't offer "Clear" mid-request; the reply would land in a cleared thread.
                    if !session.messages.is_empty()
                        && !session.is_pending()
                        && text_button(ui, theme, "Clear", "Clear conversation")
                    {
                        session.messages.clear();
                    }
                });
            });
            ui.add_space(8.0);
            ui.separator();

            // Reserve room for the composer at the bottom, give the rest to the conversation.
            let composer_h = 156.0;
            let chat_h = (ui.available_height() - composer_h).max(120.0);
            let width = ui.available_width();
            ui.allocate_ui(egui::vec2(width, chat_h), |ui| {
                if !settings.any_live() {
                    empty_state(ui, theme, open_settings);
                } else {
                    response.retry |= conversation(ui, theme, session);
                }
            });

            ui.separator();
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                ui.add_space(12.0);
                ui.vertical(|ui| {
                    ui.set_width(width - 24.0);
                    response.send = composer(ui, theme, settings, session);
                });
            });
        });
    response
}

/// The calm "connect a provider" state shown before anything is connected.
fn empty_state(ui: &mut egui::Ui, theme: &Theme, open_settings: &mut bool) {
    ui.vertical_centered(|ui| {
        ui.add_space(48.0);
        accent_dot(ui, theme, 9.0);
        ui.add_space(12.0);
        ui.label(
            RichText::new("Connect a provider to begin")
                .strong()
                .size(15.0),
        );
        ui.add_space(4.0);
        ui.label(
            RichText::new(
                "Add a free provider (Groq, Cerebras, Gemini, …) and the assistant can build \
                 and edit your circuits.",
            )
            .color(theme.label_dim),
        );
        ui.add_space(14.0);
        if accent_button(ui, theme, "  Open settings  ").clicked() {
            *open_settings = true;
        }
    });
}

/// The scrolling message list (plus a "thinking" row while a reply is in flight). Empty until
/// the first message so the panel stays calm. Returns true if a "Retry" was clicked.
fn conversation(ui: &mut egui::Ui, theme: &Theme, session: &AiSession) -> bool {
    // The row width must be captured from the bounded scroll viewport; passing it down keeps
    // the per-message right/left alignment from treating the width as unbounded (which would
    // blow the panel's size up).
    let row_width = (ui.available_width() - 24.0).max(120.0);
    let mut retry = false;
    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .stick_to_bottom(true)
        .show(ui, |ui| {
            ui.add_space(4.0);
            for (i, msg) in session.messages.iter().enumerate() {
                if msg.auto {
                    auto_note(ui, theme, msg, i);
                } else {
                    retry |= bubble(ui, theme, msg, row_width, i);
                }
                ui.add_space(8.0);
            }
            if session.is_pending() {
                if session.stream_reason.is_empty() && session.stream_content.is_empty() {
                    thinking(ui, theme, session);
                } else {
                    stream_bubble(
                        ui,
                        theme,
                        &session.stream_reason,
                        &session.stream_content,
                        row_width,
                    );
                }
            }
        });
    retry
}

/// A live assistant bubble rendered from the partial streaming buffers — reasoning dimmed above,
/// answer below — so the reply appears token by token.
fn stream_bubble(
    ui: &mut egui::Ui,
    theme: &Theme,
    reason_field: &str,
    content_field: &str,
    row_width: f32,
) {
    let (reasoning, answer) = ai_edit::live_split(reason_field, content_field);
    ui.allocate_ui_with_layout(
        egui::vec2(row_width, 0.0),
        Layout::left_to_right(Align::TOP),
        |ui| {
            ui.add_space(12.0);
            let max_w = (row_width * 0.82).max(140.0);
            Frame::NONE
                .fill(surface(theme))
                .corner_radius(CornerRadius::same(10))
                .inner_margin(Margin::symmetric(10, 8))
                .show(ui, |ui| {
                    ui.set_max_width(max_w);
                    ui.vertical(|ui| {
                        ui.horizontal(|ui| {
                            accent_dot(ui, theme, 3.0);
                            ui.add_space(5.0);
                            ui.label(RichText::new("Assistant").small().color(theme.label_dim));
                        });
                        if let Some(r) = &reasoning {
                            ui.label(RichText::new(r).italics().color(theme.label_dim).size(12.5));
                        }
                        if !answer.is_empty() {
                            ui.label(RichText::new(answer).color(theme.label));
                        }
                    });
                });
        },
    );
    // Stream smoothly.
    ui.ctx()
        .request_repaint_after(std::time::Duration::from_millis(60));
}

/// A subtle, foldable "auto-check" row for an agent-loop feedback turn — visible so the run reads
/// like a live workflow, but quiet so it never competes with the real messages.
fn auto_note(ui: &mut egui::Ui, theme: &Theme, msg: &ChatMessage, idx: usize) {
    ui.horizontal(|ui| {
        ui.add_space(12.0);
        egui::CollapsingHeader::new(
            RichText::new("Checked the circuit")
                .small()
                .italics()
                .color(theme.label_dim),
        )
        .id_salt(("ai-autocheck", idx))
        .default_open(false)
        .show(ui, |ui| {
            ui.label(RichText::new(&msg.text).small().color(theme.label_dim));
        });
    });
}

/// A left-aligned status row shown while awaiting a reply: "Thinking…", or "Working — step N…"
/// during an autonomous run.
fn thinking(ui: &mut egui::Ui, theme: &Theme, session: &AiSession) {
    let label = if session.is_running() && session.agent_step() > 0 {
        format!("Working \u{2014} step {}\u{2026}", session.agent_step())
    } else {
        "Thinking\u{2026}".to_string()
    };
    ui.horizontal(|ui| {
        ui.add_space(12.0);
        accent_dot(ui, theme, 3.5);
        ui.add_space(6.0);
        ui.label(RichText::new(label).italics().color(theme.label_dim));
    });
    // Keep animating so the reply (and this indicator) refresh promptly.
    ui.ctx()
        .request_repaint_after(std::time::Duration::from_millis(120));
}

/// A single chat bubble: user right-aligned/accent, assistant left-aligned/panel, errors muted-red.
/// Shows any `<think>` reasoning (dimmed, collapsible) above the answer and an "activity" list of
/// applied edits below it. Returns true if the bubble's "Retry" button was clicked.
fn bubble(ui: &mut egui::Ui, theme: &Theme, msg: &ChatMessage, row_width: f32, idx: usize) -> bool {
    let is_user = msg.role == Role::User;
    let mut retry = false;
    let fill = if msg.error {
        theme.conflict.linear_multiply(0.16)
    } else if is_user {
        theme.accent.linear_multiply(0.22)
    } else {
        surface(theme)
    };
    let layout = if is_user {
        Layout::right_to_left(Align::TOP)
    } else {
        Layout::left_to_right(Align::TOP)
    };
    // Bound the row to a known width so `right_to_left` has a finite right edge.
    ui.allocate_ui_with_layout(egui::vec2(row_width, 0.0), layout, |ui| {
        ui.add_space(12.0);
        let max_w = (row_width * 0.82).max(140.0);
        Frame::NONE
            .fill(fill)
            .corner_radius(CornerRadius::same(10))
            .inner_margin(Margin::symmetric(10, 8))
            .show(ui, |ui| {
                ui.set_max_width(max_w);
                ui.vertical(|ui| {
                    if msg.error {
                        ui.label(RichText::new("Error").small().color(theme.conflict));
                    } else if !is_user {
                        ui.label(RichText::new("Assistant").small().color(theme.label_dim));
                    }
                    // Reasoning first, dimmed and foldable (default open, so it's visible but calm).
                    if let Some(reasoning) = &msg.thinking {
                        egui::CollapsingHeader::new(
                            RichText::new("Reasoning").small().color(theme.label_dim),
                        )
                        .id_salt(("ai-reasoning", idx))
                        .default_open(true)
                        .show(ui, |ui| {
                            ui.label(
                                RichText::new(reasoning)
                                    .italics()
                                    .color(theme.label_dim)
                                    .size(12.5),
                            );
                        });
                    }
                    if !msg.text.trim().is_empty() {
                        ui.label(RichText::new(&msg.text).color(theme.label));
                    }
                    activity_view(ui, theme, &msg.activity);
                    if msg.error && text_button(ui, theme, "Retry", "Send this prompt again") {
                        retry = true;
                    }
                });
            });
    });
    retry
}

/// The "activity" view under an assistant reply: the concrete edits it applied to the circuit,
/// one line each, with skipped ops shown in muted-red. Nothing is drawn when the list is empty.
fn activity_view(ui: &mut egui::Ui, theme: &Theme, activity: &[String]) {
    if activity.is_empty() {
        return;
    }
    ui.add_space(6.0);
    let applied = activity
        .iter()
        .filter(|a| !a.starts_with("Skipped"))
        .count();
    ui.label(
        RichText::new(format!(
            "Applied {applied} change{}",
            if applied == 1 { "" } else { "s" }
        ))
        .small()
        .strong()
        .color(theme.label_dim),
    );
    for line in activity {
        let skipped = line.starts_with("Skipped");
        let color = if skipped {
            theme.conflict
        } else {
            theme.label_dim
        };
        ui.horizontal(|ui| {
            ui.add_space(2.0);
            colored_dot(
                ui,
                if skipped {
                    theme.conflict
                } else {
                    theme.accent
                },
                2.5,
            );
            ui.add_space(5.0);
            ui.label(RichText::new(line).small().color(color));
        });
    }
    ui.add_space(2.0);
    ui.label(
        RichText::new("Undo (Ctrl+Z) to revert")
            .small()
            .color(theme.label_dim),
    );
}

/// The bottom composer: model switcher + input + send. Returns true when the user asked to
/// send (the app then supplies circuit context and dispatches via [`AiSession::send`]).
fn composer(
    ui: &mut egui::Ui,
    theme: &Theme,
    settings: &mut AiSettings,
    session: &mut AiSession,
) -> bool {
    let live = settings.live_models();
    let ready = !live.is_empty();
    session.ensure_selection(&live);
    let pending = session.is_pending();
    let mut send = false;

    ui.add_enabled_ui(ready, |ui| {
        // Input box in a soft rounded surface.
        Frame::NONE
            .fill(surface(theme))
            .stroke(Stroke::new(1.0, theme.block_stroke))
            .corner_radius(CornerRadius::same(10))
            .inner_margin(Margin::symmetric(10, 8))
            .show(ui, |ui| {
                let hint = if !ready {
                    "Connect a provider to start\u{2026}"
                } else if pending {
                    "Waiting for a reply\u{2026}"
                } else {
                    "Ask the assistant to build or edit your circuit\u{2026}"
                };
                // Enter sends; Shift+Enter inserts a newline.
                let enter = ui.input(|i| i.key_pressed(egui::Key::Enter) && !i.modifiers.shift);
                let editor = ui.add_enabled(
                    !pending,
                    egui::TextEdit::multiline(&mut session.input)
                        .frame(Frame::NONE)
                        .desired_rows(2)
                        .hint_text(hint)
                        .desired_width(f32::INFINITY),
                );
                if enter && editor.has_focus() && !session.input.trim().is_empty() {
                    send = true;
                }
            });

        ui.add_space(6.0);
        ui.horizontal(|ui| {
            model_switcher(ui, theme, settings, session);
            // Agent toggle: when on, the assistant iterates (build → self-check → fix) on its own.
            ui.add_space(8.0);
            ui.checkbox(&mut session.agent_enabled, RichText::new("Agent").small())
                .on_hover_text(
                    "Let the assistant work on its own: build, auto-check the circuit's \
                     behavior, and fix it until it matches your request. Uses several provider \
                     calls; press Stop any time.",
                );
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                if session.is_running() {
                    // Mid-run: offer Stop instead of Send.
                    if ui
                        .add(accent_widget(theme, "Stop"))
                        .on_hover_text("Stop after the current step")
                        .clicked()
                    {
                        session.request_stop();
                    }
                } else {
                    let can_send = ready && !pending && !session.input.trim().is_empty();
                    let label = if pending { "Sending\u{2026}" } else { "Send" };
                    if ui
                        .add_enabled(can_send, accent_widget(theme, label))
                        .clicked()
                    {
                        send = true;
                    }
                }
            });
        });
    });
    // A stray newline from the Enter keypress is trimmed by `send`, but drop it here too so the
    // box doesn't briefly show one.
    if send {
        while session.input.ends_with('\n') {
            session.input.pop();
        }
    }
    send && ready && !pending && !session.input.trim().is_empty()
}

/// The in-composer model picker. Shows the current model with its provider in a fainter,
/// smaller style beside it, and lists every connected (provider, model) the same way.
fn model_switcher(
    ui: &mut egui::Ui,
    theme: &Theme,
    settings: &AiSettings,
    session: &mut AiSession,
) {
    let live = settings.live_models();
    let multi_provider = {
        // Only show provider tags when more than one provider is connected (matches the spec).
        let mut seen: Vec<usize> = Vec::new();
        for (pi, _) in &live {
            if !seen.contains(pi) {
                seen.push(*pi);
            }
        }
        seen.len() > 1
    };

    let selected_label = match &session.selected {
        Some(sel) => model_job(
            ui,
            theme,
            &sel.model,
            settings
                .providers
                .get(sel.provider)
                .map(|p| p.name.as_str()),
            multi_provider,
        ),
        None => model_job(ui, theme, "Choose a model", None, false),
    };

    ui.menu_button(selected_label, |ui| {
        ui.set_min_width(260.0);
        // A filter box once the list is long enough to be a pain to eyeball (e.g. OpenRouter).
        if live.len() > 8 {
            ui.add(
                egui::TextEdit::singleline(&mut session.model_filter)
                    .hint_text("Filter models\u{2026}")
                    .desired_width(f32::INFINITY),
            );
            ui.add_space(4.0);
        }
        let filter = session.model_filter.trim().to_lowercase();
        // Scroll so hundreds of models never run off the screen.
        egui::ScrollArea::vertical()
            .max_height(320.0)
            .auto_shrink([false, false])
            .show(ui, |ui| {
                let mut shown = 0usize;
                for (pi, model) in &live {
                    let provider = settings.providers.get(*pi).map(|p| p.name.as_str());
                    if !filter.is_empty() {
                        let hay = format!("{model} {}", provider.unwrap_or("")).to_lowercase();
                        if !hay.contains(&filter) {
                            continue;
                        }
                    }
                    shown += 1;
                    let is_sel = session
                        .selected
                        .as_ref()
                        .is_some_and(|s| s.provider == *pi && s.model == *model);
                    let row = model_job(ui, theme, model, provider, multi_provider);
                    if ui.selectable_label(is_sel, row).clicked() {
                        session.selected = Some(ModelRef {
                            provider: *pi,
                            model: model.to_string(),
                        });
                        session.model_filter.clear();
                        ui.close();
                    }
                }
                if shown == 0 {
                    ui.label(
                        RichText::new("No matching models")
                            .small()
                            .color(theme.label_dim),
                    );
                }
            });
    });
}

/// Build a two-tone label: model name in normal text, provider in a smaller, dimmer style.
fn model_job(
    ui: &egui::Ui,
    theme: &Theme,
    model: &str,
    provider: Option<&str>,
    show_provider: bool,
) -> egui::text::LayoutJob {
    use egui::text::{LayoutJob, TextFormat};
    let font = egui::TextStyle::Button.resolve(ui.style());
    let small = egui::FontId::proportional(font.size * 0.82);
    let mut job = LayoutJob::default();
    job.append(
        model,
        0.0,
        TextFormat {
            font_id: font,
            color: theme.label,
            ..Default::default()
        },
    );
    if show_provider {
        if let Some(p) = provider {
            job.append(
                &format!("   {p}"),
                0.0,
                TextFormat {
                    font_id: small,
                    color: theme.label_dim,
                    ..Default::default()
                },
            );
        }
    }
    job
}

// ===================== settings modal =====================

/// Render the provider settings window. Set `open` false to close.
pub fn settings_window(
    ctx: &egui::Context,
    theme: &Theme,
    settings: &mut AiSettings,
    open: &mut bool,
) {
    let mut keep_open = true;
    egui::Window::new(RichText::new("AI providers").strong())
        .collapsible(false)
        .resizable(true)
        .default_width(460.0)
        .max_height(560.0)
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .open(&mut keep_open)
        .show(ctx, |ui| {
            ui.label(
                RichText::new(
                    "Connect one or more providers. Keys are stored locally and never leave \
                     your machine.",
                )
                .color(theme.label_dim),
            );
            ui.add_space(8.0);
            let mut test_request: Option<usize> = None;
            let mut changed = false;
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .max_height(440.0)
                .show(ui, |ui| {
                    for i in 0..settings.providers.len() {
                        let act = provider_card(ui, theme, &mut settings.providers[i]);
                        if act.test {
                            test_request = Some(i);
                        }
                        changed |= act.changed;
                        ui.add_space(8.0);
                    }
                });
            if let Some(i) = test_request {
                settings.start_test(i);
            }
            if changed {
                settings.dirty = true;
            }

            ui.add_space(6.0);
            ui.separator();
            ui.add_space(4.0);
            ui.label(
                RichText::new(format!(
                    "Request log: {}",
                    super::ai_log::log_path_display()
                ))
                .small()
                .color(theme.label_dim),
            );
        });
    if !keep_open {
        *open = false;
        // Persist the provider setup when the modal closes.
        settings.dirty = true;
    }
}

/// What the user did to a provider card this frame.
#[derive(Default)]
struct CardAction {
    test: bool,
    changed: bool,
}

/// One provider row in settings: name, status, key field, test button.
fn provider_card(ui: &mut egui::Ui, theme: &Theme, p: &mut Provider) -> CardAction {
    let mut action = CardAction::default();
    Frame::NONE
        .fill(surface(theme))
        .stroke(Stroke::new(1.0, theme.block_stroke))
        .corner_radius(CornerRadius::same(10))
        .inner_margin(Margin::same(12))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                if ui.checkbox(&mut p.enabled, "").changed() {
                    action.changed = true;
                }
                ui.vertical(|ui| {
                    ui.label(RichText::new(&p.name).strong());
                    ui.label(RichText::new(p.kind.blurb()).small().color(theme.label_dim));
                });
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    status_chip(ui, theme, &p.status);
                });
            });

            if p.enabled {
                ui.add_space(8.0);
                if p.kind == ProviderKind::Custom {
                    labeled(ui, theme, "Name", |ui| {
                        if ui
                            .add(
                                egui::TextEdit::singleline(&mut p.name)
                                    .desired_width(f32::INFINITY),
                            )
                            .changed()
                        {
                            action.changed = true;
                        }
                    });
                    labeled(ui, theme, "Base URL", |ui| {
                        if ui
                            .add(
                                egui::TextEdit::singleline(&mut p.base_url)
                                    .hint_text("https://…/v1")
                                    .desired_width(f32::INFINITY),
                            )
                            .changed()
                        {
                            action.changed = true;
                        }
                    });
                }
                labeled(ui, theme, "API key", |ui| {
                    ui.add(
                        egui::TextEdit::singleline(&mut p.api_key)
                            .password(true)
                            .hint_text("Paste your key")
                            .desired_width(f32::INFINITY),
                    );
                });
                ui.add_space(6.0);
                ui.horizontal(|ui| {
                    let can_test = !p.api_key.trim().is_empty()
                        && p.status != ConnStatus::Testing
                        && !p.base_url.trim().is_empty();
                    if ui
                        .add_enabled(can_test, accent_widget(theme, "Test connection"))
                        .clicked()
                    {
                        action.test = true;
                    }
                    if let Some(url) = p.kind.key_url() {
                        ui.hyperlink_to(RichText::new("Get a key").color(theme.accent), url);
                    }
                });

                // Feedback line: model count when connected, or the failure reason.
                match &p.status {
                    ConnStatus::Connected => {
                        ui.add_space(4.0);
                        ui.label(
                            RichText::new(format!("{} models available", p.models.len()))
                                .small()
                                .color(theme.label_dim),
                        );
                    }
                    ConnStatus::Failed(msg) => {
                        ui.add_space(4.0);
                        ui.label(RichText::new(msg).small().color(theme.conflict));
                    }
                    _ => {}
                }
            }
        });
    action
}

fn status_chip(ui: &mut egui::Ui, theme: &Theme, status: &ConnStatus) {
    let (color, text) = match status {
        ConnStatus::Idle => (theme.label_dim, "Not connected"),
        ConnStatus::Testing => (theme.accent, "Testing\u{2026}"),
        ConnStatus::Connected => (theme.wire_high, "Connected"),
        ConnStatus::Failed(_) => (theme.conflict, "Failed"),
    };
    colored_dot(ui, color, 4.0);
    ui.add_space(4.0);
    ui.label(RichText::new(text).small().color(theme.label_dim));
}

// ===================== small shared widgets =====================

/// Paint a small filled circle inline (used instead of a `●` glyph, which egui's default
/// font doesn't include).
fn colored_dot(ui: &mut egui::Ui, color: Color32, radius: f32) {
    let (rect, _) =
        ui.allocate_exact_size(egui::vec2(radius * 2.0, radius * 2.0), egui::Sense::hover());
    ui.painter().circle_filled(rect.center(), radius, color);
}

fn accent_dot(ui: &mut egui::Ui, theme: &Theme, radius: f32) {
    colored_dot(ui, theme.accent, radius);
}

fn labeled(ui: &mut egui::Ui, theme: &Theme, label: &str, add: impl FnOnce(&mut egui::Ui)) {
    ui.add_space(4.0);
    ui.label(RichText::new(label).small().color(theme.label_dim));
    add(ui);
}

/// A subtle surface color that sits just above the panel background.
fn surface(theme: &Theme) -> Color32 {
    theme.block_fill
}

/// A borderless small text button (used for the header actions); returns true when clicked.
fn text_button(ui: &mut egui::Ui, theme: &Theme, label: &str, tip: &str) -> bool {
    ui.add(egui::Button::new(RichText::new(label).small().color(theme.label_dim)).frame(false))
        .on_hover_text(tip)
        .clicked()
}

/// An accent-filled button widget (for `add`/`add_enabled`).
fn accent_widget(theme: &Theme, text: &str) -> egui::Button<'static> {
    egui::Button::new(RichText::new(text.to_string()).color(Color32::WHITE))
        .fill(theme.accent)
        .corner_radius(CornerRadius::same(8))
}

fn accent_button(ui: &mut egui::Ui, theme: &Theme, text: &str) -> egui::Response {
    ui.add(accent_widget(theme, text))
}
