//! The AI assistant: a calm right-side panel plus a provider settings modal.
//!
//! This module owns the assistant's UI and state (providers, per-project chat session,
//! selected model) and drives provider connections through `ai_net` (off-thread HTTP) and
//! `keystore` (secure key storage). The structured circuit-edit command API, diff preview,
//! and auto-fix loop are layered on next — the types here are shaped so that slots in without
//! reshaping the UI.
//!
//! Design goals: the base app stays calm (just one toolbar toggle + this panel), and the
//! panel itself reads like a real product — a clear header, roomy chat, and a composer with
//! the model switcher right where you type.

use std::sync::mpsc::{Receiver, Sender};

use eframe::egui::{self, Align, Color32, CornerRadius, Frame, Layout, Margin, RichText, Stroke};

use llmc::io::AiProviderConfig;

use super::ai_net::{self, AiEvent};
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
    /// Constructed when the assistant's reply arrives (networking layer, added next); already
    /// rendered by `bubble`.
    #[allow(dead_code)]
    Assistant,
}

pub struct ChatMessage {
    pub role: Role,
    pub text: String,
}

/// The selected model: an index into `AiSettings::providers` plus the model id.
#[derive(Clone)]
pub struct ModelRef {
    pub provider: usize,
    pub model: String,
}

/// Per-project assistant state (each tab keeps its own conversation, so switching tabs — and,
/// later, a request running in the background — never disturbs another project).
#[derive(Default)]
pub struct AiSession {
    pub messages: Vec<ChatMessage>,
    pub input: String,
    pub selected: Option<ModelRef>,
}

// ===================== rendering =====================

/// Render the assistant panel on the right. Returns nothing; mutates session/settings and may
/// flip `open_settings` when the gear is pressed.
pub fn panel(
    ui: &mut egui::Ui,
    theme: &Theme,
    settings: &mut AiSettings,
    session: &mut AiSession,
    open_settings: &mut bool,
) {
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
                    if !session.messages.is_empty()
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
                    conversation(ui, theme, session);
                }
            });

            ui.separator();
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                ui.add_space(12.0);
                ui.vertical(|ui| {
                    ui.set_width(width - 24.0);
                    composer(ui, theme, settings, session);
                });
            });
        });
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

/// The scrolling message list.
fn conversation(ui: &mut egui::Ui, theme: &Theme, session: &AiSession) {
    // The row width must be captured from the bounded scroll viewport; passing it down keeps
    // the per-message right/left alignment from treating the width as unbounded (which would
    // blow the panel's size up).
    let row_width = (ui.available_width() - 24.0).max(120.0);
    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .stick_to_bottom(true)
        .show(ui, |ui| {
            ui.add_space(4.0);
            for msg in &session.messages {
                bubble(ui, theme, msg, row_width);
                ui.add_space(8.0);
            }
        });
}

/// A single chat bubble: user right-aligned/accent, assistant left-aligned/panel.
fn bubble(ui: &mut egui::Ui, theme: &Theme, msg: &ChatMessage, row_width: f32) {
    let is_user = msg.role == Role::User;
    let fill = if is_user {
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
                    if !is_user {
                        ui.label(RichText::new("Assistant").small().color(theme.label_dim));
                    }
                    ui.label(RichText::new(&msg.text).color(theme.label));
                });
            });
    });
}

/// The bottom composer: model switcher + input + send.
fn composer(ui: &mut egui::Ui, theme: &Theme, settings: &mut AiSettings, session: &mut AiSession) {
    let live = settings.live_models();
    let ready = !live.is_empty();

    ui.add_enabled_ui(ready, |ui| {
        // Input box in a soft rounded surface.
        Frame::NONE
            .fill(surface(theme))
            .stroke(Stroke::new(1.0, theme.block_stroke))
            .corner_radius(CornerRadius::same(10))
            .inner_margin(Margin::symmetric(10, 8))
            .show(ui, |ui| {
                let hint = if ready {
                    "Ask the assistant to build or edit your circuit\u{2026}"
                } else {
                    "Connect a provider to start\u{2026}"
                };
                ui.add(
                    egui::TextEdit::multiline(&mut session.input)
                        .frame(Frame::NONE)
                        .desired_rows(2)
                        .hint_text(hint)
                        .desired_width(f32::INFINITY),
                );
            });

        ui.add_space(6.0);
        ui.horizontal(|ui| {
            model_switcher(ui, theme, settings, session);
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                let can_send = ready && !session.input.trim().is_empty();
                if ui
                    .add_enabled(can_send, accent_widget(theme, "Send"))
                    .clicked()
                {
                    let text = std::mem::take(&mut session.input).trim().to_string();
                    session.messages.push(ChatMessage {
                        role: Role::User,
                        text,
                    });
                    // The response + circuit edits are produced by the networking/command
                    // layer added next; the UI plumbing is already in place.
                }
            });
        });
    });
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
        ui.set_min_width(230.0);
        for (pi, model) in &live {
            let is_sel = session
                .selected
                .as_ref()
                .is_some_and(|s| s.provider == *pi && s.model == *model);
            let provider = settings.providers.get(*pi).map(|p| p.name.as_str());
            let row = model_job(ui, theme, model, provider, multi_provider);
            if ui.selectable_label(is_sel, row).clicked() {
                session.selected = Some(ModelRef {
                    provider: *pi,
                    model: model.to_string(),
                });
                ui.close();
            }
        }
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
