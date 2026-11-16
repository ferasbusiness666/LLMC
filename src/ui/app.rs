//! The application shell: the window's chrome and everything that spans documents — the menu
//! bar and toolbar, document tabs, the component sidebar, the Properties/Assistant sidebar, the
//! status bar, dialogs, toasts, keyboard shortcuts, files, and the assistant's agent loop.
//!
//! Each open project is a [`Document`]; tabs are fully independent (their own circuit, undo
//! history, simulation, view, and assistant conversation). The clipboard is shared.

use std::path::{Path, PathBuf};

use eframe::egui::{
    self, Align, Align2, Color32, CornerRadius, Event, Key, KeyboardShortcut, Layout, Margin,
    Modifiers, Pos2, Rect, RichText, Sense, Stroke, StrokeKind, Ui,
};

use llmc::backend::{CircuitManager, GroupTransform};
use llmc::io::{AppConfig, GateStyle, Project, WindowState, WireStyle};
use llmc::model::{BlockType, ChipId};

use super::assistant::{self, AiSettings};
use super::canvas::{CanvasAction, CanvasSettings, Overlays, EXAMPLES};
use super::design::{self, Kind, Theme};
use super::document::{
    title_from_path, Camera, ChipEdit, Clipboard, Document, FrameRequest, NoticeKind, Tool,
    DEFAULT_ZOOM,
};
use super::icons;
use super::inspector::{self, InspectorAction};
use super::palette::{self, PaletteAction};
use super::session::{self, Autosave, Instance, Recovered};

#[derive(Clone, Copy, PartialEq, Eq)]
enum RightTab {
    Properties,
    Assistant,
}

/// What a pending close prompt is about.
#[derive(Clone, Copy, PartialEq, Eq)]
enum CloseKind {
    /// A single tab, by its stable document id.
    Tab(u64),
    /// The whole window.
    Quit,
}

enum Dialog {
    CreateChip {
        name: String,
        from_selection: bool,
        error: Option<String>,
    },
    RenameChip {
        id: ChipId,
        name: String,
        error: Option<String>,
    },
    DeleteChip {
        id: ChipId,
    },
    Close(CloseKind),
    Shortcuts,
    About,
    /// Unsaved work left behind by a crash, offered for recovery at launch.
    Recover(Vec<Recovered>),
}

struct Toast {
    text: String,
    kind: NoticeKind,
    born: f64,
}

const TOAST_SECS: f64 = 3.6;

impl Toast {
    /// Problems stay up long enough to read; long messages get extra time.
    fn life(&self) -> f64 {
        let base = match self.kind {
            NoticeKind::Warning | NoticeKind::Error => 8.0,
            _ => TOAST_SECS,
        };
        base + (self.text.chars().count().saturating_sub(80) as f64) * 0.03
    }
}

pub struct LlmcApp {
    docs: Vec<Document>,
    active: usize,
    config: AppConfig,
    dark: bool,
    clipboard: Option<Clipboard>,
    untitled_count: usize,
    next_doc_id: u64,
    ai: AiSettings,
    right_open: bool,
    right_tab: RightTab,
    dialog: Option<Dialog>,
    ai_settings_open: bool,
    allow_quit: bool,
    toasts: Vec<Toast>,
    palette_search: String,
    overlays: Overlays,
    window_title: String,
    /// Single-instance lock + inbox (other launches hand their files to this window).
    instance: Option<Instance>,
    autosave: Autosave,
    /// Manager revision last autosaved, per document id.
    autosaved: std::collections::HashMap<u64, u64>,
    /// Last known window geometry (saved on exit).
    window_state: Option<WindowState>,
    /// The window has been checked against the monitor size once.
    monitor_checked: bool,
    /// The theme last sent to the native title bar.
    title_theme: Option<bool>,
    /// The tab the strip last scrolled into view (by document id).
    tab_in_view: Option<u64>,
}

impl LlmcApp {
    pub fn new(
        cc: &eframe::CreationContext<'_>,
        config: AppConfig,
        initial: Vec<PathBuf>,
        instance: Option<Instance>,
    ) -> Self {
        design::install_fonts(&cc.egui_ctx);
        design::apply_style(&cc.egui_ctx, config.dark_mode);
        let (autosave, recovered) = Autosave::start();
        let crashed = session::take_crash_marker();
        if let Some(inbox) = instance.as_ref().map(Instance::inbox_dir) {
            // Wake the (otherwise idle) UI when another launch hands us files.
            let ctx = cc.egui_ctx.clone();
            std::thread::spawn(move || loop {
                std::thread::sleep(std::time::Duration::from_millis(400));
                let waiting = std::fs::read_dir(&inbox)
                    .map(|mut d| d.next().is_some())
                    .unwrap_or(false);
                if waiting {
                    ctx.request_repaint();
                }
            });
        }
        let window_state = config.window;
        let mut app = Self {
            docs: Vec::new(),
            active: 0,
            dark: config.dark_mode,
            config,
            clipboard: None,
            untitled_count: 0,
            next_doc_id: 0,
            ai: AiSettings::default(),
            right_open: false,
            right_tab: RightTab::Properties,
            dialog: None,
            ai_settings_open: false,
            allow_quit: false,
            toasts: Vec::new(),
            palette_search: String::new(),
            overlays: Overlays::default(),
            window_title: String::new(),
            instance,
            autosave,
            autosaved: std::collections::HashMap::new(),
            window_state,
            monitor_checked: false,
            title_theme: None,
            tab_in_view: None,
        };
        if app.docs.is_empty() {
            let doc = app.fresh_document();
            app.push_doc(doc);
        }
        // Reopen last session's projects, then whatever was passed on the command line.
        let reopen: Vec<PathBuf> = app
            .config
            .open_tabs
            .iter()
            .filter(|p| p.exists())
            .cloned()
            .collect();
        for path in reopen.iter().chain(initial.iter()) {
            app.open_path(path);
        }
        if let Some(log) = crashed {
            app.docs[app.active].notify(
                NoticeKind::Warning,
                format!(
                    "LLMC closed unexpectedly last time. Details were saved to {}.",
                    log.display()
                ),
            );
        }
        if !recovered.is_empty() {
            app.dialog = Some(Dialog::Recover(recovered));
        }
        let saved = app.config.ai_providers.clone();
        app.ai.apply_config(&saved);
        app.ai.preferred = app.config.last_model.clone();
        app.ai.fallbacks = app.config.fallback_models.clone();
        app.ai.log_prompts = app.config.log_prompts;
        app
    }

    fn theme(&self) -> Theme {
        Theme::current(self.dark)
    }

    fn canvas_settings(&self) -> CanvasSettings {
        CanvasSettings {
            gate_style: self.config.gate_style,
            wire_style: self.config.wire_style,
            scroll_zooms: self.config.scroll_zooms,
            show_grid: self.config.show_grid,
            show_minimap: self.config.show_minimap,
        }
    }

    fn toast(&mut self, ctx: &egui::Context, kind: NoticeKind, text: impl Into<String>) {
        let text = text.into();
        self.toasts.retain(|t| t.text != text);
        self.toasts.push(Toast {
            text,
            kind,
            born: ctx.input(|i| i.time),
        });
        if self.toasts.len() > 4 {
            self.toasts.remove(0);
        }
    }

    // ===================== documents =====================

    fn fresh_document(&mut self) -> Document {
        self.untitled_count += 1;
        Document::new_empty(format!("Untitled {}", self.untitled_count))
    }

    fn push_doc(&mut self, mut doc: Document) -> usize {
        doc.id = self.next_doc_id;
        self.next_doc_id += 1;
        self.docs.push(doc);
        self.docs.len() - 1
    }

    fn new_tab(&mut self) {
        let doc = self.fresh_document();
        self.active = self.push_doc(doc);
    }

    fn doc_index(&self, id: u64) -> Option<usize> {
        self.docs.iter().position(|d| d.id == id)
    }

    fn open_dialog(&mut self) {
        let mut dlg = rfd::FileDialog::new().add_filter("LLMC project", &["llmc"]);
        if let Some(dir) = &self.config.last_dir {
            dlg = dlg.set_directory(dir);
        }
        if let Some(path) = dlg.pick_file() {
            self.open_path(&path);
        }
    }

    /// Load a project into a new tab (or focus it if already open). A pristine empty
    /// "Untitled" tab is replaced rather than left behind.
    fn open_path(&mut self, path: &Path) {
        let canon = |p: &Path| std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf());
        let want = canon(path);
        if let Some(i) = self
            .docs
            .iter()
            .position(|d| d.path.as_deref().map(canon).as_ref() == Some(&want))
        {
            self.active = i;
            return;
        }
        match Project::load(path) {
            Ok(project) => {
                self.remember(path);
                let doc = Document::from_project(project, path.to_path_buf());
                let reuse = self.docs.get(self.active).is_some_and(|d| {
                    d.path.is_none()
                        && d.chip_edit.is_none()
                        && !d.is_dirty()
                        && d.manager.circuit.blocks.is_empty()
                        && d.manager.chips.is_empty()
                        && d.ai.messages.is_empty()
                });
                if reuse {
                    let id = self.docs[self.active].id;
                    self.docs[self.active] = doc;
                    self.docs[self.active].id = id;
                } else {
                    self.active = self.push_doc(doc);
                }
                let d = &mut self.docs[self.active];
                d.notify(
                    NoticeKind::Success,
                    format!("Opened {}", title_from_path(path)),
                );
            }
            Err(e) => {
                if let Some(d) = self.docs.get_mut(self.active) {
                    d.notify(
                        NoticeKind::Error,
                        format!("Couldn't open {}: {e}", path.display()),
                    );
                }
                self.config.recent_files.retain(|p| p != path);
                self.config.save();
            }
        }
    }

    fn remember(&mut self, path: &Path) {
        if let Some(parent) = path.parent() {
            self.config.last_dir = Some(parent.to_path_buf());
        }
        self.config.push_recent(path);
        self.config.save();
    }

    /// Save a tab (prompting for a location if needed). Returns whether it was written. A
    /// chip-editing tab "saves" by applying its working copy back into the source project.
    fn save_doc(&mut self, i: usize) -> bool {
        if i >= self.docs.len() {
            return false;
        }
        if self.docs[i].chip_edit.is_some() {
            return self.apply_chip_edit(i);
        }
        match self.docs[i].path.clone() {
            Some(p) => self.write_doc(i, &p),
            None => self.save_doc_as(i),
        }
    }

    fn save_doc_as(&mut self, i: usize) -> bool {
        if i >= self.docs.len() {
            return false;
        }
        if self.docs[i].chip_edit.is_some() {
            return self.apply_chip_edit(i);
        }
        let mut dlg = rfd::FileDialog::new()
            .add_filter("LLMC project", &["llmc"])
            .set_file_name(format!("{}.llmc", self.docs[i].title));
        if let Some(dir) = &self.config.last_dir {
            dlg = dlg.set_directory(dir);
        }
        match dlg.save_file() {
            Some(path) => {
                self.docs[i].path = Some(path.clone());
                self.docs[i].title = title_from_path(&path);
                self.write_doc(i, &path)
            }
            None => false,
        }
    }

    fn write_doc(&mut self, i: usize, path: &Path) -> bool {
        // The project is named after its file (not "untitled").
        if self.docs[i].chip_edit.is_none() {
            self.docs[i].manager.circuit.name = title_from_path(path);
        }
        let project = self.docs[i].to_project(self.config.save_chat);
        match project.save(path) {
            Ok(()) => {
                self.docs[i].mark_saved();
                self.autosave.remove(self.docs[i].id);
                self.autosaved.remove(&self.docs[i].id);
                self.docs[i].notify(
                    NoticeKind::Success,
                    format!("Saved {}", title_from_path(path)),
                );
                self.remember(path);
                true
            }
            Err(e) => {
                self.docs[i].notify(NoticeKind::Error, format!("Save failed: {e}"));
                false
            }
        }
    }

    /// Close a tab unconditionally. The window always keeps at least one tab.
    fn close_tab(&mut self, i: usize) {
        if i >= self.docs.len() {
            return;
        }
        self.docs[i].ai.request_stop();
        self.autosave.remove(self.docs[i].id);
        self.autosaved.remove(&self.docs[i].id);
        if self.docs.len() == 1 {
            let doc = self.fresh_document();
            self.docs.clear();
            self.push_doc(doc);
            self.active = 0;
            return;
        }
        self.docs.remove(i);
        if self.active > i || self.active >= self.docs.len() {
            self.active = self.active.saturating_sub(1).min(self.docs.len() - 1);
        }
    }

    /// Close a tab, asking first if it has unsaved work or a running assistant.
    fn request_close_tab(&mut self, i: usize) {
        let Some(doc) = self.docs.get(i) else {
            return;
        };
        if doc.is_dirty() || doc.ai.is_running() || doc.ai.is_pending() {
            self.active = i;
            self.dialog = Some(Dialog::Close(CloseKind::Tab(doc.id)));
        } else {
            self.close_tab(i);
        }
    }

    // ===================== chips =====================

    /// Open a chip's internals in a NEW tab. Chips opened from inside a chip tab still belong
    /// to the root project, so updates always land in the real library.
    fn open_chip_edit_tab(&mut self, src: usize, chip: ChipId) {
        let mut root = src;
        if let Some(e) = self.docs[src].chip_edit {
            match self.doc_index(e.source_doc) {
                Some(r) => root = r,
                None => {
                    self.docs[src].notify(NoticeKind::Warning, "Its project was closed.");
                    return;
                }
            }
        }
        let source_doc = self.docs[root].id;
        if let Some(existing) = self.docs.iter().position(|d| {
            d.chip_edit
                .is_some_and(|e| e.source_doc == source_doc && e.chip == chip)
        }) {
            self.active = existing;
            return;
        }
        let Some(def) = self.docs[root].manager.chips.get(chip) else {
            return;
        };
        let name = def.name.clone();
        let mgr =
            CircuitManager::from_parts(def.circuit.clone(), self.docs[root].manager.chips.clone());
        let mut doc = Document::new(mgr, None, name.clone(), Camera::default());
        doc.chip_edit = Some(ChipEdit { source_doc, chip });
        doc.frame_request = Some(FrameRequest::All);
        doc.mark_saved();
        doc.notify(
            NoticeKind::Info,
            format!("Editing \u{201c}{name}\u{201d} \u{2014} Update chip (Ctrl+S) applies it everywhere"),
        );
        self.active = self.push_doc(doc);
    }

    /// Apply a chip tab's working copy back into its project. Returns whether it landed.
    fn apply_chip_edit(&mut self, i: usize) -> bool {
        let Some(ChipEdit { source_doc, chip }) = self.docs[i].chip_edit else {
            return false;
        };
        let circuit = self.docs[i].manager.circuit.clone();
        let Some(src) = self.doc_index(source_doc) else {
            self.docs[i].notify(NoticeKind::Error, "Can't update: its project was closed.");
            return false;
        };
        match self.docs[src].apply_chip_circuit(chip, circuit) {
            Ok(msg) => {
                // Refresh the library copies held by other chip tabs of the same project.
                if let Some(def) = self.docs[src].manager.chips.get(chip).cloned() {
                    for (j, d) in self.docs.iter_mut().enumerate() {
                        if j != src && d.chip_edit.is_some_and(|e| e.source_doc == source_doc) {
                            d.manager.chips.insert(def.clone());
                            d.manager.touch();
                        }
                    }
                }
                self.docs[i].mark_saved();
                self.docs[i].notify(NoticeKind::Success, msg);
                true
            }
            Err(e) => {
                self.docs[i].notify(NoticeKind::Error, format!("Can't update: {e}"));
                false
            }
        }
    }

    // ===================== theme =====================

    fn toggle_theme(&mut self, ctx: &egui::Context) {
        self.dark = !self.dark;
        self.config.dark_mode = self.dark;
        design::apply_style(ctx, self.dark);
        self.config.save();
    }

    // ===================== actions =====================

    fn copy(&mut self, ctx: &egui::Context, cut: bool) {
        let doc = &mut self.docs[self.active];
        if let Some(clip) = doc.copy_selection() {
            let n = clip.blocks.len();
            ctx.copy_text(clip.to_text());
            self.clipboard = Some(clip);
            if cut {
                doc.delete_selection();
            }
            let verb = if cut { "Cut" } else { "Copied" };
            self.toast(
                ctx,
                NoticeKind::Info,
                format!("{verb} {n} block{}", if n == 1 { "" } else { "s" }),
            );
        }
    }

    fn paste(&mut self, at: Option<llmc::model::Vec2f>, from_system: Option<&str>) {
        let clip = from_system
            .and_then(Clipboard::from_text)
            .or_else(|| self.clipboard.clone());
        if let Some(c) = clip {
            self.docs[self.active].paste(&c, at);
        }
    }

    fn insert_example(&mut self, i: usize) {
        let Some((name, _, json)) = EXAMPLES.get(i) else {
            return;
        };
        let ops = llmc::ai::parse_reply(json).ops;
        let doc = &mut self.docs[self.active];
        let report = doc.apply_ai_ops(ops, true);
        // The user inserted these: they're the user's blocks, not the assistant's.
        doc.glow = None;
        for id in &report.new_blocks {
            doc.ai_blocks.remove(id);
        }
        if report.changed {
            doc.clear_selection();
            doc.frame_request = Some(FrameRequest::All);
            if !doc.running {
                doc.toggle_running();
            }
            doc.notify(
                NoticeKind::Success,
                format!("Added the {name} example \u{2014} try clicking the inputs"),
            );
        }
    }

    fn handle_canvas_action(&mut self, ctx: &egui::Context, a: CanvasAction) {
        match a {
            CanvasAction::ShowProperties => {
                self.right_open = true;
                self.right_tab = RightTab::Properties;
            }
            CanvasAction::EditChip(c) => self.open_chip_edit_tab(self.active, c),
            CanvasAction::CreateChip { from_selection } => {
                self.dialog = Some(Dialog::CreateChip {
                    name: String::new(),
                    from_selection,
                    error: None,
                })
            }
            CanvasAction::Copy => self.copy(ctx, false),
            CanvasAction::Cut => self.copy(ctx, true),
            CanvasAction::Paste(at) => self.paste(at, None),
            CanvasAction::OpenAssistant => {
                self.right_open = true;
                self.right_tab = RightTab::Assistant;
            }
            CanvasAction::InsertExample(i) => self.insert_example(i),
            CanvasAction::Open => self.open_dialog(),
            CanvasAction::ToggleMinimap => {
                self.config.show_minimap = !self.config.show_minimap;
                self.config.save();
            }
            CanvasAction::UpdateChip => {
                self.apply_chip_edit(self.active);
            }
        }
    }

    // ===================== keyboard =====================

    fn global_shortcuts(&mut self, ctx: &egui::Context) {
        let cmd = Modifiers::COMMAND;
        let cmd_shift = Modifiers::COMMAND | Modifiers::SHIFT;
        let hit = |m: Modifiers, k: Key| {
            ctx.input_mut(|i| i.consume_shortcut(&KeyboardShortcut::new(m, k)))
        };
        if hit(cmd_shift, Key::S) {
            self.save_doc_as(self.active);
        } else if hit(cmd, Key::S) {
            self.save_doc(self.active);
        }
        if hit(cmd, Key::O) {
            self.open_dialog();
        }
        if hit(cmd, Key::N) {
            self.new_tab();
        }
        if hit(cmd, Key::W) {
            self.request_close_tab(self.active);
        }
        if hit(cmd_shift, Key::Tab) {
            self.active = (self.active + self.docs.len() - 1) % self.docs.len();
        } else if hit(cmd, Key::Tab) {
            self.active = (self.active + 1) % self.docs.len();
        }
        if hit(Modifiers::NONE, Key::F5) {
            self.docs[self.active].toggle_running();
        }
        if hit(Modifiers::NONE, Key::F10) {
            self.docs[self.active].step_once();
        }
        if hit(Modifiers::NONE, Key::F1) {
            self.dialog = Some(Dialog::Shortcuts);
        }
        if hit(cmd, Key::B) {
            self.config.show_palette = !self.config.show_palette;
            self.config.save();
        }
        if hit(cmd, Key::I) {
            self.toggle_right(RightTab::Properties);
        }
        if hit(cmd, Key::L) {
            self.toggle_right(RightTab::Assistant);
        }
    }

    fn toggle_right(&mut self, tab: RightTab) {
        if self.right_open && self.right_tab == tab {
            self.right_open = false;
        } else {
            self.right_open = true;
            self.right_tab = tab;
        }
    }

    // ===================== chrome =====================

    fn top_bar(&mut self, ui: &mut Ui, ctx: &egui::Context) {
        let t = self.theme();
        egui::Panel::top("top-bar")
            .exact_size(46.0)
            .frame(
                egui::Frame::NONE
                    .fill(t.bg_app)
                    .inner_margin(Margin::symmetric(12, 0)),
            )
            .show(ui, |ui| {
                let full = ui.max_rect();
                ui.horizontal_centered(|ui| {
                    design::logo(ui, &t, 22.0);
                    ui.label(
                        RichText::new("LLMC")
                            .font(design::semibold(14.5))
                            .color(t.text),
                    );
                    ui.add_space(8.0);
                    self.menus(ui, ctx, &t);
                });
                // Centered simulation controls (hidden when the window is too narrow).
                let center = Rect::from_center_size(full.center(), egui::vec2(318.0, 36.0));
                if center.left() > full.left() + 360.0 && center.right() < full.right() - 300.0 {
                    let mut child = ui.new_child(
                        egui::UiBuilder::new()
                            .max_rect(center)
                            .layout(Layout::left_to_right(Align::Center)),
                    );
                    self.sim_controls(&mut child, &t);
                }
                let right = Rect::from_min_max(
                    Pos2::new(full.right() - 300.0, full.top()),
                    full.right_bottom(),
                );
                let mut child = ui.new_child(
                    egui::UiBuilder::new()
                        .max_rect(right)
                        .layout(Layout::right_to_left(Align::Center)),
                );
                self.right_controls(&mut child, ctx, &t);
            });
    }

    fn menus(&mut self, ui: &mut Ui, ctx: &egui::Context, t: &Theme) {
        egui::MenuBar::new().ui(ui, |ui| {
            let top = |s: &str| {
                RichText::new(s)
                    .font(design::medium(13.0))
                    .color(t.text_muted)
            };
            ui.menu_button(top("File"), |ui| self.file_menu(ui, ctx, t));
            ui.menu_button(top("Edit"), |ui| self.edit_menu(ui, ctx, t));
            ui.menu_button(top("View"), |ui| self.view_menu(ui, ctx, t));
            ui.menu_button(top("Simulate"), |ui| self.simulate_menu(ui, t));
            ui.menu_button(top("Help"), |ui| {
                ui.set_width(230.0);
                if design::menu_item(ui, t, icons::KEYBOARD, "Keyboard shortcuts", "F1") {
                    self.dialog = Some(Dialog::Shortcuts);
                }
                if design::menu_item(ui, t, icons::INFORMATION, "About LLMC", "") {
                    self.dialog = Some(Dialog::About);
                }
            });
        });
    }

    fn file_menu(&mut self, ui: &mut Ui, ctx: &egui::Context, t: &Theme) {
        ui.set_width(260.0);
        if design::menu_item(ui, t, icons::FILE_ADD, "New project", "Ctrl+N") {
            self.new_tab();
        }
        if design::menu_item(ui, t, icons::FOLDER_OPEN, "Open\u{2026}", "Ctrl+O") {
            self.open_dialog();
        }
        let recent = self.config.recent_files.clone();
        if !recent.is_empty() {
            let row = design::submenu_row(ui, t, icons::HISTORY, "Open recent");
            egui::containers::menu::SubMenu::new().show(ui, &row, |ui| {
                ui.set_width(280.0);
                for p in &recent {
                    let name = title_from_path(p);
                    let dir = p
                        .parent()
                        .map(|d| d.display().to_string())
                        .unwrap_or_default();
                    if design::menu_item(ui, t, icons::FILE, &name, "") {
                        self.open_path(p);
                    }
                    ui.horizontal(|ui| {
                        ui.add_space(26.0);
                        ui.add(
                            egui::Label::new(
                                RichText::new(dir)
                                    .font(design::regular(10.5))
                                    .color(t.text_faint),
                            )
                            .truncate(),
                        );
                    });
                }
                design::divider(ui, t);
                if design::menu_item(ui, t, icons::DELETE_BIN_6, "Clear recent", "") {
                    self.config.recent_files.clear();
                    self.config.save();
                }
            });
        }
        design::divider(ui, t);
        let chip = self.docs[self.active].chip_edit.is_some();
        if design::menu_item(
            ui,
            t,
            icons::SAVE_3,
            if chip { "Update chip" } else { "Save" },
            "Ctrl+S",
        ) {
            self.save_doc(self.active);
        }
        if design::menu_item_ex(
            ui,
            t,
            icons::SAVE_3,
            "Save as\u{2026}",
            "Ctrl+Shift+S",
            !chip,
            false,
        ) {
            self.save_doc_as(self.active);
        }
        design::divider(ui, t);
        if design::menu_item(ui, t, icons::KEY_2, "AI providers\u{2026}", "") {
            self.ai_settings_open = true;
        }
        let save_chat = self.config.save_chat;
        if design::menu_item_ex(
            ui,
            t,
            icons::CHAT_3,
            "Save chat in project files",
            "",
            true,
            save_chat,
        ) {
            self.config.save_chat = !save_chat;
            self.config.save();
        }
        design::divider(ui, t);
        if design::menu_item(ui, t, icons::CLOSE, "Close tab", "Ctrl+W") {
            self.request_close_tab(self.active);
        }
        if design::menu_item(ui, t, icons::DOOR_OPEN, "Quit", "Alt+F4") {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
    }

    fn edit_menu(&mut self, ui: &mut Ui, ctx: &egui::Context, t: &Theme) {
        ui.set_width(250.0);
        let (can_undo, can_redo, has_sel, chip_tab) = {
            let d = &self.docs[self.active];
            (
                d.manager.undo.can_undo(),
                d.manager.undo.can_redo(),
                !d.selection.is_empty(),
                d.chip_edit.is_some(),
            )
        };
        if design::menu_item_ex(
            ui,
            t,
            icons::ARROW_GO_BACK,
            "Undo",
            "Ctrl+Z",
            can_undo,
            false,
        ) {
            self.docs[self.active].undo();
        }
        if design::menu_item_ex(
            ui,
            t,
            icons::ARROW_GO_FORWARD,
            "Redo",
            "Ctrl+Y",
            can_redo,
            false,
        ) {
            self.docs[self.active].redo();
        }
        design::divider(ui, t);
        if design::menu_item_ex(ui, t, icons::SCISSORS_CUT, "Cut", "Ctrl+X", has_sel, false) {
            self.copy(ctx, true);
        }
        if design::menu_item_ex(ui, t, icons::FILE_COPY, "Copy", "Ctrl+C", has_sel, false) {
            self.copy(ctx, false);
        }
        if design::menu_item_ex(
            ui,
            t,
            icons::CLIPBOARD,
            "Paste",
            "Ctrl+V",
            self.clipboard.is_some(),
            false,
        ) {
            self.paste(None, None);
        }
        if design::menu_item_ex(
            ui,
            t,
            icons::CHECKBOX_MULTIPLE_BLANK,
            "Duplicate",
            "Ctrl+D",
            has_sel,
            false,
        ) {
            self.docs[self.active].duplicate_selection();
        }
        if design::menu_item_ex(ui, t, icons::DELETE_BIN_6, "Delete", "Del", has_sel, false) {
            self.docs[self.active].delete_selection();
        }
        design::divider(ui, t);
        if design::menu_item_ex(
            ui,
            t,
            icons::CLOCKWISE_2,
            "Rotate right",
            "R",
            has_sel,
            false,
        ) {
            self.docs[self.active].transform_selection(GroupTransform::RotateCw);
        }
        if design::menu_item_ex(
            ui,
            t,
            icons::ANTICLOCKWISE_2,
            "Rotate left",
            "Shift+R",
            has_sel,
            false,
        ) {
            self.docs[self.active].transform_selection(GroupTransform::RotateCcw);
        }
        if design::menu_item_ex(ui, t, icons::FLIP_HORIZONTAL, "Flip", "F", has_sel, false) {
            self.docs[self.active].transform_selection(GroupTransform::Flip);
        }
        design::divider(ui, t);
        if design::menu_item(ui, t, icons::CHECKBOX_CIRCLE_FILL, "Select all", "Ctrl+A") {
            self.docs[self.active].select_all();
        }
        if design::menu_item_ex(
            ui,
            t,
            icons::CPU,
            "Create chip\u{2026}",
            "",
            !chip_tab,
            false,
        ) {
            self.dialog = Some(Dialog::CreateChip {
                name: String::new(),
                from_selection: has_sel,
                error: None,
            });
        }
    }

    fn view_menu(&mut self, ui: &mut Ui, ctx: &egui::Context, t: &Theme) {
        ui.set_width(270.0);
        if design::menu_item(ui, t, icons::FOCUS_3, "Fit everything", "Home") {
            self.docs[self.active].frame_request = Some(FrameRequest::All);
        }
        let has_sel = !self.docs[self.active].selection.is_empty();
        if design::menu_item_ex(
            ui,
            t,
            icons::FOCUS_2,
            "Zoom to selection",
            "Shift+2",
            has_sel,
            false,
        ) {
            self.docs[self.active].frame_request = Some(FrameRequest::Selection);
        }
        if design::menu_item(ui, t, icons::ZOOM_IN, "Actual size (100%)", "Ctrl+0") {
            self.docs[self.active].set_zoom(DEFAULT_ZOOM);
        }
        design::divider(ui, t);
        let mut changed = false;
        let c = &mut self.config;
        if design::menu_item_ex(
            ui,
            t,
            icons::LAYOUT_LEFT,
            "Component sidebar",
            "Ctrl+B",
            true,
            c.show_palette,
        ) {
            c.show_palette = !c.show_palette;
            changed = true;
        }
        if design::menu_item_ex(ui, t, icons::GRID, "Grid", "", true, c.show_grid) {
            c.show_grid = !c.show_grid;
            changed = true;
        }
        if design::menu_item_ex(ui, t, icons::MAP_2, "Minimap", "", true, c.show_minimap) {
            c.show_minimap = !c.show_minimap;
            changed = true;
        }
        design::divider(ui, t);
        design::menu_section(ui, t, "Gates");
        if design::menu_item_ex(
            ui,
            t,
            icons::SHAPES,
            "Standard symbols",
            "",
            true,
            c.gate_style == GateStyle::Shapes,
        ) {
            c.gate_style = GateStyle::Shapes;
            changed = true;
        }
        if design::menu_item_ex(
            ui,
            t,
            icons::LAYOUT_GRID,
            "Labeled boxes",
            "",
            true,
            c.gate_style == GateStyle::Boxes,
        ) {
            c.gate_style = GateStyle::Boxes;
            changed = true;
        }
        design::menu_section(ui, t, "Wires");
        if design::menu_item_ex(
            ui,
            t,
            icons::ROUTE,
            "Curved",
            "",
            true,
            c.wire_style == WireStyle::Curved,
        ) {
            c.wire_style = WireStyle::Curved;
            changed = true;
        }
        if design::menu_item_ex(
            ui,
            t,
            icons::NODE_TREE,
            "Right angles",
            "",
            true,
            c.wire_style == WireStyle::Orthogonal,
        ) {
            c.wire_style = WireStyle::Orthogonal;
            changed = true;
        }
        design::menu_section(ui, t, "Mouse wheel");
        if design::menu_item_ex(ui, t, icons::ZOOM_IN, "Zooms", "", true, c.scroll_zooms) {
            c.scroll_zooms = true;
            changed = true;
        }
        if design::menu_item_ex(
            ui,
            t,
            icons::DRAG_MOVE_2,
            "Pans (Ctrl+wheel zooms)",
            "",
            true,
            !c.scroll_zooms,
        ) {
            c.scroll_zooms = false;
            changed = true;
        }
        if changed {
            self.config.save();
        }
        design::divider(ui, t);
        if design::menu_item_ex(
            ui,
            t,
            icons::LAYOUT_RIGHT,
            "Properties panel",
            "Ctrl+I",
            true,
            self.right_open && self.right_tab == RightTab::Properties,
        ) {
            self.toggle_right(RightTab::Properties);
        }
        if design::menu_item_ex(
            ui,
            t,
            icons::SPARKLING_2,
            "Assistant panel",
            "Ctrl+L",
            true,
            self.right_open && self.right_tab == RightTab::Assistant,
        ) {
            self.toggle_right(RightTab::Assistant);
        }
        let label = if self.dark {
            "Light theme"
        } else {
            "Dark theme"
        };
        if design::menu_item(
            ui,
            t,
            if self.dark { icons::SUN } else { icons::MOON },
            label,
            "",
        ) {
            self.toggle_theme(ctx);
        }
    }

    fn simulate_menu(&mut self, ui: &mut Ui, t: &Theme) {
        ui.set_width(240.0);
        let doc = &mut self.docs[self.active];
        let running = doc.running;
        if design::menu_item(
            ui,
            t,
            if running {
                icons::PAUSE_FILL
            } else {
                icons::PLAY_FILL
            },
            if running { "Pause" } else { "Run" },
            "F5",
        ) {
            doc.toggle_running();
        }
        if design::menu_item(ui, t, icons::SKIP_FORWARD_FILL, "Step", "F10") {
            doc.step_once();
        }
        if design::menu_item(ui, t, icons::RESTART, "Reset state", "") {
            doc.reset_sim();
            doc.notify(
                NoticeKind::Info,
                "Simulation reset: memory cleared, time back to 0",
            );
        }
        design::divider(ui, t);
        design::menu_section(ui, t, "Speed");
        for s in [0.25f32, 0.5, 1.0, 2.0, 4.0, 10.0] {
            if design::menu_item_ex(
                ui,
                t,
                icons::SPEED_UP,
                &inspector::format_speed(s),
                "",
                true,
                (doc.speed - s).abs() < 1e-4,
            ) {
                doc.speed = s;
            }
        }
    }

    fn sim_controls(&mut self, ui: &mut Ui, t: &Theme) {
        let doc = &mut self.docs[self.active];
        egui::Frame::NONE
            .fill(t.surface)
            .stroke(Stroke::new(1.0, t.border))
            .corner_radius(CornerRadius::same(design::radius::LG))
            .inner_margin(Margin::symmetric(4, 3))
            .show(ui, |ui| {
                ui.spacing_mut().item_spacing.x = 2.0;
                let running = doc.running;
                let (glyph, label) = if running {
                    (icons::PAUSE_FILL, "Pause")
                } else {
                    (icons::PLAY_FILL, "Run")
                };
                let kind = if running {
                    Kind::Secondary
                } else {
                    Kind::Primary
                };
                if design::button_ex(ui, t, kind, Some(glyph), label, true, 28.0)
                    .on_hover_text("Run / pause the simulation (F5)")
                    .clicked()
                {
                    doc.toggle_running();
                }
                if design::icon_button_sized(
                    ui,
                    t,
                    icons::SKIP_FORWARD_FILL,
                    "Step to the next clock edge (F10)",
                    false,
                    28.0,
                    true,
                )
                .clicked()
                {
                    doc.step_once();
                }
                if design::icon_button_sized(
                    ui,
                    t,
                    icons::RESTART,
                    "Reset simulation state",
                    false,
                    28.0,
                    true,
                )
                .clicked()
                {
                    doc.reset_sim();
                }
                design::vdivider(ui, t, 22.0);
                inspector::speed_menu(ui, doc, t);
                ui.label(
                    RichText::new(format!("{:>7.2}s", doc.sim.time()))
                        .font(design::mono(12.0))
                        .color(t.text_faint),
                );
            });
    }

    fn right_controls(&mut self, ui: &mut Ui, ctx: &egui::Context, t: &Theme) {
        // Laid out right-to-left.
        let ai_on = self.right_open && self.right_tab == RightTab::Assistant;
        let working = self.docs[self.active].ai.is_running();
        let label = if working {
            "Working\u{2026}"
        } else {
            "Assistant"
        };
        let kind = if ai_on {
            Kind::Primary
        } else {
            Kind::Secondary
        };
        if design::button(ui, t, kind, Some(icons::SPARKLING_2_FILL), label)
            .on_hover_text("AI assistant (Ctrl+L)")
            .clicked()
        {
            self.toggle_right(RightTab::Assistant);
        }
        let props_on = self.right_open && self.right_tab == RightTab::Properties;
        if design::icon_button(ui, t, icons::LAYOUT_RIGHT, "Properties (Ctrl+I)", props_on)
            .clicked()
        {
            self.toggle_right(RightTab::Properties);
        }
        let glyph = if self.dark { icons::SUN } else { icons::MOON };
        if design::icon_button(ui, t, glyph, "Light / dark theme", false).clicked() {
            self.toggle_theme(ctx);
        }
        design::vdivider(ui, t, 24.0);
        let (can_undo, can_redo) = {
            let u = &self.docs[self.active].manager.undo;
            (u.can_undo(), u.can_redo())
        };
        if design::icon_button_sized(
            ui,
            t,
            icons::ARROW_GO_FORWARD,
            "Redo (Ctrl+Y)",
            false,
            30.0,
            can_redo,
        )
        .clicked()
        {
            self.docs[self.active].redo();
        }
        if design::icon_button_sized(
            ui,
            t,
            icons::ARROW_GO_BACK,
            "Undo (Ctrl+Z)",
            false,
            30.0,
            can_undo,
        )
        .clicked()
        {
            self.docs[self.active].undo();
        }
    }

    fn tab_bar(&mut self, ui: &mut Ui) {
        let t = self.theme();
        let mut select = None;
        let mut close = None;
        let mut close_others = None;
        let mut add = false;
        egui::Panel::top("tabs")
            .exact_size(36.0)
            .frame(egui::Frame::NONE.fill(t.bg_app).inner_margin(Margin {
                left: 8,
                right: 8,
                top: 5,
                bottom: 0,
            }))
            .show(ui, |ui| {
                let r = ui.max_rect();
                ui.painter().line_segment(
                    [
                        Pos2::new(r.left() - 8.0, r.bottom() - 0.5),
                        Pos2::new(r.right() + 8.0, r.bottom() - 0.5),
                    ],
                    Stroke::new(1.0, t.border),
                );
                egui::ScrollArea::horizontal()
                    .scroll_bar_visibility(egui::scroll_area::ScrollBarVisibility::AlwaysHidden)
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = 2.0;
                            let active_id = self.docs[self.active].id;
                            for (i, doc) in self.docs.iter().enumerate() {
                                let (event, rect) = tab(ui, &t, doc, i == self.active);
                                // A newly active tab (new, opened, Ctrl+Tab) scrolls into view.
                                if i == self.active && self.tab_in_view != Some(active_id) {
                                    ui.scroll_to_rect(rect, None);
                                    self.tab_in_view = Some(active_id);
                                }
                                match event {
                                    TabEvent::Select => select = Some(i),
                                    TabEvent::Close => close = Some(i),
                                    TabEvent::CloseOthers => close_others = Some(i),
                                    TabEvent::None => {}
                                }
                            }
                            ui.add_space(4.0);
                            if design::icon_button_sized(
                                ui,
                                &t,
                                icons::ADD,
                                "New project (Ctrl+N)",
                                false,
                                26.0,
                                true,
                            )
                            .clicked()
                            {
                                add = true;
                            }
                        });
                    });
            });
        if let Some(i) = select {
            self.active = i;
        }
        if add {
            self.new_tab();
        }
        if let Some(i) = close {
            self.request_close_tab(i);
        }
        if let Some(keep) = close_others {
            let keep_id = self.docs[keep].id;
            let ids: Vec<u64> = self
                .docs
                .iter()
                .map(|d| d.id)
                .filter(|&id| id != keep_id)
                .collect();
            for id in ids {
                if let Some(i) = self.doc_index(id) {
                    if !self.docs[i].is_dirty() && !self.docs[i].ai.is_running() {
                        self.close_tab(i);
                    }
                }
            }
            self.active = self.doc_index(keep_id).unwrap_or(0);
        }
    }

    fn status_bar(&mut self, ui: &mut Ui) {
        let t = self.theme();
        let doc = &self.docs[self.active];
        egui::Panel::bottom("status")
            .exact_size(28.0)
            .frame(
                egui::Frame::NONE
                    .fill(t.bg_app)
                    .inner_margin(Margin::symmetric(12, 0)),
            )
            .show(ui, |ui| {
                let r = ui.max_rect();
                ui.painter().line_segment(
                    [
                        Pos2::new(r.left() - 12.0, r.top() + 0.5),
                        Pos2::new(r.right() + 12.0, r.top() + 0.5),
                    ],
                    Stroke::new(1.0, t.border),
                );
                ui.horizontal_centered(|ui| {
                    let (color, text) = inspector::sim_state(doc, &t);
                    design::pill(ui, &t, color, text);
                    design::vdivider(ui, &t, 16.0);
                    ui.label(
                        RichText::new(status_hint(doc))
                            .font(design::regular(12.0))
                            .color(t.text_muted),
                    );
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        ui.label(
                            RichText::new(format!(
                                "{:.0}%",
                                doc.camera.zoom / DEFAULT_ZOOM * 100.0
                            ))
                            .font(design::medium(12.0))
                            .color(t.text_muted),
                        );
                        design::vdivider(ui, &t, 16.0);
                        if let Some(w) = doc.cursor_world {
                            ui.label(
                                RichText::new(format!(
                                    "{}, {}",
                                    w.x.floor() as i32,
                                    w.y.floor() as i32
                                ))
                                .font(design::mono(11.5))
                                .color(t.text_faint),
                            );
                            design::vdivider(ui, &t, 16.0);
                        }
                        let c = &doc.manager.circuit;
                        ui.label(
                            RichText::new(format!(
                                "{} \u{b7} {} \u{b7} {}",
                                design::plural(c.blocks.len(), "block", "blocks"),
                                design::plural(c.connections.len(), "wire", "wires"),
                                design::plural(doc.manager.chips.len(), "chip", "chips")
                            ))
                            .font(design::regular(12.0))
                            .color(t.text_muted),
                        );
                        if doc.is_dirty() {
                            design::vdivider(ui, &t, 16.0);
                            ui.label(
                                RichText::new("Unsaved")
                                    .font(design::medium(12.0))
                                    .color(t.warning),
                            );
                        }
                    });
                });
            });
    }

    fn left_sidebar(&mut self, ui: &mut Ui, actions: &mut Vec<PaletteAction>) {
        if !self.config.show_palette {
            return;
        }
        let t = self.theme();
        let gate_style = self.config.gate_style;
        egui::Panel::left("palette")
            .resizable(true)
            .default_size(236.0)
            .min_size(200.0)
            .max_size(340.0)
            .frame(
                egui::Frame::NONE
                    .fill(t.bg_panel)
                    .inner_margin(Margin {
                        left: 10,
                        right: 10,
                        top: 0,
                        bottom: 6,
                    })
                    .stroke(Stroke::new(1.0, t.border)),
            )
            .show(ui, |ui| {
                let doc = &mut self.docs[self.active];
                let footer_h = 40.0;
                let h = (ui.available_height() - footer_h).max(100.0);
                ui.allocate_ui(egui::vec2(ui.available_width(), h), |ui| {
                    actions.extend(palette::palette(
                        ui,
                        doc,
                        &t,
                        gate_style,
                        &mut self.palette_search,
                    ));
                });
                design::divider(ui, &t);
                if let Some(a) = palette::footer(ui, &t) {
                    actions.push(a);
                }
            });
    }

    fn right_sidebar(
        &mut self,
        ui: &mut Ui,
        actions: &mut Vec<InspectorAction>,
    ) -> assistant::PanelResponse {
        let mut panel = assistant::PanelResponse::default();
        if !self.right_open {
            return panel;
        }
        let t = self.theme();
        egui::Panel::right("right-sidebar")
            .resizable(true)
            .default_size(372.0)
            .min_size(300.0)
            .max_size(620.0)
            .frame(
                egui::Frame::NONE
                    .fill(t.bg_panel)
                    .inner_margin(Margin::same(12))
                    .stroke(Stroke::new(1.0, t.border)),
            )
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    let w = ui.available_width() - 38.0;
                    ui.allocate_ui(egui::vec2(w, 28.0), |ui| {
                        let sel = if self.right_tab == RightTab::Properties {
                            0
                        } else {
                            1
                        };
                        let opts = [
                            (icons::LAYOUT_RIGHT, "Properties"),
                            (icons::SPARKLING_2, "Assistant"),
                        ];
                        if let Some(i) = design::segmented(ui, &t, &opts, sel) {
                            self.right_tab = if i == 0 {
                                RightTab::Properties
                            } else {
                                RightTab::Assistant
                            };
                        }
                    });
                    if design::icon_button(ui, &t, icons::CLOSE, "Close panel", false).clicked() {
                        self.right_open = false;
                    }
                });
                ui.add_space(10.0);
                let doc = &mut self.docs[self.active];
                match self.right_tab {
                    RightTab::Properties => actions.extend(inspector::inspector(ui, doc, &t)),
                    RightTab::Assistant => {
                        panel = assistant::panel(ui, &t, &mut self.ai, &mut doc.ai)
                    }
                }
            });
        panel
    }

    // ===================== dialogs =====================

    fn dialogs(&mut self, ctx: &egui::Context) {
        let t = self.theme();
        let Some(dialog) = self.dialog.take() else {
            return;
        };
        let backdrop = Color32::from_black_alpha(if t.dark { 140 } else { 70 });
        let modal = |id: &str| {
            egui::Modal::new(egui::Id::new(id))
                .frame(design::dialog_frame(&t))
                .backdrop_color(backdrop)
        };
        let error_line = |ui: &mut Ui, e: &str| {
            ui.add_space(6.0);
            ui.label(
                RichText::new(format!("{}  {e}", icons::ERROR_WARNING))
                    .font(design::regular(12.5))
                    .color(t.danger),
            );
        };
        let mut keep: Option<Dialog> = None;
        match dialog {
            Dialog::CreateChip {
                mut name,
                mut from_selection,
                mut error,
            } => {
                let has_sel = !self.docs[self.active].selection.is_empty();
                let (mut done, mut submit) = (false, false);
                let r = modal("create-chip").show(ctx, |ui| {
                    ui.set_width(420.0);
                    design::dialog_header(ui, &t, icons::CPU, t.accent, "Create chip", "Package a circuit as a reusable block.");
                    ui.add_space(16.0);
                    design::field_label(ui, &t, "Name");
                    let resp = design::text_field(ui, &t, &mut name, "e.g. FullAdder", None);
                    if !resp.has_focus() && !resp.lost_focus() {
                        resp.request_focus();
                    }
                    if resp.changed() {
                        error = None;
                    }
                    if resp.lost_focus() && ui.input(|i| i.key_pressed(Key::Enter)) {
                        submit = true;
                    }
                    ui.add_space(10.0);
                    design::field_label(ui, &t, "Contents");
                    let sel = if from_selection && has_sel { 1 } else { 0 };
                    let opts = [(icons::LAYOUT_GRID, "Whole canvas"), (icons::FOCUS_2, "Selection")];
                    if let Some(i) = design::segmented(ui, &t, &opts, sel) {
                        from_selection = i == 1 && has_sel;
                    }
                    ui.add_space(6.0);
                    design::hint(ui, &t, "Switches become input pins and LEDs become output pins, ordered top to bottom.");
                    // What the chip will look like, before it's made.
                    let (ins, outs) = {
                        let d = &self.docs[self.active];
                        let c = &d.manager.circuit;
                        let count = |ty: BlockType| {
                            c.iter_blocks()
                                .filter(|b| b.ty == ty)
                                .filter(|b| !(from_selection && has_sel) || d.selection.contains(&b.id))
                                .count()
                        };
                        (count(BlockType::Switch), count(BlockType::Led))
                    };
                    ui.add_space(6.0);
                    let warn = ins == 0 || outs == 0;
                    ui.label(
                        RichText::new(format!(
                            "{}  {} \u{b7} {}{}",
                            if warn { icons::ERROR_WARNING } else { icons::CHECKBOX_CIRCLE_FILL },
                            design::plural(ins, "input pin", "input pins"),
                            design::plural(outs, "output pin", "output pins"),
                            if outs == 0 {
                                " \u{2014} add at least one LED"
                            } else if ins == 0 {
                                " \u{2014} with no inputs its outputs never change"
                            } else {
                                ""
                            }
                        ))
                        .font(design::medium(12.0))
                        .color(if warn { t.warning } else { t.success }),
                    );
                    if let Some(e) = &error {
                        error_line(ui, e);
                    }
                    ui.add_space(16.0);
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        if design::button(ui, &t, Kind::Primary, Some(icons::CHECK), "Create chip").clicked() {
                            submit = true;
                        }
                        if design::button(ui, &t, Kind::Secondary, None, "Cancel").clicked() {
                            done = true;
                        }
                    });
                });
                if submit {
                    match self.docs[self.active].create_chip(&name, from_selection) {
                        Ok(_) => done = true,
                        Err(e) => error = Some(e),
                    }
                }
                if !(done || r.should_close()) {
                    keep = Some(Dialog::CreateChip {
                        name,
                        from_selection,
                        error,
                    });
                }
            }
            Dialog::RenameChip {
                id,
                mut name,
                mut error,
            } => {
                let (mut done, mut submit) = (false, false);
                let r = modal("rename-chip").show(ctx, |ui| {
                    ui.set_width(400.0);
                    design::dialog_header(
                        ui,
                        &t,
                        icons::EDIT_2,
                        t.accent,
                        "Rename chip",
                        "Every placed instance keeps working.",
                    );
                    ui.add_space(16.0);
                    let resp = design::text_field(ui, &t, &mut name, "Chip name", None);
                    if !resp.has_focus() && !resp.lost_focus() {
                        resp.request_focus();
                    }
                    if resp.changed() {
                        error = None;
                    }
                    if resp.lost_focus() && ui.input(|i| i.key_pressed(Key::Enter)) {
                        submit = true;
                    }
                    if let Some(e) = &error {
                        error_line(ui, e);
                    }
                    ui.add_space(16.0);
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        if design::button(ui, &t, Kind::Primary, None, "Rename").clicked() {
                            submit = true;
                        }
                        if design::button(ui, &t, Kind::Secondary, None, "Cancel").clicked() {
                            done = true;
                        }
                    });
                });
                if submit {
                    match self.docs[self.active].rename_chip(id, &name) {
                        Ok(()) => done = true,
                        Err(e) => error = Some(e),
                    }
                }
                if !(done || r.should_close()) {
                    keep = Some(Dialog::RenameChip { id, name, error });
                }
            }
            Dialog::DeleteChip { id } => {
                let doc = &self.docs[self.active];
                let name = doc
                    .manager
                    .chips
                    .get(id)
                    .map(|c| c.name.clone())
                    .unwrap_or_default();
                let users = doc.chip_users(id);
                let n = doc.chip_instances(id);
                let (mut done, mut confirm) = (false, false);
                let r = modal("delete-chip").show(ctx, |ui| {
                    ui.set_width(420.0);
                    design::dialog_header(
                        ui,
                        &t,
                        icons::DELETE_BIN_6,
                        t.danger,
                        &format!("Delete \u{201c}{name}\u{201d}?"),
                        "You can undo this with Ctrl+Z.",
                    );
                    ui.add_space(14.0);
                    if n > 0 {
                        design::hint(ui, &t, format!("{n} placed instance(s) on this canvas will be removed too."));
                    }
                    if !users.is_empty() {
                        ui.add_space(4.0);
                        ui.label(
                            RichText::new(format!(
                                "{}  It's used inside: {}. Those chips will stop working until you fix them.",
                                icons::ALERT,
                                users.join(", ")
                            ))
                            .font(design::regular(12.5))
                            .color(t.warning),
                        );
                    }
                    ui.add_space(16.0);
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        if design::button(ui, &t, Kind::Danger, Some(icons::DELETE_BIN_6), "Delete chip").clicked() {
                            confirm = true;
                        }
                        if design::button(ui, &t, Kind::Secondary, None, "Cancel").clicked() {
                            done = true;
                        }
                    });
                });
                if confirm {
                    self.docs[self.active].delete_chip(id);
                    done = true;
                }
                if !(done || r.should_close()) {
                    keep = Some(Dialog::DeleteChip { id });
                }
            }
            Dialog::Close(kind) => keep = self.close_dialog(ctx, &t, kind),
            Dialog::Shortcuts => {
                let r = modal("shortcuts").show(ctx, |ui| {
                    ui.set_width(640.0);
                    design::dialog_header(
                        ui,
                        &t,
                        icons::KEYBOARD,
                        t.accent,
                        "Keyboard shortcuts",
                        "Work faster without leaving the canvas.",
                    );
                    ui.add_space(14.0);
                    ui.columns(2, |cols| {
                        shortcut_group(
                            &mut cols[0],
                            &t,
                            "File",
                            &[
                                ("Ctrl+N", "New project"),
                                ("Ctrl+O", "Open"),
                                ("Ctrl+S", "Save / update chip"),
                                ("Ctrl+Shift+S", "Save as"),
                                ("Ctrl+W", "Close tab"),
                                ("Ctrl+Tab", "Next tab"),
                            ],
                        );
                        shortcut_group(
                            &mut cols[0],
                            &t,
                            "Simulation",
                            &[("F5", "Run / pause"), ("F10", "Step to next clock edge")],
                        );
                        shortcut_group(
                            &mut cols[0],
                            &t,
                            "Panels",
                            &[
                                ("Ctrl+B", "Component sidebar"),
                                ("Ctrl+I", "Properties"),
                                ("Ctrl+L", "Assistant"),
                                ("F1", "This help"),
                            ],
                        );
                        shortcut_group(
                            &mut cols[1],
                            &t,
                            "Edit",
                            &[
                                ("Ctrl+Z", "Undo"),
                                ("Ctrl+Y", "Redo"),
                                ("Ctrl+C", "Copy"),
                                ("Ctrl+X", "Cut"),
                                ("Ctrl+V", "Paste at cursor"),
                                ("Ctrl+D", "Duplicate"),
                                ("Del", "Delete"),
                                ("Ctrl+A", "Select all"),
                            ],
                        );
                        shortcut_group(
                            &mut cols[1],
                            &t,
                            "Canvas",
                            &[
                                ("R", "Rotate right"),
                                ("Shift+R", "Rotate left"),
                                ("F", "Flip"),
                                ("Arrows", "Nudge (Shift = 5)"),
                                ("Home", "Fit everything"),
                                ("Shift+2", "Zoom to selection"),
                                ("Ctrl+0", "Zoom 100%"),
                                ("Space", "Hold to pan"),
                                ("Esc", "Cancel / deselect"),
                            ],
                        );
                    });
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        if design::button(ui, &t, Kind::Primary, None, "Done").clicked() {
                            ui.close();
                        }
                    });
                });
                if !r.should_close() {
                    keep = Some(Dialog::Shortcuts);
                }
            }
            Dialog::About => {
                let r = modal("about").show(ctx, |ui| {
                    ui.set_width(400.0);
                    ui.vertical_centered(|ui| {
                        design::logo(ui, &t, 52.0);
                        ui.add_space(10.0);
                        ui.label(RichText::new("LLMC").font(design::semibold(20.0)).color(t.text));
                        ui.label(
                            RichText::new(format!("Version {}", env!("CARGO_PKG_VERSION")))
                                .font(design::regular(12.0))
                                .color(t.text_faint),
                        );
                        ui.add_space(8.0);
                        ui.label(
                            RichText::new("A digital-logic builder and simulator with an AI assistant that designs, wires, and tests circuits.")
                                .font(design::regular(13.0))
                                .color(t.text_muted),
                        );
                        ui.add_space(10.0);
                        ui.label(
                            RichText::new("Inter & JetBrains Mono (SIL OFL) \u{b7} Remix Icon (Apache 2.0)")
                                .font(design::regular(11.0))
                                .color(t.text_faint),
                        );
                        ui.add_space(14.0);
                        if design::button(ui, &t, Kind::Primary, None, "Close").clicked() {
                            ui.close();
                        }
                    });
                });
                if !r.should_close() {
                    keep = Some(Dialog::About);
                }
            }
            Dialog::Recover(items) => {
                #[derive(PartialEq)]
                enum Choice {
                    Recover,
                    Discard,
                }
                let mut choice = None;
                let r = modal("recover").show(ctx, |ui| {
                    ui.set_width(460.0);
                    design::dialog_header(
                        ui,
                        &t,
                        icons::HISTORY,
                        t.warning,
                        "Recover unsaved work?",
                        "LLMC didn't close normally last time. These projects had unsaved changes.",
                    );
                    ui.add_space(12.0);
                    for item in &items {
                        design::card(&t)
                            .inner_margin(egui::Margin::symmetric(12, 8))
                            .show(ui, |ui| {
                                ui.set_width(ui.available_width());
                                ui.horizontal(|ui| {
                                    ui.label(
                                        RichText::new(icons::FILE)
                                            .font(design::icon_font(15.0))
                                            .color(t.text_muted),
                                    );
                                    ui.label(
                                        RichText::new(&item.title)
                                            .font(design::medium(13.0))
                                            .color(t.text),
                                    );
                                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                        ui.label(
                                            RichText::new(format!("autosaved {}", item.age()))
                                                .font(design::regular(11.5))
                                                .color(t.text_faint),
                                        );
                                    });
                                });
                                let place = item
                                    .original
                                    .as_ref()
                                    .map(|p| p.display().to_string())
                                    .unwrap_or_else(|| "Never saved".to_string());
                                ui.label(
                                    RichText::new(place)
                                        .font(design::regular(11.5))
                                        .color(t.text_faint),
                                );
                            });
                        ui.add_space(6.0);
                    }
                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        if design::button(ui, &t, Kind::Ghost, None, "Discard")
                            .on_hover_text("Delete these autosaves")
                            .clicked()
                        {
                            choice = Some(Choice::Discard);
                        }
                        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                            if design::button(
                                ui,
                                &t,
                                Kind::Primary,
                                Some(icons::HISTORY),
                                "Recover",
                            )
                            .clicked()
                            {
                                choice = Some(Choice::Recover);
                            }
                        });
                    });
                });
                match choice {
                    Some(Choice::Recover) => {
                        for item in &items {
                            let Some(project) = item.load() else {
                                continue;
                            };
                            let doc = Document::recovered(
                                project,
                                item.original.clone(),
                                item.title.clone(),
                            );
                            // A tab already showing the on-disk version gives way to the newer
                            // recovered one.
                            let same = item.original.as_ref().and_then(|o| {
                                self.docs.iter().position(|d| d.path.as_ref() == Some(o))
                            });
                            match same {
                                Some(i) => {
                                    let id = self.docs[i].id;
                                    self.docs[i] = doc;
                                    self.docs[i].id = id;
                                    self.active = i;
                                }
                                None => self.active = self.push_doc(doc),
                            }
                        }
                        Autosave::discard(&items);
                        let n = items.len();
                        self.docs[self.active].notify(
                            NoticeKind::Success,
                            format!(
                                "Recovered {} \u{2014} save to keep {}",
                                design::plural(n, "project", "projects"),
                                if n == 1 { "it" } else { "them" }
                            ),
                        );
                    }
                    Some(Choice::Discard) => Autosave::discard(&items),
                    // Closed with Esc: decide next launch.
                    None if r.should_close() => {}
                    None => keep = Some(Dialog::Recover(items)),
                }
            }
        }
        if self.dialog.is_none() {
            self.dialog = keep;
        }
    }

    /// The unsaved-changes / assistant-running prompt. Returns the dialog to keep open.
    fn close_dialog(&mut self, ctx: &egui::Context, t: &Theme, kind: CloseKind) -> Option<Dialog> {
        let (title, dirty_names, running) = match kind {
            CloseKind::Tab(id) => {
                let d = &self.docs[self.doc_index(id)?];
                (
                    format!("Close \u{201c}{}\u{201d}?", d.title),
                    if d.is_dirty() {
                        vec![d.title.clone()]
                    } else {
                        vec![]
                    },
                    d.ai.is_running() || d.ai.is_pending(),
                )
            }
            CloseKind::Quit => (
                "Quit LLMC?".to_string(),
                self.docs
                    .iter()
                    .filter(|d| d.is_dirty())
                    .map(|d| d.title.clone())
                    .collect(),
                self.docs
                    .iter()
                    .any(|d| d.ai.is_running() || d.ai.is_pending()),
            ),
        };
        #[derive(PartialEq)]
        enum Choice {
            Save,
            Discard,
            Cancel,
        }
        let mut choice: Option<Choice> = None;
        let r = egui::Modal::new(egui::Id::new("close-confirm"))
            .frame(design::dialog_frame(t))
            .backdrop_color(Color32::from_black_alpha(if t.dark { 140 } else { 70 }))
            .show(ctx, |ui| {
                ui.set_width(440.0);
                design::dialog_header(ui, t, icons::ALERT, t.warning, &title, "");
                ui.add_space(12.0);
                if !dirty_names.is_empty() {
                    design::hint(
                        ui,
                        t,
                        format!("Unsaved changes in: {}", dirty_names.join(", ")),
                    );
                }
                if running {
                    ui.add_space(4.0);
                    ui.label(
                        RichText::new(format!(
                            "{}  The assistant is still working \u{2014} it will be stopped.",
                            icons::ROBOT_2
                        ))
                        .font(design::regular(12.5))
                        .color(t.warning),
                    );
                }
                ui.add_space(16.0);
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    if !dirty_names.is_empty() {
                        let label = if kind == CloseKind::Quit {
                            "Save all"
                        } else {
                            "Save"
                        };
                        if design::button(ui, t, Kind::Primary, Some(icons::SAVE_3), label)
                            .clicked()
                        {
                            choice = Some(Choice::Save);
                        }
                    }
                    let (label, k) = if dirty_names.is_empty() {
                        ("Close anyway", Kind::Primary)
                    } else {
                        ("Don't save", Kind::Secondary)
                    };
                    if design::button(ui, t, k, None, label).clicked() {
                        choice = Some(Choice::Discard);
                    }
                    if design::button(ui, t, Kind::Ghost, None, "Cancel").clicked() {
                        choice = Some(Choice::Cancel);
                    }
                });
            });
        match (choice, kind) {
            (Some(Choice::Save), CloseKind::Tab(id)) => {
                if let Some(i) = self.doc_index(id) {
                    if self.save_doc(i) {
                        if let Some(i) = self.doc_index(id) {
                            self.close_tab(i);
                        }
                    }
                }
                None
            }
            (Some(Choice::Discard), CloseKind::Tab(id)) => {
                if let Some(i) = self.doc_index(id) {
                    self.close_tab(i);
                }
                None
            }
            (Some(Choice::Save), CloseKind::Quit) => {
                if self.save_all() {
                    self.allow_quit = true;
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }
                None
            }
            (Some(Choice::Discard), CloseKind::Quit) => {
                self.allow_quit = true;
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                None
            }
            (Some(Choice::Cancel), _) => None,
            (None, _) => (!r.should_close()).then_some(Dialog::Close(kind)),
        }
    }

    /// Save every dirty tab. Chip tabs go first: applying one dirties its project, which is then
    /// saved in the second pass (doing it in tab order used to lose the chip update on quit).
    fn save_all(&mut self) -> bool {
        let chip_tabs: Vec<u64> = self
            .docs
            .iter()
            .filter(|d| d.chip_edit.is_some())
            .map(|d| d.id)
            .collect();
        for id in chip_tabs {
            if let Some(i) = self.doc_index(id) {
                self.docs[i].refresh_dirty();
                if self.docs[i].is_dirty() && !self.save_doc(i) {
                    return false;
                }
            }
        }
        for i in 0..self.docs.len() {
            self.docs[i].refresh_dirty();
            if self.docs[i].chip_edit.is_none() && self.docs[i].is_dirty() && !self.save_doc(i) {
                return false;
            }
        }
        true
    }

    fn draw_toasts(&mut self, ctx: &egui::Context) {
        let now = ctx.input(|i| i.time);
        self.toasts.retain(|t| now - t.born < t.life());
        if self.toasts.is_empty() {
            return;
        }
        let t = self.theme();
        egui::Area::new(egui::Id::new("toasts"))
            .order(egui::Order::Foreground)
            .anchor(Align2::CENTER_BOTTOM, egui::vec2(0.0, -44.0))
            .interactable(false)
            .show(ctx, |ui| {
                for toast in &self.toasts {
                    let age = (now - toast.born) as f32;
                    let alpha =
                        (age / 0.15).min(1.0) * ((toast.life() as f32 - age) / 0.4).clamp(0.0, 1.0);
                    let (glyph, color) = match toast.kind {
                        NoticeKind::Info => (icons::INFORMATION, t.accent),
                        NoticeKind::Success => (icons::CHECKBOX_CIRCLE_FILL, t.success),
                        NoticeKind::Warning => (icons::ALERT, t.warning),
                        NoticeKind::Error => (icons::ERROR_WARNING, t.danger),
                    };
                    ui.scope(|ui| {
                        ui.set_opacity(alpha);
                        design::floating(&t)
                            .inner_margin(Margin::symmetric(14, 9))
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    ui.label(
                                        RichText::new(glyph)
                                            .font(design::icon_font(16.0))
                                            .color(color),
                                    );
                                    ui.label(
                                        RichText::new(&toast.text)
                                            .font(design::medium(13.0))
                                            .color(t.text),
                                    );
                                });
                            });
                    });
                    ui.add_space(6.0);
                }
            });
        ctx.request_repaint_after(std::time::Duration::from_millis(50));
    }

    fn file_drop(&mut self, ctx: &egui::Context) {
        let (dropped, hovering) =
            ctx.input(|i| (i.raw.dropped_files.clone(), !i.raw.hovered_files.is_empty()));
        for f in dropped {
            if let Some(p) = f.path {
                if p.extension()
                    .is_some_and(|e| e.eq_ignore_ascii_case("llmc"))
                {
                    self.open_path(&p);
                } else {
                    self.toast(
                        ctx,
                        NoticeKind::Warning,
                        "Only .llmc project files can be opened",
                    );
                }
            }
        }
        if hovering {
            let t = self.theme();
            let screen = ctx.content_rect();
            let p = ctx.layer_painter(egui::LayerId::new(
                egui::Order::Foreground,
                egui::Id::new("drop"),
            ));
            p.rect_filled(screen, CornerRadius::ZERO, Color32::from_black_alpha(120));
            let card = Rect::from_center_size(screen.center(), egui::vec2(320.0, 120.0));
            p.rect_filled(
                card,
                CornerRadius::same(design::radius::XL),
                t.surface_raised,
            );
            p.rect_stroke(
                card,
                CornerRadius::same(design::radius::XL),
                Stroke::new(2.0, t.accent),
                StrokeKind::Inside,
            );
            p.text(
                card.center() - egui::vec2(0.0, 14.0),
                Align2::CENTER_CENTER,
                icons::FOLDER_OPEN,
                design::icon_font(28.0),
                t.accent,
            );
            p.text(
                card.center() + egui::vec2(0.0, 22.0),
                Align2::CENTER_CENTER,
                "Drop to open project",
                design::semibold(15.0),
                t.text,
            );
        }
    }

    /// The assistant's agent loop for every tab (requests keep running in background tabs).
    fn run_agents(&mut self, ctx: &egui::Context) {
        let settings = &mut self.ai;
        let mut any_pending = false;
        let mut any_waiting = false;
        for doc in &mut self.docs {
            let reply = doc.ai.poll_chat();
            if let Some(ops) = doc.ai.take_pending_ops() {
                // The user's own blocks may only be removed when their request asked for
                // changes ("replace", "fix", "delete", …).
                let allow = llmc::ai::user_allows_removal(doc.ai.last_user_text());
                let mut report = doc.apply_ai_ops(ops, allow);
                report.errors.extend(doc.ai.take_problems());
                // Build the feedback from the full report BEFORE handing activity to the chat.
                let feedback = doc.agent_observation(&report);
                doc.ai.attach_edit_result(report.activity, report.errors);
                if doc.ai.should_continue() {
                    let prompt = doc.system_prompt();
                    doc.ai.continue_agent(settings, prompt, feedback);
                } else {
                    doc.ai.end_agent();
                }
            } else if reply {
                let prompt = doc.system_prompt();
                doc.ai.handle_reply_without_edits(settings, prompt);
            }
            // A rate-limited model hands over to the next one in the fallback chain at once.
            doc.ai.try_fallback(settings);
            if doc.ai.retry_due() {
                let prompt = doc.system_prompt();
                doc.ai.retry_now(settings, prompt);
            }
            any_pending |= doc.ai.is_pending();
            any_waiting |= doc.ai.retry_scheduled();
        }
        if any_pending {
            ctx.request_repaint_after(std::time::Duration::from_millis(120));
        } else if any_waiting {
            ctx.request_repaint_after(std::time::Duration::from_millis(250));
        }
    }

    fn update_title(&mut self, ctx: &egui::Context) {
        let d = &self.docs[self.active];
        let mut title = format!(
            "{}{} \u{2014} LLMC",
            if d.chip_edit.is_some() {
                "\u{270e} "
            } else {
                ""
            },
            d.title
        );
        if d.is_dirty() {
            title.push_str(" \u{2022}");
        }
        if title != self.window_title {
            ctx.send_viewport_cmd(egui::ViewportCommand::Title(title.clone()));
            self.window_title = title;
        }
    }
}

enum TabEvent {
    None,
    Select,
    Close,
    CloseOthers,
}

/// One document tab: icon, title, and a close button (a dot when unsaved, like most editors).
fn tab(ui: &mut Ui, t: &Theme, doc: &Document, active: bool) -> (TabEvent, Rect) {
    let font = design::medium(12.5);
    let title = super::glyphs::ellipsize(ui.painter(), &doc.title, &font, 180.0);
    let galley = ui
        .painter()
        .layout_no_wrap(title.clone(), font.clone(), t.text);
    let w = galley.size().x + 62.0;
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(w, 31.0), Sense::click());
    let hovered = resp.hovered();
    let p = ui.painter();
    let rad = CornerRadius {
        nw: 8,
        ne: 8,
        sw: 0,
        se: 0,
    };
    if active {
        p.rect_filled(rect, rad, t.bg_panel);
        p.rect_stroke(rect, rad, Stroke::new(1.0, t.border), StrokeKind::Inside);
        // Merge the active tab into the content below, and mark it with an accent line.
        p.line_segment(
            [
                rect.left_bottom() + egui::vec2(1.0, -0.5),
                rect.right_bottom() + egui::vec2(-1.0, -0.5),
            ],
            Stroke::new(2.0, t.bg_panel),
        );
        p.line_segment(
            [
                rect.left_top() + egui::vec2(8.0, 0.5),
                rect.right_top() + egui::vec2(-8.0, 0.5),
            ],
            Stroke::new(2.0, t.accent),
        );
    } else if hovered {
        p.rect_filled(
            rect.shrink2(egui::vec2(0.0, 2.0)),
            CornerRadius::same(7),
            t.surface_hover,
        );
    }
    let (glyph, gcolor) = if doc.chip_edit.is_some() {
        (icons::CPU, t.accent)
    } else if doc.ai.is_running() {
        (icons::SPARKLING_2_FILL, t.accent)
    } else {
        (
            icons::FILE,
            if active { t.text_muted } else { t.text_faint },
        )
    };
    p.text(
        Pos2::new(rect.left() + 16.0, rect.center().y),
        Align2::CENTER_CENTER,
        glyph,
        design::icon_font(14.0),
        gcolor,
    );
    p.text(
        Pos2::new(rect.left() + 30.0, rect.center().y),
        Align2::LEFT_CENTER,
        title,
        font,
        if active { t.text } else { t.text_muted },
    );
    // Close button / unsaved dot.
    let cr = Rect::from_center_size(
        Pos2::new(rect.right() - 15.0, rect.center().y),
        egui::vec2(18.0, 18.0),
    );
    let close_resp = ui.interact(cr, resp.id.with("close"), Sense::click());
    let over = hovered || close_resp.hovered();
    if close_resp.hovered() {
        ui.painter()
            .rect_filled(cr, CornerRadius::same(5), t.surface_active);
    }
    if doc.is_dirty() && !over {
        ui.painter().circle_filled(cr.center(), 3.5, t.text_muted);
    } else if over || active {
        ui.painter().text(
            cr.center(),
            Align2::CENTER_CENTER,
            icons::CLOSE,
            design::icon_font(13.0),
            t.text_muted,
        );
    }
    let mut ev = TabEvent::None;
    if close_resp.clicked() || resp.middle_clicked() {
        ev = TabEvent::Close;
    } else if resp.clicked() {
        ev = TabEvent::Select;
    }
    let path = doc
        .path
        .as_ref()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| {
            if doc.chip_edit.is_some() {
                "Chip editor".into()
            } else {
                "Not saved yet".into()
            }
        });
    resp.on_hover_text(path).context_menu(|ui| {
        if design::menu_item(ui, t, icons::CLOSE, "Close", "Ctrl+W") {
            ev = TabEvent::Close;
        }
        if design::menu_item(ui, t, icons::CLOSE_CIRCLE, "Close other saved tabs", "") {
            ev = TabEvent::CloseOthers;
        }
    });
    (ev, rect)
}

fn shortcut_group(ui: &mut Ui, t: &Theme, title: &str, rows: &[(&str, &str)]) {
    design::section_label(ui, t, title);
    ui.add_space(2.0);
    for (keys, what) in rows {
        ui.horizontal(|ui| {
            ui.label(
                RichText::new(*what)
                    .font(design::regular(12.5))
                    .color(t.text_muted),
            );
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                ui.spacing_mut().item_spacing.x = 3.0;
                for k in keys.split('+').collect::<Vec<_>>().into_iter().rev() {
                    design::kbd(ui, t, k);
                }
            });
        });
    }
    ui.add_space(10.0);
}

/// The contextual hint in the status bar.
fn status_hint(doc: &Document) -> String {
    if let Some(ty) = doc.palette_drag {
        return format!("Drop to place {}", ty.display_name());
    }
    match doc.tool {
        Tool::Place(ty) => {
            let name = match ty {
                BlockType::Chip(c) => doc
                    .manager
                    .chips
                    .get(c)
                    .map(|d| d.name.clone())
                    .unwrap_or_default(),
                t => t.display_name().to_string(),
            };
            format!(
                "Click to place {name} \u{b7} Shift+click to place several \u{b7} Esc to cancel"
            )
        }
        Tool::Select if doc.pending_wire.is_some() => {
            "Click a target port to finish the wire \u{b7} Esc to cancel".into()
        }
        Tool::Select => match (doc.selection.len(), doc.selected_conns.len()) {
            (0, 0) => "Drag from a port to wire \u{b7} Space+drag to pan \u{b7} F5 to run".into(),
            (1, 0) => "R rotate \u{b7} F flip \u{b7} Del delete \u{b7} Enter for properties".into(),
            (n, w) => format!(
                "{n} block(s), {w} wire(s) selected \u{b7} Ctrl+D duplicate \u{b7} Del delete"
            ),
        },
    }
}

impl LlmcApp {
    /// Per-frame housekeeping for the process-level safety nets: files handed over by another
    /// launch, autosave, window geometry, and the native title bar's theme.
    fn housekeeping(&mut self, ctx: &egui::Context) {
        if let Some(paths) = self.instance.as_mut().and_then(Instance::poll) {
            for p in &paths {
                self.open_path(p);
            }
            ctx.send_viewport_cmd(egui::ViewportCommand::Minimized(false));
            ctx.send_viewport_cmd(egui::ViewportCommand::Focus);
        }

        if self.autosave.due() {
            for d in &self.docs {
                // Chip-editor tabs belong to their project; only whole projects are autosaved.
                if d.chip_edit.is_some() {
                    continue;
                }
                if d.is_dirty() {
                    let rev = d.manager.revision();
                    if self.autosaved.get(&d.id) != Some(&rev) {
                        let project = d.to_project(true);
                        self.autosave
                            .write(d.id, &project, &d.title, d.path.as_deref());
                        self.autosaved.insert(d.id, rev);
                    }
                } else if self.autosaved.remove(&d.id).is_some() {
                    self.autosave.remove(d.id);
                }
            }
        }
        // Autosave needs a frame now and then even when nothing moves.
        ctx.request_repaint_after(session::AUTOSAVE_EVERY);

        let (outer, inner, maximized, monitor) = ctx.input(|i| {
            let v = i.viewport();
            (v.outer_rect, v.inner_rect, v.maximized, v.monitor_size)
        });
        if maximized == Some(true) {
            if let Some(w) = &mut self.window_state {
                w.maximized = true;
            } else {
                self.window_state = Some(WindowState {
                    x: 40.0,
                    y: 40.0,
                    width: 1440.0,
                    height: 900.0,
                    maximized: true,
                });
            }
        } else if let (Some(o), Some(i)) = (outer, inner) {
            self.window_state = Some(WindowState {
                x: o.min.x,
                y: o.min.y,
                width: i.width(),
                height: i.height(),
                maximized: false,
            });
        }
        // A window restored from a bigger screen must not hang off this one.
        if !self.monitor_checked {
            if let (Some(m), Some(i)) = (monitor, inner) {
                self.monitor_checked = true;
                if maximized != Some(true) && (i.width() > m.x + 1.0 || i.height() > m.y - 40.0) {
                    let size = egui::vec2(
                        (m.x * 0.92).max(960.0).min(m.x),
                        (m.y * 0.88).max(600.0).min(m.y),
                    );
                    ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(size));
                    ctx.send_viewport_cmd(egui::ViewportCommand::OuterPosition(Pos2::new(
                        ((m.x - size.x) / 2.0).max(0.0),
                        ((m.y - size.y) / 2.0).max(0.0),
                    )));
                }
            }
        }

        if self.title_theme != Some(self.dark) {
            self.title_theme = Some(self.dark);
            ctx.send_viewport_cmd(egui::ViewportCommand::SetTheme(if self.dark {
                egui::SystemTheme::Dark
            } else {
                egui::SystemTheme::Light
            }));
        }
    }
}

impl eframe::App for LlmcApp {
    /// A clean exit: remember the window and open projects, and drop the autosaves (there is
    /// nothing to recover — unsaved work was either saved or knowingly discarded).
    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        self.config.window = self.window_state;
        self.config.open_tabs = self
            .docs
            .iter()
            .filter(|d| d.chip_edit.is_none())
            .filter_map(|d| d.path.clone())
            .collect();
        self.config.save();
        self.autosave.clear_session();
    }

    fn ui(&mut self, ui: &mut Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.housekeeping(&ctx);
        self.ai.poll();
        if self.ai.any_testing() {
            ctx.request_repaint_after(std::time::Duration::from_millis(120));
        }
        self.run_agents(&ctx);
        if self.ai.take_dirty() {
            self.config.ai_providers = self.ai.config_snapshot();
            self.config.last_model = self.ai.preferred.clone();
            self.config.fallback_models = self.ai.fallbacks.clone();
            self.config.log_prompts = self.ai.log_prompts;
            self.config.save();
        }

        // Advance the active document's simulation.
        let dt = ctx.input(|i| i.stable_dt).clamp(0.0, 0.1) as f64;
        {
            let doc = &mut self.docs[self.active];
            doc.tick(dt);
            if doc.running {
                ctx.request_repaint();
            }
        }

        // Keyboard: app-wide shortcuts always; canvas keys only when no text field has focus
        // (typing in the chat must never edit the circuit) and no dialog is open.
        if self.dialog.is_none() && !self.ai_settings_open {
            self.global_shortcuts(&ctx);
            if !ctx.text_edit_focused() {
                let actions = self.docs[self.active].handle_keys(&ctx);
                for a in actions {
                    self.handle_canvas_action(&ctx, a);
                }
                let pastes: Vec<String> = ctx.input(|i| {
                    i.events
                        .iter()
                        .filter_map(|e| match e {
                            Event::Paste(s) => Some(s.clone()),
                            _ => None,
                        })
                        .collect()
                });
                for p in pastes {
                    self.paste(None, Some(&p));
                }
            } else {
                self.docs[self.active].space_down = false;
            }
        }
        self.file_drop(&ctx);

        // Chrome.
        self.top_bar(ui, &ctx);
        self.tab_bar(ui);
        self.status_bar(ui);
        let mut palette_actions = Vec::new();
        self.left_sidebar(ui, &mut palette_actions);
        let mut inspector_actions = Vec::new();
        let panel = self.right_sidebar(ui, &mut inspector_actions);

        // Canvas.
        let t = self.theme();
        let settings = self.canvas_settings();
        let has_clip = self.clipboard.is_some();
        let mut canvas_actions = Vec::new();
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE.fill(t.bg_canvas))
            .show(ui, |ui| {
                let doc = &mut self.docs[self.active];
                canvas_actions = doc.canvas(ui, &t, &settings, &mut self.overlays, has_clip);
            });

        // Route actions.
        for a in canvas_actions {
            self.handle_canvas_action(&ctx, a);
        }
        for a in palette_actions {
            match a {
                PaletteAction::CreateChip => {
                    self.dialog = Some(Dialog::CreateChip {
                        name: String::new(),
                        from_selection: false,
                        error: None,
                    })
                }
                PaletteAction::EditChip(c) => self.open_chip_edit_tab(self.active, c),
                PaletteAction::RenameChip(id) => {
                    let name = self.docs[self.active]
                        .manager
                        .chips
                        .get(id)
                        .map(|c| c.name.clone())
                        .unwrap_or_default();
                    self.dialog = Some(Dialog::RenameChip {
                        id,
                        name,
                        error: None,
                    });
                }
                PaletteAction::DeleteChip(id) => self.dialog = Some(Dialog::DeleteChip { id }),
                PaletteAction::ShowShortcuts => self.dialog = Some(Dialog::Shortcuts),
            }
        }
        for a in inspector_actions {
            match a {
                InspectorAction::EditChip(c) => self.open_chip_edit_tab(self.active, c),
                InspectorAction::CreateChip { from_selection } => {
                    self.dialog = Some(Dialog::CreateChip {
                        name: String::new(),
                        from_selection,
                        error: None,
                    })
                }
            }
        }
        if panel.open_settings {
            self.ai_settings_open = true;
        }
        if panel.continue_reply {
            let doc = &mut self.docs[self.active];
            doc.ai.input = "Continue exactly where you left off.".to_string();
            let prompt = doc.system_prompt();
            doc.ai.send(&self.ai, prompt);
        } else if panel.send {
            let doc = &mut self.docs[self.active];
            let prompt = doc.system_prompt();
            doc.ai.send(&self.ai, prompt);
        } else if panel.retry {
            let doc = &mut self.docs[self.active];
            let prompt = doc.system_prompt();
            doc.ai.resend(&self.ai, prompt);
        }

        // Notices → toasts; dirty flags (cheap: only re-hashed when content changed).
        let mut notices = Vec::new();
        for d in &mut self.docs {
            d.refresh_dirty();
            notices.append(&mut d.notices);
        }
        for n in notices {
            self.toast(&ctx, n.kind, n.text);
        }

        // Modals and overlays.
        if self.ai_settings_open {
            let t = self.theme();
            assistant::settings_dialog(&ctx, &t, &mut self.ai, &mut self.ai_settings_open);
        }
        self.dialogs(&ctx);
        self.draw_toasts(&ctx);
        self.update_title(&ctx);

        // Window close: ask first if there is unsaved work or a running assistant.
        if ctx.input(|i| i.viewport().close_requested()) && !self.allow_quit {
            let busy = self
                .docs
                .iter()
                .any(|d| d.is_dirty() || d.ai.is_running() || d.ai.is_pending());
            if busy {
                ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
                self.dialog = Some(Dialog::Close(CloseKind::Quit));
            }
        }
    }
}
