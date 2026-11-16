//! The LLMC design system: color tokens (dark + light), typography (Inter / JetBrains Mono /
//! Remix icons), the egui style built from them, and the small set of shared widgets every screen
//! uses — so the whole app reads as one family.
//!
//! Visual language: neutral greys stepped up for elevation (not inverted), one blue accent, a
//! lime "logic high" signal color, 1px hairline borders, 8–10px radii, very soft multi-layer
//! shadows, an accent ring (not a fill) for selection, gradient primary buttons with an inner top
//! highlight, and keycap-style shortcut hints.

use eframe::egui::{
    self, epaint, Align, Align2, Color32, CornerRadius, FontFamily, FontId, Frame, Layout, Margin,
    Pos2, Rect, Response, RichText, Sense, Shadow, Stroke, StrokeKind, Ui, Vec2,
};

use super::icons;

// ===================== tokens =====================

#[derive(Clone, Copy, Debug)]
pub struct Theme {
    pub dark: bool,
    /// Window chrome (top bar, tab strip).
    pub bg_app: Color32,
    /// Side panels.
    pub bg_panel: Color32,
    /// The canvas.
    pub bg_canvas: Color32,
    /// Cards, inputs, secondary buttons, block bodies.
    pub surface: Color32,
    pub surface_hover: Color32,
    pub surface_active: Color32,
    /// Raised surfaces (menus, popovers, tooltips).
    pub surface_raised: Color32,
    pub border: Color32,
    pub border_strong: Color32,
    pub text: Color32,
    pub text_muted: Color32,
    pub text_faint: Color32,
    pub accent: Color32,
    pub accent_strong: Color32,
    /// Accent at low opacity (selected rows, focus halos).
    pub accent_soft: Color32,
    pub on_accent: Color32,
    pub success: Color32,
    pub warning: Color32,
    pub danger: Color32,
    pub danger_soft: Color32,
    // Canvas-specific.
    pub grid: Color32,
    pub grid_strong: Color32,
    pub wire_low: Color32,
    pub wire_high: Color32,
    pub gate_stroke: Color32,
    pub port: Color32,
    pub led_on: Color32,
    pub led_off: Color32,
    pub kbd_bg: Color32,
}

const fn rgb(hex: u32) -> Color32 {
    Color32::from_rgb((hex >> 16) as u8, (hex >> 8) as u8, hex as u8)
}

impl Theme {
    pub fn dark() -> Self {
        let accent = rgb(0x3392ff);
        Self {
            dark: true,
            bg_app: rgb(0x111111),
            bg_panel: rgb(0x161616),
            bg_canvas: rgb(0x0c0c0c),
            surface: rgb(0x1e1e1e),
            surface_hover: rgb(0x272727),
            surface_active: rgb(0x303030),
            surface_raised: rgb(0x1f1f1f),
            border: rgb(0x262626),
            border_strong: rgb(0x383838),
            text: rgb(0xf5f5f5),
            text_muted: rgb(0xa3a3a3),
            text_faint: rgb(0x6b6b6b),
            accent,
            accent_strong: rgb(0x2473fe),
            accent_soft: Color32::from_rgba_unmultiplied(0x33, 0x92, 0xff, 38),
            on_accent: Color32::WHITE,
            success: rgb(0xa3e635),
            warning: rgb(0xfbbf24),
            danger: rgb(0xf87171),
            danger_soft: Color32::from_rgba_unmultiplied(0xf8, 0x71, 0x71, 30),
            grid: rgb(0x262626),
            grid_strong: rgb(0x3a3a3a),
            wire_low: rgb(0x4a4a4a),
            wire_high: rgb(0xa3e635),
            gate_stroke: rgb(0x9a9a9a),
            port: rgb(0x6b6b6b),
            led_on: rgb(0xa3e635),
            led_off: rgb(0x2a2a2a),
            kbd_bg: rgb(0x2b2b2b),
        }
    }

    pub fn light() -> Self {
        let accent = rgb(0x2473fe);
        Self {
            dark: false,
            bg_app: rgb(0xf3f3f3),
            bg_panel: rgb(0xfbfbfb),
            bg_canvas: rgb(0xf7f7f7),
            surface: rgb(0xffffff),
            surface_hover: rgb(0xf2f2f2),
            surface_active: rgb(0xe8e8e8),
            surface_raised: rgb(0xffffff),
            border: rgb(0xe6e6e6),
            border_strong: rgb(0xd0d0d0),
            text: rgb(0x0a0a0a),
            text_muted: rgb(0x5c5c5c),
            text_faint: rgb(0x9a9a9a),
            accent,
            accent_strong: rgb(0x0450e2),
            accent_soft: Color32::from_rgba_unmultiplied(0x24, 0x73, 0xfe, 28),
            on_accent: Color32::WHITE,
            success: rgb(0x65a30d),
            warning: rgb(0xd97706),
            danger: rgb(0xef4444),
            danger_soft: Color32::from_rgba_unmultiplied(0xef, 0x44, 0x44, 24),
            grid: rgb(0xdcdcdc),
            grid_strong: rgb(0xc2c2c2),
            wire_low: rgb(0xa8a8a8),
            wire_high: rgb(0x65a30d),
            gate_stroke: rgb(0x525252),
            port: rgb(0xa3a3a3),
            led_on: rgb(0x84cc16),
            led_off: rgb(0xe5e5e5),
            kbd_bg: rgb(0xececec),
        }
    }

    pub fn current(dark: bool) -> Self {
        if dark {
            Self::dark()
        } else {
            Self::light()
        }
    }

    /// Soft drop shadow for floating things (menus, popovers, the minimap).
    pub fn shadow(&self) -> Shadow {
        Shadow {
            offset: [0, 6],
            blur: 20,
            spread: 0,
            color: if self.dark {
                Color32::from_black_alpha(110)
            } else {
                Color32::from_black_alpha(26)
            },
        }
    }

    /// Bigger, softer shadow for windows and dialogs.
    pub fn shadow_large(&self) -> Shadow {
        Shadow {
            offset: [0, 14],
            blur: 40,
            spread: 0,
            color: if self.dark {
                Color32::from_black_alpha(150)
            } else {
                Color32::from_black_alpha(38)
            },
        }
    }
}

/// Corner radii.
pub mod radius {
    pub const SM: u8 = 6;
    pub const MD: u8 = 8;
    pub const LG: u8 = 10;
    pub const XL: u8 = 14;
}

// ===================== typography =====================

pub const MEDIUM: &str = "inter-medium";
pub const SEMIBOLD: &str = "inter-semibold";
pub const ICON_FAMILY: &str = "icons";

pub fn medium(size: f32) -> FontId {
    FontId::new(size, FontFamily::Name(MEDIUM.into()))
}

pub fn semibold(size: f32) -> FontId {
    FontId::new(size, FontFamily::Name(SEMIBOLD.into()))
}

pub fn regular(size: f32) -> FontId {
    FontId::proportional(size)
}

pub fn mono(size: f32) -> FontId {
    FontId::monospace(size)
}

pub fn icon_font(size: f32) -> FontId {
    FontId::new(size, FontFamily::Name(ICON_FAMILY.into()))
}

/// Install the bundled fonts: Inter (regular/medium/semibold) as the UI face, JetBrains Mono for
/// code, Remix Icon glyphs as a fallback in every family (so icons can sit inline in any text),
/// and egui's defaults kept as further fallbacks for wide script/emoji coverage.
pub fn install_fonts(ctx: &egui::Context) {
    use std::sync::Arc;
    let mut fonts = egui::FontDefinitions::default();
    let add = |fonts: &mut egui::FontDefinitions, name: &str, bytes: &'static [u8]| {
        fonts.font_data.insert(
            name.to_string(),
            Arc::new(egui::FontData::from_static(bytes)),
        );
    };
    add(
        &mut fonts,
        "inter",
        include_bytes!("../../assets/fonts/Inter-Regular.ttf"),
    );
    add(
        &mut fonts,
        MEDIUM,
        include_bytes!("../../assets/fonts/Inter-Medium.ttf"),
    );
    add(
        &mut fonts,
        SEMIBOLD,
        include_bytes!("../../assets/fonts/Inter-SemiBold.ttf"),
    );
    add(
        &mut fonts,
        "jetbrains",
        include_bytes!("../../assets/fonts/JetBrainsMono-Regular.ttf"),
    );
    add(
        &mut fonts,
        ICON_FAMILY,
        include_bytes!("../../assets/fonts/remixicon-subset.ttf"),
    );

    let defaults_prop = fonts
        .families
        .get(&FontFamily::Proportional)
        .cloned()
        .unwrap_or_default();
    let defaults_mono = fonts
        .families
        .get(&FontFamily::Monospace)
        .cloned()
        .unwrap_or_default();

    // The icon font comes FIRST: all of its glyphs live in the Private Use Area, and Inter
    // ships a few PUA glyphs of its own that would otherwise shadow the icons.
    let chain = |primary: &str| -> Vec<String> {
        let mut v = vec![ICON_FAMILY.to_string(), primary.to_string()];
        v.extend(defaults_prop.iter().cloned());
        v
    };
    fonts
        .families
        .insert(FontFamily::Proportional, chain("inter"));
    fonts
        .families
        .insert(FontFamily::Name(MEDIUM.into()), chain(MEDIUM));
    fonts
        .families
        .insert(FontFamily::Name(SEMIBOLD.into()), chain(SEMIBOLD));
    let mut mono_chain = vec![ICON_FAMILY.to_string(), "jetbrains".to_string()];
    mono_chain.extend(defaults_mono);
    fonts.families.insert(FontFamily::Monospace, mono_chain);
    fonts.families.insert(
        FontFamily::Name(ICON_FAMILY.into()),
        vec![ICON_FAMILY.to_string(), "inter".to_string()],
    );
    ctx.set_fonts(fonts);
}

// ===================== egui style =====================

/// Apply the theme to egui's widgets, spacing, and text styles.
pub fn apply_style(ctx: &egui::Context, dark: bool) {
    let t = Theme::current(dark);
    let mut v = if dark {
        egui::Visuals::dark()
    } else {
        egui::Visuals::light()
    };
    v.panel_fill = t.bg_panel;
    v.window_fill = t.surface_raised;
    v.window_stroke = Stroke::new(1.0, t.border_strong);
    v.window_corner_radius = CornerRadius::same(radius::XL);
    v.window_shadow = t.shadow_large();
    v.popup_shadow = t.shadow();
    v.menu_corner_radius = CornerRadius::same(radius::LG);
    v.extreme_bg_color = if dark { rgb(0x141414) } else { rgb(0xffffff) };
    v.faint_bg_color = t.surface;
    v.code_bg_color = t.surface;
    v.hyperlink_color = t.accent;
    v.selection.bg_fill = t.accent.linear_multiply(0.35);
    v.selection.stroke = Stroke::new(1.0, t.accent);
    v.override_text_color = None;
    v.text_cursor.stroke = Stroke::new(1.5, t.accent);
    v.striped = false;
    v.slider_trailing_fill = true;
    v.handle_shape = egui::style::HandleShape::Circle;
    v.collapsing_header_frame = false;
    v.indent_has_left_vline = false;

    let r = CornerRadius::same(radius::MD);
    let w = &mut v.widgets;
    w.noninteractive.bg_fill = t.bg_panel;
    w.noninteractive.weak_bg_fill = t.bg_panel;
    w.noninteractive.bg_stroke = Stroke::new(1.0, t.border);
    w.noninteractive.fg_stroke = Stroke::new(1.0, t.text_muted);
    w.noninteractive.corner_radius = r;

    // Rails (sliders) and DragValue wells read against panels one step up from `surface`.
    w.inactive.bg_fill = t.surface_active;
    w.inactive.weak_bg_fill = t.surface;
    w.inactive.bg_stroke = Stroke::new(1.0, t.border_strong);
    w.inactive.fg_stroke = Stroke::new(1.0, t.text);
    w.inactive.corner_radius = r;
    w.inactive.expansion = 0.0;

    w.hovered.bg_fill = t.surface_hover;
    w.hovered.weak_bg_fill = t.surface_hover;
    w.hovered.bg_stroke = Stroke::new(1.0, t.border_strong);
    w.hovered.fg_stroke = Stroke::new(1.0, t.text);
    w.hovered.corner_radius = r;
    w.hovered.expansion = 0.0;

    w.active.bg_fill = t.surface_active;
    w.active.weak_bg_fill = t.surface_active;
    w.active.bg_stroke = Stroke::new(1.0, t.accent);
    w.active.fg_stroke = Stroke::new(1.0, t.text);
    w.active.corner_radius = r;
    w.active.expansion = 0.0;

    w.open.bg_fill = t.surface_active;
    w.open.weak_bg_fill = t.surface_hover;
    w.open.bg_stroke = Stroke::new(1.0, t.border_strong);
    w.open.fg_stroke = Stroke::new(1.0, t.text);
    w.open.corner_radius = r;

    let (theme, pref) = if dark {
        (egui::Theme::Dark, egui::ThemePreference::Dark)
    } else {
        (egui::Theme::Light, egui::ThemePreference::Light)
    };
    ctx.set_theme(pref);
    ctx.set_visuals_of(theme, v);
    ctx.all_styles_mut(|style| {
        use egui::TextStyle;
        style.text_styles = [
            (TextStyle::Small, regular(11.5)),
            (TextStyle::Body, regular(13.5)),
            (TextStyle::Button, regular(13.5)),
            (TextStyle::Heading, semibold(17.0)),
            (TextStyle::Monospace, mono(12.5)),
        ]
        .into();
        let s = &mut style.spacing;
        s.item_spacing = egui::vec2(8.0, 6.0);
        s.button_padding = egui::vec2(10.0, 5.0);
        s.interact_size = egui::vec2(28.0, 28.0);
        s.icon_width = 15.0;
        s.icon_width_inner = 9.0;
        s.icon_spacing = 7.0;
        s.slider_width = 150.0;
        s.slider_rail_height = 4.0;
        s.combo_width = 120.0;
        s.text_edit_width = 220.0;
        s.menu_margin = Margin::same(6);
        s.window_margin = Margin::same(18);
        s.indent = 14.0;
        s.tooltip_width = 300.0;
        s.scroll = egui::style::ScrollStyle::thin();
        style.interaction.tooltip_delay = 0.35;
        style.interaction.selectable_labels = false;
        style.animation_time = 0.12;
    });
    // Ctrl +/- zoom the canvas, not the whole UI.
    ctx.options_mut(|o| o.zoom_with_keyboard = false);
}

// ===================== shared widgets =====================

/// Paint an icon glyph centered in `rect`.
pub fn paint_icon(ui: &Ui, rect: Rect, glyph: &str, size: f32, color: Color32) {
    ui.painter().text(
        rect.center(),
        Align2::CENTER_CENTER,
        glyph,
        icon_font(size),
        color,
    );
}

/// A square, frameless icon button that shows a soft background on hover and an accent tint when
/// `active`. The workhorse of toolbars.
pub fn icon_button(ui: &mut Ui, t: &Theme, glyph: &str, tip: &str, active: bool) -> Response {
    icon_button_sized(ui, t, glyph, tip, active, 30.0, true)
}

pub fn icon_button_sized(
    ui: &mut Ui,
    t: &Theme,
    glyph: &str,
    tip: &str,
    active: bool,
    size: f32,
    enabled: bool,
) -> Response {
    let sense = if enabled {
        Sense::click()
    } else {
        Sense::hover()
    };
    let (rect, resp) = ui.allocate_exact_size(Vec2::splat(size), sense);
    let hovered = enabled && resp.hovered();
    let pressed = enabled && resp.is_pointer_button_down_on();
    let anim = ui
        .ctx()
        .animate_bool_with_time(resp.id.with("hover"), hovered || active, 0.10);
    let bg = if active {
        t.accent_soft
    } else if pressed {
        t.surface_active
    } else {
        t.surface_hover.gamma_multiply(anim)
    };
    if active || anim > 0.0 {
        ui.painter()
            .rect_filled(rect.shrink(1.0), CornerRadius::same(radius::MD), bg);
    }
    if active {
        ui.painter().rect_stroke(
            rect.shrink(1.0),
            CornerRadius::same(radius::MD),
            Stroke::new(1.0, t.accent.gamma_multiply(0.55)),
            StrokeKind::Inside,
        );
    }
    let color = if !enabled {
        t.text_faint
    } else if active {
        t.accent
    } else if hovered {
        t.text
    } else {
        t.text_muted
    };
    paint_icon(ui, rect, glyph, (size * 0.55).round(), color);
    if !tip.is_empty() {
        resp.on_hover_text(tip)
    } else {
        resp
    }
}

/// Button flavors.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// Gradient accent with an inner top highlight.
    Primary,
    /// Surface with a hairline border.
    Secondary,
    /// Text-only until hovered.
    Ghost,
    /// Destructive action.
    Danger,
}

/// A styled button: optional leading icon, label, and flavor. Painted by hand so every button in
/// the app shares exact metrics.
pub fn button(ui: &mut Ui, t: &Theme, kind: Kind, glyph: Option<&str>, label: &str) -> Response {
    button_ex(ui, t, kind, glyph, label, true, 30.0)
}

pub fn button_ex(
    ui: &mut Ui,
    t: &Theme,
    kind: Kind,
    glyph: Option<&str>,
    label: &str,
    enabled: bool,
    height: f32,
) -> Response {
    let font = medium(13.0);
    let galley = ui
        .painter()
        .layout_no_wrap(label.to_string(), font.clone(), Color32::WHITE);
    let icon_w = if glyph.is_some() {
        if label.is_empty() {
            16.0
        } else {
            22.0
        }
    } else {
        0.0
    };
    let pad = if label.is_empty() { 8.0 } else { 12.0 };
    let width = (galley.size().x + icon_w + pad * 2.0).max(height);
    let sense = if enabled {
        Sense::click()
    } else {
        Sense::hover()
    };
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(width, height), sense);
    let hovered = enabled && resp.hovered();
    let pressed = enabled && resp.is_pointer_button_down_on();
    let p = ui.painter();
    let rad = CornerRadius::same(radius::MD);

    let (text_color, icon_color) = match kind {
        Kind::Primary => {
            let (top, bottom) = if !enabled {
                (t.surface_active, t.surface_active)
            } else if pressed {
                (t.accent_strong, t.accent_strong.gamma_multiply(0.9))
            } else if hovered {
                (t.accent.gamma_multiply(1.08), t.accent)
            } else {
                (t.accent, t.accent_strong)
            };
            vertical_gradient(p, rect, rad, top, bottom);
            if enabled {
                // Inner top highlight + hairline edge give the button a physical feel.
                p.line_segment(
                    [
                        rect.left_top() + egui::vec2(4.0, 1.0),
                        rect.right_top() + egui::vec2(-4.0, 1.0),
                    ],
                    Stroke::new(1.0, Color32::from_white_alpha(64)),
                );
                p.rect_stroke(
                    rect,
                    rad,
                    Stroke::new(1.0, t.accent_strong.gamma_multiply(0.9)),
                    StrokeKind::Inside,
                );
            }
            let c = if enabled { t.on_accent } else { t.text_faint };
            (c, c)
        }
        Kind::Secondary => {
            let fill = if pressed {
                t.surface_active
            } else if hovered {
                t.surface_hover
            } else {
                t.surface
            };
            p.rect_filled(rect, rad, fill);
            p.rect_stroke(
                rect,
                rad,
                Stroke::new(1.0, if hovered { t.border_strong } else { t.border }),
                StrokeKind::Inside,
            );
            if enabled {
                (t.text, t.text_muted)
            } else {
                (t.text_faint, t.text_faint)
            }
        }
        Kind::Ghost => {
            if pressed {
                p.rect_filled(rect, rad, t.surface_active);
            } else if hovered {
                p.rect_filled(rect, rad, t.surface_hover);
            }
            if enabled {
                (t.text, t.text_muted)
            } else {
                (t.text_faint, t.text_faint)
            }
        }
        Kind::Danger => {
            let fill = if pressed {
                t.danger.gamma_multiply(0.85)
            } else if hovered {
                t.danger
            } else {
                t.danger.gamma_multiply(0.9)
            };
            if enabled {
                vertical_gradient(p, rect, rad, fill, fill.gamma_multiply(0.85));
                (Color32::WHITE, Color32::WHITE)
            } else {
                p.rect_filled(rect, rad, t.surface_active);
                (t.text_faint, t.text_faint)
            }
        }
    };

    let content_w = icon_w + galley.size().x;
    let mut x = rect.center().x - content_w / 2.0;
    if let Some(g) = glyph {
        let r = Rect::from_center_size(Pos2::new(x + 8.0, rect.center().y), Vec2::splat(16.0));
        p.text(
            r.center(),
            Align2::CENTER_CENTER,
            g,
            icon_font(15.0),
            icon_color,
        );
        x += icon_w;
    }
    if !label.is_empty() {
        p.text(
            Pos2::new(x, rect.center().y),
            Align2::LEFT_CENTER,
            label,
            font,
            text_color,
        );
    }
    if hovered {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }
    resp
}

/// Fill a rounded rect with a smooth top-to-bottom gradient: tessellate the rounded rect once,
/// then recolor every vertex by its height (keeping the anti-aliasing feather intact).
pub fn vertical_gradient(
    p: &egui::Painter,
    rect: Rect,
    rad: CornerRadius,
    top: Color32,
    bottom: Color32,
) {
    let ppp = p.ctx().pixels_per_point();
    let mut tess = epaint::Tessellator::new(
        ppp,
        epaint::TessellationOptions::default(),
        [1, 1],
        Vec::new(),
    );
    let mut mesh = epaint::Mesh::default();
    tess.tessellate_rect(
        &epaint::RectShape::filled(rect, rad, Color32::WHITE),
        &mut mesh,
    );
    let h = rect.height().max(1.0);
    for v in &mut mesh.vertices {
        let k = ((v.pos.y - rect.top()) / h).clamp(0.0, 1.0);
        let coverage = v.color.a() as f32 / 255.0;
        v.color = lerp_color(top, bottom, k).linear_multiply(coverage);
    }
    p.add(egui::Shape::mesh(mesh));
}

/// An uppercase, letter-spaced section label ("COMPONENTS").
pub fn section_label(ui: &mut Ui, t: &Theme, text: &str) {
    let mut job = egui::text::LayoutJob::default();
    job.append(
        &text.to_uppercase(),
        0.0,
        egui::TextFormat {
            font_id: semibold(10.5),
            color: t.text_faint,
            extra_letter_spacing: 0.8,
            ..Default::default()
        },
    );
    ui.label(job);
}

/// A section label inside a menu, aligned with the item text.
pub fn menu_section(ui: &mut Ui, t: &Theme, text: &str) {
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        ui.add_space(10.0);
        section_label(ui, t, text);
    });
}

/// A keycap ("Ctrl", "S").
pub fn kbd(ui: &mut Ui, t: &Theme, key: &str) -> Response {
    let font = medium(11.0);
    let galley = ui
        .painter()
        .layout_no_wrap(key.to_string(), font.clone(), t.text_muted);
    let size = egui::vec2((galley.size().x + 10.0).max(20.0), 20.0);
    let (rect, resp) = ui.allocate_exact_size(size, Sense::hover());
    let p = ui.painter();
    p.rect_filled(rect, CornerRadius::same(5), t.kbd_bg);
    p.rect_stroke(
        rect,
        CornerRadius::same(5),
        Stroke::new(1.0, t.border_strong),
        StrokeKind::Inside,
    );
    // Bottom edge a touch darker, like a physical key.
    p.line_segment(
        [
            rect.left_bottom() + egui::vec2(3.0, -0.5),
            rect.right_bottom() + egui::vec2(-3.0, -0.5),
        ],
        Stroke::new(1.0, t.border_strong),
    );
    p.text(
        rect.center(),
        Align2::CENTER_CENTER,
        key,
        font,
        t.text_muted,
    );
    resp
}

/// A small rounded status pill with a colored dot.
pub fn pill(ui: &mut Ui, t: &Theme, color: Color32, text: &str) -> Response {
    let font = medium(11.5);
    let galley = ui
        .painter()
        .layout_no_wrap(text.to_string(), font.clone(), t.text);
    let size = egui::vec2(galley.size().x + 26.0, 20.0);
    let (rect, resp) = ui.allocate_exact_size(size, Sense::hover());
    let p = ui.painter();
    p.rect_filled(rect, CornerRadius::same(10), color.gamma_multiply(0.14));
    p.rect_stroke(
        rect,
        CornerRadius::same(10),
        Stroke::new(1.0, color.gamma_multiply(0.35)),
        StrokeKind::Inside,
    );
    p.circle_filled(Pos2::new(rect.left() + 10.0, rect.center().y), 3.5, color);
    p.text(
        Pos2::new(rect.left() + 18.0, rect.center().y),
        Align2::LEFT_CENTER,
        text,
        font,
        t.text,
    );
    resp
}

/// A small neutral badge ("3 → 2", "New").
pub fn badge(ui: &mut Ui, t: &Theme, text: &str) -> Response {
    let font = medium(10.5);
    let galley = ui
        .painter()
        .layout_no_wrap(text.to_string(), font.clone(), t.text_muted);
    let size = egui::vec2(galley.size().x + 10.0, 17.0);
    let (rect, resp) = ui.allocate_exact_size(size, Sense::hover());
    let p = ui.painter();
    p.rect_filled(rect, CornerRadius::same(5), t.surface_active);
    p.text(
        rect.center(),
        Align2::CENTER_CENTER,
        text,
        font,
        t.text_muted,
    );
    resp
}

/// A card container: surface fill, hairline border, rounded.
pub fn card(t: &Theme) -> Frame {
    Frame::NONE
        .fill(t.surface)
        .stroke(Stroke::new(1.0, t.border))
        .corner_radius(CornerRadius::same(radius::LG))
        .inner_margin(Margin::same(12))
}

/// A floating overlay container (minimap, zoom bar, toasts): raised surface + soft shadow.
pub fn floating(t: &Theme) -> Frame {
    Frame::NONE
        .fill(t.surface_raised)
        .stroke(Stroke::new(1.0, t.border_strong))
        .corner_radius(CornerRadius::same(radius::LG))
        .shadow(t.shadow())
        .inner_margin(Margin::same(4))
}

/// The frame for modal dialogs.
pub fn dialog_frame(t: &Theme) -> Frame {
    Frame::NONE
        .fill(t.surface_raised)
        .stroke(Stroke::new(1.0, t.border_strong))
        .corner_radius(CornerRadius::same(radius::XL))
        .shadow(t.shadow_large())
        .inner_margin(Margin::same(22))
}

/// Dialog header: an icon tile, a title, and an optional subtitle.
pub fn dialog_header(
    ui: &mut Ui,
    t: &Theme,
    glyph: &str,
    tint: Color32,
    title: &str,
    subtitle: &str,
) {
    ui.horizontal(|ui| {
        let (rect, _) = ui.allocate_exact_size(Vec2::splat(36.0), Sense::hover());
        ui.painter().rect_filled(
            rect,
            CornerRadius::same(radius::LG),
            tint.gamma_multiply(0.16),
        );
        ui.painter().rect_stroke(
            rect,
            CornerRadius::same(radius::LG),
            Stroke::new(1.0, tint.gamma_multiply(0.4)),
            StrokeKind::Inside,
        );
        paint_icon(ui, rect, glyph, 18.0, tint);
        ui.add_space(4.0);
        ui.vertical(|ui| {
            ui.spacing_mut().item_spacing.y = 2.0;
            ui.label(RichText::new(title).font(semibold(15.5)).color(t.text));
            if !subtitle.is_empty() {
                ui.label(
                    RichText::new(subtitle)
                        .font(regular(12.5))
                        .color(t.text_muted),
                );
            }
        });
    });
}

/// A full-width text field with an optional leading icon, drawn as an input "well".
pub fn text_field(
    ui: &mut Ui,
    t: &Theme,
    value: &mut String,
    hint: &str,
    glyph: Option<&str>,
) -> Response {
    let width = ui.available_width();
    let height = 30.0;
    let (rect, _) = ui.allocate_exact_size(egui::vec2(width, height), Sense::hover());
    let id = ui.next_auto_id();
    let focused = ui.ctx().memory(|m| m.has_focus(id));
    let p = ui.painter();
    p.rect_filled(rect, CornerRadius::same(radius::MD), t.surface);
    p.rect_stroke(
        rect,
        CornerRadius::same(radius::MD),
        Stroke::new(1.0, if focused { t.accent } else { t.border_strong }),
        StrokeKind::Inside,
    );
    if focused {
        p.rect_stroke(
            rect.expand(2.0),
            CornerRadius::same(radius::MD + 2),
            Stroke::new(2.0, t.accent_soft),
            StrokeKind::Inside,
        );
    }
    let mut inner = rect.shrink2(egui::vec2(9.0, 0.0));
    if let Some(g) = glyph {
        let ir = Rect::from_min_size(inner.left_top(), egui::vec2(16.0, height));
        paint_icon(ui, ir, g, 14.0, t.text_faint);
        inner.min.x += 22.0;
    }
    let edit = egui::TextEdit::singleline(value)
        .id(id)
        .frame(Frame::NONE)
        .hint_text(RichText::new(hint).color(t.text_faint))
        .vertical_align(Align::Center)
        .desired_width(inner.width());
    ui.put(inner, edit)
}

/// A row of mutually-exclusive options drawn as a segmented control. Returns the clicked index.
pub fn segmented(
    ui: &mut Ui,
    t: &Theme,
    options: &[(&str, &str)],
    selected: usize,
) -> Option<usize> {
    let height = 28.0;
    let width = ui.available_width();
    let (rect, _) = ui.allocate_exact_size(egui::vec2(width, height), Sense::hover());
    let p = ui.painter();
    p.rect_filled(rect, CornerRadius::same(radius::MD), t.surface);
    p.rect_stroke(
        rect,
        CornerRadius::same(radius::MD),
        Stroke::new(1.0, t.border),
        StrokeKind::Inside,
    );
    let n = options.len().max(1) as f32;
    let seg_w = (rect.width() - 4.0) / n;
    let mut clicked = None;
    for (i, (glyph, label)) in options.iter().enumerate() {
        let r = Rect::from_min_size(
            Pos2::new(rect.left() + 2.0 + seg_w * i as f32, rect.top() + 2.0),
            egui::vec2(seg_w, height - 4.0),
        );
        let resp = ui.interact(r, ui.id().with(("seg", i, *label)), Sense::click());
        let on = i == selected;
        if on {
            ui.painter()
                .rect_filled(r, CornerRadius::same(radius::SM), t.surface_active);
            ui.painter().rect_stroke(
                r,
                CornerRadius::same(radius::SM),
                Stroke::new(1.0, t.border_strong),
                StrokeKind::Inside,
            );
        } else if resp.hovered() {
            ui.painter()
                .rect_filled(r, CornerRadius::same(radius::SM), t.surface_hover);
        }
        let color = if on { t.text } else { t.text_muted };
        let text = if glyph.is_empty() {
            label.to_string()
        } else {
            format!("{glyph}  {label}")
        };
        ui.painter()
            .text(r.center(), Align2::CENTER_CENTER, text, medium(12.5), color);
        if resp.clicked() {
            clicked = Some(i);
        }
    }
    clicked
}

/// An iOS-style toggle switch with a label on the left. Returns true when flipped.
pub fn toggle_row(ui: &mut Ui, t: &Theme, label: &str, hint: &str, on: &mut bool) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.vertical(|ui| {
            ui.spacing_mut().item_spacing.y = 1.0;
            ui.label(RichText::new(label).color(t.text));
            if !hint.is_empty() {
                ui.label(RichText::new(hint).font(regular(11.5)).color(t.text_faint));
            }
        });
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            if toggle(ui, t, *on).clicked() {
                *on = !*on;
                changed = true;
            }
        });
    });
    changed
}

/// The toggle switch itself.
pub fn toggle(ui: &mut Ui, t: &Theme, on: bool) -> Response {
    let size = egui::vec2(34.0, 20.0);
    let (rect, resp) = ui.allocate_exact_size(size, Sense::click());
    let k = ui
        .ctx()
        .animate_bool_with_time(resp.id.with("toggle"), on, 0.12);
    let p = ui.painter();
    let track = lerp_color(t.surface_active, t.accent, k);
    p.rect_filled(rect, CornerRadius::same(10), track);
    p.rect_stroke(
        rect,
        CornerRadius::same(10),
        Stroke::new(1.0, lerp_color(t.border_strong, t.accent_strong, k)),
        StrokeKind::Inside,
    );
    let x = egui::lerp(rect.left() + 10.0..=rect.right() - 10.0, k);
    p.circle_filled(
        Pos2::new(x, rect.center().y + 0.5),
        7.5,
        Color32::from_black_alpha(40),
    );
    p.circle_filled(Pos2::new(x, rect.center().y), 7.0, Color32::WHITE);
    if resp.hovered() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }
    resp
}

pub fn lerp_color(a: Color32, b: Color32, k: f32) -> Color32 {
    let l = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * k).round() as u8;
    Color32::from_rgba_unmultiplied(
        l(a.r(), b.r()),
        l(a.g(), b.g()),
        l(a.b(), b.b()),
        l(a.a(), b.a()),
    )
}

/// A menu row: icon, label, and a right-aligned shortcut hint. Closes the menu when clicked.
pub fn menu_item(ui: &mut Ui, t: &Theme, glyph: &str, label: &str, shortcut_hint: &str) -> bool {
    menu_item_ex(ui, t, glyph, label, shortcut_hint, true, false)
}

pub fn menu_item_ex(
    ui: &mut Ui,
    t: &Theme,
    glyph: &str,
    label: &str,
    shortcut_hint: &str,
    enabled: bool,
    checked: bool,
) -> bool {
    let width = ui.max_rect().width().max(200.0);
    let height = 28.0;
    let sense = if enabled {
        Sense::click()
    } else {
        Sense::hover()
    };
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(width, height), sense);
    let hovered = enabled && resp.hovered();
    let p = ui.painter();
    if hovered {
        p.rect_filled(rect, CornerRadius::same(radius::SM), t.surface_hover);
    }
    let (fg, icon_c) = if !enabled {
        (t.text_faint, t.text_faint)
    } else if hovered {
        (t.text, t.text)
    } else {
        (t.text, t.text_muted)
    };
    let glyph = if checked { icons::CHECK } else { glyph };
    if !glyph.is_empty() {
        p.text(
            Pos2::new(rect.left() + 16.0, rect.center().y),
            Align2::CENTER_CENTER,
            glyph,
            icon_font(15.0),
            if checked { t.accent } else { icon_c },
        );
    }
    p.text(
        Pos2::new(rect.left() + 32.0, rect.center().y),
        Align2::LEFT_CENTER,
        label,
        regular(13.0),
        fg,
    );
    if !shortcut_hint.is_empty() {
        p.text(
            Pos2::new(rect.right() - 10.0, rect.center().y),
            Align2::RIGHT_CENTER,
            shortcut_hint,
            regular(12.0),
            t.text_faint,
        );
    }
    let clicked = resp.clicked();
    if clicked {
        ui.close();
    }
    clicked
}

/// A menu row that opens a submenu (same layout as [`menu_item_ex`], with a chevron). Pass the
/// response to `egui::containers::menu::SubMenu::show`.
pub fn submenu_row(ui: &mut Ui, t: &Theme, glyph: &str, label: &str) -> egui::Response {
    let width = ui.max_rect().width().max(200.0);
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(width, 28.0), Sense::click());
    let open = egui::containers::menu::SubMenu::id_from_widget_id(resp.id);
    let active = resp.hovered() || egui::Popup::is_id_open(ui.ctx(), open);
    let p = ui.painter();
    if active {
        p.rect_filled(rect, CornerRadius::same(radius::SM), t.surface_hover);
    }
    p.text(
        Pos2::new(rect.left() + 16.0, rect.center().y),
        Align2::CENTER_CENTER,
        glyph,
        icon_font(15.0),
        if active { t.text } else { t.text_muted },
    );
    p.text(
        Pos2::new(rect.left() + 32.0, rect.center().y),
        Align2::LEFT_CENTER,
        label,
        regular(13.0),
        t.text,
    );
    p.text(
        Pos2::new(rect.right() - 12.0, rect.center().y),
        Align2::CENTER_CENTER,
        icons::ARROW_RIGHT_S,
        icon_font(14.0),
        t.text_faint,
    );
    resp
}

/// A thin horizontal divider inside menus/panels.
pub fn divider(ui: &mut Ui, t: &Theme) {
    let w = ui.available_width();
    let (rect, _) = ui.allocate_exact_size(egui::vec2(w, 9.0), Sense::hover());
    ui.painter().line_segment(
        [
            Pos2::new(rect.left() + 2.0, rect.center().y),
            Pos2::new(rect.right() - 2.0, rect.center().y),
        ],
        Stroke::new(1.0, t.border),
    );
}

/// A vertical divider for toolbars.
pub fn vdivider(ui: &mut Ui, t: &Theme, height: f32) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(9.0, height), Sense::hover());
    ui.painter().line_segment(
        [
            Pos2::new(rect.center().x, rect.top() + 4.0),
            Pos2::new(rect.center().x, rect.bottom() - 4.0),
        ],
        Stroke::new(1.0, t.border_strong),
    );
}

/// Muted helper text.
pub fn hint(ui: &mut Ui, t: &Theme, text: impl Into<String>) {
    ui.label(
        RichText::new(text.into())
            .font(regular(12.0))
            .color(t.text_muted),
    );
}

/// A label/value row for property panels ("Inputs     2").
pub fn field_label(ui: &mut Ui, t: &Theme, text: &str) {
    ui.label(RichText::new(text).font(medium(12.0)).color(t.text_muted));
}

/// A tiny, crisp brand mark: an accent tile with a logic "pulse" glyph.
pub fn logo(ui: &mut Ui, t: &Theme, size: f32) -> Response {
    let (rect, resp) = ui.allocate_exact_size(Vec2::splat(size), Sense::hover());
    let rad = CornerRadius::same((size * 0.28) as u8);
    vertical_gradient(ui.painter(), rect, rad, t.accent, t.accent_strong);
    ui.painter().line_segment(
        [
            rect.left_top() + egui::vec2(size * 0.25, 1.0),
            rect.right_top() + egui::vec2(-size * 0.25, 1.0),
        ],
        Stroke::new(1.0, Color32::from_white_alpha(70)),
    );
    // A square wave: the "signal" of a logic simulator.
    let s = size;
    let o = rect.left_top();
    let pts = [
        (0.20, 0.64),
        (0.36, 0.64),
        (0.36, 0.36),
        (0.56, 0.36),
        (0.56, 0.64),
        (0.72, 0.64),
        (0.72, 0.36),
        (0.82, 0.36),
    ]
    .iter()
    .map(|(x, y)| o + egui::vec2(x * s, y * s))
    .collect();
    ui.painter().add(egui::Shape::line(
        pts,
        Stroke::new((size * 0.085).max(1.5), Color32::WHITE),
    ));
    resp
}

/// "1 block", "3 blocks".
pub fn plural(n: usize, one: &str, many: &str) -> String {
    format!("{n} {}", if n == 1 { one } else { many })
}
