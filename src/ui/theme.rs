//! A calm, modern palette and the egui style tweaks that make the app feel polished
//! rather than default. One dark theme (primary) and a light variant.

use eframe::egui::{self, Color32};

#[derive(Clone, Copy)]
pub struct Theme {
    pub bg: Color32,
    pub grid: Color32,
    pub grid_strong: Color32,
    pub block_fill: Color32,
    pub block_stroke: Color32,
    pub label: Color32,
    pub label_dim: Color32,
    pub selected: Color32,
    pub hover: Color32,
    pub port: Color32,
    pub port_out: Color32,
    pub wire_low: Color32,
    pub wire_high: Color32,
    pub conflict: Color32,
    pub led_on: Color32,
    pub led_off: Color32,
    pub accent: Color32,
}

impl Theme {
    pub fn dark() -> Self {
        Self {
            bg: Color32::from_rgb(0x15, 0x17, 0x1c),
            grid: Color32::from_rgb(0x22, 0x25, 0x2d),
            grid_strong: Color32::from_rgb(0x2c, 0x30, 0x3b),
            block_fill: Color32::from_rgb(0x24, 0x29, 0x34),
            block_stroke: Color32::from_rgb(0x3c, 0x44, 0x55),
            label: Color32::from_rgb(0xe6, 0xe9, 0xf0),
            label_dim: Color32::from_rgb(0x9a, 0xa2, 0xb2),
            selected: Color32::from_rgb(0x5b, 0x9d, 0xf9),
            hover: Color32::from_rgb(0x84, 0xb4, 0xff),
            port: Color32::from_rgb(0x8b, 0x93, 0xa5),
            port_out: Color32::from_rgb(0xb6, 0xbe, 0xcf),
            wire_low: Color32::from_rgb(0x55, 0x5e, 0x70),
            wire_high: Color32::from_rgb(0x36, 0xd3, 0x99),
            conflict: Color32::from_rgb(0xef, 0x44, 0x44),
            led_on: Color32::from_rgb(0x3d, 0xdc, 0x84),
            led_off: Color32::from_rgb(0x33, 0x39, 0x45),
            accent: Color32::from_rgb(0x5b, 0x9d, 0xf9),
        }
    }

    pub fn light() -> Self {
        Self {
            bg: Color32::from_rgb(0xf5, 0xf6, 0xf8),
            grid: Color32::from_rgb(0xe4, 0xe7, 0xec),
            grid_strong: Color32::from_rgb(0xd2, 0xd7, 0xdf),
            block_fill: Color32::from_rgb(0xff, 0xff, 0xff),
            block_stroke: Color32::from_rgb(0xc2, 0xc8, 0xd2),
            label: Color32::from_rgb(0x1b, 0x1f, 0x27),
            label_dim: Color32::from_rgb(0x5d, 0x65, 0x72),
            selected: Color32::from_rgb(0x2f, 0x6f, 0xed),
            hover: Color32::from_rgb(0x4c, 0x8d, 0xff),
            port: Color32::from_rgb(0x7a, 0x83, 0x95),
            port_out: Color32::from_rgb(0x4a, 0x52, 0x63),
            wire_low: Color32::from_rgb(0x9a, 0xa2, 0xb0),
            wire_high: Color32::from_rgb(0x18, 0xa9, 0x63),
            conflict: Color32::from_rgb(0xd7, 0x2b, 0x2b),
            led_on: Color32::from_rgb(0x22, 0xc5, 0x5e),
            led_off: Color32::from_rgb(0xd6, 0xda, 0xe1),
            accent: Color32::from_rgb(0x2f, 0x6f, 0xed),
        }
    }
}

/// Apply global style: theme visuals, rounded widgets, comfortable spacing.
pub fn apply_style(ctx: &egui::Context, dark: bool) {
    let mut visuals = if dark {
        egui::Visuals::dark()
    } else {
        egui::Visuals::light()
    };
    let theme = if dark { Theme::dark() } else { Theme::light() };

    visuals.panel_fill = if dark {
        Color32::from_rgb(0x1a, 0x1d, 0x24)
    } else {
        Color32::from_rgb(0xec, 0xee, 0xf2)
    };
    visuals.window_fill = visuals.panel_fill;
    visuals.extreme_bg_color = theme.bg;
    visuals.selection.bg_fill = theme.accent.linear_multiply(0.35);
    visuals.selection.stroke = egui::Stroke::new(1.0, theme.accent);
    visuals.widgets.noninteractive.corner_radius = 6u8.into();
    visuals.widgets.inactive.corner_radius = 6u8.into();
    visuals.widgets.hovered.corner_radius = 6u8.into();
    visuals.widgets.active.corner_radius = 6u8.into();

    // egui 0.35 keeps a style per theme; force the chosen one and set its visuals.
    let (egui_theme, pref) = if dark {
        (egui::Theme::Dark, egui::ThemePreference::Dark)
    } else {
        (egui::Theme::Light, egui::ThemePreference::Light)
    };
    ctx.set_theme(pref);
    ctx.set_visuals_of(egui_theme, visuals);
    ctx.all_styles_mut(|style| {
        style.spacing.item_spacing = egui::vec2(8.0, 8.0);
        style.spacing.button_padding = egui::vec2(8.0, 5.0);
    });
}
