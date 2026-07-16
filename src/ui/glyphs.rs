//! Drawing of individual blocks. Bodies are axis-aligned rounded rects (rotation is
//! conveyed by where the ports sit, which keeps labels upright and legible), with LEDs
//! and switches getting bespoke glyphs. A block may carry a custom color and a user
//! name (shown above the type label).

use eframe::egui::{self, Align2, Color32, FontId, Pos2, Rect, Stroke, StrokeKind};

use llmc::model::BlockType;

use super::theme::Theme;

pub struct BlockStyle<'a> {
    pub rect: Rect,
    pub ty: BlockType,
    pub on: bool,
    pub selected: bool,
    pub hovered: bool,
    pub running: bool,
    pub theme: &'a Theme,
    pub zoom: f32,
    pub chip_name: Option<&'a str>,
    pub color: Option<Color32>,
    pub label: Option<&'a str>,
}

fn blend(a: Color32, b: Color32, t: f32) -> Color32 {
    let l = |x: u8, y: u8| (x as f32 * (1.0 - t) + y as f32 * t).round() as u8;
    Color32::from_rgb(l(a.r(), b.r()), l(a.g(), b.g()), l(a.b(), b.b()))
}

/// Pick a readable text color for a given background.
fn contrast(bg: Color32) -> Color32 {
    let luma = 0.299 * bg.r() as f32 + 0.587 * bg.g() as f32 + 0.114 * bg.b() as f32;
    if luma > 140.0 {
        Color32::from_rgb(0x18, 0x1b, 0x22)
    } else {
        Color32::from_rgb(0xe6, 0xe9, 0xf0)
    }
}

fn rounding(zoom: f32) -> f32 {
    (zoom * 0.2).clamp(3.0, 9.0)
}

fn name_font(zoom: f32, len: usize) -> FontId {
    let base = (zoom * 0.44).clamp(9.0, 17.0);
    let scale = if len >= 8 {
        0.62
    } else if len >= 5 {
        0.82
    } else {
        1.0
    };
    FontId::proportional(base * scale)
}

fn type_font(zoom: f32) -> FontId {
    FontId::proportional((zoom * 0.3).clamp(7.0, 12.0))
}

fn type_only_font(zoom: f32, len: usize) -> FontId {
    let base = (zoom * 0.52).clamp(8.0, 20.0);
    let scale = if len >= 5 {
        0.62
    } else if len >= 4 {
        0.78
    } else {
        1.0
    };
    FontId::proportional(base * scale)
}

pub fn draw_block(painter: &egui::Painter, s: &BlockStyle) {
    let t = s.theme;
    let stroke_color = if s.selected {
        t.selected
    } else if s.hovered {
        t.hover
    } else {
        t.block_stroke
    };
    let stroke_w = if s.selected { 2.0 } else { 1.4 };

    match s.ty {
        BlockType::Led => draw_led(painter, s, stroke_color, stroke_w),
        BlockType::Switch | BlockType::Button => draw_switch(painter, s, stroke_color, stroke_w),
        BlockType::Clock => draw_clock(painter, s, stroke_color, stroke_w),
        _ => draw_boxed(painter, s, stroke_color, stroke_w),
    }
}

/// Draw the name (if any) above the `type_str`, or just `type_str` centered.
fn draw_name_and_type(
    painter: &egui::Painter,
    s: &BlockStyle,
    type_str: &str,
    name_color: Color32,
    type_color: Color32,
) {
    let c = s.rect.center();
    match s.label {
        Some(name) if !name.is_empty() => {
            let gap = (s.zoom * 0.18).clamp(4.0, 12.0);
            painter.text(
                Pos2::new(c.x, c.y - gap * 0.9),
                Align2::CENTER_CENTER,
                name,
                name_font(s.zoom, name.chars().count()),
                name_color,
            );
            painter.text(
                Pos2::new(c.x, c.y + gap),
                Align2::CENTER_CENTER,
                type_str,
                type_font(s.zoom),
                type_color,
            );
        }
        _ => {
            painter.text(
                c,
                Align2::CENTER_CENTER,
                type_str,
                type_only_font(s.zoom, type_str.len()),
                name_color,
            );
        }
    }
}

/// Draw the name as a small caption just below the block (for round/glyph blocks).
fn draw_caption_below(painter: &egui::Painter, s: &BlockStyle) {
    if let Some(name) = s.label {
        if !name.is_empty() {
            let pos = Pos2::new(
                s.rect.center().x,
                s.rect.max.y + (s.zoom * 0.1).clamp(2.0, 7.0),
            );
            painter.text(
                pos,
                Align2::CENTER_TOP,
                name,
                name_font(s.zoom, name.chars().count()),
                s.theme.label,
            );
        }
    }
}

fn body_fill(s: &BlockStyle) -> Color32 {
    let base = s.color.unwrap_or(s.theme.block_fill);
    if s.on && s.running {
        blend(base, s.theme.wire_high, 0.16)
    } else {
        base
    }
}

fn draw_boxed(painter: &egui::Painter, s: &BlockStyle, stroke_color: Color32, stroke_w: f32) {
    let t = s.theme;
    let r = rounding(s.zoom);
    let fill = body_fill(s);
    painter.rect_filled(s.rect, r, fill);
    painter.rect_stroke(
        s.rect,
        r,
        Stroke::new(stroke_w, stroke_color),
        StrokeKind::Inside,
    );

    let type_str: String = match s.ty {
        BlockType::Chip(_) => s
            .chip_name
            .map(|n| n.chars().take(6).collect())
            .unwrap_or_else(|| "IC".to_string()),
        other => other.short_label().to_string(),
    };
    let (name_color, type_color) = if s.color.is_some() {
        (contrast(fill), contrast(fill).linear_multiply(0.75))
    } else {
        (t.label, t.label_dim)
    };
    draw_name_and_type(painter, s, &type_str, name_color, type_color);
}

fn draw_led(painter: &egui::Painter, s: &BlockStyle, stroke_color: Color32, stroke_w: f32) {
    let t = s.theme;
    let c = s.rect.center();
    let radius = s.rect.width().min(s.rect.height()) * 0.32;
    let on_color = s.color.unwrap_or(t.led_on);
    let fill = if s.on { on_color } else { t.led_off };
    if s.on {
        painter.circle_filled(c, radius * 1.7, on_color.linear_multiply(0.28));
    }
    painter.circle_filled(c, radius, fill);
    painter.circle_stroke(c, radius, Stroke::new(stroke_w, stroke_color));
    draw_caption_below(painter, s);
}

fn draw_switch(painter: &egui::Painter, s: &BlockStyle, stroke_color: Color32, stroke_w: f32) {
    let t = s.theme;
    let r = rounding(s.zoom);
    let on_tint = s.color.unwrap_or(t.led_on);
    let fill = if s.on {
        blend(s.color.unwrap_or(t.block_fill), on_tint, 0.30)
    } else {
        s.color.unwrap_or(t.block_fill)
    };
    painter.rect_filled(s.rect, r, fill);
    painter.rect_stroke(
        s.rect,
        r,
        Stroke::new(stroke_w, stroke_color),
        StrokeKind::Inside,
    );

    let value = if s.on { "1" } else { "0" };
    let value_color = if s.color.is_some() {
        contrast(fill)
    } else if s.on {
        t.led_on
    } else {
        t.label_dim
    };
    let name_color = if s.color.is_some() {
        contrast(fill)
    } else {
        t.label
    };
    draw_name_and_type(painter, s, value, name_color, value_color);
}

fn draw_clock(painter: &egui::Painter, s: &BlockStyle, stroke_color: Color32, stroke_w: f32) {
    let t = s.theme;
    let r = rounding(s.zoom);
    let fill = body_fill(s);
    painter.rect_filled(s.rect, r, fill);
    painter.rect_stroke(
        s.rect,
        r,
        Stroke::new(stroke_w, stroke_color),
        StrokeKind::Inside,
    );

    let dim = if s.color.is_some() {
        contrast(fill).linear_multiply(0.75)
    } else {
        t.label_dim
    };

    if s.label.map(|n| !n.is_empty()).unwrap_or(false) {
        let name_color = if s.color.is_some() {
            contrast(fill)
        } else {
            t.label
        };
        draw_name_and_type(painter, s, "CLK", name_color, dim);
        return;
    }

    // Unlabeled: a small square-wave icon.
    let rect = s.rect.shrink(s.rect.width().min(s.rect.height()) * 0.24);
    let (x0, x1) = (rect.left(), rect.right());
    let (yl, yh) = (rect.bottom(), rect.top());
    let xa = x0 + (x1 - x0) * 0.33;
    let xb = x0 + (x1 - x0) * 0.66;
    let pts = [
        Pos2::new(x0, yl),
        Pos2::new(x0, yh),
        Pos2::new(xa, yh),
        Pos2::new(xa, yl),
        Pos2::new(xb, yl),
        Pos2::new(xb, yh),
        Pos2::new(x1, yh),
    ];
    let stroke = Stroke::new((s.zoom * 0.06).clamp(1.2, 2.4), dim);
    for w in pts.windows(2) {
        painter.line_segment([w[0], w[1]], stroke);
    }
}
