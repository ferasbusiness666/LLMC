//! Drawing of individual blocks. Bodies are axis-aligned rounded rects (rotation is
//! conveyed by where the ports sit, which keeps labels upright and legible), with LEDs
//! and switches getting bespoke glyphs.

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
}

fn blend(a: Color32, b: Color32, t: f32) -> Color32 {
    let l = |x: u8, y: u8| (x as f32 * (1.0 - t) + y as f32 * t).round() as u8;
    Color32::from_rgb(l(a.r(), b.r()), l(a.g(), b.g()), l(a.b(), b.b()))
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

fn rounding(zoom: f32) -> f32 {
    (zoom * 0.2).clamp(3.0, 9.0)
}

fn label_font(zoom: f32, len: usize) -> FontId {
    let base = (zoom * 0.52).clamp(8.0, 20.0);
    // Shrink long labels (NAND, XNOR, chip names) so they fit.
    let scale = if len >= 5 {
        0.62
    } else if len >= 4 {
        0.78
    } else {
        1.0
    };
    FontId::proportional(base * scale)
}

fn draw_boxed(painter: &egui::Painter, s: &BlockStyle, stroke_color: Color32, stroke_w: f32) {
    let t = s.theme;
    let r = rounding(s.zoom);
    let fill = if s.on && s.running {
        blend(t.block_fill, t.wire_high, 0.16)
    } else {
        t.block_fill
    };
    painter.rect_filled(s.rect, r, fill);
    painter.rect_stroke(
        s.rect,
        r,
        Stroke::new(stroke_w, stroke_color),
        StrokeKind::Inside,
    );

    let text: String = match s.ty {
        BlockType::Chip(_) => s
            .chip_name
            .map(|n| n.chars().take(6).collect())
            .unwrap_or_else(|| "IC".to_string()),
        other => other.short_label().to_string(),
    };
    painter.text(
        s.rect.center(),
        Align2::CENTER_CENTER,
        &text,
        label_font(s.zoom, text.len()),
        t.label,
    );
}

fn draw_led(painter: &egui::Painter, s: &BlockStyle, stroke_color: Color32, stroke_w: f32) {
    let t = s.theme;
    let c = s.rect.center();
    let radius = s.rect.width().min(s.rect.height()) * 0.32;
    let fill = if s.on { t.led_on } else { t.led_off };
    if s.on {
        painter.circle_filled(c, radius * 1.7, t.led_on.linear_multiply(0.22));
    }
    painter.circle_filled(c, radius, fill);
    painter.circle_stroke(c, radius, Stroke::new(stroke_w, stroke_color));
}

fn draw_switch(painter: &egui::Painter, s: &BlockStyle, stroke_color: Color32, stroke_w: f32) {
    let t = s.theme;
    let r = rounding(s.zoom);
    let fill = if s.on {
        blend(t.block_fill, t.led_on, 0.30)
    } else {
        t.block_fill
    };
    painter.rect_filled(s.rect, r, fill);
    painter.rect_stroke(
        s.rect,
        r,
        Stroke::new(stroke_w, stroke_color),
        StrokeKind::Inside,
    );

    let label = if s.on { "1" } else { "0" };
    let color = if s.on { t.led_on } else { t.label_dim };
    painter.text(
        s.rect.center(),
        Align2::CENTER_CENTER,
        label,
        FontId::proportional((s.zoom * 0.66).clamp(10.0, 24.0)),
        color,
    );
}

fn draw_clock(painter: &egui::Painter, s: &BlockStyle, stroke_color: Color32, stroke_w: f32) {
    let t = s.theme;
    let r = rounding(s.zoom);
    let fill = if s.on && s.running {
        blend(t.block_fill, t.wire_high, 0.16)
    } else {
        t.block_fill
    };
    painter.rect_filled(s.rect, r, fill);
    painter.rect_stroke(
        s.rect,
        r,
        Stroke::new(stroke_w, stroke_color),
        StrokeKind::Inside,
    );

    // A small square wave icon.
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
    let stroke = Stroke::new((s.zoom * 0.06).clamp(1.2, 2.4), t.label_dim);
    for w in pts.windows(2) {
        painter.line_segment([w[0], w[1]], stroke);
    }
}
