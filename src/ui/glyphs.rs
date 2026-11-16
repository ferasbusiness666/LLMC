//! Drawing individual blocks.
//!
//! Gates are drawn either as standard distinctive-shape symbols (AND's "D", OR's shield, XOR's
//! double back, inversion bubbles) or as labeled boxes; I/O blocks get bespoke glyphs (a toggle
//! switch, a push button, a glowing LED, a square-wave clock); chips are cards with pin labels.
//!
//! Geometry is authored in the block's *local, unrotated* cell space and mapped to the screen
//! through [`Placement`], which applies the block's orientation — the exact same transform the
//! model uses for port positions, so bodies and ports always line up. Text stays upright.
//! Gate outlines are triangulated once per (kind, input count) and cached.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use eframe::egui::{self, epaint, Align2, Color32, CornerRadius, Pos2, Rect, Stroke, StrokeKind};

use llmc::io::GateStyle;
use llmc::model::{BlockType, Orientation, Vec2f};

use super::design::{self, Theme};

/// Maps a block's local (unrotated) cell coordinates to screen space.
#[derive(Clone, Copy)]
pub struct Placement {
    /// Screen position of the block's (rotated) footprint top-left.
    pub origin: Pos2,
    /// Pixels per cell.
    pub zoom: f32,
    pub orient: Orientation,
    /// Unrotated layout size in cells.
    pub size: Vec2f,
}

impl Placement {
    pub fn map(&self, local: Vec2f) -> Pos2 {
        let p = self.orient.transform(local, self.size);
        self.origin + egui::vec2(p.x * self.zoom, p.y * self.zoom)
    }

    /// The screen rect of the rotated footprint.
    pub fn rect(&self) -> Rect {
        let s = self.orient.transformed_size(self.size);
        Rect::from_min_size(self.origin, egui::vec2(s.x * self.zoom, s.y * self.zoom))
    }

    /// Screen rect of a local rect (axis-aligned after any 90° rotation).
    pub fn map_rect(&self, min: Vec2f, max: Vec2f) -> Rect {
        Rect::from_two_pos(self.map(min), self.map(max))
    }
}

/// Everything needed to draw one block.
pub struct BlockStyle<'a> {
    pub ty: BlockType,
    pub gate_style: GateStyle,
    /// Primary signal for the glyph: LED lit, switch/button level, constant value.
    pub on: bool,
    pub selected: bool,
    pub hovered: bool,
    pub theme: &'a Theme,
    pub label: Option<&'a str>,
    pub color: Option<Color32>,
    /// Number of inputs (variable gates), for the shape height.
    pub inputs: usize,
    /// Live level of each input and of output 0 (stub coloring).
    pub input_levels: &'a [bool],
    pub output_level: bool,
    /// Live level of every output (chips have several).
    pub output_levels: &'a [bool],
    /// For chips: name and pin labels.
    pub chip_name: Option<&'a str>,
    pub chip_inputs: &'a [&'a str],
    pub chip_outputs: &'a [&'a str],
    /// Clock frequency (Hz), shown under the clock glyph.
    pub hz: Option<f32>,
    /// A held momentary button.
    pub pressed: bool,
}

fn level_color(t: &Theme, high: bool) -> Color32 {
    if high {
        t.wire_high
    } else {
        t.wire_low
    }
}

fn stroke_w(zoom: f32) -> f32 {
    (zoom * 0.065).clamp(1.2, 2.4)
}

/// Draw selection/hover chrome around the footprint.
fn chrome(p: &egui::Painter, pl: &Placement, s: &BlockStyle) {
    let t = s.theme;
    let rect = pl.rect().expand(pl.zoom * 0.12);
    let rad = CornerRadius::same((pl.zoom * 0.22).clamp(4.0, 10.0) as u8);
    if s.selected {
        p.rect_filled(rect, rad, t.accent_soft);
        p.rect_stroke(rect, rad, Stroke::new(1.5, t.accent), StrokeKind::Outside);
    } else if s.hovered {
        p.rect_stroke(
            rect,
            rad,
            Stroke::new(1.0, t.border_strong),
            StrokeKind::Outside,
        );
    }
}

/// Draw a block (body, stubs, labels). Ports are drawn by the canvas on top.
pub fn draw_block(p: &egui::Painter, pl: &Placement, s: &BlockStyle) {
    chrome(p, pl, s);
    match s.ty {
        BlockType::Led => draw_led(p, pl, s),
        BlockType::Switch => draw_switch(p, pl, s),
        BlockType::Button => draw_button(p, pl, s),
        BlockType::Clock => draw_clock(p, pl, s),
        BlockType::ConstantHigh | BlockType::ConstantLow => draw_constant(p, pl, s),
        BlockType::Chip(_) => draw_chip(p, pl, s),
        _ => match s.gate_style {
            GateStyle::Shapes => draw_gate_shape(p, pl, s),
            GateStyle::Boxes => draw_gate_box(p, pl, s),
        },
    }
    draw_caption(p, pl, s);
}

/// The user's name for the block, centered above its footprint.
fn draw_caption(p: &egui::Painter, pl: &Placement, s: &BlockStyle) {
    let Some(label) = s.label.filter(|l| !l.trim().is_empty()) else {
        return;
    };
    if pl.zoom < 9.0 {
        return;
    }
    let rect = pl.rect();
    let size = (pl.zoom * 0.42).clamp(9.5, 15.0);
    p.text(
        Pos2::new(rect.center().x, rect.top() - size * 0.35),
        Align2::CENTER_BOTTOM,
        label.trim(),
        design::medium(size),
        s.theme.text_muted,
    );
}

fn body_fill(s: &BlockStyle) -> Color32 {
    match s.color {
        Some(c) => design::lerp_color(s.theme.surface, c, 0.55),
        None => s.theme.surface,
    }
}

fn outline(s: &BlockStyle) -> Color32 {
    if s.selected {
        s.theme.accent
    } else {
        s.theme.gate_stroke
    }
}

// ---------- gate shapes ----------

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Family {
    And,
    Or,
    Xor,
    Buf,
}

fn family(ty: BlockType) -> Option<(Family, bool)> {
    Some(match ty {
        BlockType::And => (Family::And, false),
        BlockType::Nand => (Family::And, true),
        BlockType::Or => (Family::Or, false),
        BlockType::Nor => (Family::Or, true),
        BlockType::Xor => (Family::Xor, false),
        BlockType::Xnor => (Family::Xor, true),
        BlockType::Buffer => (Family::Buf, false),
        BlockType::Not => (Family::Buf, true),
        _ => return None,
    })
}

/// A gate outline in local cell space, triangulated once.
struct GateShape {
    outline: Vec<Vec2f>,
    tris: Vec<u32>,
    /// XOR's detached back curve.
    extra_back: Option<Vec<Vec2f>>,
    /// Inversion bubble (center, radius).
    bubble: Option<(Vec2f, f32)>,
    /// Where each input stub meets the body/spine (x at that input's y).
    input_attach: Vec<f32>,
    /// x where the output stub starts.
    output_attach: f32,
    /// Vertical spine for tall gates (from, to), at x = spine_x.
    spine: Option<(f32, f32, f32)>,
}

const X0: f32 = 0.42;
const X1: f32 = 1.62;
const BUBBLE_R: f32 = 0.13;

fn quad(a: Vec2f, c: Vec2f, b: Vec2f, t: f32) -> Vec2f {
    let u = 1.0 - t;
    Vec2f::new(
        u * u * a.x + 2.0 * u * t * c.x + t * t * b.x,
        u * u * a.y + 2.0 * u * t * c.y + t * t * b.y,
    )
}

fn build_shape(fam: Family, bubble: bool, inputs: usize, h: f32) -> GateShape {
    let mid = h / 2.0;
    // Symbol body height: grows with a couple of inputs, capped so tall gates keep a classic
    // proportion and extend a spine to reach the outer inputs instead.
    let bh = match fam {
        Family::Buf => 1.35,
        _ => (h - 0.24).min(2.3),
    };
    let top = mid - bh / 2.0;
    let bot = mid + bh / 2.0;
    let tip = if bubble { X1 - 2.0 * BUBBLE_R } else { X1 };
    let n = 24;
    let mut pts: Vec<Vec2f> = Vec::new();
    let mut extra_back = None;
    let back_x: Box<dyn Fn(f32) -> f32>;
    match fam {
        Family::And => {
            let xm = X0 + 0.32;
            let rx = tip - xm;
            let ry = bh / 2.0;
            pts.push(Vec2f::new(X0, top));
            pts.push(Vec2f::new(xm, top));
            for i in 1..n {
                let th = -std::f32::consts::FRAC_PI_2 + std::f32::consts::PI * i as f32 / n as f32;
                pts.push(Vec2f::new(xm + rx * th.cos(), mid + ry * th.sin()));
            }
            pts.push(Vec2f::new(xm, bot));
            pts.push(Vec2f::new(X0, bot));
            back_x = Box::new(|_| X0);
        }
        Family::Or | Family::Xor => {
            let shift = if fam == Family::Xor { 0.14 } else { 0.0 };
            let bx = X0 + shift;
            let bulge = 0.2;
            let ctrl = 0.62;
            // Front: top-back → tip → bottom-back.
            for i in 0..=n {
                let t = i as f32 / n as f32;
                pts.push(quad(
                    Vec2f::new(bx, top),
                    Vec2f::new(bx + ctrl, top),
                    Vec2f::new(tip, mid),
                    t,
                ));
            }
            for i in 1..=n {
                let t = i as f32 / n as f32;
                pts.push(quad(
                    Vec2f::new(tip, mid),
                    Vec2f::new(bx + ctrl, bot),
                    Vec2f::new(bx, bot),
                    t,
                ));
            }
            // Concave back: bottom → top.
            for i in 1..n {
                let sft = 1.0 - i as f32 / n as f32;
                let y = top + bh * sft;
                let x = bx + bulge * (std::f32::consts::PI * sft).sin();
                pts.push(Vec2f::new(x, y));
            }
            if fam == Family::Xor {
                let back: Vec<Vec2f> = (0..=n)
                    .map(|i| {
                        let sft = i as f32 / n as f32;
                        Vec2f::new(
                            X0 - 0.02 + bulge * (std::f32::consts::PI * sft).sin(),
                            top + bh * sft,
                        )
                    })
                    .collect();
                extra_back = Some(back);
            }
            // Stubs end at the outermost back curve (XOR's detached one, else the body's).
            let back_ref = if fam == Family::Xor { X0 - 0.02 } else { X0 };
            back_x = Box::new(move |y: f32| {
                if y <= top || y >= bot {
                    back_ref
                } else {
                    let sft = (y - top) / bh;
                    back_ref + bulge * (std::f32::consts::PI * sft).sin()
                }
            });
        }
        Family::Buf => {
            pts.push(Vec2f::new(X0, top));
            pts.push(Vec2f::new(tip, mid));
            pts.push(Vec2f::new(X0, bot));
            back_x = Box::new(|_| X0);
        }
    }
    let tris = triangulate(&pts);
    // Input positions follow the model's layout: variable gates have inputs at y = i + 0.5,
    // single-input gates at y = 1.
    let ys: Vec<f32> = if inputs == 1 {
        vec![1.0]
    } else {
        (0..inputs).map(|i| i as f32 + 0.5).collect()
    };
    let input_attach = ys.iter().map(|&y| back_x(y)).collect();
    let spine = (h > bh + 0.3).then(|| {
        let x = back_x(-1.0);
        (
            x,
            ys.first().copied().unwrap_or(top),
            ys.last().copied().unwrap_or(bot),
        )
    });
    GateShape {
        outline: pts,
        tris,
        extra_back,
        bubble: bubble.then(|| (Vec2f::new(tip + BUBBLE_R, mid), BUBBLE_R)),
        input_attach,
        output_attach: X1,
        spine,
    }
}

/// Ear-clipping triangulation of a simple polygon (run once per shape, then cached).
fn triangulate(pts: &[Vec2f]) -> Vec<u32> {
    let n = pts.len();
    if n < 3 {
        return Vec::new();
    }
    let area: f32 = (0..n)
        .map(|i| {
            let (a, b) = (pts[i], pts[(i + 1) % n]);
            a.x * b.y - b.x * a.y
        })
        .sum();
    let ccw = area > 0.0;
    let mut idx: Vec<usize> = (0..n).collect();
    let mut out = Vec::with_capacity((n - 2) * 3);
    let cross =
        |o: Vec2f, a: Vec2f, b: Vec2f| (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
    let mut guard = 0;
    while idx.len() > 3 && guard < n * n {
        guard += 1;
        let m = idx.len();
        let mut clipped = false;
        for i in 0..m {
            let (ia, ib, ic) = (idx[(i + m - 1) % m], idx[i], idx[(i + 1) % m]);
            let (a, b, c) = (pts[ia], pts[ib], pts[ic]);
            let c_ = cross(a, b, c);
            let convex = if ccw { c_ > 1e-7 } else { c_ < -1e-7 };
            if !convex {
                continue;
            }
            let inside = idx.iter().any(|&j| {
                if j == ia || j == ib || j == ic {
                    return false;
                }
                let p = pts[j];
                let d1 = cross(a, b, p);
                let d2 = cross(b, c, p);
                let d3 = cross(c, a, p);
                if ccw {
                    d1 >= 0.0 && d2 >= 0.0 && d3 >= 0.0
                } else {
                    d1 <= 0.0 && d2 <= 0.0 && d3 <= 0.0
                }
            });
            if inside {
                continue;
            }
            out.extend([ia as u32, ib as u32, ic as u32]);
            idx.remove(i);
            clipped = true;
            break;
        }
        if !clipped {
            break;
        }
    }
    if idx.len() == 3 {
        out.extend(idx.iter().map(|&i| i as u32));
    }
    out
}

thread_local! {
    static SHAPES: RefCell<HashMap<(Family, bool, usize), Rc<GateShape>>> = RefCell::new(HashMap::new());
}

fn shape_for(fam: Family, bubble: bool, inputs: usize, h: f32) -> Rc<GateShape> {
    SHAPES.with(|cache| {
        cache
            .borrow_mut()
            .entry((fam, bubble, inputs))
            .or_insert_with(|| Rc::new(build_shape(fam, bubble, inputs, h)))
            .clone()
    })
}

fn draw_gate_shape(p: &egui::Painter, pl: &Placement, s: &BlockStyle) {
    let Some((fam, bubble)) = family(s.ty) else {
        return;
    };
    let t = s.theme;
    let inputs = if fam == Family::Buf {
        1
    } else {
        s.inputs.max(2)
    };
    let shape = shape_for(fam, bubble, inputs, pl.size.y);
    let sw = stroke_w(pl.zoom);
    let mid = pl.size.y / 2.0;

    // Stubs first (under the body).
    let ys: Vec<f32> = if inputs == 1 {
        vec![1.0]
    } else {
        (0..inputs).map(|i| i as f32 + 0.5).collect()
    };
    for (i, (&y, &ax)) in ys.iter().zip(&shape.input_attach).enumerate() {
        let high = s.input_levels.get(i).copied().unwrap_or(false);
        p.line_segment(
            [pl.map(Vec2f::new(0.0, y)), pl.map(Vec2f::new(ax, y))],
            Stroke::new(sw, level_color(t, high)),
        );
    }
    p.line_segment(
        [
            pl.map(Vec2f::new(shape.output_attach, mid)),
            pl.map(Vec2f::new(pl.size.x, mid)),
        ],
        Stroke::new(sw, level_color(t, s.output_level)),
    );
    if let Some((x, y0, y1)) = shape.spine {
        p.line_segment(
            [pl.map(Vec2f::new(x, y0)), pl.map(Vec2f::new(x, y1))],
            Stroke::new(sw, outline(s)),
        );
    }

    // Body fill (cached triangulation) + crisp outline.
    let screen: Vec<Pos2> = shape.outline.iter().map(|&v| pl.map(v)).collect();
    let fill = body_fill(s);
    let mut mesh = epaint::Mesh::default();
    for &pt in &screen {
        mesh.colored_vertex(pt, fill);
    }
    for tri in shape.tris.chunks_exact(3) {
        mesh.add_triangle(tri[0], tri[1], tri[2]);
    }
    p.add(egui::Shape::mesh(mesh));
    p.add(egui::Shape::closed_line(
        screen,
        Stroke::new(sw, outline(s)),
    ));
    if let Some(back) = &shape.extra_back {
        let pts: Vec<Pos2> = back.iter().map(|&v| pl.map(v)).collect();
        p.add(egui::Shape::line(pts, Stroke::new(sw, outline(s))));
    }
    if let Some((c, r)) = shape.bubble {
        let center = pl.map(c);
        let rad = r * pl.zoom;
        p.circle_filled(center, rad, fill);
        p.circle_stroke(center, rad, Stroke::new(sw, outline(s)));
    }
}

fn draw_gate_box(p: &egui::Painter, pl: &Placement, s: &BlockStyle) {
    let t = s.theme;
    let sw = stroke_w(pl.zoom);
    let inputs = if matches!(s.ty, BlockType::Not | BlockType::Buffer) {
        1
    } else {
        s.inputs.max(2)
    };
    let ys: Vec<f32> = if inputs == 1 {
        vec![1.0]
    } else {
        (0..inputs).map(|i| i as f32 + 0.5).collect()
    };
    let x0 = 0.32;
    let x1 = pl.size.x - 0.32;
    for (i, &y) in ys.iter().enumerate() {
        let high = s.input_levels.get(i).copied().unwrap_or(false);
        p.line_segment(
            [pl.map(Vec2f::new(0.0, y)), pl.map(Vec2f::new(x0, y))],
            Stroke::new(sw, level_color(t, high)),
        );
    }
    let mid = pl.size.y / 2.0;
    p.line_segment(
        [
            pl.map(Vec2f::new(x1, mid)),
            pl.map(Vec2f::new(pl.size.x, mid)),
        ],
        Stroke::new(sw, level_color(t, s.output_level)),
    );
    let body = pl.map_rect(Vec2f::new(x0, 0.08), Vec2f::new(x1, pl.size.y - 0.08));
    let rad = CornerRadius::same((pl.zoom * 0.18).clamp(3.0, 8.0) as u8);
    p.rect_filled(body, rad, body_fill(s));
    p.rect_stroke(
        body,
        rad,
        Stroke::new(sw * 0.8, outline(s)),
        StrokeKind::Inside,
    );
    if pl.zoom >= 10.0 {
        let text = s.ty.short_label();
        let size = (pl.zoom * if text.len() >= 4 { 0.36 } else { 0.46 }).clamp(7.0, 16.0);
        p.text(
            body.center(),
            Align2::CENTER_CENTER,
            text,
            design::semibold(size),
            t.text,
        );
    }
}

// ---------- I/O glyphs ----------

fn io_body(p: &egui::Painter, pl: &Placement, s: &BlockStyle, inset: f32) -> Rect {
    let rect = pl.rect().shrink(pl.zoom * inset);
    let rad = CornerRadius::same((pl.zoom * 0.24).clamp(4.0, 10.0) as u8);
    p.rect_filled(rect, rad, body_fill(s));
    p.rect_stroke(
        rect,
        rad,
        Stroke::new(
            1.0,
            if s.selected {
                s.theme.accent
            } else {
                s.theme.border_strong
            },
        ),
        StrokeKind::Inside,
    );
    rect
}

fn draw_switch(p: &egui::Painter, pl: &Placement, s: &BlockStyle) {
    let t = s.theme;
    let body = io_body(p, pl, s, 0.12);
    // A toggle pill inside the body.
    let w = body.width() * 0.66;
    let h = (w * 0.56).min(body.height() * 0.5);
    let track = Rect::from_center_size(body.center(), egui::vec2(w, h));
    let on = s.on;
    p.rect_filled(
        track,
        CornerRadius::same((h / 2.0) as u8),
        if on { t.wire_high } else { t.surface_active },
    );
    p.rect_stroke(
        track,
        CornerRadius::same((h / 2.0) as u8),
        Stroke::new(1.0, if on { t.wire_high } else { t.border_strong }),
        StrokeKind::Inside,
    );
    let r = h * 0.38;
    let kx = if on {
        track.right() - h / 2.0
    } else {
        track.left() + h / 2.0
    };
    p.circle_filled(
        Pos2::new(kx, track.center().y + 0.6),
        r,
        Color32::from_black_alpha(50),
    );
    p.circle_filled(Pos2::new(kx, track.center().y), r, Color32::WHITE);
    stub_out(p, pl, s);
}

fn draw_button(p: &egui::Painter, pl: &Placement, s: &BlockStyle) {
    let t = s.theme;
    let body = io_body(p, pl, s, 0.12);
    let r = body.width().min(body.height()) * 0.30;
    let c = body.center();
    let lit = s.on || s.pressed;
    p.circle_filled(c + egui::vec2(0.0, r * 0.12), r * 1.12, t.border_strong);
    p.circle_filled(
        c + if s.pressed {
            egui::vec2(0.0, r * 0.1)
        } else {
            egui::Vec2::ZERO
        },
        r,
        if lit { t.wire_high } else { t.surface_active },
    );
    if !lit {
        p.circle_stroke(c, r, Stroke::new(1.0, t.border_strong));
    }
    stub_out(p, pl, s);
}

fn draw_led(p: &egui::Painter, pl: &Placement, s: &BlockStyle) {
    let t = s.theme;
    let rect = pl.rect();
    let c = pl.map(Vec2f::new(1.15, 1.0));
    let r = rect.width().min(rect.height()) * 0.30;
    let on_color = s.color.unwrap_or(t.led_on);
    // Input stub.
    p.line_segment(
        [
            pl.map(Vec2f::new(0.0, 1.0)),
            pl.map(Vec2f::new(1.15 - 0.3, 1.0)),
        ],
        Stroke::new(stroke_w(pl.zoom), level_color(t, s.on)),
    );
    if s.on {
        for (k, a) in [(2.3, 18u8), (1.7, 34), (1.3, 60)] {
            p.circle_filled(
                c,
                r * k,
                Color32::from_rgba_unmultiplied(on_color.r(), on_color.g(), on_color.b(), a),
            );
        }
    }
    p.circle_filled(c, r, if s.on { on_color } else { t.led_off });
    p.circle_stroke(
        c,
        r,
        Stroke::new(
            1.2,
            if s.selected {
                t.accent
            } else if s.on {
                on_color
            } else {
                t.border_strong
            },
        ),
    );
    if s.on {
        // Specular highlight.
        p.circle_filled(
            c + egui::vec2(-r * 0.3, -r * 0.32),
            r * 0.28,
            Color32::from_white_alpha(120),
        );
    }
}

fn draw_clock(p: &egui::Painter, pl: &Placement, s: &BlockStyle) {
    let t = s.theme;
    let body = io_body(p, pl, s, 0.12);
    let w = body.width() * 0.62;
    let h = body.height() * 0.30;
    let o = body.center()
        - egui::vec2(
            w / 2.0,
            h / 2.0
                + if pl.zoom >= 14.0 {
                    body.height() * 0.08
                } else {
                    0.0
                },
        );
    let pts: Vec<Pos2> = [
        (0.0, 1.0),
        (0.25, 1.0),
        (0.25, 0.0),
        (0.5, 0.0),
        (0.5, 1.0),
        (0.75, 1.0),
        (0.75, 0.0),
        (1.0, 0.0),
    ]
    .iter()
    .map(|(x, y)| o + egui::vec2(x * w, y * h))
    .collect();
    p.add(egui::Shape::line(
        pts,
        Stroke::new(
            stroke_w(pl.zoom),
            if s.on { t.wire_high } else { t.text_muted },
        ),
    ));
    if pl.zoom >= 14.0 {
        let hz = s.hz.unwrap_or(1.0);
        let text = if hz >= 10.0 || hz.fract() == 0.0 {
            format!("{hz:.0} Hz")
        } else {
            format!("{hz:.2} Hz")
        };
        p.text(
            Pos2::new(body.center().x, body.bottom() - body.height() * 0.16),
            Align2::CENTER_BOTTOM,
            text,
            design::medium((pl.zoom * 0.28).clamp(7.0, 11.0)),
            t.text_faint,
        );
    }
    stub_out(p, pl, s);
}

fn draw_constant(p: &egui::Painter, pl: &Placement, s: &BlockStyle) {
    let t = s.theme;
    let body = io_body(p, pl, s, 0.22);
    let high = s.ty == BlockType::ConstantHigh;
    p.text(
        body.center(),
        Align2::CENTER_CENTER,
        if high { "1" } else { "0" },
        design::mono((pl.zoom * 0.6).clamp(8.0, 22.0)),
        if high { t.wire_high } else { t.text_muted },
    );
    stub_out(p, pl, s);
}

/// The short output stub of a 2×2 source block.
fn stub_out(p: &egui::Painter, pl: &Placement, s: &BlockStyle) {
    let x0 = pl.size.x - 0.24;
    p.line_segment(
        [
            pl.map(Vec2f::new(x0, 1.0)),
            pl.map(Vec2f::new(pl.size.x, 1.0)),
        ],
        Stroke::new(stroke_w(pl.zoom), level_color(s.theme, s.output_level)),
    );
}

// ---------- chips ----------

fn draw_chip(p: &egui::Painter, pl: &Placement, s: &BlockStyle) {
    let t = s.theme;
    let sw = stroke_w(pl.zoom);
    let (w, h) = (pl.size.x, pl.size.y);
    let inset = 0.3;
    let nin = s.chip_inputs.len();
    let nout = s.chip_outputs.len();
    let ys = |n: usize| -> Vec<f32> {
        (0..n)
            .map(|i| h * (i as f32 + 1.0) / (n as f32 + 1.0))
            .collect()
    };
    for (i, y) in ys(nin).into_iter().enumerate() {
        let high = s.input_levels.get(i).copied().unwrap_or(false);
        p.line_segment(
            [pl.map(Vec2f::new(0.0, y)), pl.map(Vec2f::new(inset, y))],
            Stroke::new(sw, level_color(t, high)),
        );
    }
    for (i, y) in ys(nout).into_iter().enumerate() {
        let high = s.output_levels.get(i).copied().unwrap_or(false);
        p.line_segment(
            [pl.map(Vec2f::new(w - inset, y)), pl.map(Vec2f::new(w, y))],
            Stroke::new(sw, level_color(t, high)),
        );
    }
    let body = pl.map_rect(Vec2f::new(inset, 0.06), Vec2f::new(w - inset, h - 0.06));
    let rad = CornerRadius::same((pl.zoom * 0.22).clamp(4.0, 10.0) as u8);
    p.rect_filled(body, rad, body_fill(s));
    // A subtle header band in the chip's accent.
    let band_color = s.color.unwrap_or(t.accent);
    p.rect_stroke(
        body,
        rad,
        Stroke::new(
            1.2,
            if s.selected {
                t.accent
            } else {
                t.border_strong
            },
        ),
        StrokeKind::Inside,
    );
    // A thin accent band along the top edge of the unrotated body (chip identity color).
    let band = pl.map_rect(
        Vec2f::new(inset + 0.12, 0.06),
        Vec2f::new(w - inset - 0.12, 0.06 + 0.1),
    );
    p.rect_filled(band, CornerRadius::same(2), band_color.gamma_multiply(0.8));

    let name = s.chip_name.unwrap_or("IC");
    if pl.zoom >= 8.0 {
        // Chip name above the body (when the user hasn't given the instance its own label).
        if !s.label.is_some_and(|l| !l.trim().is_empty()) {
            let size = (pl.zoom * 0.42).clamp(9.5, 15.0);
            let r = pl.rect();
            p.text(
                Pos2::new(r.center().x, r.top() - size * 0.35),
                Align2::CENTER_BOTTOM,
                name,
                design::semibold(size),
                t.text,
            );
        }
    }
    // Pin labels inside the body, near their pins.
    if pl.zoom >= 15.0 {
        let size = (pl.zoom * 0.34).clamp(8.0, 12.5);
        let font = design::medium(size);
        let max_w = body.width() * 0.48;
        let pin_label = |text: &str, local: Vec2f, align: Align2| {
            if text.is_empty() {
                return;
            }
            let shown = ellipsize(p, text, &font, max_w);
            p.text(pl.map(local), align, shown, font.clone(), t.text_muted);
        };
        let rot = pl.orient;
        // Horizontal pins (unrotated or 180°) get left/right aligned labels; vertical pins
        // (90°/270°) get top/bottom aligned ones.
        let horizontal = matches!(
            rot.rotation,
            llmc::model::Rotation::R0 | llmc::model::Rotation::R180
        );
        for (i, y) in ys(nin).into_iter().enumerate() {
            let at = pl.map(Vec2f::new(inset, y));
            let center = body.center();
            let align = pin_align(at, center, horizontal);
            pin_label(
                s.chip_inputs.get(i).copied().unwrap_or(""),
                Vec2f::new(inset + 0.16, y),
                align,
            );
        }
        for (i, y) in ys(nout).into_iter().enumerate() {
            let at = pl.map(Vec2f::new(w - inset, y));
            let align = pin_align(at, body.center(), horizontal);
            pin_label(
                s.chip_outputs.get(i).copied().unwrap_or(""),
                Vec2f::new(w - inset - 0.16, y),
                align,
            );
        }
    } else if pl.zoom >= 9.0 && body.height() > 16.0 {
        // Too small for pin names: show a short name inside instead.
        let shown = ellipsize(p, name, &design::semibold(10.0), body.width() - 4.0);
        p.text(
            body.center(),
            Align2::CENTER_CENTER,
            shown,
            design::semibold(10.0),
            t.text_muted,
        );
    }
}

/// Align a pin label so it reads inward from its pin.
fn pin_align(pin: Pos2, center: Pos2, horizontal: bool) -> Align2 {
    if horizontal {
        if pin.x < center.x {
            Align2::LEFT_CENTER
        } else {
            Align2::RIGHT_CENTER
        }
    } else if pin.y < center.y {
        Align2::CENTER_TOP
    } else {
        Align2::CENTER_BOTTOM
    }
}

/// Truncate `text` with an ellipsis to fit `max_w` pixels.
pub fn ellipsize(p: &egui::Painter, text: &str, font: &egui::FontId, max_w: f32) -> String {
    let fits = |s: &str| {
        p.layout_no_wrap(s.to_string(), font.clone(), Color32::WHITE)
            .size()
            .x
            <= max_w
    };
    if fits(text) {
        return text.to_string();
    }
    let chars: Vec<char> = text.chars().collect();
    let mut n = chars.len();
    while n > 1 {
        n -= 1;
        let s: String = chars[..n].iter().collect::<String>() + "\u{2026}";
        if fits(&s) {
            return s;
        }
    }
    "\u{2026}".to_string()
}

/// A small static preview of a block type for the palette (drawn into `rect`).
pub fn draw_thumbnail(
    p: &egui::Painter,
    rect: Rect,
    ty: BlockType,
    gate_style: GateStyle,
    t: &Theme,
    active: bool,
) {
    let size = match ty {
        BlockType::Chip(_) => Vec2f::new(3.0, 3.0),
        _ => Vec2f::new(2.0, 2.0),
    };
    let zoom = (rect.width() / size.x).min(rect.height() / size.y);
    let origin = rect.center() - egui::vec2(size.x * zoom / 2.0, size.y * zoom / 2.0);
    let pl = Placement {
        origin,
        zoom,
        orient: Orientation::default(),
        size,
    };
    let style = BlockStyle {
        ty,
        gate_style,
        on: active && !matches!(ty, BlockType::ConstantLow),
        selected: false,
        hovered: false,
        theme: t,
        label: None,
        color: None,
        inputs: 2,
        input_levels: &[],
        output_level: false,
        output_levels: &[],
        chip_name: None,
        chip_inputs: &["", ""],
        chip_outputs: &[""],
        hz: None,
        pressed: false,
    };
    match ty {
        BlockType::Chip(_) => {
            let body = rect.shrink(rect.width() * 0.16);
            let rad = CornerRadius::same(4);
            p.rect_filled(body, rad, t.surface);
            p.rect_stroke(
                body,
                rad,
                Stroke::new(1.2, t.gate_stroke),
                StrokeKind::Inside,
            );
            for i in 0..3 {
                let y = body.top() + body.height() * (i as f32 + 1.0) / 4.0;
                p.line_segment(
                    [Pos2::new(rect.left() + 1.0, y), Pos2::new(body.left(), y)],
                    Stroke::new(1.2, t.gate_stroke),
                );
                p.line_segment(
                    [Pos2::new(body.right(), y), Pos2::new(rect.right() - 1.0, y)],
                    Stroke::new(1.2, t.gate_stroke),
                );
            }
        }
        _ => draw_block(p, &pl, &style),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn triangulation_covers_concave_or_shape() {
        let s = build_shape(Family::Or, false, 2, 2.0);
        assert_eq!(
            s.tris.len(),
            (s.outline.len() - 2) * 3,
            "every vertex clipped"
        );
        let x = build_shape(Family::Xor, true, 4, 4.0);
        assert!(x.extra_back.is_some() && x.bubble.is_some());
        assert_eq!(x.input_attach.len(), 4);
        assert!(
            x.spine.is_some(),
            "tall gates extend a spine to their outer inputs"
        );
    }

    #[test]
    fn and_shape_is_convex_and_fully_triangulated() {
        let s = build_shape(Family::And, true, 2, 2.0);
        assert_eq!(s.tris.len(), (s.outline.len() - 2) * 3);
    }
}
