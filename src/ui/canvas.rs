//! The interactive canvas: drawing (grid, wires, blocks, ports, overlays) and pointer/keyboard
//! interaction for a [`Document`].
//!
//! Every frame starts by building a [`Geo`] cache — each block's screen placement and port
//! positions, and each wire's path — once. Drawing, hit-testing and culling all read from it,
//! so cost scales with what's visible rather than with ports × wires.

use std::collections::{HashMap, HashSet};

use eframe::egui::{
    self, Align2, Color32, CornerRadius, CursorIcon, Event, Key, PointerButton, Pos2, Rect,
    Response, Sense, Stroke, StrokeKind, Ui, Vec2,
};

use llmc::backend::GroupTransform;
use llmc::io::{GateStyle, WireStyle};
use llmc::model::{BlockId, BlockType, ChipId, ConnId, Port, PortKind, Pos, Vec2f};

use super::design::{self, Theme};
use super::document::{Document, FrameRequest, Interaction, Tool, DEFAULT_ZOOM};
use super::glyphs::{draw_block, BlockStyle, Placement};
use super::icons;

/// View options that affect the canvas.
#[derive(Clone, Copy)]
pub struct CanvasSettings {
    pub gate_style: GateStyle,
    pub wire_style: WireStyle,
    pub scroll_zooms: bool,
    pub show_grid: bool,
    pub show_minimap: bool,
}

/// Things the canvas asks the shell to do.
#[derive(Clone, Debug, PartialEq)]
pub enum CanvasAction {
    ShowProperties,
    EditChip(ChipId),
    CreateChip {
        from_selection: bool,
    },
    Copy,
    Cut,
    Paste(Option<Vec2f>),
    OpenAssistant,
    InsertExample(usize),
    Open,
    ToggleMinimap,
    /// Apply this chip-editing tab back to its project.
    UpdateChip,
}

struct BlockGeo {
    id: BlockId,
    pl: Placement,
    rect: Rect,
    /// (screen position, outward direction) per port.
    inputs: Vec<(Pos2, Vec2)>,
    outputs: Vec<(Pos2, Vec2)>,
}

struct WireGeo {
    id: ConnId,
    pts: Vec<Pos2>,
    bbox: Rect,
    high: bool,
}

/// Per-frame geometry cache.
struct Geo {
    blocks: Vec<BlockGeo>,
    index: HashMap<BlockId, usize>,
    wires: Vec<WireGeo>,
    /// Live level at each wired input port (OR of its drivers).
    input_levels: HashMap<(BlockId, u16), bool>,
}

impl Geo {
    fn block(&self, id: BlockId) -> Option<&BlockGeo> {
        self.index.get(&id).map(|&i| &self.blocks[i])
    }

    fn port(&self, port: Port) -> Option<(Pos2, Vec2)> {
        let b = self.block(port.block)?;
        match port.kind {
            PortKind::Input => b.inputs.get(port.index as usize).copied(),
            PortKind::Output => b.outputs.get(port.index as usize).copied(),
        }
    }
}

/// Example circuits offered on the empty canvas, as assistant command scripts (applied through
/// the same undoable path the AI uses).
pub const EXAMPLES: [(&str, &str, &str); 3] = [
    (
        "Half adder",
        "XOR + AND: adds two bits",
        r#"{"commands":[
 {"op":"add","ref":"a","kind":"switch","label":"A","x":0,"y":0},
 {"op":"add","ref":"b","kind":"switch","label":"B","x":0,"y":5},
 {"op":"add","ref":"x","kind":"xor","x":8,"y":0},
 {"op":"add","ref":"n","kind":"and","x":8,"y":5},
 {"op":"add","ref":"s","kind":"led","label":"Sum","x":15,"y":0},
 {"op":"add","ref":"c","kind":"led","label":"Carry","x":15,"y":5},
 {"op":"connect","from":"a","to":"x"},{"op":"connect","from":"b","to":"x"},
 {"op":"connect","from":"a","to":"n"},{"op":"connect","from":"b","to":"n"},
 {"op":"connect","from":"x","to":"s"},{"op":"connect","from":"n","to":"c"}]}"#,
    ),
    (
        "SR latch",
        "Two cross-coupled NOR gates: 1 bit of memory",
        r#"{"commands":[
 {"op":"add","ref":"s","kind":"button","label":"Set","x":0,"y":0},
 {"op":"add","ref":"r","kind":"button","label":"Reset","x":0,"y":6},
 {"op":"add","ref":"q","kind":"nor","x":8,"y":0},
 {"op":"add","ref":"qn","kind":"nor","x":8,"y":6},
 {"op":"add","ref":"lq","kind":"led","label":"Q","x":15,"y":0},
 {"op":"connect","from":"r","to":"q","to_port":0},{"op":"connect","from":"qn","to":"q","to_port":1},
 {"op":"connect","from":"s","to":"qn","to_port":0},{"op":"connect","from":"q","to":"qn","to_port":1},
 {"op":"connect","from":"q","to":"lq"}]}"#,
    ),
    (
        "Blinker",
        "A 2 Hz clock driving an LED and its inverse",
        r#"{"commands":[
 {"op":"add","ref":"c","kind":"clock","label":"CLK","hz":2,"x":0,"y":0},
 {"op":"add","ref":"n","kind":"not","x":7,"y":4},
 {"op":"add","ref":"l1","kind":"led","label":"Tick","x":14,"y":0},
 {"op":"add","ref":"l2","kind":"led","label":"Tock","x":14,"y":4},
 {"op":"connect","from":"c","to":"l1"},{"op":"connect","from":"c","to":"n"},
 {"op":"connect","from":"n","to":"l2"}]}"#,
    ),
];

fn normalize(v: Vec2) -> Vec2 {
    if v.length() > 1e-4 {
        v.normalized()
    } else {
        Vec2::new(1.0, 0.0)
    }
}

fn bezier(p0: Pos2, p1: Pos2, d0: Vec2, d1: Vec2, zoom: f32) -> Vec<Pos2> {
    let dist = (p1 - p0).length();
    let h = (dist * 0.4).clamp(zoom * 0.6, zoom * 6.0);
    let (c0, c1) = (p0 + d0 * h, p1 + d1 * h);
    let n = ((dist / 14.0) as usize).clamp(8, 32);
    (0..=n)
        .map(|i| {
            let t = i as f32 / n as f32;
            let u = 1.0 - t;
            let (w0, w1, w2, w3) = (u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t);
            Pos2::new(
                w0 * p0.x + w1 * c0.x + w2 * c1.x + w3 * p1.x,
                w0 * p0.y + w1 * c0.y + w2 * c1.y + w3 * p1.y,
            )
        })
        .collect()
}

/// A wire that runs BACKWARDS (feedback into an earlier stage, or a block feeding itself):
/// instead of cutting straight through the blocks, it leaves the output, drops below both blocks
/// (`below` is that y), runs back, and climbs into the input. Rounded for the curved style.
fn loop_around(
    p0: Pos2,
    p1: Pos2,
    d0: Vec2,
    d1: Vec2,
    below: f32,
    zoom: f32,
    curved: bool,
) -> Vec<Pos2> {
    let stub = zoom * 0.9;
    let a = p0 + d0 * stub;
    let b = p1 + d1 * stub;
    let pts = vec![p0, a, Pos2::new(a.x, below), Pos2::new(b.x, below), b, p1];
    if curved {
        round_corners(&pts, zoom * 0.7)
    } else {
        pts
    }
}

/// Replace each interior corner of a polyline with a short quadratic arc of radius ≈ `r`.
fn round_corners(pts: &[Pos2], r: f32) -> Vec<Pos2> {
    if pts.len() < 3 {
        return pts.to_vec();
    }
    let mut out = vec![pts[0]];
    for i in 1..pts.len() - 1 {
        let (prev, c, next) = (pts[i - 1], pts[i], pts[i + 1]);
        let (lin, lout) = ((c - prev).length(), (next - c).length());
        if lin < 1e-3 || lout < 1e-3 {
            continue;
        }
        let rr = r.min(lin * 0.5).min(lout * 0.5);
        let s = c - (c - prev) / lin * rr;
        let e = c + (next - c) / lout * rr;
        for k in 0..=6 {
            let t = k as f32 / 6.0;
            let u = 1.0 - t;
            out.push(Pos2::new(
                u * u * s.x + 2.0 * u * t * c.x + t * t * e.x,
                u * u * s.y + 2.0 * u * t * c.y + t * t * e.y,
            ));
        }
    }
    out.push(pts[pts.len() - 1]);
    out
}

/// Right-angle routing between two ports, with a short stub out of each port. `lane` nudges the
/// middle segment so parallel wires don't sit exactly on top of each other.
fn orthogonal(p0: Pos2, p1: Pos2, d0: Vec2, d1: Vec2, zoom: f32, lane: f32) -> Vec<Pos2> {
    let stub = zoom * 0.7;
    let a = p0 + d0 * stub;
    let b = p1 + d1 * stub;
    let h0 = d0.x.abs() >= d0.y.abs();
    let h1 = d1.x.abs() >= d1.y.abs();
    let mut pts = vec![p0];
    match (h0, h1) {
        (true, true) => {
            let forward = (b.x - a.x) * d0.x.signum() >= 0.0 || d0.x.signum() == d1.x.signum();
            if forward {
                let mx = (a.x + b.x) / 2.0 + lane;
                pts.extend([Pos2::new(mx, p0.y), Pos2::new(mx, p1.y)]);
            } else {
                let my = (a.y + b.y) / 2.0 + lane;
                pts.extend([a, Pos2::new(a.x, my), Pos2::new(b.x, my), b]);
            }
        }
        (false, false) => {
            let my = (a.y + b.y) / 2.0 + lane;
            pts.extend([Pos2::new(p0.x, my), Pos2::new(p1.x, my)]);
        }
        (true, false) => pts.extend([a, Pos2::new(b.x, a.y), b]),
        (false, true) => pts.extend([a, Pos2::new(a.x, b.y), b]),
    }
    pts.push(p1);
    pts.dedup_by(|x, y| (*x - *y).length_sq() < 0.01);
    pts
}

fn dist_sq_point_seg(p: Pos2, a: Pos2, b: Pos2) -> f32 {
    let ab = b - a;
    let len2 = ab.length_sq();
    if len2 <= 1e-6 {
        return (p - a).length_sq();
    }
    let t = ((p - a).dot(ab) / len2).clamp(0.0, 1.0);
    (p - (a + ab * t)).length_sq()
}

/// Does segment `a`–`b` touch `rect`?
fn seg_intersects_rect(a: Pos2, b: Pos2, rect: Rect) -> bool {
    if rect.contains(a) || rect.contains(b) {
        return true;
    }
    let c = [
        rect.left_top(),
        rect.right_top(),
        rect.right_bottom(),
        rect.left_bottom(),
    ];
    (0..4).any(|i| segs_cross(a, b, c[i], c[(i + 1) % 4]))
}

fn segs_cross(p1: Pos2, p2: Pos2, p3: Pos2, p4: Pos2) -> bool {
    let cross = |o: Pos2, x: Pos2, y: Pos2| (x.x - o.x) * (y.y - o.y) - (x.y - o.y) * (y.x - o.x);
    let d1 = cross(p3, p4, p1);
    let d2 = cross(p3, p4, p2);
    let d3 = cross(p1, p2, p3);
    let d4 = cross(p1, p2, p4);
    ((d1 > 0.0 && d2 < 0.0) || (d1 < 0.0 && d2 > 0.0))
        && ((d3 > 0.0 && d4 < 0.0) || (d3 < 0.0 && d4 > 0.0))
}

/// Screen rects (last frame) of overlays that sit on the canvas, so canvas input ignores them.
#[derive(Default, Clone)]
pub struct Overlays {
    pub rects: Vec<Rect>,
}

impl Document {
    fn build_geo(&self, origin: Pos2, s: &CanvasSettings) -> Geo {
        let c = &self.manager.circuit;
        let chips = &self.manager.chips;
        let drag: Option<(Pos, HashSet<BlockId>)> = match &self.interaction {
            Interaction::DragBlocks {
                offset, original, ..
            } => Some((*offset, original.iter().map(|(b, _)| *b).collect())),
            _ => None,
        };
        let mut blocks = Vec::with_capacity(c.blocks.len());
        let mut index = HashMap::with_capacity(c.blocks.len());
        for b in c.iter_blocks() {
            let mut base = b.pos.as_vec();
            if let Some((off, set)) = &drag {
                if set.contains(&b.id) {
                    base = base + Vec2f::new(off.x as f32, off.y as f32);
                }
            }
            let layout = b.layout(chips);
            let pl = Placement {
                origin: self.camera.to_screen(base, origin),
                zoom: self.camera.zoom,
                orient: b.orientation,
                size: layout.size,
            };
            let port = |l: &Vec2f, dx: f32| {
                let at = pl.map(*l);
                let dir = normalize(pl.map(Vec2f::new(l.x + dx, l.y)) - at);
                (at, dir)
            };
            let inputs = layout.inputs.iter().map(|l| port(l, -1.0)).collect();
            let outputs = layout.outputs.iter().map(|l| port(l, 1.0)).collect();
            index.insert(b.id, blocks.len());
            blocks.push(BlockGeo {
                id: b.id,
                rect: pl.rect(),
                pl,
                inputs,
                outputs,
            });
        }
        let mut geo = Geo {
            blocks,
            index,
            wires: Vec::with_capacity(c.connections.len()),
            input_levels: HashMap::new(),
        };
        for w in c.iter_connections() {
            let high = self.sim.output_value(w.from).unwrap_or(false);
            if high {
                geo.input_levels.insert((w.to.block, w.to.index), true);
            } else {
                geo.input_levels
                    .entry((w.to.block, w.to.index))
                    .or_insert(false);
            }
            let (Some((p0, d0)), Some((p1, d1))) = (geo.port(w.from), geo.port(w.to)) else {
                continue;
            };
            let z = self.camera.zoom;
            // Feedback (into an earlier stage, or a block into itself) loops around underneath.
            let facing = d0.x.abs() > 0.9 && d1.x.abs() > 0.9 && d0.x * d1.x < 0.0;
            let backward = facing && ((p1.x + d1.x * z) - (p0.x + d0.x * z)) * d0.x < z * 0.5;
            let pts = if backward || w.from.block == w.to.block {
                let rect_of = |id: BlockId| geo.index.get(&id).map(|&i| geo.blocks[i].rect);
                let bottom = [rect_of(w.from.block), rect_of(w.to.block)]
                    .into_iter()
                    .flatten()
                    .map(|r| r.bottom())
                    .fold(p0.y.max(p1.y), f32::max);
                let lane = (w.to.index % 3) as f32 * z * 0.3;
                loop_around(
                    p0,
                    p1,
                    d0,
                    d1,
                    bottom + z * 0.9 + lane,
                    z,
                    s.wire_style == WireStyle::Curved,
                )
            } else {
                match s.wire_style {
                    WireStyle::Curved => bezier(p0, p1, d0, d1, z),
                    WireStyle::Orthogonal => {
                        let lane = ((w.to.index as i32 % 3) - 1) as f32 * z * 0.12;
                        orthogonal(p0, p1, d0, d1, z, lane)
                    }
                }
            };
            let bbox = Rect::from_points(&pts);
            geo.wires.push(WireGeo {
                id: w.id,
                pts,
                bbox,
                high,
            });
        }
        geo
    }

    fn hit_port(&self, geo: &Geo, cursor: Pos2) -> Option<(Port, bool)> {
        let radius = (self.camera.zoom * 0.45).clamp(9.0, 16.0);
        let mut best_d = radius * radius;
        let mut best = None;
        for b in &geo.blocks {
            if !b.rect.expand(radius).contains(cursor) {
                continue;
            }
            // Over a block's body, only a click right on a pin counts as the pin: when zoomed
            // out, a generous pin radius would otherwise swallow a whole switch and start a
            // wire instead of toggling it.
            let inside = b.rect.shrink(1.0).contains(cursor);
            let near = (b.rect.width().min(b.rect.height()) * 0.22).max(4.0);
            let limit = if inside { near * near } else { f32::INFINITY };
            for (i, (p, _)) in b.outputs.iter().enumerate() {
                let d = p.distance_sq(cursor);
                if d < best_d && d < limit {
                    best_d = d;
                    best = Some((Port::output(b.id, i as u16), true));
                }
            }
            for (i, (p, _)) in b.inputs.iter().enumerate() {
                let d = p.distance_sq(cursor);
                if d < best_d && d < limit {
                    best_d = d;
                    best = Some((Port::input(b.id, i as u16), false));
                }
            }
        }
        best
    }

    fn hit_block(&self, geo: &Geo, cursor: Pos2) -> Option<BlockId> {
        geo.blocks
            .iter()
            .rev()
            .find(|b| b.rect.contains(cursor))
            .map(|b| b.id)
    }

    fn hit_wire(&self, geo: &Geo, cursor: Pos2) -> Option<ConnId> {
        let thresh = 6.0f32;
        let mut best_d = thresh * thresh;
        let mut best = None;
        for w in &geo.wires {
            if !w.bbox.expand(thresh).contains(cursor) {
                continue;
            }
            for seg in w.pts.windows(2) {
                let d = dist_sq_point_seg(cursor, seg[0], seg[1]);
                if d < best_d {
                    best_d = d;
                    best = Some(w.id);
                }
            }
        }
        best
    }

    /// Draw and drive the canvas for this frame. Returns actions for the shell.
    pub fn canvas(
        &mut self,
        ui: &mut Ui,
        t: &Theme,
        s: &CanvasSettings,
        overlays: &mut Overlays,
        has_clipboard: bool,
    ) -> Vec<CanvasAction> {
        let mut actions = Vec::new();
        let (response, painter) = ui.allocate_painter(ui.available_size(), Sense::click_and_drag());
        let rect = response.rect;
        let origin = rect.min;
        self.canvas_rect = rect;
        self.resolve_frame_request();
        let ctx = ui.ctx().clone();

        painter.rect_filled(rect, CornerRadius::ZERO, t.bg_canvas);

        let pointer = ctx.input(|i| i.pointer.hover_pos());
        let over_overlay = pointer.is_some_and(|p| overlays.rects.iter().any(|r| r.contains(p)));
        let cursor = if response.hovered() && !over_overlay {
            response.hover_pos()
        } else {
            None
        };
        let world = cursor.map(|c| self.camera.to_world(c, origin));
        if world.map(|w| (w.x.round(), w.y.round()))
            != self.cursor_world.map(|w| (w.x.round(), w.y.round()))
        {
            self.cursor_moved();
        }
        self.cursor_world = world;

        // ---- navigation (wheel / pinch) ----
        if cursor.is_some() {
            let (scroll, zoomd, shift) =
                ctx.input(|i| (i.smooth_scroll_delta, i.zoom_delta(), i.modifiers.shift));
            let anchor = cursor.unwrap_or_else(|| rect.center());
            if s.scroll_zooms && !shift {
                if scroll.y.abs() > 0.0 {
                    self.zoom_at(anchor, (scroll.y * 0.0018).exp());
                }
                self.camera.pan.x += scroll.x;
            } else {
                self.camera.pan += scroll;
            }
            if (zoomd - 1.0).abs() > 1e-3 {
                self.zoom_at(anchor, zoomd);
            }
        }

        let geo = self.build_geo(origin, s);
        if !over_overlay {
            self.handle_input(&response, &geo, origin, &ctx, &mut actions);
        }
        let geo = self.build_geo(origin, s); // reflect this frame's edits/drags

        // ---- cursor ----
        let hover_port = cursor.and_then(|c| self.hit_port(&geo, c));
        let hover_block = if hover_port.is_some() {
            None
        } else {
            cursor.and_then(|c| self.hit_block(&geo, c))
        };
        let hover_wire = if hover_port.is_none() && hover_block.is_none() {
            cursor.and_then(|c| self.hit_wire(&geo, c))
        } else {
            None
        };
        if cursor.is_some() {
            let icon = match (&self.interaction, self.tool) {
                (Interaction::Panning, _) => CursorIcon::Grabbing,
                (Interaction::DragBlocks { .. }, _) => CursorIcon::Grabbing,
                (_, Tool::Place(_)) => CursorIcon::Crosshair,
                _ if self.space_down => CursorIcon::Grab,
                _ if self.pending_wire.is_some() || hover_port.is_some() => CursorIcon::Crosshair,
                _ if hover_block.is_some() => CursorIcon::PointingHand,
                _ => CursorIcon::Default,
            };
            ctx.set_cursor_icon(icon);
        }

        // ---- drawing ----
        let clip = rect.expand(40.0);
        let painter = painter.with_clip_rect(rect);
        if s.show_grid {
            self.draw_grid(&painter, rect, origin, t);
        }
        self.draw_wires(&painter, &geo, clip, hover_wire, t);
        self.draw_glow(&painter, &geo, &ctx, t);
        self.draw_blocks(&painter, &geo, clip, hover_block, s, t);
        self.draw_ports(&painter, &geo, clip, hover_port, t);
        self.draw_overlay(&painter, &geo, origin, cursor, s, t);

        // ---- floating UI on top of the canvas ----
        overlays.rects.clear();
        if self.manager.circuit.blocks.is_empty() && self.chip_edit.is_none() {
            self.empty_state(ui, rect, t, overlays, &mut actions);
        }
        if s.show_minimap && !self.manager.circuit.blocks.is_empty() {
            self.minimap(ui, rect, &geo, t, overlays);
        }
        self.zoom_bar(ui, rect, t, s, overlays, &mut actions);
        if let Some(e) = self.chip_edit {
            if self.chip_banner(ui, rect, e.chip, t, overlays) {
                actions.push(CanvasAction::UpdateChip);
            }
        }

        // ---- tooltip ----
        if matches!(self.interaction, Interaction::Idle)
            && self.palette_drag.is_none()
            && !egui::Popup::is_any_open(&ctx)
        {
            if let Some(c) = cursor {
                let _ = c;
                self.hover_tooltip(ui, hover_port, hover_block, &response);
            }
        }

        // ---- context menu ----
        if matches!(self.tool, Tool::Select) && self.pending_wire.is_none() && !over_overlay {
            self.context_menu(&response, t, has_clipboard, &mut actions);
        }
        actions
    }

    // ===================== input =====================

    fn handle_input(
        &mut self,
        response: &Response,
        geo: &Geo,
        origin: Pos2,
        ctx: &egui::Context,
        actions: &mut Vec<CanvasAction>,
    ) {
        let shift = ctx.input(|i| i.modifiers.shift);
        let pointer = response.interact_pointer_pos();

        // Palette drag-to-place: drop on release over the canvas.
        if let Some(ty) = self.palette_drag {
            if ctx.input(|i| i.pointer.any_released()) {
                if let Some(p) = response.hover_pos() {
                    let w = self.camera.to_world(p, origin);
                    if let Some(id) = self.add_block_at(ty, w) {
                        self.selection = [id].into_iter().collect();
                        self.selected_conns.clear();
                    }
                }
                self.palette_drag = None;
            }
            return;
        }

        // Momentary buttons: high only while the primary button is held on them.
        let (pressed, released) =
            ctx.input(|i| (i.pointer.primary_pressed(), i.pointer.primary_released()));
        if pressed && !self.space_down {
            if let Some(p) = pointer.or_else(|| response.hover_pos()) {
                if self.hit_port(geo, p).is_none() {
                    if let Some(bid) = self.hit_block(geo, p) {
                        if self.manager.circuit.block(bid).map(|b| b.ty) == Some(BlockType::Button)
                        {
                            self.pressed_button = Some(bid);
                            self.set_input(bid, true, false);
                        }
                    }
                }
            }
        }
        if released {
            if let Some(bid) = self.pressed_button.take() {
                self.set_input(bid, false, false);
            }
        }

        // Pan with middle or secondary drag at any time.
        if response.dragged_by(PointerButton::Middle)
            || response.dragged_by(PointerButton::Secondary)
        {
            self.camera.pan += response.drag_delta();
        }

        // Right-click (no drag): cancel the place tool / click-wire, else select for the menu.
        if response.secondary_clicked() {
            let pos = pointer.or_else(|| response.hover_pos());
            if matches!(self.tool, Tool::Place(_)) {
                self.tool = Tool::Select;
            } else if self.pending_wire.is_some() {
                self.pending_wire = None;
            } else if let Some(c) = pos {
                self.menu_world = self.camera.to_world(c, origin);
                if let Some(bid) = self.hit_block(geo, c) {
                    if !self.selection.contains(&bid) {
                        self.clear_selection();
                        self.selection.insert(bid);
                    }
                } else if let Some(cid) = self.hit_wire(geo, c) {
                    if !self.selected_conns.contains(&cid) {
                        self.clear_selection();
                        self.selected_conns.insert(cid);
                    }
                }
            }
        }

        if response.drag_started_by(PointerButton::Primary) {
            // The drag begins a few pixels after the press: start from where the press was.
            let press = ctx
                .input(|i| i.pointer.press_origin())
                .or(response.interact_pointer_pos());
            if let Some(press) = press {
                if self.space_down {
                    self.interaction = Interaction::Panning;
                } else {
                    self.begin_primary(geo, press, origin, shift);
                }
            }
        }
        if response.dragged_by(PointerButton::Primary) {
            if matches!(self.interaction, Interaction::Panning) {
                self.camera.pan += response.drag_delta();
            } else if let Some(p) = pointer {
                self.update_primary(p, origin);
            }
        }
        if response.drag_stopped_by(PointerButton::Primary) {
            self.end_primary(geo, pointer, shift);
        }
        if response.double_clicked() {
            if let Some(p) = pointer.or_else(|| response.hover_pos()) {
                if let Some(bid) = self.hit_block(geo, p) {
                    match self.manager.circuit.block(bid).map(|b| b.ty) {
                        Some(BlockType::Chip(cid)) => actions.push(CanvasAction::EditChip(cid)),
                        Some(_) => actions.push(CanvasAction::ShowProperties),
                        None => {}
                    }
                    return;
                }
            }
        }
        if response.clicked() {
            if let Some(p) = pointer.or_else(|| response.hover_pos()) {
                self.handle_click(geo, p, origin, shift);
            }
        }
    }

    fn begin_primary(&mut self, geo: &Geo, press: Pos2, origin: Pos2, shift: bool) {
        self.pending_wire = None;
        if matches!(self.tool, Tool::Place(_)) {
            return; // placement happens on click
        }
        if let Some((port, is_out)) = self.hit_port(geo, press) {
            self.interaction = Interaction::DrawWire {
                from: port,
                from_output: is_out,
            };
            return;
        }
        if let Some(bid) = self.hit_block(geo, press) {
            if !self.selection.contains(&bid) {
                if !shift {
                    self.clear_selection();
                }
                self.selection.insert(bid);
            }
            let original: Vec<(BlockId, Pos)> = self
                .selection
                .iter()
                .filter_map(|&id| self.manager.circuit.block(id).map(|b| (id, b.pos)))
                .collect();
            self.interaction = Interaction::DragBlocks {
                grab: self.camera.to_world(press, origin),
                offset: Pos::new(0, 0),
                original,
                moved: false,
            };
            return;
        }
        if !shift {
            self.clear_selection();
        }
        self.interaction = Interaction::BoxSelect {
            start_screen: press,
        };
    }

    fn update_primary(&mut self, pointer: Pos2, origin: Pos2) {
        let cam = self.camera;
        if let Interaction::DragBlocks {
            grab,
            offset,
            moved,
            ..
        } = &mut self.interaction
        {
            let cur = cam.to_world(pointer, origin);
            let snap = Pos::new(
                (cur.x - grab.x).round() as i32,
                (cur.y - grab.y).round() as i32,
            );
            if snap != *offset {
                *offset = snap;
                if snap.x != 0 || snap.y != 0 {
                    *moved = true;
                }
            }
        }
    }

    fn end_primary(&mut self, geo: &Geo, pointer: Option<Pos2>, shift: bool) {
        match std::mem::replace(&mut self.interaction, Interaction::Idle) {
            Interaction::DragBlocks {
                original,
                offset,
                moved,
                ..
            } => {
                if moved && (offset.x != 0 || offset.y != 0) {
                    let finals: Vec<(BlockId, Pos)> = original
                        .iter()
                        .map(|(id, p)| (*id, Pos::new(p.x + offset.x, p.y + offset.y)))
                        .collect();
                    self.manager.move_blocks(&finals);
                }
            }
            Interaction::BoxSelect { start_screen } => {
                if let Some(end) = pointer {
                    let box_rect = Rect::from_two_pos(start_screen, end);
                    if !shift {
                        self.clear_selection();
                    }
                    for b in &geo.blocks {
                        if b.rect.intersects(box_rect) {
                            self.selection.insert(b.id);
                        }
                    }
                    for w in &geo.wires {
                        if w.bbox.intersects(box_rect)
                            && w.pts
                                .windows(2)
                                .any(|s| seg_intersects_rect(s[0], s[1], box_rect))
                        {
                            self.selected_conns.insert(w.id);
                        }
                    }
                }
            }
            Interaction::DrawWire { from, from_output } => {
                if let Some(end) = pointer {
                    if let Some((target, to_out)) = self.hit_port(geo, end) {
                        if from_output && !to_out {
                            self.connect(from, target);
                        } else if !from_output && to_out {
                            self.connect(target, from);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    fn handle_click(&mut self, geo: &Geo, p: Pos2, origin: Pos2, shift: bool) {
        if let Tool::Place(ty) = self.tool {
            let world = self.camera.to_world(p, origin);
            if let Some(id) = self.add_block_at(ty, world) {
                if !shift {
                    self.clear_selection();
                }
                self.selection.insert(id);
                // Shift keeps the tool armed for placing several; otherwise one and done.
                if !shift {
                    self.tool = Tool::Select;
                }
            }
            return;
        }
        // Click-to-wire: click a port, then a compatible target port.
        if let Some((port, is_out)) = self.hit_port(geo, p) {
            match self.pending_wire {
                None => self.pending_wire = Some((port, is_out)),
                Some((from, from_out)) => {
                    if from == port {
                        self.pending_wire = None;
                    } else if from_out && !is_out {
                        self.connect(from, port);
                        self.pending_wire = None;
                    } else if !from_out && is_out {
                        self.connect(port, from);
                        self.pending_wire = None;
                    } else {
                        self.pending_wire = Some((port, is_out));
                    }
                }
            }
            return;
        }
        if self.pending_wire.take().is_some() {
            return;
        }
        if let Some(bid) = self.hit_block(geo, p) {
            // A plain click toggles a switch; Shift+click only selects it.
            let sw = self
                .manager
                .circuit
                .block(bid)
                .filter(|b| b.ty == BlockType::Switch)
                .map(|b| self.sim.switch_value(b.id));
            if let (Some(level), false) = (sw, shift) {
                self.set_input(bid, !level, true);
            }
            if shift {
                if !self.selection.remove(&bid) {
                    self.selection.insert(bid);
                }
            } else {
                self.clear_selection();
                self.selection.insert(bid);
            }
            self.new_series();
            return;
        }
        if let Some(cid) = self.hit_wire(geo, p) {
            if !shift {
                self.clear_selection();
            }
            self.selected_conns.insert(cid);
            return;
        }
        if !shift {
            self.clear_selection();
        }
        self.new_series();
    }

    /// Canvas keyboard shortcuts (only called when no text field has focus).
    pub fn handle_keys(&mut self, ctx: &egui::Context) -> Vec<CanvasAction> {
        let mut actions = Vec::new();
        let (events, space) = ctx.input(|i| (i.events.clone(), i.key_down(Key::Space)));
        self.space_down = space;
        for e in &events {
            match e {
                Event::Copy => actions.push(CanvasAction::Copy),
                Event::Cut => actions.push(CanvasAction::Cut),
                Event::Paste(_) => {}
                Event::Key {
                    key,
                    pressed: true,
                    modifiers,
                    ..
                } => {
                    let cmd = modifiers.command;
                    let sh = modifiers.shift;
                    match key {
                        Key::Delete | Key::Backspace => self.delete_selection(),
                        Key::Z if cmd && !sh => self.undo(),
                        Key::Z if cmd && sh => self.redo(),
                        Key::Y if cmd => self.redo(),
                        Key::D if cmd => self.duplicate_selection(),
                        Key::A if cmd => self.select_all(),
                        Key::C if cmd => actions.push(CanvasAction::Copy),
                        Key::X if cmd => actions.push(CanvasAction::Cut),
                        Key::V if cmd => actions.push(CanvasAction::Paste(None)),
                        Key::R if !cmd => self.transform_selection(if sh {
                            GroupTransform::RotateCcw
                        } else {
                            GroupTransform::RotateCw
                        }),
                        Key::F if !cmd => self.transform_selection(GroupTransform::Flip),
                        Key::Home => self.frame_request = Some(FrameRequest::All),
                        Key::Num1 if sh => self.frame_request = Some(FrameRequest::All),
                        Key::Num2 if sh => {
                            if !self.selection.is_empty() {
                                self.frame_request = Some(FrameRequest::Selection)
                            }
                        }
                        Key::Num0 if cmd => self.set_zoom(DEFAULT_ZOOM),
                        Key::Plus | Key::Equals if cmd => {
                            let c = self.canvas_rect.center();
                            self.zoom_at(c, 1.25)
                        }
                        Key::Minus if cmd => {
                            let c = self.canvas_rect.center();
                            self.zoom_at(c, 0.8)
                        }
                        Key::Enter => {
                            if self.selection.len() == 1 {
                                actions.push(CanvasAction::ShowProperties)
                            }
                        }
                        Key::ArrowLeft | Key::ArrowRight | Key::ArrowUp | Key::ArrowDown
                            if !self.selection.is_empty() =>
                        {
                            let step = if sh { 5 } else { 1 };
                            let (dx, dy) = match key {
                                Key::ArrowLeft => (-step, 0),
                                Key::ArrowRight => (step, 0),
                                Key::ArrowUp => (0, -step),
                                _ => (0, step),
                            };
                            self.nudge(dx, dy);
                        }
                        Key::Escape => {
                            self.tool = Tool::Select;
                            self.interaction = Interaction::Idle;
                            self.pending_wire = None;
                            self.palette_drag = None;
                            self.clear_selection();
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }
        // Ctrl+V arrives as a Paste event carrying the system clipboard's text; the shell handles
        // it because it owns the shared clipboard.
        actions
    }

    // ===================== drawing =====================

    fn draw_grid(&self, p: &egui::Painter, rect: Rect, origin: Pos2, t: &Theme) {
        let tl = self.camera.to_world(rect.min, origin);
        let br = self.camera.to_world(rect.max, origin);
        let mut step = 1;
        while (self.camera.zoom * step as f32) < 18.0 {
            step *= 2;
        }
        let x0 = (tl.x.floor() as i32).div_euclid(step) * step;
        let y0 = (tl.y.floor() as i32).div_euclid(step) * step;
        let (x1, y1) = (br.x.ceil() as i32, br.y.ceil() as i32);
        let major = step * 8;
        let r_minor = if self.camera.zoom * step as f32 > 40.0 {
            1.3
        } else {
            1.0
        };
        let mut x = x0;
        while x <= x1 {
            let mut y = y0;
            while y <= y1 {
                let pt = self
                    .camera
                    .to_screen(Vec2f::new(x as f32, y as f32), origin);
                let strong = x.rem_euclid(major) == 0 && y.rem_euclid(major) == 0;
                p.circle_filled(
                    pt,
                    if strong { 1.6 } else { r_minor },
                    if strong { t.grid_strong } else { t.grid },
                );
                y += step;
            }
            x += step;
        }
    }

    fn draw_wires(
        &self,
        p: &egui::Painter,
        geo: &Geo,
        clip: Rect,
        hover: Option<ConnId>,
        t: &Theme,
    ) {
        let zoom = self.camera.zoom;
        let w_base = (zoom * 0.075).clamp(1.4, 2.6);
        // Low wires first, high on top, selected/hovered last.
        let mut order: Vec<&WireGeo> = geo
            .wires
            .iter()
            .filter(|w| w.bbox.intersects(clip))
            .collect();
        order.sort_by_key(|w| {
            (
                self.selected_conns.contains(&w.id) || hover == Some(w.id),
                w.high,
            )
        });
        for w in order {
            let selected = self.selected_conns.contains(&w.id);
            let hovered = hover == Some(w.id);
            let color = if selected {
                t.accent
            } else if w.high {
                t.wire_high
            } else if hovered {
                t.text_muted
            } else {
                t.wire_low
            };
            if w.high && !selected {
                // A soft glow under live wires.
                p.add(egui::Shape::line(
                    w.pts.clone(),
                    Stroke::new(w_base * 3.2, t.wire_high.gamma_multiply(0.12)),
                ));
            }
            if selected {
                p.add(egui::Shape::line(
                    w.pts.clone(),
                    Stroke::new(w_base * 3.4, t.accent_soft),
                ));
            }
            p.add(egui::Shape::line(
                w.pts.clone(),
                Stroke::new(
                    if selected || hovered {
                        w_base + 0.8
                    } else {
                        w_base
                    },
                    color,
                ),
            ));
        }
    }

    fn draw_blocks(
        &self,
        p: &egui::Painter,
        geo: &Geo,
        clip: Rect,
        hover: Option<BlockId>,
        s: &CanvasSettings,
        t: &Theme,
    ) {
        let chips = &self.manager.chips;
        for bg in &geo.blocks {
            if !bg.rect.intersects(clip) {
                continue;
            }
            let Some(b) = self.manager.circuit.block(bg.id) else {
                continue;
            };
            let input_levels: Vec<bool> = (0..bg.inputs.len())
                .map(|i| {
                    geo.input_levels
                        .get(&(b.id, i as u16))
                        .copied()
                        .unwrap_or(false)
                })
                .collect();
            let output_levels: Vec<bool> = (0..bg.outputs.len())
                .map(|i| {
                    self.sim
                        .output_value(Port::output(b.id, i as u16))
                        .unwrap_or(false)
                })
                .collect();
            let output_level = output_levels.first().copied().unwrap_or(false);
            let (chip_name, ins, outs): (Option<&str>, Vec<&str>, Vec<&str>) = match b.ty {
                BlockType::Chip(cid) => match chips.get(cid) {
                    Some(d) => (
                        Some(d.name.as_str()),
                        (0..d.input_blocks.len()).map(|i| d.input_name(i)).collect(),
                        (0..d.output_blocks.len())
                            .map(|i| d.output_name(i))
                            .collect(),
                    ),
                    None => (Some("Missing chip"), Vec::new(), Vec::new()),
                },
                _ => (None, Vec::new(), Vec::new()),
            };
            let on = match b.ty {
                BlockType::Led => self.sim.led_value(b.id).unwrap_or(false),
                BlockType::Switch | BlockType::Button => self.sim.switch_value(b.id),
                BlockType::ConstantHigh => true,
                _ => output_level,
            };
            let style = BlockStyle {
                ty: b.ty,
                gate_style: s.gate_style,
                on,
                selected: self.selection.contains(&b.id),
                hovered: hover == Some(b.id),
                theme: t,
                label: b.label.as_deref(),
                color: b.color.map(|[r, g, bl]| Color32::from_rgb(r, g, bl)),
                inputs: b.gate_inputs(),
                input_levels: &input_levels,
                output_level,
                output_levels: &output_levels,
                chip_name,
                chip_inputs: &ins,
                chip_outputs: &outs,
                hz: b.freq_hz,
                pressed: self.pressed_button == Some(b.id),
            };
            draw_block(p, &bg.pl, &style);
        }
    }

    /// A soft fading halo around blocks the assistant just built or pointed at, plus its note.
    fn draw_glow(&self, p: &egui::Painter, geo: &Geo, ctx: &egui::Context, t: &Theme) {
        let Some(glow) = &self.glow else {
            return;
        };
        let age = glow.since.elapsed().as_secs_f32();
        if age > glow.duration {
            return;
        }
        ctx.request_repaint();
        // Pulse in quickly, hold, then fade out over the last second.
        let fade = (age / 0.25).min(1.0) * ((glow.duration - age).min(1.0));
        let color = if glow.note.is_some() {
            t.warning
        } else {
            t.accent
        };
        let z = self.camera.zoom;
        let mut first: Option<Rect> = None;
        for id in &glow.blocks {
            let Some(&i) = geo.index.get(id) else {
                continue;
            };
            let r = geo.blocks[i].rect.expand(z * 0.25);
            first.get_or_insert(r);
            for (k, a) in [(3.0, 26.0), (1.5, 60.0)] {
                p.rect_stroke(
                    r.expand(k),
                    CornerRadius::same((z * 0.35 + k) as u8),
                    Stroke::new(k * 1.6, color.gamma_multiply(fade * a / 255.0 * 4.0)),
                    egui::StrokeKind::Outside,
                );
            }
            p.rect_stroke(
                r,
                CornerRadius::same((z * 0.3) as u8),
                Stroke::new(1.5, color.gamma_multiply(fade)),
                egui::StrokeKind::Outside,
            );
        }
        if let (Some(note), Some(r)) = (&glow.note, first) {
            let galley = p.layout(note.clone(), super::design::medium(12.0), t.text, 260.0);
            let pos = Pos2::new(r.left(), r.top() - galley.size().y - 14.0);
            let bg = Rect::from_min_size(pos, galley.size()).expand2(Vec2::new(9.0, 6.0));
            p.rect_filled(bg, CornerRadius::same(8), t.surface.gamma_multiply(fade));
            p.rect_stroke(
                bg,
                CornerRadius::same(8),
                Stroke::new(1.0, color.gamma_multiply(fade)),
                egui::StrokeKind::Inside,
            );
            p.galley_with_override_text_color(pos, galley, t.text.gamma_multiply(fade));
        }
    }

    fn draw_ports(
        &self,
        p: &egui::Painter,
        geo: &Geo,
        clip: Rect,
        hover: Option<(Port, bool)>,
        t: &Theme,
    ) {
        let r = (self.camera.zoom * 0.11).clamp(2.2, 5.0);
        let wired: HashSet<(BlockId, u16, bool)> = self
            .manager
            .circuit
            .iter_connections()
            .flat_map(|c| {
                [
                    (c.to.block, c.to.index, false),
                    (c.from.block, c.from.index, true),
                ]
            })
            .collect();
        for bg in &geo.blocks {
            if !bg.rect.expand(r * 2.0).intersects(clip) {
                continue;
            }
            for (i, (pos, _)) in bg.outputs.iter().enumerate() {
                let high = self
                    .sim
                    .output_value(Port::output(bg.id, i as u16))
                    .unwrap_or(false);
                let is_wired = wired.contains(&(bg.id, i as u16, true));
                draw_port(
                    p,
                    *pos,
                    r,
                    high,
                    is_wired,
                    hover == Some((Port::output(bg.id, i as u16), true)),
                    t,
                );
            }
            for (i, (pos, _)) in bg.inputs.iter().enumerate() {
                let high = geo
                    .input_levels
                    .get(&(bg.id, i as u16))
                    .copied()
                    .unwrap_or(false);
                let is_wired = wired.contains(&(bg.id, i as u16, false));
                draw_port(
                    p,
                    *pos,
                    r,
                    high,
                    is_wired,
                    hover == Some((Port::input(bg.id, i as u16), false)),
                    t,
                );
            }
        }
    }

    fn draw_overlay(
        &self,
        p: &egui::Painter,
        geo: &Geo,
        origin: Pos2,
        cursor: Option<Pos2>,
        s: &CanvasSettings,
        t: &Theme,
    ) {
        match &self.interaction {
            Interaction::BoxSelect { start_screen } => {
                if let Some(c) = cursor {
                    let r = Rect::from_two_pos(*start_screen, c);
                    p.rect_filled(r, CornerRadius::same(2), t.accent_soft);
                    p.rect_stroke(
                        r,
                        CornerRadius::same(2),
                        Stroke::new(1.0, t.accent),
                        StrokeKind::Inside,
                    );
                }
            }
            Interaction::DrawWire { from, from_output } => {
                if let Some(c) = cursor {
                    self.wire_preview(p, geo, *from, *from_output, c, s, t, false);
                }
            }
            _ => {}
        }
        if let (Some((from, out)), Some(c)) = (self.pending_wire, cursor) {
            self.wire_preview(p, geo, from, out, c, s, t, true);
        }
        // Ghost of the block being placed (place tool or palette drag).
        let ghost = match (self.tool, self.palette_drag) {
            (_, Some(ty)) => Some(ty),
            (Tool::Place(ty), None) => Some(ty),
            _ => None,
        };
        if let (Some(ty), Some(c)) = (ghost, cursor) {
            let world = self.camera.to_world(c, origin);
            let sz = ty.layout(&self.manager.chips).size;
            let tl = Vec2f::new(
                (world.x - sz.x / 2.0).round(),
                (world.y - sz.y / 2.0).round(),
            );
            let r = Rect::from_min_max(
                self.camera.to_screen(tl, origin),
                self.camera.to_screen(tl + sz, origin),
            );
            let pl = Placement {
                origin: r.min,
                zoom: self.camera.zoom,
                orient: Default::default(),
                size: sz,
            };
            let (name, ins, outs): (Option<&str>, Vec<&str>, Vec<&str>) = match ty {
                BlockType::Chip(cid) => match self.manager.chips.get(cid) {
                    Some(d) => (
                        Some(d.name.as_str()),
                        (0..d.input_blocks.len()).map(|i| d.input_name(i)).collect(),
                        (0..d.output_blocks.len())
                            .map(|i| d.output_name(i))
                            .collect(),
                    ),
                    None => (None, vec![], vec![]),
                },
                _ => (None, vec![], vec![]),
            };
            let style = BlockStyle {
                ty,
                gate_style: s.gate_style,
                on: false,
                selected: true,
                hovered: false,
                theme: t,
                label: None,
                color: None,
                inputs: 2,
                input_levels: &[],
                output_level: false,
                output_levels: &[],
                chip_name: name,
                chip_inputs: &ins,
                chip_outputs: &outs,
                hz: None,
                pressed: false,
            };
            let mut ghost_painter = p.clone();
            ghost_painter.set_opacity(0.65);
            draw_block(&ghost_painter, &pl, &style);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn wire_preview(
        &self,
        p: &egui::Painter,
        geo: &Geo,
        from: Port,
        from_output: bool,
        cursor: Pos2,
        s: &CanvasSettings,
        t: &Theme,
        armed: bool,
    ) {
        let Some((p0, d0)) = geo.port(from) else {
            return;
        };
        let (end, valid) = match self.hit_port(geo, cursor) {
            Some((tp, to_out)) if to_out != from_output => {
                (geo.port(tp).map(|x| x.0).unwrap_or(cursor), true)
            }
            _ => (cursor, false),
        };
        let d1 = Vec2::new(-d0.x, -d0.y);
        let pts = match s.wire_style {
            WireStyle::Curved => bezier(p0, end, d0, d1, self.camera.zoom),
            WireStyle::Orthogonal => orthogonal(p0, end, d0, d1, self.camera.zoom, 0.0),
        };
        let color = if valid { t.accent } else { t.text_muted };
        let stroke = Stroke::new(2.0, color);
        if valid {
            p.add(egui::Shape::line(pts, stroke));
        } else {
            p.extend(egui::Shape::dashed_line(&pts, stroke, 6.0, 4.0));
        }
        p.circle_filled(p0, 4.5, t.accent);
        if armed {
            p.circle_stroke(p0, 8.0, Stroke::new(1.5, t.accent));
        }
        if valid {
            p.circle_stroke(end, 7.0, Stroke::new(2.0, t.accent));
        }
    }

    // ===================== overlays =====================

    fn empty_state(
        &mut self,
        ui: &mut Ui,
        rect: Rect,
        t: &Theme,
        overlays: &mut Overlays,
        actions: &mut Vec<CanvasAction>,
    ) {
        let w = 440.0f32.min(rect.width() - 40.0);
        let area = Rect::from_center_size(rect.center(), egui::vec2(w, 300.0));
        let mut child = ui.new_child(egui::UiBuilder::new().max_rect(area));
        let resp = design::card(t)
            .inner_margin(egui::Margin::same(24))
            .shadow(t.shadow())
            .fill(t.bg_panel)
            .show(&mut child, |ui| {
                ui.set_width(w - 48.0);
                ui.vertical_centered(|ui| {
                    design::logo(ui, t, 40.0);
                    ui.add_space(10.0);
                    ui.label(
                        egui::RichText::new("Start building")
                            .font(design::semibold(18.0))
                            .color(t.text),
                    );
                    ui.add_space(2.0);
                    design::hint(
                        ui,
                        t,
                        "Drag a component in from the sidebar, or let the Assistant build it.",
                    );
                });
                ui.add_space(14.0);
                design::section_label(ui, t, "Examples");
                ui.add_space(2.0);
                for (i, (name, desc, _)) in EXAMPLES.iter().enumerate() {
                    if example_row(ui, t, name, desc).clicked() {
                        actions.push(CanvasAction::InsertExample(i));
                    }
                }
                ui.add_space(12.0);
                ui.horizontal(|ui| {
                    if design::button(
                        ui,
                        t,
                        design::Kind::Primary,
                        Some(icons::SPARKLING_2_FILL),
                        "Ask the Assistant",
                    )
                    .clicked()
                    {
                        actions.push(CanvasAction::OpenAssistant);
                    }
                    if design::button(
                        ui,
                        t,
                        design::Kind::Secondary,
                        Some(icons::FOLDER_OPEN),
                        "Open project",
                    )
                    .clicked()
                    {
                        actions.push(CanvasAction::Open);
                    }
                });
            });
        overlays.rects.push(resp.response.rect);
    }

    fn zoom_bar(
        &mut self,
        ui: &mut Ui,
        rect: Rect,
        t: &Theme,
        s: &CanvasSettings,
        overlays: &mut Overlays,
        actions: &mut Vec<CanvasAction>,
    ) {
        let size = egui::vec2(196.0, 38.0);
        let area = Rect::from_min_size(rect.right_bottom() - size - egui::vec2(14.0, 14.0), size);
        let mut child = ui.new_child(egui::UiBuilder::new().max_rect(area));
        let resp = design::floating(t).show(&mut child, |ui| {
            ui.horizontal_centered(|ui| {
                ui.spacing_mut().item_spacing.x = 2.0;
                if design::icon_button_sized(
                    ui,
                    t,
                    icons::SUBTRACT,
                    "Zoom out (Ctrl+\u{2212})",
                    false,
                    28.0,
                    true,
                )
                .clicked()
                {
                    let c = self.canvas_rect.center();
                    self.zoom_at(c, 0.8);
                }
                let pct = format!("{:.0}%", self.camera.zoom / DEFAULT_ZOOM * 100.0);
                let r = ui.add_sized(
                    egui::vec2(52.0, 28.0),
                    egui::Button::new(
                        egui::RichText::new(pct)
                            .font(design::medium(12.5))
                            .color(t.text),
                    )
                    .frame(false),
                );
                if r.on_hover_text("Reset to 100% (Ctrl+0)").clicked() {
                    self.set_zoom(DEFAULT_ZOOM);
                }
                if design::icon_button_sized(
                    ui,
                    t,
                    icons::ADD,
                    "Zoom in (Ctrl+=)",
                    false,
                    28.0,
                    true,
                )
                .clicked()
                {
                    let c = self.canvas_rect.center();
                    self.zoom_at(c, 1.25);
                }
                design::vdivider(ui, t, 24.0);
                if design::icon_button_sized(
                    ui,
                    t,
                    icons::FOCUS_3,
                    "Fit everything (Home)",
                    false,
                    28.0,
                    true,
                )
                .clicked()
                {
                    self.frame_request = Some(FrameRequest::All);
                }
                if design::icon_button_sized(
                    ui,
                    t,
                    icons::MAP_2,
                    "Minimap",
                    s.show_minimap,
                    28.0,
                    true,
                )
                .clicked()
                {
                    actions.push(CanvasAction::ToggleMinimap);
                }
            });
        });
        overlays.rects.push(resp.response.rect);
    }

    fn minimap(&mut self, ui: &mut Ui, rect: Rect, geo: &Geo, t: &Theme, overlays: &mut Overlays) {
        let size = egui::vec2(196.0, 128.0);
        let area = Rect::from_min_size(
            rect.right_bottom() - size - egui::vec2(14.0, 14.0 + 38.0 + 8.0),
            size,
        );
        let Some((mut min, mut max)) = self.bounds_of(None) else {
            return;
        };
        let origin = rect.min;
        let view_min = self.camera.to_world(rect.min, origin);
        let view_max = self.camera.to_world(rect.max, origin);
        min = Vec2f::new(min.x.min(view_min.x), min.y.min(view_min.y));
        max = Vec2f::new(max.x.max(view_max.x), max.y.max(view_max.y));
        let inner = area.shrink(8.0);
        let ww = (max.x - min.x).max(1.0);
        let wh = (max.y - min.y).max(1.0);
        let k = (inner.width() / ww).min(inner.height() / wh);
        let off = inner.center() - egui::vec2((min.x + max.x) / 2.0 * k, (min.y + max.y) / 2.0 * k);
        let to_map = |w: Vec2f| Pos2::new(off.x + w.x * k, off.y + w.y * k);

        let p = ui.painter().clone();
        p.rect_filled(
            area.translate(egui::vec2(0.0, 4.0)).expand(2.0),
            CornerRadius::same(12),
            Color32::from_black_alpha(if t.dark { 70 } else { 18 }),
        );
        p.rect_filled(
            area,
            CornerRadius::same(design::radius::LG),
            t.surface_raised,
        );
        p.rect_stroke(
            area,
            CornerRadius::same(design::radius::LG),
            Stroke::new(1.0, t.border_strong),
            StrokeKind::Inside,
        );
        let p = p.with_clip_rect(area.shrink(2.0));
        for bg in &geo.blocks {
            let Some(b) = self.manager.circuit.block(bg.id) else {
                continue;
            };
            let sz = b.footprint(&self.manager.chips);
            let r = Rect::from_min_max(to_map(b.pos.as_vec()), to_map(b.pos.as_vec() + sz));
            let color = if self.selection.contains(&b.id) {
                t.accent
            } else if matches!(b.ty, BlockType::Chip(_)) {
                t.text_faint
            } else {
                t.gate_stroke.gamma_multiply(0.6)
            };
            p.rect_filled(r.expand(0.5), CornerRadius::same(1), color);
        }
        let vr = Rect::from_min_max(to_map(view_min), to_map(view_max));
        p.rect_filled(vr, CornerRadius::same(3), t.accent_soft);
        p.rect_stroke(
            vr,
            CornerRadius::same(3),
            Stroke::new(1.2, t.accent),
            StrokeKind::Inside,
        );

        let resp = ui.interact(area, ui.id().with("minimap"), Sense::click_and_drag());
        if resp.hovered() {
            ui.ctx().set_cursor_icon(CursorIcon::Move);
        }
        if resp.dragged() || resp.clicked() {
            if let Some(pp) = resp.interact_pointer_pos() {
                let w = Vec2f::new((pp.x - off.x) / k, (pp.y - off.y) / k);
                self.camera.pan = Vec2::new(
                    rect.width() / 2.0 - w.x * self.camera.zoom,
                    rect.height() / 2.0 - w.y * self.camera.zoom,
                );
            }
        }
        overlays.rects.push(area);
    }

    /// The chip-editing banner. Returns true when "Update chip" is clicked.
    fn chip_banner(
        &mut self,
        ui: &mut Ui,
        rect: Rect,
        chip: ChipId,
        t: &Theme,
        overlays: &mut Overlays,
    ) -> bool {
        let name = self
            .manager
            .chips
            .get(chip)
            .map(|c| c.name.clone())
            .unwrap_or_else(|| "chip".to_string());
        let dirty = self.is_dirty();
        let w = 560.0f32.min(rect.width() - 28.0);
        let area = Rect::from_min_size(
            Pos2::new(rect.center().x - w / 2.0, rect.top() + 12.0),
            egui::vec2(w, 48.0),
        );
        let mut child = ui.new_child(egui::UiBuilder::new().max_rect(area));
        let mut update = false;
        let resp = design::floating(t)
            .inner_margin(egui::Margin::symmetric(12, 7))
            .show(&mut child, |ui| {
                ui.set_width(w - 24.0);
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new(icons::CPU)
                            .font(design::icon_font(16.0))
                            .color(t.accent),
                    );
                    ui.vertical(|ui| {
                        ui.spacing_mut().item_spacing.y = 0.0;
                        ui.label(
                            egui::RichText::new(format!("Editing chip \u{201c}{name}\u{201d}"))
                                .font(design::semibold(13.0))
                                .color(t.text),
                        );
                        ui.label(
                            egui::RichText::new(if dirty {
                                "Unapplied changes \u{b7} switches = inputs, LEDs = outputs"
                            } else {
                                "Switches = input pins \u{b7} LEDs = output pins"
                            })
                            .font(design::regular(11.5))
                            .color(if dirty {
                                t.warning
                            } else {
                                t.text_muted
                            }),
                        );
                    });
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let r = design::button_ex(
                            ui,
                            t,
                            if dirty {
                                design::Kind::Primary
                            } else {
                                design::Kind::Secondary
                            },
                            Some(icons::CHECK),
                            "Update chip",
                            dirty,
                            30.0,
                        )
                        .on_hover_text("Apply these changes to every placed instance (Ctrl+S)");
                        update = dirty && r.clicked();
                    });
                });
            });
        overlays.rects.push(resp.response.rect);
        update
    }

    fn hover_tooltip(
        &self,
        ui: &mut Ui,
        hover_port: Option<(Port, bool)>,
        hover_block: Option<BlockId>,
        response: &Response,
    ) {
        let still = ui
            .ctx()
            .input(|i| i.pointer.time_since_last_movement() > 0.45);
        if !still {
            return;
        }
        let text: Option<String> = if let Some((port, is_out)) = hover_port {
            let b = self.manager.circuit.block(port.block);
            let pin = b.and_then(|b| match b.ty {
                BlockType::Chip(cid) => self.manager.chips.get(cid).map(|d| {
                    let n = if is_out {
                        d.output_name(port.index as usize)
                    } else {
                        d.input_name(port.index as usize)
                    };
                    n.to_string()
                }),
                _ => None,
            });
            let level = if is_out {
                self.sim.output_value(port).unwrap_or(false)
            } else {
                self.manager
                    .circuit
                    .connections_into(port)
                    .any(|c| self.sim.output_value(c.from).unwrap_or(false))
            };
            let kind = if is_out { "Output" } else { "Input" };
            let name = match pin.filter(|p| !p.is_empty()) {
                Some(p) => format!("{kind} \u{201c}{p}\u{201d}"),
                None => format!("{kind} {}", port.index),
            };
            Some(format!(
                "{name}  \u{b7}  {}\nDrag to wire, or click then click a target.",
                if level { "1 (high)" } else { "0 (low)" }
            ))
        } else if let Some(bid) = hover_block {
            self.manager.circuit.block(bid).map(|b| {
                let ty = match b.ty {
                    BlockType::Chip(cid) => self
                        .manager
                        .chips
                        .get(cid)
                        .map(|d| format!("Chip \u{b7} {}", d.name))
                        .unwrap_or_else(|| "Chip".into()),
                    t => t.display_name().to_string(),
                };
                let name = b
                    .label
                    .as_deref()
                    .map(|l| format!("{l}  \u{2014}  "))
                    .unwrap_or_default();
                let hint = match b.ty {
                    BlockType::Switch => "Click to toggle \u{b7} Shift+click to select",
                    BlockType::Button => "Hold to press",
                    BlockType::Chip(_) => "Double-click to edit the chip",
                    _ => "Double-click for properties",
                };
                // The id is how the assistant's auto-check refers to unnamed blocks.
                format!("{name}{ty}  #{}\n{}\n{hint}", b.id, b.ty.description())
            })
        } else {
            None
        };
        if let Some(text) = text {
            egui::Tooltip::always_open(
                ui.ctx().clone(),
                ui.layer_id(),
                response.id.with("canvas-tip"),
                egui::PopupAnchor::Pointer,
            )
            .gap(14.0)
            .show(|ui| {
                ui.label(egui::RichText::new(text).font(design::regular(12.5)));
            });
        }
    }

    fn context_menu(
        &mut self,
        response: &Response,
        t: &Theme,
        has_clipboard: bool,
        actions: &mut Vec<CanvasAction>,
    ) {
        let world = self.menu_world;
        response.context_menu(|ui| {
            ui.set_width(250.0);
            let has_sel = self.has_selection();
            let has_blocks = !self.selection.is_empty();
            let one = self.selection.len() == 1 && self.selected_conns.is_empty();
            let chip_inst = if one {
                self.selection
                    .iter()
                    .next()
                    .and_then(|id| self.manager.circuit.block(*id))
                    .and_then(|b| match b.ty {
                        BlockType::Chip(c) => Some(c),
                        _ => None,
                    })
            } else {
                None
            };
            if design::menu_item_ex(ui, t, icons::SETTINGS_3, "Properties", "Enter", one, false) {
                actions.push(CanvasAction::ShowProperties);
            }
            if let Some(c) = chip_inst {
                if design::menu_item(ui, t, icons::PENCIL, "Edit chip\u{2026}", "Double-click") {
                    actions.push(CanvasAction::EditChip(c));
                }
            }
            design::divider(ui, t);
            if design::menu_item_ex(
                ui,
                t,
                icons::SCISSORS_CUT,
                "Cut",
                "Ctrl+X",
                has_blocks,
                false,
            ) {
                actions.push(CanvasAction::Cut);
            }
            if design::menu_item_ex(ui, t, icons::FILE_COPY, "Copy", "Ctrl+C", has_blocks, false) {
                actions.push(CanvasAction::Copy);
            }
            if design::menu_item_ex(
                ui,
                t,
                icons::CLIPBOARD,
                "Paste here",
                "Ctrl+V",
                has_clipboard,
                false,
            ) {
                actions.push(CanvasAction::Paste(Some(world)));
            }
            if design::menu_item_ex(
                ui,
                t,
                icons::CHECKBOX_MULTIPLE_BLANK,
                "Duplicate",
                "Ctrl+D",
                has_blocks,
                false,
            ) {
                self.duplicate_selection();
            }
            design::divider(ui, t);
            if design::menu_item_ex(
                ui,
                t,
                icons::CLOCKWISE_2,
                "Rotate right",
                "R",
                has_blocks,
                false,
            ) {
                self.transform_selection(GroupTransform::RotateCw);
            }
            if design::menu_item_ex(
                ui,
                t,
                icons::ANTICLOCKWISE_2,
                "Rotate left",
                "Shift+R",
                has_blocks,
                false,
            ) {
                self.transform_selection(GroupTransform::RotateCcw);
            }
            if design::menu_item_ex(
                ui,
                t,
                icons::FLIP_HORIZONTAL,
                "Flip",
                "F",
                has_blocks,
                false,
            ) {
                self.transform_selection(GroupTransform::Flip);
            }
            design::divider(ui, t);
            if design::menu_item_ex(
                ui,
                t,
                icons::CPU,
                "Create chip from selection\u{2026}",
                "",
                has_blocks,
                false,
            ) {
                actions.push(CanvasAction::CreateChip {
                    from_selection: true,
                });
            }
            if design::menu_item_ex(
                ui,
                t,
                icons::FOCUS_2,
                "Zoom to selection",
                "Shift+2",
                has_blocks,
                false,
            ) {
                self.frame_request = Some(FrameRequest::Selection);
            }
            if design::menu_item(ui, t, icons::CHECKBOX_CIRCLE_FILL, "Select all", "Ctrl+A") {
                self.select_all();
            }
            design::divider(ui, t);
            if design::menu_item_ex(ui, t, icons::DELETE_BIN_6, "Delete", "Del", has_sel, false) {
                self.delete_selection();
            }
        });
    }
}

fn draw_port(
    p: &egui::Painter,
    pos: Pos2,
    r: f32,
    high: bool,
    wired: bool,
    hovered: bool,
    t: &Theme,
) {
    if hovered {
        p.circle_filled(pos, r + 5.0, t.accent_soft);
        p.circle_stroke(pos, r + 3.5, Stroke::new(1.5, t.accent));
    }
    if high {
        p.circle_filled(pos, r, t.wire_high);
    } else if wired {
        p.circle_filled(pos, r, t.wire_low);
    } else {
        // Unwired ports are hollow, so floating inputs stand out.
        p.circle_filled(pos, r, t.bg_canvas);
        p.circle_stroke(pos, r - 0.5, Stroke::new(1.2, t.port));
    }
}

fn example_row(ui: &mut Ui, t: &Theme, name: &str, desc: &str) -> Response {
    let w = ui.available_width();
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(w, 40.0), Sense::click());
    let p = ui.painter();
    if resp.hovered() {
        p.rect_filled(
            rect,
            CornerRadius::same(design::radius::MD),
            t.surface_hover,
        );
        ui.ctx().set_cursor_icon(CursorIcon::PointingHand);
    }
    let icon_r = Rect::from_min_size(
        rect.left_top() + egui::vec2(6.0, 6.0),
        egui::vec2(28.0, 28.0),
    );
    p.rect_filled(
        icon_r,
        CornerRadius::same(design::radius::MD),
        t.surface_active,
    );
    p.text(
        icon_r.center(),
        Align2::CENTER_CENTER,
        icons::FLOW_CHART,
        design::icon_font(15.0),
        t.accent,
    );
    p.text(
        Pos2::new(icon_r.right() + 10.0, rect.top() + 13.0),
        Align2::LEFT_CENTER,
        name,
        design::medium(13.0),
        t.text,
    );
    p.text(
        Pos2::new(icon_r.right() + 10.0, rect.top() + 28.0),
        Align2::LEFT_CENTER,
        desc,
        design::regular(11.5),
        t.text_muted,
    );
    p.text(
        Pos2::new(rect.right() - 10.0, rect.center().y),
        Align2::RIGHT_CENTER,
        icons::ADD_CIRCLE,
        design::icon_font(16.0),
        if resp.hovered() {
            t.accent
        } else {
            t.text_faint
        },
    );
    resp
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn box_catches_wire_crossing_it() {
        let r = Rect::from_min_max(Pos2::new(10.0, 10.0), Pos2::new(20.0, 20.0));
        assert!(seg_intersects_rect(
            Pos2::new(0.0, 15.0),
            Pos2::new(30.0, 15.0),
            r
        ));
        assert!(!seg_intersects_rect(
            Pos2::new(0.0, 0.0),
            Pos2::new(30.0, 0.0),
            r
        ));
    }

    #[test]
    fn orthogonal_routes_are_axis_aligned() {
        let pts = orthogonal(
            Pos2::new(0.0, 0.0),
            Pos2::new(100.0, 40.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(-1.0, 0.0),
            20.0,
            0.0,
        );
        for s in pts.windows(2) {
            assert!(
                (s[0].x - s[1].x).abs() < 1e-3 || (s[0].y - s[1].y).abs() < 1e-3,
                "diagonal segment {s:?}"
            );
        }
        assert_eq!(*pts.first().unwrap(), Pos2::new(0.0, 0.0));
        assert_eq!(*pts.last().unwrap(), Pos2::new(100.0, 40.0));
    }

    #[test]
    fn examples_parse_and_build_cleanly() {
        for (name, _, json) in EXAMPLES {
            let mut m = llmc::backend::CircuitManager::new();
            let r = llmc::ai::apply_ops(&mut m, llmc::ai::parse_reply(json).ops);
            assert!(r.errors.is_empty(), "{name}: {:?}", r.errors);
            assert!(!m.circuit.blocks.is_empty());
        }
    }
}
