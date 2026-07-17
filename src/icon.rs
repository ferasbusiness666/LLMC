//! The app's logo, drawn procedurally as RGBA pixels — no image assets to ship.
//!
//! It's a line-art AND gate (the universal logic symbol) in the app's accent green on a
//! dark rounded square, with two input pins and one output pin. The binary hands the pixels
//! to eframe as the window/taskbar icon; `examples/icon_png.rs` renders the same pixels to a
//! PNG for previews. Everything is anti-aliased by 4×4 supersampling.

/// Icon side length in pixels (a multiple of 4, as eframe wants).
pub const SIZE: u32 = 256;

/// Supersampling factor per axis (16 samples per pixel).
const SS: u32 = 4;

type Rgb = (f32, f32, f32);

const BG: Rgb = (0.070, 0.082, 0.106); // #12151b — matches the canvas background
const BORDER: Rgb = (0.125, 0.149, 0.192); // a touch lighter, so it reads on dark taskbars
const GREEN: Rgb = (0.290, 0.871, 0.502); // #4ade80 — the signal/accent green
const BRIGHT: Rgb = (0.525, 0.937, 0.675); // #86efac — the "live output" node

#[derive(Clone, Copy)]
struct P {
    x: f32,
    y: f32,
}

fn p(x: f32, y: f32) -> P {
    P { x, y }
}

fn dist(a: P, b: P) -> f32 {
    ((a.x - b.x).powi(2) + (a.y - b.y).powi(2)).sqrt()
}

/// Distance from point `p` to segment `a`–`b`.
fn dist_seg(pt: P, a: P, b: P) -> f32 {
    let (dx, dy) = (b.x - a.x, b.y - a.y);
    let len2 = dx * dx + dy * dy;
    if len2 <= 1e-6 {
        return dist(pt, a);
    }
    let t = (((pt.x - a.x) * dx + (pt.y - a.y) * dy) / len2).clamp(0.0, 1.0);
    dist(pt, p(a.x + dx * t, a.y + dy * t))
}

/// Inside test for a rounded rectangle: within the bounds, and within `r` of the corner
/// centers in the corner regions.
fn in_round_rect(pt: P, min: P, max: P, r: f32) -> bool {
    if pt.x < min.x || pt.x > max.x || pt.y < min.y || pt.y > max.y {
        return false;
    }
    // How far the point pokes past the straight edges into a corner region (0 if not).
    let qx = (min.x + r - pt.x).max(pt.x - (max.x - r)).max(0.0);
    let qy = (min.y + r - pt.y).max(pt.y - (max.y - r)).max(0.0);
    qx * qx + qy * qy <= r * r
}

/// Resolve the (straight-alpha) color at a sample point, compositing the layers top-down.
fn sample(pt: P) -> (Rgb, f32) {
    let min = p(12.0, 12.0);
    let max = p(244.0, 244.0);

    // Outside the rounded silhouette is transparent.
    if !in_round_rect(pt, min, max, 54.0) {
        return (BG, 0.0);
    }

    // --- gate geometry (a flat-backed, round-fronted AND-gate outline) ---
    let top = 74.0;
    let bot = 182.0;
    let left = 84.0;
    let arc_cx = 142.0;
    let cy = 128.0;
    let r = 54.0; // == (bot - top) / 2
    let tip = arc_cx + r; // 196
    let hw = 5.0; // stroke half-width for the gate body
    let pw = 4.5; // stroke half-width for the pins

    let in1 = p(44.0, 101.0);
    let in2 = p(44.0, 155.0);
    let out = p(228.0, cy);

    // Nodes sit on top of the wires.
    if dist(pt, out) <= 11.0 {
        return (BRIGHT, 1.0);
    }
    if dist(pt, in1) <= 10.0 || dist(pt, in2) <= 10.0 {
        return (GREEN, 1.0);
    }

    // Pins.
    let on_pin = dist_seg(pt, in1, p(left, in1.y)) <= pw
        || dist_seg(pt, in2, p(left, in2.y)) <= pw
        || dist_seg(pt, p(tip, cy), out) <= pw;
    if on_pin {
        return (GREEN, 1.0);
    }

    // Gate outline: flat left edge, flat top/bottom, and the right semicircle.
    let on_flat = dist_seg(pt, p(left, top), p(left, bot)) <= hw
        || dist_seg(pt, p(left, top), p(arc_cx, top)) <= hw
        || dist_seg(pt, p(left, bot), p(arc_cx, bot)) <= hw;
    let on_arc = pt.x >= arc_cx && (dist(pt, p(arc_cx, cy)) - r).abs() <= hw;
    if on_flat || on_arc {
        return (GREEN, 1.0);
    }

    // A slim inner border ring, then the flat background fill.
    if !in_round_rect(pt, p(15.0, 15.0), p(241.0, 241.0), 51.0) {
        return (BORDER, 1.0);
    }
    (BG, 1.0)
}

/// Render the icon and return `(rgba, width, height)`.
pub fn rgba() -> (Vec<u8>, u32, u32) {
    let n = SIZE as usize;
    let mut out = vec![0u8; n * n * 4];
    let inv = 1.0 / (SS * SS) as f32;
    for py in 0..n {
        for px in 0..n {
            // Accumulate premultiplied color over the sub-samples for clean edges.
            let (mut ar, mut ag, mut ab, mut aa) = (0.0f32, 0.0, 0.0, 0.0);
            for sy in 0..SS {
                for sx in 0..SS {
                    let fx = px as f32 + (sx as f32 + 0.5) / SS as f32;
                    let fy = py as f32 + (sy as f32 + 0.5) / SS as f32;
                    let ((r, g, b), a) = sample(p(fx, fy));
                    ar += r * a;
                    ag += g * a;
                    ab += b * a;
                    aa += a;
                }
            }
            ar *= inv;
            ag *= inv;
            ab *= inv;
            aa *= inv;
            // Un-premultiply back to straight alpha.
            let (r, g, b) = if aa > 1e-4 {
                (ar / aa, ag / aa, ab / aa)
            } else {
                (0.0, 0.0, 0.0)
            };
            let i = (py * n + px) * 4;
            out[i] = (r * 255.0).round().clamp(0.0, 255.0) as u8;
            out[i + 1] = (g * 255.0).round().clamp(0.0, 255.0) as u8;
            out[i + 2] = (b * 255.0).round().clamp(0.0, 255.0) as u8;
            out[i + 3] = (aa * 255.0).round().clamp(0.0, 255.0) as u8;
        }
    }
    (out, SIZE, SIZE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn icon_has_expected_dimensions() {
        let (rgba, w, h) = rgba();
        assert_eq!(w, SIZE);
        assert_eq!(h, SIZE);
        assert_eq!(rgba.len(), (SIZE * SIZE * 4) as usize);
    }

    #[test]
    fn corners_transparent_center_opaque() {
        let (rgba, w, _h) = rgba();
        let at = |x: u32, y: u32| rgba[((y * w + x) * 4 + 3) as usize];
        // The rounded corners are outside the silhouette → transparent.
        assert_eq!(at(0, 0), 0);
        assert_eq!(at(SIZE - 1, SIZE - 1), 0);
        // The middle of the icon is solid.
        assert_eq!(at(SIZE / 2, SIZE / 2), 255);
    }
}
