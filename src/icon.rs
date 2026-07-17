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

/// Render the icon at the default [`SIZE`] and return `(rgba, width, height)`.
pub fn rgba() -> (Vec<u8>, u32, u32) {
    rgba_at(SIZE)
}

/// Render the icon at an arbitrary square `size`. The glyph is defined in a fixed 256-unit
/// design space and sampled into `size`×`size`, so the same logo scales to any icon size
/// (16/32/48… for the Windows `.ico`, 256 for the window icon).
pub fn rgba_at(size: u32) -> (Vec<u8>, u32, u32) {
    let n = size.max(1) as usize;
    let mut out = vec![0u8; n * n * 4];
    let inv = 1.0 / (SS * SS) as f32;
    // Map an output pixel (plus sub-sample offset) into the 256-unit design space.
    let scale = SIZE as f32 / size as f32;
    for py in 0..n {
        for px in 0..n {
            // Accumulate premultiplied color over the sub-samples for clean edges.
            let (mut ar, mut ag, mut ab, mut aa) = (0.0f32, 0.0, 0.0, 0.0);
            for sy in 0..SS {
                for sx in 0..SS {
                    let fx = (px as f32 + (sx as f32 + 0.5) / SS as f32) * scale;
                    let fy = (py as f32 + (sy as f32 + 0.5) / SS as f32) * scale;
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
    (out, size, size)
}

/// The icon sizes packed into the Windows `.ico` (small ones for the taskbar, large for
/// Explorer / high-DPI). Windows picks the closest match per context.
pub const ICO_SIZES: &[u32] = &[16, 24, 32, 48, 64, 128, 256];

/// Encode the app icon as a multi-resolution Windows `.ico` (PNG-compressed entries, which
/// Windows Vista+ supports at every size). This is what `build.rs` embeds into `llmc.exe`.
pub fn ico_bytes() -> Vec<u8> {
    let images: Vec<(u32, Vec<u8>)> = ICO_SIZES
        .iter()
        .map(|&s| {
            let (rgba, w, h) = rgba_at(s);
            (s, encode_png(&rgba, w, h))
        })
        .collect();

    let mut out = Vec::new();
    // ICONDIR: reserved, type=1 (icon), image count.
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&(images.len() as u16).to_le_bytes());

    // ICONDIRENTRY per image, then the concatenated PNG blobs.
    let mut offset = 6 + 16 * images.len();
    for (size, png) in &images {
        let dim = if *size >= 256 { 0u8 } else { *size as u8 };
        out.push(dim); // width  (0 means 256)
        out.push(dim); // height (0 means 256)
        out.push(0); // palette color count
        out.push(0); // reserved
        out.extend_from_slice(&1u16.to_le_bytes()); // color planes
        out.extend_from_slice(&32u16.to_le_bytes()); // bits per pixel
        out.extend_from_slice(&(png.len() as u32).to_le_bytes()); // bytes in resource
        out.extend_from_slice(&(offset as u32).to_le_bytes()); // offset from file start
        offset += png.len();
    }
    for (_, png) in &images {
        out.extend_from_slice(png);
    }
    out
}

/// Encode 8-bit RGBA pixels as a PNG using uncompressed (stored) DEFLATE blocks — no
/// dependencies, deterministic output. Used for the `.ico` entries and previews.
pub fn encode_png(rgba: &[u8], w: u32, h: u32) -> Vec<u8> {
    let mut out = vec![0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];

    // IHDR: 8-bit, color type 6 (RGBA), no interlace.
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&w.to_be_bytes());
    ihdr.extend_from_slice(&h.to_be_bytes());
    ihdr.extend_from_slice(&[8, 6, 0, 0, 0]);
    png_chunk(&mut out, b"IHDR", &ihdr);

    // Raw image data: each row prefixed with filter byte 0.
    let mut raw = Vec::with_capacity((h * (w * 4 + 1)) as usize);
    for y in 0..h as usize {
        raw.push(0);
        let start = y * w as usize * 4;
        raw.extend_from_slice(&rgba[start..start + w as usize * 4]);
    }
    png_chunk(&mut out, b"IDAT", &zlib_store(&raw));

    png_chunk(&mut out, b"IEND", &[]);
    out
}

fn png_chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(kind);
    out.extend_from_slice(data);
    let mut crc_data = Vec::with_capacity(4 + data.len());
    crc_data.extend_from_slice(kind);
    crc_data.extend_from_slice(data);
    out.extend_from_slice(&crc32(&crc_data).to_be_bytes());
}

/// Wrap `data` in a zlib stream using only stored (uncompressed) DEFLATE blocks.
fn zlib_store(data: &[u8]) -> Vec<u8> {
    let mut out = vec![0x78, 0x01]; // zlib header: 32K window, default level
    let mut i = 0;
    while i < data.len() {
        let block = (data.len() - i).min(0xffff);
        let final_block = i + block >= data.len();
        out.push(u8::from(final_block));
        let len = block as u16;
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&(!len).to_le_bytes());
        out.extend_from_slice(&data[i..i + block]);
        i += block;
    }
    if data.is_empty() {
        out.extend_from_slice(&[1, 0, 0, 0xff, 0xff]);
    }
    out.extend_from_slice(&adler32(data).to_be_bytes());
    out
}

fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xffff_ffffu32;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xedb8_8320 & mask);
        }
    }
    !crc
}

fn adler32(data: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for &byte in data {
        a = (a + byte as u32) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
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

    #[test]
    fn rgba_at_respects_requested_size() {
        let (rgba, w, h) = rgba_at(32);
        assert_eq!((w, h), (32, 32));
        assert_eq!(rgba.len(), 32 * 32 * 4);
    }

    #[test]
    fn png_has_signature_and_ihdr() {
        let (rgba, w, h) = rgba_at(16);
        let png = encode_png(&rgba, w, h);
        assert_eq!(&png[..8], &[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]);
        // First chunk after the signature is IHDR.
        assert_eq!(&png[12..16], b"IHDR");
    }

    #[test]
    fn ico_header_matches_entry_count() {
        let ico = ico_bytes();
        assert_eq!(&ico[0..2], &0u16.to_le_bytes()); // reserved
        assert_eq!(&ico[2..4], &1u16.to_le_bytes()); // type = icon
        let count = u16::from_le_bytes([ico[4], ico[5]]);
        assert_eq!(count as usize, ICO_SIZES.len());
        // Every entry's declared offset+length must stay within the file.
        for k in 0..count as usize {
            let e = 6 + k * 16;
            let len = u32::from_le_bytes([ico[e + 8], ico[e + 9], ico[e + 10], ico[e + 11]]);
            let off = u32::from_le_bytes([ico[e + 12], ico[e + 13], ico[e + 14], ico[e + 15]]);
            assert!(off as usize + len as usize <= ico.len());
        }
    }
}
