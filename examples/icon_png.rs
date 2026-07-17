//! Render the procedural app icon (`llmc::icon`) to a PNG. Usage:
//! `cargo run --example icon_png -- icon.png`. Handy for previews or as the source image
//! for a Windows `.ico`. Uses a tiny built-in PNG writer so there are no extra dependencies.

fn main() {
    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "icon.png".to_string());
    let (rgba, w, h) = llmc::icon::rgba();
    let png = encode_png(&rgba, w, h);
    std::fs::write(&out, png).expect("write png");
    println!("wrote {out} ({w}x{h})");
}

/// Encode 8-bit RGBA pixels as a PNG using uncompressed (stored) DEFLATE blocks.
fn encode_png(rgba: &[u8], w: u32, h: u32) -> Vec<u8> {
    let mut out = vec![0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];

    // IHDR: 8-bit, color type 6 (RGBA), no interlace.
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&w.to_be_bytes());
    ihdr.extend_from_slice(&h.to_be_bytes());
    ihdr.extend_from_slice(&[8, 6, 0, 0, 0]);
    chunk(&mut out, b"IHDR", &ihdr);

    // Raw image data: each row prefixed with filter byte 0.
    let mut raw = Vec::with_capacity((h * (w * 4 + 1)) as usize);
    for y in 0..h as usize {
        raw.push(0);
        let start = y * w as usize * 4;
        raw.extend_from_slice(&rgba[start..start + w as usize * 4]);
    }
    chunk(&mut out, b"IDAT", &zlib_store(&raw));

    chunk(&mut out, b"IEND", &[]);
    out
}

fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(kind);
    out.extend_from_slice(data);
    let mut crc_data = Vec::with_capacity(4 + data.len());
    crc_data.extend_from_slice(kind);
    crc_data.extend_from_slice(data);
    out.extend_from_slice(&crc32(&crc_data).to_be_bytes());
}

/// Wrap `data` in a zlib stream that uses only stored (uncompressed) DEFLATE blocks.
fn zlib_store(data: &[u8]) -> Vec<u8> {
    let mut out = vec![0x78, 0x01]; // zlib header: 32K window, default level
    let mut i = 0;
    while i < data.len() {
        let block = (data.len() - i).min(0xffff);
        let final_block = i + block >= data.len();
        out.push(if final_block { 1 } else { 0 });
        let len = block as u16;
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&(!len).to_le_bytes());
        out.extend_from_slice(&data[i..i + block]);
        i += block;
    }
    // Empty input still needs one final stored block.
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
