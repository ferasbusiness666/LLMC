//! Render the procedural app icon (`llmc::icon`) to a PNG for previews.
//! Usage: `cargo run --example icon_png -- icon.png`.

fn main() {
    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "icon.png".to_string());
    let (rgba, w, h) = llmc::icon::rgba();
    let png = llmc::icon::encode_png(&rgba, w, h);
    std::fs::write(&out, png).expect("write png");
    println!("wrote {out} ({w}x{h})");
}
