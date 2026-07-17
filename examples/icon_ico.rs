//! Regenerate the Windows icon that `build.rs` embeds into `llmc.exe`.
//! Usage: `cargo run --example icon_ico` writes `assets/icon.ico`.
//!
//! The icon is committed so the Windows build doesn't need to render it, but it is fully
//! derived from `llmc::icon`, so re-run this whenever the logo changes.

fn main() {
    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "assets/icon.ico".to_string());
    if let Some(parent) = std::path::Path::new(&out).parent() {
        std::fs::create_dir_all(parent).expect("create assets dir");
    }
    let ico = llmc::icon::ico_bytes();
    std::fs::write(&out, &ico).expect("write ico");
    println!(
        "wrote {out} ({} bytes, sizes {:?})",
        ico.len(),
        llmc::icon::ICO_SIZES
    );
}
