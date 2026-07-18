//! The eframe/egui GUI (binary-only). Everything here depends on egui; the pure logic
//! lives in the `llmc` library crate.

mod app;
mod assistant;
mod glyphs;
mod theme;

/// Launch the native window and run the app. An optional `.llmc` path may be passed as
/// the first CLI argument (also enables double-click / drag-to-open on Windows).
pub fn run() -> eframe::Result<()> {
    let config = llmc::io::AppConfig::load();
    let initial = std::env::args_os().nth(1).map(std::path::PathBuf::from);
    let (rgba, width, height) = llmc::icon::rgba();
    let icon = eframe::egui::IconData {
        rgba,
        width,
        height,
    };
    let native_options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_title("LLMC — Logic Builder")
            .with_icon(std::sync::Arc::new(icon))
            .with_inner_size([1280.0, 820.0])
            .with_min_inner_size([860.0, 560.0]),
        ..Default::default()
    };
    eframe::run_native(
        "LLMC",
        native_options,
        Box::new(move |cc| Ok(Box::new(app::LlmcApp::new(cc, config, initial)))),
    )
}
