//! The eframe/egui GUI (binary-only). Everything here depends on egui; the pure logic
//! lives in the `llmc` library crate.

mod ai_log;
mod ai_net;
mod app;
mod assistant;
mod canvas;
mod design;
mod document;
mod glyphs;
mod icons;
mod inspector;
mod keystore;
mod palette;
mod session;

/// Launch the native window and run the app. `.llmc` paths may be passed as arguments (that's
/// how double-clicking a file opens it); if LLMC is already running, they open there instead.
pub fn run() -> eframe::Result<()> {
    session::install_crash_handler();
    let paths: Vec<std::path::PathBuf> = std::env::args_os()
        .skip(1)
        .map(std::path::PathBuf::from)
        .collect();
    let instance = match session::startup(&paths) {
        session::Startup::Forwarded => return Ok(()),
        session::Startup::Primary(instance) => instance,
    };
    let config = llmc::io::AppConfig::load();
    let (rgba, width, height) = llmc::icon::rgba();
    let icon = eframe::egui::IconData {
        rgba,
        width,
        height,
    };
    let mut viewport = eframe::egui::ViewportBuilder::default()
        .with_title("LLMC")
        .with_icon(std::sync::Arc::new(icon))
        .with_min_inner_size([960.0, 600.0]);
    viewport = match config.window.filter(|w| {
        w.width.is_finite() && w.height.is_finite() && w.width >= 400.0 && w.height >= 300.0
    }) {
        // Restore last session's size and position (a position far off any screen is dropped).
        Some(w) => {
            let v = viewport
                .with_inner_size([w.width, w.height])
                .with_maximized(w.maximized);
            if (-200.0..10_000.0).contains(&w.x) && (-200.0..10_000.0).contains(&w.y) {
                v.with_position([w.x.max(0.0), w.y.max(0.0)])
            } else {
                v
            }
        }
        // First run: fill the screen, whatever its size (a fixed 1440×900 window spills off a
        // 1366×768 laptop screen).
        None => viewport
            .with_inner_size([1440.0, 900.0])
            .with_maximized(true),
    };
    let native_options = eframe::NativeOptions {
        viewport,
        ..Default::default()
    };
    eframe::run_native(
        "LLMC",
        native_options,
        Box::new(move |cc| Ok(Box::new(app::LlmcApp::new(cc, config, paths, instance)))),
    )
}
