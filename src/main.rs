//! LLMC — a clean, powerful digital-logic builder and simulator.

// Don't spawn a console window alongside the GUI on Windows release builds.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod ui;

fn main() -> eframe::Result<()> {
    ui::run()
}
