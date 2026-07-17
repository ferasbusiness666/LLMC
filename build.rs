//! Embed the app icon into `llmc.exe` on Windows so the executable itself carries the logo
//! in Explorer, the taskbar, and shortcuts (the runtime `with_icon` call only affects the
//! live window). No-op on other platforms and if the resource compiler is unavailable, so
//! Linux/macOS builds and CI are unaffected.

fn main() {
    println!("cargo:rerun-if-changed=assets/icon.ico");
    println!("cargo:rerun-if-changed=build.rs");

    #[cfg(windows)]
    {
        let mut res = winresource::WindowsResource::new();
        res.set_icon("assets/icon.ico");
        if let Err(e) = res.compile() {
            // Don't fail the build over the icon; just note it.
            println!("cargo:warning=could not embed the app icon: {e}");
        }
    }
}
