//! Process-level safety nets: one running instance, autosave with crash recovery, and a crash
//! log.
//!
//! * **Single instance** — the first LLMC holds a lock file in the config directory. A second
//!   launch (e.g. double-clicking a `.llmc` file) drops its file paths into an inbox folder and
//!   exits; the running window picks them up, opens them as tabs, and comes to the front.
//! * **Autosave** — every 30 s each project with unsaved changes is copied to an autosave
//!   folder. A clean exit clears it; if LLMC crashes or the PC loses power, the next launch
//!   offers to recover those copies.
//! * **Crash log** — a panic writes its message, location and backtrace to `crash.log`, and the
//!   next launch says so.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use llmc::io::{write_atomic, AppConfig, Project};

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ===================== crash log =====================

fn crash_log_path() -> Option<PathBuf> {
    AppConfig::dir().map(|d| d.join("crash.log"))
}

/// Install a panic hook that appends the panic (message, location, backtrace) to `crash.log`
/// before the default hook runs.
pub fn install_crash_handler() {
    let default = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        if let Some(path) = crash_log_path() {
            if let Some(dir) = path.parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            let bt = std::backtrace::Backtrace::force_capture();
            let entry = format!(
                "===== LLMC {} crashed (unix time {}) =====\n{info}\n\nBacktrace:\n{bt}\n\n",
                env!("CARGO_PKG_VERSION"),
                unix_now(),
            );
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
            {
                use std::io::Write;
                let _ = f.write_all(entry.as_bytes());
            }
            // Mark the crash for the next launch.
            let _ = std::fs::write(path.with_extension("pending"), b"1");
        }
        default(info);
    }));
}

/// If the previous run crashed, returns the crash log's path (and clears the marker).
pub fn take_crash_marker() -> Option<PathBuf> {
    let log = crash_log_path()?;
    let marker = log.with_extension("pending");
    if marker.exists() {
        let _ = std::fs::remove_file(marker);
        Some(log)
    } else {
        None
    }
}

// ===================== single instance =====================

/// How this launch should proceed.
pub enum Startup {
    /// This is the only LLMC: run, and watch the inbox (`None` if locking isn't possible —
    /// then we simply run without single-instance behavior).
    Primary(Option<Instance>),
    /// Another LLMC is running and has been handed the files: exit now.
    Forwarded,
}

/// The running instance's lock (held for the process lifetime) and inbox.
pub struct Instance {
    _lock: File,
    inbox: PathBuf,
    last_poll: Instant,
}

/// Become the primary instance, or hand `paths` to the one already running.
pub fn startup(paths: &[PathBuf]) -> Startup {
    let Some(dir) = AppConfig::dir() else {
        return Startup::Primary(None);
    };
    if std::fs::create_dir_all(&dir).is_err() {
        return Startup::Primary(None);
    }
    let inbox = dir.join("inbox");
    let Ok(lock) = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(dir.join("instance.lock"))
    else {
        return Startup::Primary(None);
    };
    match lock.try_lock() {
        Ok(()) => {
            // Requests left over from a crashed session are stale.
            let _ = std::fs::remove_dir_all(&inbox);
            let _ = std::fs::create_dir_all(&inbox);
            Startup::Primary(Some(Instance {
                _lock: lock,
                inbox,
                last_poll: Instant::now(),
            }))
        }
        Err(std::fs::TryLockError::WouldBlock) => {
            let _ = std::fs::create_dir_all(&inbox);
            let body: String = paths
                .iter()
                .filter_map(|p| std::fs::canonicalize(p).ok().or_else(|| Some(p.clone())))
                .map(|p| format!("{}\n", p.display()))
                .collect();
            let name = format!("{}-{}.txt", std::process::id(), unix_now());
            if write_atomic(&inbox.join(name), body.as_bytes()).is_ok() {
                Startup::Forwarded
            } else {
                Startup::Primary(None)
            }
        }
        Err(_) => Startup::Primary(None),
    }
}

impl Instance {
    /// The folder other launches drop their requests into.
    pub fn inbox_dir(&self) -> PathBuf {
        self.inbox.clone()
    }

    /// Files another launch asked us to open (checked at most every half second). `Some` —
    /// even with no paths — means "come to the front".
    pub fn poll(&mut self) -> Option<Vec<PathBuf>> {
        if self.last_poll.elapsed() < Duration::from_millis(500) {
            return None;
        }
        self.last_poll = Instant::now();
        let entries = std::fs::read_dir(&self.inbox).ok()?;
        let mut paths = Vec::new();
        let mut any = false;
        for e in entries.flatten() {
            let p = e.path();
            if p.extension().is_some_and(|x| x == "txt") {
                if let Ok(text) = std::fs::read_to_string(&p) {
                    any = true;
                    paths.extend(
                        text.lines()
                            .map(str::trim)
                            .filter(|l| !l.is_empty())
                            .map(PathBuf::from),
                    );
                }
                let _ = std::fs::remove_file(&p);
            }
        }
        any.then_some(paths)
    }
}

// ===================== autosave =====================

/// Metadata stored next to each autosaved project.
#[derive(serde::Serialize, serde::Deserialize, Clone)]
struct Meta {
    title: String,
    original: Option<PathBuf>,
    saved_at: u64,
}

/// An autosaved project found at launch (left behind by a crash).
#[derive(Clone)]
pub struct Recovered {
    pub title: String,
    pub original: Option<PathBuf>,
    pub saved_at: u64,
    file: PathBuf,
}

impl Recovered {
    pub fn load(&self) -> Option<Project> {
        Project::load(&self.file).ok()
    }

    /// "3 min ago" style age.
    pub fn age(&self) -> String {
        let s = unix_now().saturating_sub(self.saved_at);
        if s < 90 {
            "just now".to_string()
        } else if s < 3600 {
            format!("{} min ago", s / 60)
        } else if s < 86_400 {
            format!("{} h ago", s / 3600)
        } else {
            format!("{} days ago", s / 86_400)
        }
    }
}

/// Every 30 s, copies projects with unsaved changes to the autosave folder.
pub struct Autosave {
    dir: Option<PathBuf>,
    /// Distinguishes this run's files from a previous (crashed) run's.
    session: u64,
    last: Instant,
}

/// How often unsaved work is copied to the autosave folder.
pub const AUTOSAVE_EVERY: Duration = Duration::from_secs(30);

impl Autosave {
    /// Set up autosave and return the projects a previous, crashed run left behind.
    pub fn start() -> (Self, Vec<Recovered>) {
        let dir = AppConfig::dir().map(|d| d.join("autosave"));
        let mut found = Vec::new();
        if let Some(dir) = &dir {
            let _ = std::fs::create_dir_all(dir);
            if let Ok(entries) = std::fs::read_dir(dir) {
                for e in entries.flatten() {
                    let p = e.path();
                    if p.extension().is_some_and(|x| x == "json") {
                        let file = p.with_extension("llmc");
                        let meta = std::fs::read_to_string(&p)
                            .ok()
                            .and_then(|s| serde_json::from_str::<Meta>(&s).ok());
                        match meta {
                            Some(m) if file.exists() => found.push(Recovered {
                                title: m.title,
                                original: m.original,
                                saved_at: m.saved_at,
                                file,
                            }),
                            _ => {
                                let _ = std::fs::remove_file(&p);
                                let _ = std::fs::remove_file(&file);
                            }
                        }
                    }
                }
            }
        }
        found.sort_by_key(|r| std::cmp::Reverse(r.saved_at));
        (
            Self {
                dir,
                session: unix_now(),
                last: Instant::now(),
            },
            found,
        )
    }

    fn stem(&self, doc_id: u64) -> Option<PathBuf> {
        Some(
            self.dir
                .as_ref()?
                .join(format!("{}-{doc_id}", self.session)),
        )
    }

    /// Is it time for the next round?
    pub fn due(&mut self) -> bool {
        if self.last.elapsed() >= AUTOSAVE_EVERY {
            self.last = Instant::now();
            true
        } else {
            false
        }
    }

    /// Store one project's unsaved state.
    pub fn write(&self, doc_id: u64, project: &Project, title: &str, original: Option<&Path>) {
        let Some(stem) = self.stem(doc_id) else {
            return;
        };
        let Ok(json) = project.to_json() else {
            return;
        };
        if write_atomic(&stem.with_extension("llmc"), json.as_bytes()).is_err() {
            return;
        }
        let meta = Meta {
            title: title.to_string(),
            original: original.map(Path::to_path_buf),
            saved_at: unix_now(),
        };
        if let Ok(m) = serde_json::to_string(&meta) {
            let _ = write_atomic(&stem.with_extension("json"), m.as_bytes());
        }
    }

    /// Drop a project's autosave (it was saved, closed, or is clean again).
    pub fn remove(&self, doc_id: u64) {
        if let Some(stem) = self.stem(doc_id) {
            let _ = std::fs::remove_file(stem.with_extension("json"));
            let _ = std::fs::remove_file(stem.with_extension("llmc"));
        }
    }

    /// Forget recovered files once the user chose what to do with them.
    pub fn discard(items: &[Recovered]) {
        for r in items {
            let _ = std::fs::remove_file(&r.file);
            let _ = std::fs::remove_file(r.file.with_extension("json"));
        }
    }

    /// A clean exit: nothing needs recovering.
    pub fn clear_session(&self) {
        let Some(dir) = &self.dir else {
            return;
        };
        let prefix = format!("{}-", self.session);
        if let Ok(entries) = std::fs::read_dir(dir) {
            for e in entries.flatten() {
                if e.file_name().to_string_lossy().starts_with(&prefix) {
                    let _ = std::fs::remove_file(e.path());
                }
            }
        }
    }
}
