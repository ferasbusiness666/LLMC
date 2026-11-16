//! Turning an assistant reply into circuit edits.
//!
//! Providers answer in free text, so this module does the tolerant parsing: it pulls any
//! `<think>` reasoning out for separate display, and extracts a `{"commands":[…]}` block (in a
//! ```json fence or raw) into a list of loosely-typed [`RawOp`]s. Applying those ops to a real
//! circuit — allocating ids, resolving refs, validating ports — happens in [`super::apply`].

use std::ops::Range;

use crate::model::BlockType;

/// What an `add` op places: a primitive block, or an instance of a library chip (by name).
#[derive(Debug, Clone, PartialEq)]
pub enum AddKind {
    Prim(BlockType),
    Chip(String),
}

/// How a connect op names a port: by index, or by pin name (resolved against a chip's pins).
/// `None` on an input side means "pick the first free input" — auto-assignment.
#[derive(Debug, Clone, PartialEq)]
pub enum PortSel {
    Index(u16),
    Name(String),
}

/// One edit the model asked for, still in "handle" form (refs/ids as strings) and unvalidated.
/// The app resolves handles to real block ids and checks ports before applying.
#[derive(Debug, Clone, PartialEq)]
pub enum RawOp {
    Add {
        r#ref: Option<String>,
        kind: AddKind,
        x: i32,
        y: i32,
        inputs: Option<u16>,
        label: Option<String>,
        state: bool,
        /// Clock frequency in Hz (clocks only).
        hz: Option<f32>,
        /// Custom body color.
        color: Option<[u8; 3]>,
    },
    Connect {
        from: String,
        from_port: Option<PortSel>,
        to: String,
        to_port: Option<PortSel>,
    },
    Remove {
        target: String,
    },
    Move {
        target: String,
        x: i32,
        y: i32,
    },
    Label {
        target: String,
        text: Option<String>,
    },
    /// Define a reusable chip: declared pins plus the ops that build its internals. The app
    /// creates the pin blocks automatically (refs = pin names), runs the inner ops against a
    /// scratch circuit, registers the chip, and truth-table-verifies it in isolation.
    DefChip {
        name: String,
        inputs: Vec<String>,
        outputs: Vec<String>,
        ops: Vec<RawOp>,
    },
    /// Change properties of an existing block: input count, clock frequency, color, a switch's
    /// default level, or its label. Unset fields are left alone.
    Set {
        target: String,
        inputs: Option<u16>,
        hz: Option<f32>,
        /// `Some(None)` clears a custom color.
        color: Option<Option<[u8; 3]>>,
        state: Option<bool>,
        label: Option<String>,
    },
    /// A scripted test: each step sets switches/buttons (by handle), optionally pulses some
    /// (high then low) and/or lets time pass, then reads every LED. This is how sequential
    /// circuits (latches, registers, memory, clocked CPUs) get verified.
    Test {
        steps: Vec<TestStep>,
    },
    /// Remove wires between two blocks (one port pair, or every wire from `from` to `to`).
    Disconnect {
        from: String,
        from_port: Option<PortSel>,
        to: String,
        to_port: Option<PortSel>,
    },
    /// Read-only: report blocks' current pin values and what drives each input.
    Inspect {
        targets: Vec<String>,
    },
    /// Read-only: walk back from a block through everything that feeds it, with live values
    /// ("why is CARRY off?").
    Trace {
        target: String,
    },
    /// Point at blocks on the canvas (no edit) — used when explaining a circuit.
    Highlight {
        targets: Vec<String>,
        note: Option<String>,
    },
    /// Delete a chip definition (and every placed instance of it).
    DeleteChip {
        name: String,
    },
}

/// One step of a scripted test.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TestStep {
    /// Switch/button levels to set (they persist into later steps).
    pub set: Vec<(String, bool)>,
    /// Inputs to pulse: set high, settle, set low, settle (a manual clock edge).
    pub pulse: Vec<String>,
    /// Seconds of simulated time to let pass (clocks tick).
    pub wait: Option<f64>,
    /// Number of clock edges (transitions of the fastest clock) to advance.
    pub ticks: Option<u32>,
    /// LED levels this step should end with; mismatches are reported as failures.
    pub expect: Vec<(String, bool)>,
}

/// A parsed assistant reply: the reasoning (if any), the prose to show, and edit ops.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct ParsedReply {
    pub thinking: Option<String>,
    pub text: String,
    pub ops: Vec<RawOp>,
    /// Commands that were present but couldn't be read (unknown op, missing field, broken
    /// JSON) — fed back to the model instead of failing silently.
    pub problems: Vec<String>,
}

/// Split reasoning, strip the command block, and parse it into ops.
pub fn parse_reply(raw: &str) -> ParsedReply {
    let (thinking, body) = split_reasoning(raw);
    let (ops, text, problems) = extract_ops(&body);
    ParsedReply {
        thinking,
        text,
        ops,
        problems,
    }
}

/// Live view of a *partial* streaming reply, for display while tokens are still arriving.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct LiveView {
    /// Reasoning so far (from a separate reasoning stream or an inline `<think>` tag).
    pub reasoning: Option<String>,
    /// The prose answer so far, cut just before any command block.
    pub answer: String,
    /// The raw command JSON typed so far (fence markers stripped) — shown in a collapsible
    /// "Writing commands…" section so the user can watch the build being written.
    pub commands: Option<String>,
}

/// Split a partial streaming reply into reasoning / answer / in-progress command text.
/// `reason_field` is a separate reasoning stream (some providers emit one); otherwise an inline
/// `<think>` tag in `content` is used, open or closed.
pub fn live_split(reason_field: &str, content: &str) -> LiveView {
    let opt = |s: &str| {
        let t = s.trim();
        (!t.is_empty()).then(|| t.to_string())
    };
    let view = |reasoning: Option<String>, body: &str| {
        let (answer, commands) = split_live_answer(body);
        LiveView {
            reasoning,
            answer,
            commands,
        }
    };
    if !reason_field.trim().is_empty() {
        return view(opt(reason_field), content);
    }
    for (open_tag, close_tag) in [("<think>", "</think>"), ("<thinking>", "</thinking>")] {
        if let Some(open) = content.find(open_tag) {
            let after = open + open_tag.len();
            return match content[after..].find(close_tag) {
                Some(rel) => {
                    let cpos = after + rel;
                    let body =
                        format!("{}{}", &content[..open], &content[cpos + close_tag.len()..]);
                    view(opt(&content[after..cpos]), &body)
                }
                None => view(opt(&content[after..]), &content[..open]),
            };
        }
    }
    view(None, content)
}

/// Split partial body text into (prose answer, in-progress command text). The answer stops at
/// the first code fence or `{"commands"` object; everything from there on — with fence markers
/// stripped — is the live command text.
fn split_live_answer(s: &str) -> (String, Option<String>) {
    let mut cut = s.len();
    if let Some(p) = s.find("```") {
        cut = cut.min(p);
    }
    if let Some(p) = s.find("{\"commands\"") {
        cut = cut.min(p);
    }
    let answer = s[..cut].trim_end().to_string();
    if cut == s.len() {
        return (answer, None);
    }
    // Strip the opening fence line (```json) and any closing fence from the tail.
    let mut tail = s[cut..].trim_start();
    if let Some(rest) = tail.strip_prefix("```") {
        tail = match rest.find('\n') {
            Some(nl) => &rest[nl + 1..],
            None => "", // fence opened but its line isn't complete yet
        };
    }
    let tail = tail.trim_end().trim_end_matches("```").trim_end();
    (answer, (!tail.is_empty()).then(|| tail.to_string()))
}

/// Pull the contents of any `<think>…</think>` / `<thinking>…</thinking>` blocks out of `raw`,
/// returning (joined reasoning, body with those blocks removed). Unclosed tags are left as-is.
fn split_reasoning(raw: &str) -> (Option<String>, String) {
    let mut thoughts: Vec<String> = Vec::new();
    let mut body = String::with_capacity(raw.len());
    let lower = raw.to_lowercase();
    let mut i = 0;
    while i < raw.len() {
        // Find the next opening tag at or after i.
        let next = ["<think>", "<thinking>"]
            .iter()
            .filter_map(|tag| lower[i..].find(tag).map(|p| (i + p, tag.len())))
            .min_by_key(|(p, _)| *p);
        let Some((open, open_len)) = next else {
            body.push_str(&raw[i..]);
            break;
        };
        let content_start = open + open_len;
        // Matching close tag.
        let close = ["</think>", "</thinking>"]
            .iter()
            .filter_map(|tag| {
                lower[content_start..]
                    .find(tag)
                    .map(|p| (content_start + p, tag.len()))
            })
            .min_by_key(|(p, _)| *p);
        let Some((close_pos, close_len)) = close else {
            // No closing tag — keep the rest verbatim.
            body.push_str(&raw[i..]);
            break;
        };
        body.push_str(&raw[i..open]);
        let thought = raw[content_start..close_pos].trim();
        if !thought.is_empty() {
            thoughts.push(thought.to_string());
        }
        i = close_pos + close_len;
    }
    let thinking = if thoughts.is_empty() {
        None
    } else {
        Some(thoughts.join("\n\n"))
    };
    (thinking, body.trim().to_string())
}

/// Extract commands from `body`, returning (ops, body-without-the-command-block).
///
/// Tolerant on purpose: many models emit "JSONC" — `//` comments, trailing commas, or a reply
/// that got truncated mid-array. We first try the located command block (sanitized), then, if
/// that yields nothing, recover individual `{"op":…}` objects wherever they appear.
fn extract_ops(body: &str) -> (Vec<RawOp>, String, Vec<String>) {
    if let Some((json, rest)) = find_command_json(body) {
        let (ops, mut problems) = ops_from_text(&json);
        if !ops.is_empty() {
            return (ops, rest.trim().to_string(), problems);
        }
        if problems.is_empty() {
            let detail = serde_json::from_str::<serde_json::Value>(&sanitize_json(&json))
                .err()
                .map(|e| format!(" ({e})"))
                .unwrap_or_default();
            problems.push(format!(
                "your command block could not be read as JSON{detail} \u{2014} nothing was \
                 applied; resend it as strict JSON"
            ));
        }
        // Keep the prose; the broken block isn't shown.
        let fallback = ops_from_text(body).0;
        if !fallback.is_empty() {
            return (fallback, rest.trim().to_string(), Vec::new());
        }
        return (Vec::new(), rest.trim().to_string(), problems);
    }
    // No usable block was located (missing fence, etc.) — scan the whole reply for loose op
    // objects and, if any are found, trim the JSON-ish tail from what we display.
    let (ops, problems) = ops_from_text(body);
    if !ops.is_empty() {
        let cut = body
            .find("```")
            .or_else(|| body.find("{\""))
            .unwrap_or(body.len());
        return (ops, body[..cut].trim().to_string(), problems);
    }
    (Vec::new(), body.trim().to_string(), Vec::new())
}

/// Parse ops from a chunk of text that should contain a command block. First tries the whole
/// thing as JSON (after stripping comments / trailing commas), then falls back to scanning for
/// individual op objects so a malformed or truncated array still yields its valid commands.
/// Also returns a note for every command that was present but unreadable.
fn ops_from_text(region: &str) -> (Vec<RawOp>, Vec<String>) {
    let sanitized = sanitize_json(region);
    if let Ok(val) = serde_json::from_str::<serde_json::Value>(&sanitized) {
        if let Some(arr) = val
            .get("commands")
            .and_then(|c| c.as_array())
            .or_else(|| val.as_array())
        {
            let mut ops = Vec::new();
            let mut problems = Vec::new();
            for (i, item) in arr.iter().enumerate() {
                match parse_op(item) {
                    Some(op) => ops.push(op),
                    None => problems.push(describe_unreadable(i + 1, item)),
                }
            }
            if !ops.is_empty() || !problems.is_empty() {
                return (ops, problems);
            }
        }
    }
    (scan_op_objects(&sanitized), Vec::new())
}

/// Why command `n` couldn't be used, in words the model can act on.
fn describe_unreadable(n: usize, v: &serde_json::Value) -> String {
    let mut snippet = v.to_string();
    if snippet.chars().count() > 90 {
        snippet = format!("{}\u{2026}", snippet.chars().take(90).collect::<String>());
    }
    match v
        .get("op")
        .or_else(|| v.get("action"))
        .and_then(|o| o.as_str())
    {
        None => format!("command {n} has no \"op\" field: {snippet}"),
        Some(op) if !KNOWN_OPS.contains(&normalize(op).as_str()) => format!(
            "command {n}: unknown op \"{op}\" (use add, connect, disconnect, set, remove, move, \
             label, defchip, delete_chip, test, inspect, trace, highlight)"
        ),
        Some(op) => format!("command {n} (\"{op}\") is missing a required field: {snippet}"),
    }
}

/// Every op spelling [`parse_op`] accepts (normalized).
const KNOWN_OPS: &[&str] = &[
    "add",
    "addblock",
    "place",
    "create",
    "new",
    "instance",
    "set",
    "setprops",
    "props",
    "properties",
    "configure",
    "update",
    "modify",
    "connect",
    "wire",
    "addconnection",
    "link",
    "join",
    "remove",
    "removeblock",
    "delete",
    "deleteblock",
    "erase",
    "move",
    "moveblock",
    "reposition",
    "label",
    "setlabel",
    "rename",
    "defchip",
    "definechip",
    "chip",
    "definecomponent",
    "defmodule",
    "test",
    "simulate",
    "run",
    "check",
    "verify",
    "disconnect",
    "unwire",
    "removewire",
    "deletewire",
    "removeconnection",
    "deleteconnection",
    "inspect",
    "probe",
    "read",
    "query",
    "trace",
    "why",
    "highlight",
    "point",
    "show",
    "focus",
    "deletechip",
    "removechip",
];

/// Recover ops by scanning for balanced `{…}` objects, parsing each independently. An object
/// that parses as a single op (including a `defchip`, which legitimately contains an inner
/// `"commands"` array) is consumed whole; a wrapper object is stepped into instead.
fn scan_op_objects(s: &str) -> Vec<RawOp> {
    let b = s.as_bytes();
    let mut ops = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'{' {
            if let Some(end) = balanced_end(b, i) {
                let obj = &s[i..=end];
                if obj.contains("\"op\"") {
                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(obj) {
                        if let Some(op) = parse_op(&v) {
                            ops.push(op);
                            i = end + 1; // consume this object (and its nested braces)
                            continue;
                        }
                        // Valid JSON but not an op itself — a {"commands":[…]} wrapper: take
                        // its ops directly and consume it.
                        if let Some(arr) = v.get("commands").and_then(|c| c.as_array()) {
                            ops.extend(arr.iter().filter_map(parse_op));
                            i = end + 1;
                            continue;
                        }
                    }
                    if !obj.contains("\"commands\"") {
                        i = end + 1; // malformed lone op — skip it entirely
                        continue;
                    }
                    // Malformed wrapper: step inside to salvage its inner ops.
                }
            }
        }
        i += 1;
    }
    ops
}

/// Strip `//` line comments, `/* … */` block comments, and trailing commas (`,}` / `,]`) from a
/// JSON-ish string, leaving string contents untouched — so models' "JSONC" output parses.
fn sanitize_json(s: &str) -> String {
    let b = s.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(b.len());
    let mut in_str = false;
    let mut esc = false;
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        if in_str {
            out.push(c);
            if esc {
                esc = false;
            } else if c == b'\\' {
                esc = true;
            } else if c == b'"' {
                in_str = false;
            }
            i += 1;
            continue;
        }
        match c {
            b'"' => {
                in_str = true;
                out.push(c);
                i += 1;
            }
            b'/' if b.get(i + 1) == Some(&b'/') => {
                while i < b.len() && b[i] != b'\n' {
                    i += 1;
                }
            }
            b'/' if b.get(i + 1) == Some(&b'*') => {
                i += 2;
                while i + 1 < b.len() && !(b[i] == b'*' && b[i + 1] == b'/') {
                    i += 1;
                }
                i += 2; // consume the closing */
            }
            _ => {
                out.push(c);
                i += 1;
            }
        }
    }
    strip_trailing_commas(&String::from_utf8(out).unwrap_or_default())
}

/// Remove commas that directly precede a `}` or `]` (ignoring whitespace), outside of strings.
fn strip_trailing_commas(s: &str) -> String {
    let b = s.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(b.len());
    let mut in_str = false;
    let mut esc = false;
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        if in_str {
            out.push(c);
            if esc {
                esc = false;
            } else if c == b'\\' {
                esc = true;
            } else if c == b'"' {
                in_str = false;
            }
            i += 1;
            continue;
        }
        if c == b'"' {
            in_str = true;
            out.push(c);
            i += 1;
            continue;
        }
        if c == b',' {
            let mut j = i + 1;
            while j < b.len() && b[j].is_ascii_whitespace() {
                j += 1;
            }
            if j < b.len() && (b[j] == b'}' || b[j] == b']') {
                i += 1; // drop the trailing comma
                continue;
            }
        }
        out.push(c);
        i += 1;
    }
    String::from_utf8(out).unwrap_or_default()
}

/// Locate the command JSON: first a ```-fenced block that mentions `"commands"`, else a raw
/// brace-balanced object that does. Returns (json text, body with that span removed).
fn find_command_json(body: &str) -> Option<(String, String)> {
    // Fenced code blocks.
    let mut from = 0;
    while let Some(rel) = body[from..].find("```") {
        let open = from + rel;
        let after = open + 3;
        let Some(nl) = body[after..].find('\n') else {
            break;
        };
        let content_start = after + nl + 1;
        let Some(crel) = body[content_start..].find("```") else {
            break;
        };
        let close = content_start + crel;
        let inner = body[content_start..close].trim();
        if inner.contains("\"commands\"") {
            let rest = format!("{}{}", &body[..open], &body[close + 3..]);
            return Some((inner.to_string(), rest));
        }
        from = close + 3;
    }
    // Raw braced object.
    if let Some(range) = braced_with_commands(body) {
        let json = body[range.clone()].to_string();
        let rest = format!("{}{}", &body[..range.start], &body[range.end..]);
        return Some((json, rest));
    }
    None
}

/// Byte range of the first `{…}` (brace-balanced, string-aware) that contains `"commands"`.
fn braced_with_commands(s: &str) -> Option<Range<usize>> {
    let b = s.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'{' {
            if let Some(end) = balanced_end(b, i) {
                if s[i..=end].contains("\"commands\"") {
                    return Some(i..end + 1);
                }
                i = end + 1;
                continue;
            }
        }
        i += 1;
    }
    None
}

/// Index of the `}` closing the `{` at `start`, honoring quoted strings and escapes.
fn balanced_end(b: &[u8], start: usize) -> Option<usize> {
    let mut depth = 0i32;
    let mut in_str = false;
    let mut esc = false;
    for (i, &c) in b.iter().enumerate().skip(start) {
        if in_str {
            if esc {
                esc = false;
            } else if c == b'\\' {
                esc = true;
            } else if c == b'"' {
                in_str = false;
            }
        } else {
            match c {
                b'"' => in_str = true,
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(i);
                    }
                }
                _ => {}
            }
        }
    }
    None
}

fn parse_op(v: &serde_json::Value) -> Option<RawOp> {
    let op = v
        .get("op")
        .or_else(|| v.get("action"))
        .and_then(|o| o.as_str())?;
    match normalize(op).as_str() {
        "add" | "addblock" | "place" | "create" | "new" | "instance" => {
            let kind_str = v
                .get("kind")
                .or_else(|| v.get("type"))
                .or_else(|| v.get("block"))
                .or_else(|| v.get("chip"))
                .and_then(|k| k.as_str())?;
            // A primitive name, else a chip instance ("chip:Name" or just the chip's name).
            let kind = match kind_from_str(kind_str) {
                Some(prim) => AddKind::Prim(prim),
                None => {
                    let name = kind_str.trim().trim_start_matches("chip:").trim();
                    if name.is_empty() {
                        return None;
                    }
                    AddKind::Chip(name.to_string())
                }
            };
            let (x, y) = xy(v);
            Some(RawOp::Add {
                r#ref: v
                    .get("ref")
                    .or_else(|| v.get("id"))
                    .or_else(|| v.get("name"))
                    .and_then(as_handle),
                kind,
                x,
                y,
                inputs: v
                    .get("inputs")
                    .or_else(|| v.get("num_inputs"))
                    .and_then(|n| n.as_u64())
                    .map(|n| n.min(u16::MAX as u64) as u16),
                label: v.get("label").and_then(|l| l.as_str()).map(str::to_string),
                state: v
                    .get("state")
                    .or_else(|| v.get("value"))
                    .and_then(as_bool)
                    .unwrap_or(false),
                hz: hz_of(v),
                color: v.get("color").and_then(as_color),
            })
        }
        "set" | "setprops" | "props" | "properties" | "configure" | "update" | "modify" => {
            Some(RawOp::Set {
                target: target_handle(v)?,
                inputs: v
                    .get("inputs")
                    .or_else(|| v.get("num_inputs"))
                    .and_then(|n| n.as_u64())
                    .map(|n| n.min(u16::MAX as u64) as u16),
                hz: hz_of(v),
                color: v.get("color").map(|c| {
                    if c.is_null()
                        || c.as_str()
                            .is_some_and(|s| s.trim().is_empty() || s.trim() == "none")
                    {
                        None
                    } else {
                        as_color(c)
                    }
                }),
                state: v.get("state").or_else(|| v.get("value")).and_then(as_bool),
                label: v.get("label").and_then(|l| l.as_str()).map(str::to_string),
            })
        }
        "connect" | "wire" | "addconnection" | "link" | "join" => {
            let from = v
                .get("from")
                .or_else(|| v.get("source"))
                .or_else(|| v.get("output_block"))
                .and_then(as_handle)?;
            let to = v
                .get("to")
                .or_else(|| v.get("target"))
                .or_else(|| v.get("dest"))
                .or_else(|| v.get("input_block"))
                .and_then(as_handle)?;
            Some(RawOp::Connect {
                from,
                from_port: port_sel(
                    v,
                    &[
                        "from_port",
                        "output",
                        "out_port",
                        "output_index",
                        "fromport",
                    ],
                ),
                to,
                to_port: port_sel(
                    v,
                    &[
                        "to_port",
                        "input",
                        "in_port",
                        "port",
                        "input_index",
                        "toport",
                    ],
                ),
            })
        }
        "remove" | "removeblock" | "delete" | "deleteblock" | "erase" => Some(RawOp::Remove {
            target: target_handle(v)?,
        }),
        "move" | "moveblock" | "reposition" => {
            let (x, y) = xy(v);
            Some(RawOp::Move {
                target: target_handle(v)?,
                x,
                y,
            })
        }
        "label" | "setlabel" | "rename" => Some(RawOp::Label {
            target: target_handle(v)?,
            text: v
                .get("text")
                .or_else(|| v.get("label"))
                .or_else(|| v.get("name"))
                .and_then(|t| t.as_str())
                .map(str::to_string),
        }),
        "defchip" | "definechip" | "chip" | "definecomponent" | "defmodule" => {
            let name = v
                .get("name")
                .or_else(|| v.get("chip"))
                .and_then(|n| n.as_str())
                .map(str::trim)
                .filter(|n| !n.is_empty())?
                .to_string();
            let pin_list = |key: &str| -> Vec<String> {
                v.get(key)
                    .and_then(|a| a.as_array())
                    .map(|a| {
                        a.iter()
                            .filter_map(|p| p.as_str())
                            .map(|p| p.trim().to_string())
                            .filter(|p| !p.is_empty())
                            .collect()
                    })
                    .unwrap_or_default()
            };
            let ops = v
                .get("commands")
                .or_else(|| v.get("ops"))
                .or_else(|| v.get("body"))
                .and_then(|a| a.as_array())
                .map(|a| a.iter().filter_map(parse_op).collect())
                .unwrap_or_default();
            Some(RawOp::DefChip {
                name,
                inputs: pin_list("inputs"),
                outputs: pin_list("outputs"),
                ops,
            })
        }
        "test" | "simulate" | "run" | "check" | "verify" => {
            let mut steps: Vec<TestStep> = Vec::new();
            if let Some(arr) = v.get("steps").and_then(|s| s.as_array()) {
                for step in arr {
                    steps.push(test_step(step));
                }
            } else {
                let step = test_step(v);
                if step != TestStep::default() {
                    steps.push(step);
                }
            }
            (!steps.is_empty()).then_some(RawOp::Test { steps })
        }
        "disconnect" | "unwire" | "removewire" | "deletewire" | "removeconnection"
        | "deleteconnection" => {
            let from = v
                .get("from")
                .or_else(|| v.get("source"))
                .and_then(as_handle)?;
            let to = v
                .get("to")
                .or_else(|| v.get("target"))
                .or_else(|| v.get("dest"))
                .and_then(as_handle)?;
            Some(RawOp::Disconnect {
                from,
                from_port: port_sel(v, &["from_port", "output", "out_port", "fromport"]),
                to,
                to_port: port_sel(v, &["to_port", "input", "in_port", "port", "toport"]),
            })
        }
        "inspect" | "probe" | "read" | "query" => {
            let targets = handle_list(v);
            (!targets.is_empty()).then_some(RawOp::Inspect { targets })
        }
        "trace" | "why" => Some(RawOp::Trace {
            target: target_handle(v)?,
        }),
        "highlight" | "point" | "show" | "focus" => {
            let targets = handle_list(v);
            (!targets.is_empty()).then(|| RawOp::Highlight {
                targets,
                note: v
                    .get("note")
                    .or_else(|| v.get("text"))
                    .and_then(|n| n.as_str())
                    .map(str::to_string),
            })
        }
        "deletechip" | "removechip" => Some(RawOp::DeleteChip {
            name: v
                .get("name")
                .or_else(|| v.get("chip"))
                .and_then(|n| n.as_str())
                .map(str::trim)
                .filter(|n| !n.is_empty())?
                .to_string(),
        }),
        _ => None,
    }
}

/// Block handles from `targets` (array or single), else a single `target`.
fn handle_list(v: &serde_json::Value) -> Vec<String> {
    match v.get("targets").or_else(|| v.get("blocks")) {
        Some(serde_json::Value::Array(a)) => a.iter().filter_map(as_handle).collect(),
        Some(one) => as_handle(one).into_iter().collect(),
        None => target_handle(v).into_iter().collect(),
    }
}

/// One test step: `{"set":{…},"pulse":["CLK"],"wait":0.5,"ticks":2}` — or, for backwards
/// compatibility, a bare `{"A":1,"B":0}` map of switch levels.
fn test_step(step: &serde_json::Value) -> TestStep {
    const KEYS: [&str; 6] = ["set", "pulse", "wait", "ticks", "expect", "assert"];
    let structured = step
        .as_object()
        .is_some_and(|o| o.keys().any(|k| KEYS.contains(&k.as_str())));
    if !structured {
        return TestStep {
            set: set_map(step),
            ..Default::default()
        };
    }
    let pulse = match step.get("pulse") {
        Some(serde_json::Value::Array(a)) => a.iter().filter_map(as_handle).collect(),
        Some(v) => as_handle(v).into_iter().collect(),
        None => Vec::new(),
    };
    TestStep {
        set: step.get("set").map(set_map).unwrap_or_default(),
        pulse,
        wait: step
            .get("wait")
            .and_then(|w| w.as_f64())
            .map(|w| w.clamp(0.0, 3600.0)),
        ticks: step
            .get("ticks")
            .or_else(|| step.get("tick"))
            .and_then(|t| t.as_u64())
            .map(|t| t.min(100_000) as u32),
        expect: step
            .get("expect")
            .or_else(|| step.get("assert"))
            .map(set_map)
            .unwrap_or_default(),
    }
}

/// Parse a `{"handle": value}` map of switch settings; values may be bool, 0/1, or "0"/"1".
fn set_map(v: &serde_json::Value) -> Vec<(String, bool)> {
    let Some(obj) = v.as_object() else {
        return Vec::new();
    };
    obj.iter()
        .filter_map(|(k, val)| Some((k.trim().to_string(), as_bool(val)?)))
        .collect()
}

/// A level given as bool, 0/1, or "0"/"1"/"true"/"false".
fn as_bool(val: &serde_json::Value) -> Option<bool> {
    val.as_bool()
        .or_else(|| val.as_u64().map(|n| n != 0))
        .or_else(|| match val.as_str()?.trim() {
            "1" | "true" | "high" | "on" => Some(true),
            "0" | "false" | "low" | "off" => Some(false),
            _ => None,
        })
}

/// A clock frequency from `hz` / `freq` / `frequency`, kept in a sane range.
fn hz_of(v: &serde_json::Value) -> Option<f32> {
    v.get("hz")
        .or_else(|| v.get("freq"))
        .or_else(|| v.get("frequency"))
        .and_then(|h| h.as_f64())
        .filter(|h| h.is_finite() && *h > 0.0)
        .map(|h| h.clamp(0.01, 1000.0) as f32)
}

/// A color as `"#rrggbb"` / `"rrggbb"` or `[r, g, b]`.
fn as_color(v: &serde_json::Value) -> Option<[u8; 3]> {
    if let Some(a) = v.as_array() {
        let c = |i: usize| a.get(i).and_then(|x| x.as_u64()).map(|x| x.min(255) as u8);
        return Some([c(0)?, c(1)?, c(2)?]);
    }
    let s = v.as_str()?.trim().trim_start_matches('#');
    if s.len() != 6 {
        return None;
    }
    let p = |i: usize| u8::from_str_radix(&s[i..i + 2], 16).ok();
    Some([p(0)?, p(2)?, p(4)?])
}

fn target_handle(v: &serde_json::Value) -> Option<String> {
    v.get("target")
        .or_else(|| v.get("id"))
        .or_else(|| v.get("ref"))
        .or_else(|| v.get("block"))
        .and_then(as_handle)
}

/// A block handle can be given as a string ref or a numeric id — normalize both to a string.
fn as_handle(v: &serde_json::Value) -> Option<String> {
    if let Some(s) = v.as_str() {
        let t = s.trim();
        return (!t.is_empty()).then(|| t.to_string());
    }
    v.as_u64().map(|n| n.to_string())
}

/// Read `x`/`y`, or a `pos`/`position` array `[x, y]`. Defaults to (0, 0).
fn xy(v: &serde_json::Value) -> (i32, i32) {
    if let Some(arr) = v
        .get("pos")
        .or_else(|| v.get("position"))
        .and_then(|p| p.as_array())
    {
        let get = |i: usize| arr.get(i).and_then(as_i32).unwrap_or(0);
        return (get(0), get(1));
    }
    (
        v.get("x").and_then(as_i32).unwrap_or(0),
        v.get("y").and_then(as_i32).unwrap_or(0),
    )
}

/// Coordinates are kept within ±100 000 cells: a model emitting an absurd number gets a block
/// far away rather than one wrapped around to a random spot.
fn as_i32(v: &serde_json::Value) -> Option<i32> {
    const LIMIT: i64 = 100_000;
    v.as_i64()
        .or_else(|| {
            v.as_f64()
                .filter(|n| n.is_finite())
                .map(|n| n.round() as i64)
        })
        .map(|n| n.clamp(-LIMIT, LIMIT) as i32)
}

/// Read a port selector from the first present key: a number is an index, a numeric string is
/// an index, any other string is a pin name. Absent → `None` (auto-assign on the input side).
fn port_sel(v: &serde_json::Value, keys: &[&str]) -> Option<PortSel> {
    for k in keys {
        let Some(val) = v.get(*k) else {
            continue;
        };
        if let Some(n) = val.as_u64() {
            return Some(PortSel::Index(n as u16));
        }
        if let Some(s) = val.as_str() {
            let t = s.trim();
            if t.is_empty() {
                continue;
            }
            return Some(match t.parse::<u16>() {
                Ok(n) => PortSel::Index(n),
                Err(_) => PortSel::Name(t.to_string()),
            });
        }
    }
    None
}

/// Lowercase and drop anything but a–z/0–9, so `"Add_Block"`, `"add-block"`, `"AddBlock"` all match.
fn normalize(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .map(|c| c.to_ascii_lowercase())
        .collect()
}

/// Map a model-supplied block name to a [`BlockType`], accepting common synonyms.
pub fn kind_from_str(s: &str) -> Option<BlockType> {
    use BlockType::*;
    Some(match normalize(s).as_str() {
        "and" => And,
        "or" => Or,
        "not" | "inverter" | "inv" => Not,
        "nand" => Nand,
        "nor" => Nor,
        "xor" => Xor,
        "xnor" => Xnor,
        "buffer" | "buf" => Buffer,
        "switch" | "sw" | "toggle" | "input" | "in" => Switch,
        "button" | "btn" | "push" => Button,
        "led" | "lamp" | "output" | "out" | "bulb" | "light" => Led,
        "constanthigh" | "constant1" | "const1" | "high" | "vcc" | "one" | "true" => ConstantHigh,
        "constantlow" | "constant0" | "const0" | "low" | "gnd" | "zero" | "false" => ConstantLow,
        "clock" | "clk" | "oscillator" | "osc" => Clock,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_fenced_commands_and_keeps_prose() {
        let raw = "Sure, here's an AND gate.\n\n```json\n{\"commands\":[\
            {\"op\":\"add\",\"ref\":\"g\",\"kind\":\"and\",\"x\":4,\"y\":1}]}\n```";
        let parsed = parse_reply(raw);
        assert_eq!(parsed.text, "Sure, here's an AND gate.");
        assert_eq!(parsed.ops.len(), 1);
        assert!(matches!(
            parsed.ops[0],
            RawOp::Add {
                kind: AddKind::Prim(BlockType::And),
                x: 4,
                y: 1,
                ..
            }
        ));
    }

    #[test]
    fn parses_raw_json_without_fence() {
        let raw = "{\"commands\":[{\"op\":\"connect\",\"from\":\"a\",\"to\":\"b\",\"to_port\":1}]}";
        let parsed = parse_reply(raw);
        assert_eq!(parsed.ops.len(), 1);
        match &parsed.ops[0] {
            RawOp::Connect {
                from,
                from_port,
                to,
                to_port,
            } => {
                assert_eq!(from, "a");
                assert_eq!(*from_port, None, "absent from_port stays unset");
                assert_eq!(to, "b");
                assert_eq!(*to_port, Some(PortSel::Index(1)));
            }
            _ => panic!("expected connect"),
        }
    }

    #[test]
    fn parses_defchip_with_inner_ops_and_chip_instance() {
        let raw = r#"{"commands":[
            {"op":"defchip","name":"And2","inputs":["a","b"],"outputs":["y"],"commands":[
                {"op":"add","ref":"g","kind":"and","x":6,"y":0},
                {"op":"connect","from":"a","to":"g"},
                {"op":"connect","from":"b","to":"g"},
                {"op":"connect","from":"g","to":"y"}
            ]},
            {"op":"add","ref":"u1","kind":"And2","x":6,"y":0},
            {"op":"connect","from":"u1","from_port":"y","to":"led1","to_port":"0"}
        ]}"#;
        let parsed = parse_reply(raw);
        assert_eq!(parsed.ops.len(), 3);
        match &parsed.ops[0] {
            RawOp::DefChip {
                name,
                inputs,
                outputs,
                ops,
            } => {
                assert_eq!(name, "And2");
                assert_eq!(inputs, &["a", "b"]);
                assert_eq!(outputs, &["y"]);
                assert_eq!(ops.len(), 4);
            }
            _ => panic!("expected defchip"),
        }
        assert!(matches!(
            &parsed.ops[1],
            RawOp::Add { kind: AddKind::Chip(n), .. } if n == "And2"
        ));
        match &parsed.ops[2] {
            RawOp::Connect {
                from_port, to_port, ..
            } => {
                assert_eq!(*from_port, Some(PortSel::Name("y".to_string())));
                // A numeric string is an index, not a name.
                assert_eq!(*to_port, Some(PortSel::Index(0)));
            }
            _ => panic!("expected connect"),
        }
    }

    #[test]
    fn parses_test_op_steps() {
        let raw = r#"{"commands":[{"op":"test","steps":[
            {"set":{"EN":1,"D":true}},
            {"D":0}
        ]}]}"#;
        let parsed = parse_reply(raw);
        match &parsed.ops[0] {
            RawOp::Test { steps } => {
                assert_eq!(steps.len(), 2);
                assert!(steps[0].set.contains(&("EN".to_string(), true)));
                assert!(steps[0].set.contains(&("D".to_string(), true)));
                assert_eq!(steps[1].set, vec![("D".to_string(), false)]);
            }
            _ => panic!("expected test"),
        }
    }

    #[test]
    fn extracts_reasoning() {
        let raw = "<think>I should add a switch.</think>Here you go.";
        let parsed = parse_reply(raw);
        assert_eq!(parsed.thinking.as_deref(), Some("I should add a switch."));
        assert_eq!(parsed.text, "Here you go.");
    }

    #[test]
    fn reasoning_with_commands_together() {
        let raw = "<thinking>plan</thinking>Done.\n```json\n{\"commands\":[\
            {\"op\":\"add\",\"kind\":\"led\",\"x\":0,\"y\":0}]}\n```";
        let parsed = parse_reply(raw);
        assert_eq!(parsed.thinking.as_deref(), Some("plan"));
        assert_eq!(parsed.text, "Done.");
        assert_eq!(parsed.ops.len(), 1);
    }

    #[test]
    fn numeric_handles_and_synonyms() {
        let raw = "{\"commands\":[{\"op\":\"wire\",\"from\":3,\"to\":5}]}";
        let parsed = parse_reply(raw);
        match &parsed.ops[0] {
            RawOp::Connect { from, to, .. } => {
                assert_eq!(from, "3");
                assert_eq!(to, "5");
            }
            _ => panic!("expected connect"),
        }
        assert_eq!(kind_from_str("Switch"), Some(BlockType::Switch));
        assert_eq!(kind_from_str("inverter"), Some(BlockType::Not));
        assert_eq!(kind_from_str("output"), Some(BlockType::Led));
        assert_eq!(kind_from_str("nonsense"), None);
    }

    #[test]
    fn no_commands_is_plain_chat() {
        let parsed = parse_reply("An AND gate outputs 1 only when both inputs are 1.");
        assert!(parsed.ops.is_empty());
        assert!(parsed.thinking.is_none());
        assert_eq!(
            parsed.text,
            "An AND gate outputs 1 only when both inputs are 1."
        );
    }

    #[test]
    fn live_split_separate_reasoning_field() {
        let v = live_split("thinking about it", "Here is the plan");
        assert_eq!(v.reasoning.as_deref(), Some("thinking about it"));
        assert_eq!(v.answer, "Here is the plan");
        assert_eq!(v.commands, None);
    }

    #[test]
    fn live_split_inline_open_think() {
        // Mid-stream: the think tag has opened but not closed yet.
        let v = live_split("", "<think>still reasoning");
        assert_eq!(v.reasoning.as_deref(), Some("still reasoning"));
        assert_eq!(v.answer, "");
    }

    #[test]
    fn live_split_exposes_partial_command_json() {
        let v = live_split("", "Adding a gate.\n```json\n{\"commands\":[{\"op\":");
        assert_eq!(v.answer, "Adding a gate.", "prose stops before the fence");
        assert_eq!(
            v.commands.as_deref(),
            Some("{\"commands\":[{\"op\":"),
            "the in-progress JSON is exposed for the live 'Writing commands' view"
        );
    }

    #[test]
    fn tolerates_json_comments_and_trailing_commas() {
        // The exact shape weak models produce: // comments and a trailing comma.
        let raw = "Let's build it.\n```json\n{\n  \"commands\": [\n    // Enable switch\n    \
                   {\"op\":\"add\",\"ref\":\"EN\",\"kind\":\"switch\",\"x\":0,\"y\":70},\n\n    \
                   // Bit 0\n    {\"op\":\"add\",\"ref\":\"D0\",\"kind\":\"switch\",\"x\":0,\"y\":0},\n  ]\n}\n```";
        let parsed = parse_reply(raw);
        assert_eq!(parsed.ops.len(), 2, "comments + trailing comma still parse");
        assert_eq!(parsed.text, "Let's build it.");
    }

    #[test]
    fn recovers_ops_from_truncated_array() {
        // Reply cut off mid-stream: the array never closes, but the complete objects survive.
        let raw = "```json\n{\"commands\":[\
                   {\"op\":\"add\",\"ref\":\"a\",\"kind\":\"and\",\"x\":0,\"y\":0},\
                   {\"op\":\"connect\",\"from\":\"a\",\"to\":\"b\"},\
                   {\"op\":\"add\",\"ref\":\"b\",\"ki";
        let parsed = parse_reply(raw);
        assert_eq!(parsed.ops.len(), 2, "the two complete ops are recovered");
    }

    #[test]
    fn comment_stripper_leaves_urls_in_strings_intact() {
        // `//` inside a JSON string value must NOT be treated as a comment.
        let raw =
            "{\"commands\":[{\"op\":\"label\",\"target\":\"1\",\"text\":\"see http://x/y\"}]}";
        let parsed = parse_reply(raw);
        match &parsed.ops[0] {
            RawOp::Label { text, .. } => assert_eq!(text.as_deref(), Some("see http://x/y")),
            _ => panic!("expected label"),
        }
    }

    #[test]
    fn parses_set_op_and_add_properties() {
        let raw = r##"{"commands":[
            {"op":"add","ref":"c","kind":"clock","x":0,"y":0,"hz":4,"color":"#ff8800"},
            {"op":"set","target":"c","hz":2.5,"color":null},
            {"op":"set","target":"g1","inputs":4,"state":"1","label":"AND4"}
        ]}"##;
        let ops = parse_reply(raw).ops;
        assert!(
            matches!(&ops[0], RawOp::Add { hz: Some(h), color: Some([255, 136, 0]), .. } if *h == 4.0)
        );
        assert!(matches!(&ops[1], RawOp::Set { hz: Some(h), color: Some(None), .. } if *h == 2.5));
        match &ops[2] {
            RawOp::Set {
                target,
                inputs,
                state,
                label,
                color,
                ..
            } => {
                assert_eq!(target, "g1");
                assert_eq!(*inputs, Some(4));
                assert_eq!(*state, Some(true));
                assert_eq!(label.as_deref(), Some("AND4"));
                assert_eq!(*color, None, "absent color is left alone");
            }
            other => panic!("expected set, got {other:?}"),
        }
    }

    #[test]
    fn parses_pulse_wait_and_tick_test_steps() {
        let raw = r#"{"commands":[{"op":"test","steps":[
            {"set":{"D":1}},
            {"pulse":"CLK"},
            {"set":{"EN":0},"pulse":["CLK","LOAD"],"wait":0.25},
            {"ticks":4}
        ]}]}"#;
        let ops = parse_reply(raw).ops;
        let RawOp::Test { steps } = &ops[0] else {
            panic!("expected test");
        };
        assert_eq!(steps[0].set, vec![("D".to_string(), true)]);
        assert_eq!(steps[1].pulse, vec!["CLK".to_string()]);
        assert_eq!(steps[2].pulse.len(), 2);
        assert_eq!(steps[2].wait, Some(0.25));
        assert_eq!(steps[3].ticks, Some(4));
    }

    #[test]
    fn absurd_coordinates_are_clamped_not_wrapped() {
        let raw = r#"{"commands":[{"op":"add","kind":"and","x":99999999999,"y":-5e12}]}"#;
        let ops = parse_reply(raw).ops;
        assert!(matches!(
            ops[0],
            RawOp::Add {
                x: 100_000,
                y: -100_000,
                ..
            }
        ));
    }

    #[test]
    fn unreadable_commands_are_reported_not_dropped_silently() {
        let r = parse_reply(
            "ok\n```json\n{\"commands\":[{\"op\":\"add\",\"kind\":\"and\"},{\"op\":\"nand3\"},{\"kind\":\"or\"}]}\n```",
        );
        assert_eq!(r.ops.len(), 1);
        assert!(
            r.problems[0].contains("unknown op \"nand3\""),
            "{:?}",
            r.problems
        );
        assert!(
            r.problems[1].contains("no \"op\" field"),
            "{:?}",
            r.problems
        );
        let broken =
            parse_reply("```json\n{\"commands\": [ {\"op\": \"add\", \"kind\" \"and\"} ]}\n```");
        assert!(broken.ops.is_empty());
        assert!(
            broken.problems[0].contains("could not be read as JSON"),
            "{:?}",
            broken.problems
        );
    }

    #[test]
    fn new_ops_parse() {
        let r = parse_reply(
            r#"{"commands":[{"op":"disconnect","from":"a","to":"b","to_port":1},
              {"op":"inspect","targets":["a",3]},{"op":"trace","target":"Y"},
              {"op":"highlight","targets":"g","note":"n"},{"op":"delete_chip","name":"X"},
              {"op":"test","steps":[{"set":{"A":1},"expect":{"Y":0}}]}]}"#,
        );
        assert!(r.problems.is_empty(), "{:?}", r.problems);
        assert_eq!(r.ops.len(), 6);
        assert!(matches!(&r.ops[1], RawOp::Inspect { targets } if targets == &["a", "3"]));
        match &r.ops[5] {
            RawOp::Test { steps } => assert_eq!(steps[0].expect, vec![("Y".to_string(), false)]),
            other => panic!("{other:?}"),
        }
    }
}
