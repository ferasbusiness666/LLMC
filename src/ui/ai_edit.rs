//! Turning an assistant reply into circuit edits.
//!
//! Providers answer in free text, so this module does the tolerant parsing: it pulls any
//! `<think>` reasoning out for separate display, and extracts a `{"commands":[…]}` block (in a
//! ```json fence or raw) into a list of loosely-typed [`RawOp`]s. Applying those ops to a real
//! circuit — allocating ids, resolving refs, validating ports — happens in the app layer, which
//! owns the circuit; keeping the parsing here (egui-free) lets it be unit-tested.

use std::ops::Range;

use llmc::model::BlockType;

/// One edit the model asked for, still in "handle" form (refs/ids as strings) and unvalidated.
/// The app resolves handles to real block ids and checks ports before applying.
#[derive(Debug, Clone, PartialEq)]
pub enum RawOp {
    Add {
        r#ref: Option<String>,
        kind: BlockType,
        x: i32,
        y: i32,
        inputs: Option<u16>,
        label: Option<String>,
        state: bool,
    },
    Connect {
        from: String,
        from_port: u16,
        to: String,
        to_port: u16,
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
}

/// A parsed assistant reply: the reasoning (if any), the prose to show, and edit ops.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct ParsedReply {
    pub thinking: Option<String>,
    pub text: String,
    pub ops: Vec<RawOp>,
}

/// Split reasoning, strip the command block, and parse it into ops.
pub fn parse_reply(raw: &str) -> ParsedReply {
    let (thinking, body) = split_reasoning(raw);
    let (ops, text) = extract_ops(&body);
    ParsedReply {
        thinking,
        text,
        ops,
    }
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

/// Extract a `{"commands":[…]}` block from `body`, returning (ops, body-without-the-block).
fn extract_ops(body: &str) -> (Vec<RawOp>, String) {
    if let Some((json, rest)) = find_command_json(body) {
        if let Ok(val) = serde_json::from_str::<serde_json::Value>(&json) {
            if let Some(arr) = val.get("commands").and_then(|c| c.as_array()) {
                let ops: Vec<RawOp> = arr.iter().filter_map(parse_op).collect();
                if !ops.is_empty() {
                    return (ops, rest.trim().to_string());
                }
            }
        }
    }
    (Vec::new(), body.trim().to_string())
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
        "add" | "addblock" | "place" | "create" | "new" => {
            let kind = kind_from_str(
                v.get("kind")
                    .or_else(|| v.get("type"))
                    .or_else(|| v.get("block"))
                    .and_then(|k| k.as_str())?,
            )?;
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
                    .map(|n| n as u16),
                label: v.get("label").and_then(|l| l.as_str()).map(str::to_string),
                state: v
                    .get("state")
                    .or_else(|| v.get("value"))
                    .and_then(|s| s.as_bool())
                    .unwrap_or(false),
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
                from_port: port(
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
                to_port: port(
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
        "label" | "setlabel" | "rename" | "name" => Some(RawOp::Label {
            target: target_handle(v)?,
            text: v
                .get("text")
                .or_else(|| v.get("label"))
                .or_else(|| v.get("name"))
                .and_then(|t| t.as_str())
                .map(str::to_string),
        }),
        _ => None,
    }
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

fn as_i32(v: &serde_json::Value) -> Option<i32> {
    v.as_i64()
        .map(|n| n as i32)
        .or_else(|| v.as_f64().map(|n| n.round() as i32))
}

fn port(v: &serde_json::Value, keys: &[&str]) -> u16 {
    for k in keys {
        if let Some(n) = v.get(*k).and_then(|n| n.as_u64()) {
            return n as u16;
        }
    }
    0
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
                kind: BlockType::And,
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
                assert_eq!(*from_port, 0);
                assert_eq!(to, "b");
                assert_eq!(*to_port, 1);
            }
            _ => panic!("expected connect"),
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
}
