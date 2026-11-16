//! The Properties inspector (right sidebar): what's selected, its live signals, and every
//! editable property. Each committed change is one undo step (slider drags and typing collapse
//! into a single step per series).

use eframe::egui::{
    self, Align, Align2, Color32, CornerRadius, Layout, RichText, Sense, Stroke, Ui,
};

use llmc::backend::GroupTransform;
use llmc::model::{BlockType, ChipId, Port, MAX_GATE_INPUTS};

use super::design::{self, Kind, Theme};
use super::document::{Document, FrameRequest};
use super::glyphs;
use super::icons;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum InspectorAction {
    EditChip(ChipId),
    CreateChip { from_selection: bool },
}

/// Preset body colors (BoardUI-style chart palette, readable in both themes).
const SWATCHES: [[u8; 3]; 8] = [
    [0x33, 0x92, 0xff],
    [0x2d, 0xd4, 0xbf],
    [0xa3, 0xe6, 0x35],
    [0xfa, 0xcc, 0x15],
    [0xfb, 0x92, 0x3c],
    [0xf4, 0x72, 0xb6],
    [0xa7, 0x8b, 0xfa],
    [0x94, 0xa3, 0xb8],
];

pub fn inspector(ui: &mut Ui, doc: &mut Document, t: &Theme) -> Vec<InspectorAction> {
    let mut actions = Vec::new();
    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            ui.add_space(4.0);
            let nblocks = doc.selection.len();
            let nwires = doc.selected_conns.len();
            if nblocks == 1 && nwires == 0 {
                let id = *doc.selection.iter().next().unwrap();
                single_block(ui, doc, t, id, &mut actions);
            } else if nblocks > 0 || nwires > 0 {
                multi(ui, doc, t, &mut actions);
            } else {
                overview(ui, doc, t);
            }
            ui.add_space(16.0);
        });
    actions
}

fn heading(ui: &mut Ui, t: &Theme, text: &str) {
    ui.add_space(14.0);
    design::section_label(ui, t, text);
    ui.add_space(4.0);
}

fn stat(ui: &mut Ui, t: &Theme, label: &str, value: String) {
    ui.horizontal(|ui| {
        ui.label(
            RichText::new(label)
                .font(design::regular(12.5))
                .color(t.text_muted),
        );
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            ui.label(
                RichText::new(value)
                    .font(design::medium(12.5))
                    .color(t.text),
            );
        });
    });
}

fn overview(ui: &mut Ui, doc: &mut Document, t: &Theme) {
    design::card(t).show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.horizontal(|ui| {
            let (r, _) = ui.allocate_exact_size(egui::vec2(32.0, 32.0), Sense::hover());
            ui.painter()
                .rect_filled(r, CornerRadius::same(design::radius::MD), t.accent_soft);
            design::paint_icon(
                ui,
                r,
                if doc.chip_edit.is_some() {
                    icons::CPU
                } else {
                    icons::FILE
                },
                16.0,
                t.accent,
            );
            ui.vertical(|ui| {
                ui.spacing_mut().item_spacing.y = 1.0;
                ui.label(
                    RichText::new(&doc.title)
                        .font(design::semibold(14.0))
                        .color(t.text),
                );
                let sub = match &doc.path {
                    Some(p) => p.display().to_string(),
                    None if doc.chip_edit.is_some() => "Chip editor".to_string(),
                    None => "Not saved yet".to_string(),
                };
                ui.label(
                    RichText::new(sub)
                        .font(design::regular(11.5))
                        .color(t.text_faint),
                );
            });
        });
        ui.add_space(10.0);
        let c = &doc.manager.circuit;
        stat(ui, t, "Blocks", c.blocks.len().to_string());
        stat(ui, t, "Wires", c.connections.len().to_string());
        stat(
            ui,
            t,
            "Chips in library",
            doc.manager.chips.len().to_string(),
        );
        stat(
            ui,
            t,
            "Primitive gates (flattened)",
            doc.sim.gate_count().to_string(),
        );
    });

    heading(ui, t, "Simulation");
    design::card(t).show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.horizontal(|ui| {
            let (color, text) = sim_state(doc, t);
            design::pill(ui, t, color, text);
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                ui.label(
                    RichText::new(format!("t = {:.3} s", doc.sim.time()))
                        .font(design::mono(12.0))
                        .color(t.text_muted),
                );
            });
        });
        ui.add_space(8.0);
        ui.horizontal(|ui| {
            design::field_label(ui, t, "Speed");
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                speed_menu(ui, doc, t);
            });
        });
    });

    heading(ui, t, "Getting around");
    for (keys, what) in [
        ("Click", "select \u{b7} toggle a switch"),
        ("Drag port", "draw a wire"),
        ("Space+Drag", "pan the view"),
        ("Scroll", "zoom around the cursor"),
        ("Home", "fit everything"),
        ("R / F", "rotate / flip selection"),
    ] {
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 4.0;
            for k in keys.split(" / ") {
                design::kbd(ui, t, k);
            }
            ui.add_space(4.0);
            ui.label(
                RichText::new(what)
                    .font(design::regular(12.0))
                    .color(t.text_muted),
            );
        });
    }
}

pub fn sim_state(doc: &Document, t: &Theme) -> (Color32, &'static str) {
    if doc.sim.has_recursive_chips() {
        (t.danger, "Recursive chip")
    } else if doc.running && doc.sim.is_unstable() {
        (t.warning, "Oscillating")
    } else if doc.running {
        (t.success, "Running")
    } else {
        (t.text_faint, "Paused")
    }
}

pub fn speed_menu(ui: &mut Ui, doc: &mut Document, t: &Theme) {
    let label = format_speed(doc.speed);
    let btn = ui.add(
        egui::Button::new(
            RichText::new(format!("{label}  {}", icons::ARROW_DOWN_S)).font(design::medium(12.5)),
        )
        .min_size(egui::vec2(64.0, 26.0)),
    );
    egui::Popup::menu(&btn).show(|ui| {
        ui.set_min_width(140.0);
        for s in [0.1f32, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0] {
            if design::menu_item_ex(
                ui,
                t,
                "",
                &format_speed(s),
                "",
                true,
                (doc.speed - s).abs() < 1e-4,
            ) {
                doc.speed = s;
            }
        }
    });
}

pub fn format_speed(s: f32) -> String {
    if s >= 1.0 {
        format!("{s:.0}\u{d7}")
    } else {
        format!("{s}\u{d7}")
    }
}

fn single_block(
    ui: &mut Ui,
    doc: &mut Document,
    t: &Theme,
    id: llmc::model::BlockId,
    actions: &mut Vec<InspectorAction>,
) {
    let Some(block) = doc.manager.circuit.block(id).cloned() else {
        return;
    };
    let type_name = match block.ty {
        BlockType::Chip(cid) => doc
            .manager
            .chips
            .get(cid)
            .map(|d| d.name.clone())
            .unwrap_or_else(|| "Missing chip".into()),
        ty => ty.display_name().to_string(),
    };

    // Header: thumbnail tile, type, id.
    ui.horizontal(|ui| {
        let (r, _) = ui.allocate_exact_size(egui::vec2(44.0, 40.0), Sense::hover());
        ui.painter()
            .rect_filled(r, CornerRadius::same(design::radius::LG), t.bg_canvas);
        ui.painter().rect_stroke(
            r,
            CornerRadius::same(design::radius::LG),
            Stroke::new(1.0, t.border),
            egui::StrokeKind::Inside,
        );
        glyphs::draw_thumbnail(
            ui.painter(),
            r.shrink(6.0),
            block.ty,
            llmc::io::GateStyle::Shapes,
            t,
            false,
        );
        ui.vertical(|ui| {
            ui.spacing_mut().item_spacing.y = 1.0;
            ui.label(
                RichText::new(&type_name)
                    .font(design::semibold(15.0))
                    .color(t.text),
            );
            let kind = match block.ty {
                BlockType::Chip(_) => "Chip instance".to_string(),
                ty if ty.is_gate() => "Logic gate".to_string(),
                _ => "Input / output".to_string(),
            };
            ui.label(
                RichText::new(format!("{kind} \u{b7} #{}", block.id))
                    .font(design::regular(11.5))
                    .color(t.text_faint),
            );
        });
    });

    heading(ui, t, "Name");
    let mut name = block.label.clone().unwrap_or_default();
    let resp = design::text_field(
        ui,
        t,
        &mut name,
        &format!("{type_name} (no name)"),
        Some(icons::HASHTAG),
    );
    if resp.changed() {
        let mut b = block.clone();
        let n = name.trim();
        b.label = (!n.is_empty()).then(|| name.clone());
        doc.update_block(b);
    }
    if resp.lost_focus() {
        doc.new_series();
    }

    // Live signals.
    heading(ui, t, "Signals");
    design::card(t)
        .inner_margin(egui::Margin::same(10))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            signals(ui, doc, t, &block);
        });

    // Type-specific properties.
    let mut b = doc
        .manager
        .circuit
        .block(id)
        .cloned()
        .unwrap_or(block.clone());
    if b.ty.variable_inputs() {
        heading(ui, t, "Inputs");
        ui.horizontal(|ui| {
            let n = b.gate_inputs();
            if design::icon_button_sized(ui, t, icons::SUBTRACT, "Fewer inputs", false, 28.0, n > 2)
                .clicked()
            {
                b.inputs = Some((n - 1) as u16);
                doc.new_series();
                doc.update_block(b.clone());
                doc.new_series();
            }
            ui.label(
                RichText::new(format!("{n}"))
                    .font(design::semibold(15.0))
                    .color(t.text),
            );
            if design::icon_button_sized(
                ui,
                t,
                icons::ADD,
                "More inputs",
                false,
                28.0,
                n < MAX_GATE_INPUTS,
            )
            .clicked()
            {
                b.inputs = Some((n + 1) as u16);
                doc.new_series();
                doc.update_block(b.clone());
                doc.new_series();
            }
        });
        design::hint(ui, t, "Removing inputs disconnects their wires.");
    }
    if b.ty == BlockType::Clock {
        heading(ui, t, "Frequency");
        ui.horizontal(|ui| {
            let mut hz = b.freq_hz.unwrap_or(1.0);
            let r = ui.add(
                egui::Slider::new(&mut hz, 0.1..=200.0)
                    .logarithmic(true)
                    .suffix(" Hz")
                    .max_decimals(2),
            );
            if r.changed() {
                b.freq_hz = Some(hz);
                doc.update_block(b.clone());
            }
            if r.drag_stopped() || r.lost_focus() {
                doc.new_series();
            }
        });
        ui.horizontal_wrapped(|ui| {
            for hz in [0.5f32, 1.0, 2.0, 5.0, 10.0, 50.0] {
                if design::button_ex(
                    ui,
                    t,
                    Kind::Secondary,
                    None,
                    &format!("{hz} Hz"),
                    true,
                    24.0,
                )
                .clicked()
                {
                    b.freq_hz = Some(hz);
                    doc.new_series();
                    doc.update_block(b.clone());
                    doc.new_series();
                }
            }
        });
    }
    if matches!(b.ty, BlockType::Switch | BlockType::Button) {
        heading(ui, t, "Behavior");
        let mut on = b.state;
        let label = if b.ty == BlockType::Switch {
            "Starts on"
        } else {
            "Default level is 1"
        };
        if design::toggle_row(ui, t, label, "Level used when the project opens", &mut on) {
            b.state = on;
            doc.new_series();
            doc.update_block(b.clone());
            doc.set_input(b.id, on, false);
            doc.new_series();
        }
    }
    if let BlockType::Chip(cid) = b.ty {
        heading(ui, t, "Chip");
        design::card(t)
            .inner_margin(egui::Margin::same(10))
            .show(ui, |ui| {
                ui.set_width(ui.available_width());
                let n = doc.chip_instances(cid);
                stat(ui, t, "Placed here", format!("{n}\u{d7}"));
                if let Some(d) = doc.manager.chips.get(cid) {
                    stat(
                        ui,
                        t,
                        "Pins",
                        format!(
                            "{} in \u{b7} {} out",
                            d.input_blocks.len(),
                            d.output_blocks.len()
                        ),
                    );
                    stat(ui, t, "Blocks inside", d.circuit.blocks.len().to_string());
                }
                ui.add_space(8.0);
                if design::button(ui, t, Kind::Primary, Some(icons::PENCIL), "Edit chip").clicked()
                {
                    actions.push(InspectorAction::EditChip(cid));
                }
            });
    }

    heading(ui, t, "Appearance");
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing = egui::vec2(6.0, 6.0);
        // "None" swatch.
        let none_sel = b.color.is_none();
        if swatch(ui, t, None, none_sel)
            .on_hover_text("Default")
            .clicked()
            && !none_sel
        {
            b.color = None;
            doc.new_series();
            doc.update_block(b.clone());
            doc.new_series();
        }
        for c in SWATCHES {
            let sel = b.color == Some(c);
            if swatch(ui, t, Some(c), sel).clicked() && !sel {
                b.color = Some(c);
                doc.new_series();
                doc.update_block(b.clone());
                doc.new_series();
            }
        }
        let mut custom = b.color.unwrap_or([0x33, 0x92, 0xff]);
        let r = ui
            .color_edit_button_srgb(&mut custom)
            .on_hover_text("Custom color");
        if r.changed() {
            b.color = Some(custom);
            doc.update_block(b.clone());
        }
    });

    heading(ui, t, "Transform");
    ui.horizontal(|ui| {
        design::field_label(ui, t, "X");
        let mut x = b.pos.x;
        let rx = ui.add(egui::DragValue::new(&mut x).speed(0.2));
        design::field_label(ui, t, "Y");
        let mut y = b.pos.y;
        let ry = ui.add(egui::DragValue::new(&mut y).speed(0.2));
        if rx.changed() || ry.changed() {
            b.pos = llmc::model::Pos::new(x, y);
            doc.update_block(b.clone());
        }
        if rx.drag_stopped() || ry.drag_stopped() || rx.lost_focus() || ry.lost_focus() {
            doc.new_series();
        }
    });
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        if design::button_ex(
            ui,
            t,
            Kind::Secondary,
            Some(icons::ANTICLOCKWISE_2),
            "",
            true,
            30.0,
        )
        .on_hover_text("Rotate left (Shift+R)")
        .clicked()
        {
            doc.transform_selection(GroupTransform::RotateCcw);
        }
        if design::button_ex(
            ui,
            t,
            Kind::Secondary,
            Some(icons::CLOCKWISE_2),
            "",
            true,
            30.0,
        )
        .on_hover_text("Rotate right (R)")
        .clicked()
        {
            doc.transform_selection(GroupTransform::RotateCw);
        }
        if design::button_ex(
            ui,
            t,
            Kind::Secondary,
            Some(icons::FLIP_HORIZONTAL),
            "",
            true,
            30.0,
        )
        .on_hover_text("Flip (F)")
        .clicked()
        {
            doc.transform_selection(GroupTransform::Flip);
        }
        if design::button_ex(ui, t, Kind::Secondary, Some(icons::FOCUS_2), "", true, 30.0)
            .on_hover_text("Zoom to it (Shift+2)")
            .clicked()
        {
            doc.frame_request = Some(FrameRequest::Selection);
        }
    });

    ui.add_space(18.0);
    if design::button(
        ui,
        t,
        Kind::Ghost,
        Some(icons::DELETE_BIN_6),
        "Delete block",
    )
    .clicked()
    {
        doc.delete_selection();
    }
}

/// Live input/output levels of one block.
fn signals(ui: &mut Ui, doc: &mut Document, t: &Theme, b: &llmc::model::Block) {
    let nin = b.input_count(&doc.manager.chips);
    let nout = b.output_count(&doc.manager.chips);
    let names = |input: bool, i: usize| -> String {
        match b.ty {
            BlockType::Chip(cid) => doc
                .manager
                .chips
                .get(cid)
                .map(|d| {
                    if input {
                        d.input_name(i)
                    } else {
                        d.output_name(i)
                    }
                    .to_string()
                })
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| format!("{}{i}", if input { "in" } else { "out" })),
            _ => format!("{}{i}", if input { "in" } else { "out" }),
        }
    };
    if !doc.running && doc.sim.time() == 0.0 {
        design::hint(ui, t, "Paused \u{2014} press Run (F5) to see live values.");
        ui.add_space(4.0);
    }
    if matches!(b.ty, BlockType::Switch | BlockType::Button) {
        let level = doc.sim.switch_value(b.id);
        ui.horizontal(|ui| {
            level_chip(ui, t, "level", level);
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                if b.ty == BlockType::Switch && design::toggle(ui, t, level).clicked() {
                    doc.set_input(b.id, !level, true);
                }
            });
        });
        return;
    }
    if b.ty == BlockType::Led {
        level_chip(
            ui,
            t,
            if doc.sim.led_value(b.id).unwrap_or(false) {
                "lit"
            } else {
                "off"
            },
            doc.sim.led_value(b.id).unwrap_or(false),
        );
        return;
    }
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing = egui::vec2(4.0, 4.0);
        for i in 0..nin {
            let port = Port::input(b.id, i as u16);
            let high = doc
                .manager
                .circuit
                .connections_into(port)
                .any(|c| doc.sim.output_value(c.from).unwrap_or(false));
            level_chip(ui, t, &names(true, i), high);
        }
        if nin > 0 && nout > 0 {
            ui.label(RichText::new("\u{2192}").color(t.text_faint));
        }
        for i in 0..nout {
            let high = doc
                .sim
                .output_value(Port::output(b.id, i as u16))
                .unwrap_or(false);
            level_chip(ui, t, &names(false, i), high);
        }
    });
}

fn level_chip(ui: &mut Ui, t: &Theme, name: &str, high: bool) {
    let font = design::medium(11.5);
    let text = format!("{name}  {}", if high { 1 } else { 0 });
    let galley = ui
        .painter()
        .layout_no_wrap(text.clone(), font.clone(), t.text);
    let (r, _) = ui.allocate_exact_size(egui::vec2(galley.size().x + 14.0, 22.0), Sense::hover());
    let color = if high { t.wire_high } else { t.text_faint };
    ui.painter()
        .rect_filled(r, CornerRadius::same(6), color.gamma_multiply(0.14));
    ui.painter().rect_stroke(
        r,
        CornerRadius::same(6),
        Stroke::new(1.0, color.gamma_multiply(0.45)),
        egui::StrokeKind::Inside,
    );
    ui.painter().text(
        r.center(),
        Align2::CENTER_CENTER,
        text,
        font,
        if high { t.text } else { t.text_muted },
    );
}

fn swatch(ui: &mut Ui, t: &Theme, color: Option<[u8; 3]>, selected: bool) -> egui::Response {
    let (r, resp) = ui.allocate_exact_size(egui::vec2(22.0, 22.0), Sense::click());
    let p = ui.painter();
    match color {
        Some([r_, g, b]) => {
            p.circle_filled(r.center(), 9.0, Color32::from_rgb(r_, g, b));
        }
        None => {
            p.circle_filled(r.center(), 9.0, t.surface);
            p.circle_stroke(r.center(), 8.5, Stroke::new(1.0, t.border_strong));
            p.line_segment(
                [
                    r.center() + egui::vec2(-5.5, 5.5),
                    r.center() + egui::vec2(5.5, -5.5),
                ],
                Stroke::new(1.2, t.danger),
            );
        }
    }
    if selected {
        p.circle_stroke(r.center(), 11.0, Stroke::new(2.0, t.accent));
    } else if resp.hovered() {
        p.circle_stroke(r.center(), 11.0, Stroke::new(1.0, t.border_strong));
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }
    resp
}

fn multi(ui: &mut Ui, doc: &mut Document, t: &Theme, actions: &mut Vec<InspectorAction>) {
    let nb = doc.selection.len();
    let nw = doc.selected_conns.len();
    ui.horizontal(|ui| {
        let (r, _) = ui.allocate_exact_size(egui::vec2(36.0, 36.0), Sense::hover());
        ui.painter()
            .rect_filled(r, CornerRadius::same(design::radius::LG), t.accent_soft);
        design::paint_icon(
            ui,
            r,
            if nb == 0 { icons::ROUTE } else { icons::STACK },
            17.0,
            t.accent,
        );
        ui.vertical(|ui| {
            ui.spacing_mut().item_spacing.y = 1.0;
            let title = match (nb, nw) {
                (0, w) => design::plural(w, "wire", "wires"),
                (b, 0) => design::plural(b, "block", "blocks"),
                (b, w) => format!(
                    "{}, {}",
                    design::plural(b, "block", "blocks"),
                    design::plural(w, "wire", "wires")
                ),
            };
            ui.label(
                RichText::new(title)
                    .font(design::semibold(15.0))
                    .color(t.text),
            );
            ui.label(
                RichText::new("Selection")
                    .font(design::regular(11.5))
                    .color(t.text_faint),
            );
        });
    });

    if nb == 0 && nw == 1 {
        let cid = *doc.selected_conns.iter().next().unwrap();
        if let Some(c) = doc.manager.circuit.connection(cid).copied() {
            heading(ui, t, "Wire");
            design::card(t)
                .inner_margin(egui::Margin::same(10))
                .show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    let name = |id: llmc::model::BlockId| {
                        doc.manager
                            .circuit
                            .block(id)
                            .map(|b| {
                                b.label
                                    .clone()
                                    .unwrap_or_else(|| format!("{} #{}", b.ty.display_name(), b.id))
                            })
                            .unwrap_or_default()
                    };
                    stat(
                        ui,
                        t,
                        "From",
                        format!("{} \u{b7} out {}", name(c.from.block), c.from.index),
                    );
                    stat(
                        ui,
                        t,
                        "To",
                        format!("{} \u{b7} in {}", name(c.to.block), c.to.index),
                    );
                    ui.add_space(4.0);
                    level_chip(
                        ui,
                        t,
                        "signal",
                        doc.sim.output_value(c.from).unwrap_or(false),
                    );
                });
        }
    }

    if nb > 0 {
        heading(ui, t, "Contents");
        let mut counts: std::collections::BTreeMap<String, usize> = Default::default();
        for id in &doc.selection {
            if let Some(b) = doc.manager.circuit.block(*id) {
                let name = match b.ty {
                    BlockType::Chip(cid) => doc
                        .manager
                        .chips
                        .get(cid)
                        .map(|d| d.name.clone())
                        .unwrap_or_else(|| "Chip".into()),
                    ty => ty.display_name().to_string(),
                };
                *counts.entry(name).or_default() += 1;
            }
        }
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing = egui::vec2(4.0, 4.0);
            for (name, n) in counts {
                design::badge(ui, t, &format!("{name} \u{d7}{n}"));
            }
        });
        heading(ui, t, "Arrange");
        ui.horizontal(|ui| {
            if design::button_ex(
                ui,
                t,
                Kind::Secondary,
                Some(icons::ANTICLOCKWISE_2),
                "",
                true,
                30.0,
            )
            .on_hover_text("Rotate left")
            .clicked()
            {
                doc.transform_selection(GroupTransform::RotateCcw);
            }
            if design::button_ex(
                ui,
                t,
                Kind::Secondary,
                Some(icons::CLOCKWISE_2),
                "",
                true,
                30.0,
            )
            .on_hover_text("Rotate right")
            .clicked()
            {
                doc.transform_selection(GroupTransform::RotateCw);
            }
            if design::button_ex(
                ui,
                t,
                Kind::Secondary,
                Some(icons::FLIP_HORIZONTAL),
                "",
                true,
                30.0,
            )
            .on_hover_text("Flip")
            .clicked()
            {
                doc.transform_selection(GroupTransform::Flip);
            }
            if design::button_ex(
                ui,
                t,
                Kind::Secondary,
                Some(icons::CHECKBOX_MULTIPLE_BLANK),
                "",
                true,
                30.0,
            )
            .on_hover_text("Duplicate (Ctrl+D)")
            .clicked()
            {
                doc.duplicate_selection();
            }
            if design::button_ex(ui, t, Kind::Secondary, Some(icons::FOCUS_2), "", true, 30.0)
                .on_hover_text("Zoom to selection")
                .clicked()
            {
                doc.frame_request = Some(FrameRequest::Selection);
            }
        });
        heading(ui, t, "Reuse");
        if design::button(
            ui,
            t,
            Kind::Secondary,
            Some(icons::CPU),
            "Create chip from selection",
        )
        .clicked()
        {
            actions.push(InspectorAction::CreateChip {
                from_selection: true,
            });
        }
        design::hint(
            ui,
            t,
            "Switches become input pins, LEDs become output pins.",
        );
    }
    ui.add_space(18.0);
    if design::button(
        ui,
        t,
        Kind::Ghost,
        Some(icons::DELETE_BIN_6),
        "Delete selection",
    )
    .clicked()
    {
        doc.delete_selection();
    }
}
