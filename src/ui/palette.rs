//! The left sidebar: a searchable component library (sources, outputs, gates, and the project's
//! chips) with live thumbnails. Click an item to arm placement, or drag it onto the canvas.

use eframe::egui::{
    self, Align2, CornerRadius, CursorIcon, Pos2, Rect, Response, Sense, Stroke, StrokeKind, Ui,
};

use llmc::io::GateStyle;
use llmc::model::{BlockType, ChipId};

use super::design::{self, Theme};
use super::document::{Document, Tool};
use super::glyphs;
use super::icons;

/// What the palette asks the shell to do.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PaletteAction {
    CreateChip,
    EditChip(ChipId),
    RenameChip(ChipId),
    DeleteChip(ChipId),
    ShowShortcuts,
}

const SOURCES: [BlockType; 5] = [
    BlockType::Switch,
    BlockType::Button,
    BlockType::Clock,
    BlockType::ConstantHigh,
    BlockType::ConstantLow,
];
const OUTPUTS: [BlockType; 1] = [BlockType::Led];
const GATES: [BlockType; 8] = [
    BlockType::And,
    BlockType::Or,
    BlockType::Not,
    BlockType::Nand,
    BlockType::Nor,
    BlockType::Xor,
    BlockType::Xnor,
    BlockType::Buffer,
];

fn matches(query: &str, name: &str, desc: &str) -> bool {
    let q = query.trim().to_lowercase();
    q.is_empty() || name.to_lowercase().contains(&q) || desc.to_lowercase().contains(&q)
}

pub fn palette(
    ui: &mut Ui,
    doc: &mut Document,
    t: &Theme,
    gate_style: GateStyle,
    search: &mut String,
) -> Vec<PaletteAction> {
    let mut actions = Vec::new();
    ui.add_space(12.0);
    ui.horizontal(|ui| {
        ui.add_space(2.0);
        ui.label(
            egui::RichText::new("Components")
                .font(design::semibold(14.0))
                .color(t.text),
        );
    });
    ui.add_space(8.0);
    design::text_field(
        ui,
        t,
        search,
        "Search components\u{2026}",
        Some(icons::SEARCH),
    );
    ui.add_space(6.0);

    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            let q = search.clone();
            let mut any = false;
            any |= section(ui, doc, t, gate_style, "Inputs", &SOURCES, &q);
            any |= section(ui, doc, t, gate_style, "Outputs", &OUTPUTS, &q);
            any |= section(ui, doc, t, gate_style, "Logic gates", &GATES, &q);
            any |= chips_section(ui, doc, t, &q, &mut actions);
            if !any {
                ui.add_space(20.0);
                ui.vertical_centered(|ui| {
                    design::hint(
                        ui,
                        t,
                        format!("Nothing matches \u{201c}{}\u{201d}", q.trim()),
                    );
                });
            }
            ui.add_space(12.0);
        });
    actions
}

/// A collapsible section; returns whether it showed anything.
fn section(
    ui: &mut Ui,
    doc: &mut Document,
    t: &Theme,
    gate_style: GateStyle,
    title: &str,
    items: &[BlockType],
    query: &str,
) -> bool {
    let shown: Vec<BlockType> = items
        .iter()
        .copied()
        .filter(|ty| matches(query, ty.display_name(), ty.description()))
        .collect();
    if shown.is_empty() {
        return false;
    }
    let open = section_header(ui, t, title, None, !query.trim().is_empty());
    if open {
        for ty in shown {
            let selected = doc.tool == Tool::Place(ty) || doc.palette_drag == Some(ty);
            let resp = item_row(
                ui,
                t,
                selected,
                |p, r| glyphs::draw_thumbnail(p, r, ty, gate_style, t, selected),
                ty.display_name(),
                None,
            );
            handle_item(doc, &resp, ty);
            resp.on_hover_text(format!(
                "{}\n\nClick to place, or drag onto the canvas.",
                ty.description()
            ));
        }
    }
    true
}

fn handle_item(doc: &mut Document, resp: &Response, ty: BlockType) {
    if resp.drag_started() {
        doc.palette_drag = Some(ty);
        doc.tool = Tool::Select;
    }
    if resp.clicked() {
        doc.tool = if doc.tool == Tool::Place(ty) {
            Tool::Select
        } else {
            Tool::Place(ty)
        };
    }
    if resp.dragged() {
        resp.ctx.set_cursor_icon(CursorIcon::Grabbing);
    } else if resp.hovered() {
        resp.ctx.set_cursor_icon(CursorIcon::PointingHand);
    }
}

fn chips_section(
    ui: &mut Ui,
    doc: &mut Document,
    t: &Theme,
    query: &str,
    actions: &mut Vec<PaletteAction>,
) -> bool {
    let chips: Vec<(ChipId, String, usize, usize)> = doc
        .manager
        .chips
        .iter()
        .filter(|c| doc.placeable(c.id))
        .filter(|c| matches(query, &c.name, "chip"))
        .map(|c| {
            (
                c.id,
                c.name.clone(),
                c.input_blocks.len(),
                c.output_blocks.len(),
            )
        })
        .collect();
    let searching = !query.trim().is_empty();
    if searching && chips.is_empty() {
        return false;
    }
    let count = doc.manager.chips.len().to_string();
    let open = section_header(ui, t, "Chips", Some(&count), searching);
    if !open {
        return true;
    }
    if chips.is_empty() {
        design::card(t)
            .inner_margin(egui::Margin::same(10))
            .show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.label(
                    egui::RichText::new("No chips yet")
                        .font(design::medium(12.5))
                        .color(t.text),
                );
                design::hint(
                    ui,
                    t,
                    "Package a circuit as a reusable chip, then drop it in like any gate.",
                );
            });
        ui.add_space(6.0);
    }
    for (id, name, nin, nout) in chips {
        let ty = BlockType::Chip(id);
        let selected = doc.tool == Tool::Place(ty) || doc.palette_drag == Some(ty);
        let badge = format!("{nin}\u{2192}{nout}");
        let resp = item_row(
            ui,
            t,
            selected,
            |p, r| glyphs::draw_thumbnail(p, r, ty, GateStyle::Shapes, t, selected),
            &name,
            Some(&badge),
        );
        handle_item(doc, &resp, ty);
        let users = doc.chip_users(id);
        let used_by = if users.is_empty() {
            String::new()
        } else {
            format!("\nUsed inside: {}", users.join(", "))
        };
        let resp = resp.on_hover_text(format!(
            "{name} \u{b7} {nin} input(s), {nout} output(s){used_by}\n\nClick to place \u{b7} drag onto the canvas \u{b7} right-click for options"
        ));
        resp.context_menu(|ui| {
            ui.set_min_width(200.0);
            if design::menu_item(ui, t, icons::ADD_CIRCLE, "Place", "") {
                doc.tool = Tool::Place(ty);
            }
            if design::menu_item(ui, t, icons::PENCIL, "Edit internals\u{2026}", "") {
                actions.push(PaletteAction::EditChip(id));
            }
            if design::menu_item(ui, t, icons::EDIT_2, "Rename\u{2026}", "") {
                actions.push(PaletteAction::RenameChip(id));
            }
            design::divider(ui, t);
            if design::menu_item(ui, t, icons::DELETE_BIN_6, "Delete\u{2026}", "") {
                actions.push(PaletteAction::DeleteChip(id));
            }
        });
    }
    ui.add_space(4.0);
    if doc.chip_edit.is_none()
        && design::button_ex(
            ui,
            t,
            design::Kind::Ghost,
            Some(icons::ADD),
            "New chip from canvas",
            true,
            28.0,
        )
        .on_hover_text("Package the current circuit (switches = inputs, LEDs = outputs) as a chip")
        .clicked()
    {
        actions.push(PaletteAction::CreateChip);
    }
    true
}

/// A section header with a chevron; returns whether it's open. `force_open` (while searching)
/// shows contents regardless of the remembered state.
fn section_header(
    ui: &mut Ui,
    t: &Theme,
    title: &str,
    count: Option<&str>,
    force_open: bool,
) -> bool {
    let id = ui.id().with(("palette-section", title));
    let mut open = ui.ctx().data_mut(|d| *d.get_persisted_mut_or(id, true));
    ui.add_space(8.0);
    let w = ui.available_width();
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(w, 22.0), Sense::click());
    if resp.clicked() && !force_open {
        open = !open;
        ui.ctx().data_mut(|d| d.insert_persisted(id, open));
    }
    let show = open || force_open;
    let p = ui.painter();
    p.text(
        Pos2::new(rect.left() + 8.0, rect.center().y),
        Align2::CENTER_CENTER,
        if show {
            icons::ARROW_DOWN_S
        } else {
            icons::ARROW_RIGHT_S
        },
        design::icon_font(14.0),
        t.text_faint,
    );
    let mut job = egui::text::LayoutJob::default();
    job.append(
        &title.to_uppercase(),
        0.0,
        egui::TextFormat {
            font_id: design::semibold(10.5),
            color: if resp.hovered() {
                t.text_muted
            } else {
                t.text_faint
            },
            extra_letter_spacing: 0.8,
            ..Default::default()
        },
    );
    let galley = ui.painter().layout_job(job);
    ui.painter().galley(
        Pos2::new(rect.left() + 18.0, rect.center().y - galley.size().y / 2.0),
        galley,
        t.text_faint,
    );
    if let Some(c) = count {
        ui.painter().text(
            Pos2::new(rect.right() - 6.0, rect.center().y),
            Align2::RIGHT_CENTER,
            c,
            design::medium(11.0),
            t.text_faint,
        );
    }
    show
}

/// One palette row: thumbnail, name, optional badge. Selected rows get the accent ring.
fn item_row(
    ui: &mut Ui,
    t: &Theme,
    selected: bool,
    thumb: impl FnOnce(&egui::Painter, Rect),
    name: &str,
    badge: Option<&str>,
) -> Response {
    let w = ui.available_width();
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(w, 34.0), Sense::click_and_drag());
    let hovered = resp.hovered();
    let anim = ui
        .ctx()
        .animate_bool_with_time(resp.id.with("hover"), hovered, 0.10);
    let p = ui.painter();
    let rad = CornerRadius::same(design::radius::MD);
    if selected {
        p.rect_filled(rect, rad, t.accent_soft);
        p.rect_stroke(rect, rad, Stroke::new(1.0, t.accent), StrokeKind::Inside);
        // Inner top highlight, like a pressed key.
        p.line_segment(
            [
                rect.left_top() + egui::vec2(6.0, 1.5),
                rect.right_top() + egui::vec2(-6.0, 1.5),
            ],
            Stroke::new(1.0, egui::Color32::from_white_alpha(28)),
        );
    } else if anim > 0.0 {
        p.rect_filled(rect, rad, t.surface_hover.gamma_multiply(anim));
    }
    let thumb_rect = Rect::from_min_size(
        rect.left_top() + egui::vec2(6.0, 5.0),
        egui::vec2(30.0, 24.0),
    );
    p.rect_filled(
        thumb_rect,
        CornerRadius::same(design::radius::SM),
        t.bg_canvas,
    );
    p.rect_stroke(
        thumb_rect,
        CornerRadius::same(design::radius::SM),
        Stroke::new(1.0, t.border),
        StrokeKind::Inside,
    );
    thumb(&p.with_clip_rect(thumb_rect), thumb_rect.shrink(3.0));
    let text_x = thumb_rect.right() + 10.0;
    let max_w = rect.right() - text_x - if badge.is_some() { 44.0 } else { 8.0 };
    let shown = glyphs::ellipsize(p, name, &design::medium(13.0), max_w);
    p.text(
        Pos2::new(text_x, rect.center().y),
        Align2::LEFT_CENTER,
        shown,
        design::medium(13.0),
        if selected || hovered {
            t.text
        } else {
            t.text_muted
        },
    );
    if let Some(b) = badge {
        let font = design::medium(10.5);
        let galley = p.layout_no_wrap(b.to_string(), font.clone(), t.text_faint);
        let br = Rect::from_min_size(
            Pos2::new(rect.right() - galley.size().x - 16.0, rect.center().y - 8.5),
            egui::vec2(galley.size().x + 10.0, 17.0),
        );
        p.rect_filled(br, CornerRadius::same(5), t.surface_active);
        p.text(br.center(), Align2::CENTER_CENTER, b, font, t.text_muted);
    }
    resp
}

/// The small footer under the palette (shortcuts reference).
pub fn footer(ui: &mut Ui, t: &Theme) -> Option<PaletteAction> {
    let mut action = None;
    ui.horizontal(|ui| {
        if design::button_ex(
            ui,
            t,
            design::Kind::Ghost,
            Some(icons::KEYBOARD),
            "Shortcuts",
            true,
            26.0,
        )
        .clicked()
        {
            action = Some(PaletteAction::ShowShortcuts);
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            design::kbd(ui, t, "F1");
        });
    });
    action
}
