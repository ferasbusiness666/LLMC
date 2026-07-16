//! Block types and their port layouts.

use serde::{Deserialize, Serialize};

use super::chip::ChipLibrary;
use super::connection::{Port, PortKind};
use super::geometry::{Orientation, Pos, Vec2f};
use super::ids::{BlockId, ChipId};

/// Every kind of block that can be placed on a circuit.
///
/// `Junction` is intentionally absent: fan-out is done by drawing several wires from
/// one output, so no bidirectional node is required.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BlockType {
    And,
    Or,
    Not,
    Nand,
    Nor,
    Xor,
    Xnor,
    Buffer,
    /// A user-toggleable input.
    Switch,
    /// A momentary input (high only while held).
    Button,
    /// An output indicator.
    Led,
    ConstantHigh,
    ConstantLow,
    /// A free-running oscillator.
    Clock,
    /// An instance of a reusable sub-circuit from the chip library.
    Chip(ChipId),
}

impl BlockType {
    /// A short label suitable for drawing inside a small glyph.
    pub fn short_label(&self) -> &'static str {
        match self {
            BlockType::And => "AND",
            BlockType::Or => "OR",
            BlockType::Not => "NOT",
            BlockType::Nand => "NAND",
            BlockType::Nor => "NOR",
            BlockType::Xor => "XOR",
            BlockType::Xnor => "XNOR",
            BlockType::Buffer => "BUF",
            BlockType::Switch => "SW",
            BlockType::Button => "BTN",
            BlockType::Led => "LED",
            BlockType::ConstantHigh => "1",
            BlockType::ConstantLow => "0",
            BlockType::Clock => "CLK",
            BlockType::Chip(_) => "IC",
        }
    }

    /// True for the primitive logic gates (not I/O or chips).
    pub fn is_gate(&self) -> bool {
        matches!(
            self,
            BlockType::And
                | BlockType::Or
                | BlockType::Not
                | BlockType::Nand
                | BlockType::Nor
                | BlockType::Xor
                | BlockType::Xnor
                | BlockType::Buffer
        )
    }

    /// The full set of primitive block types, in palette order.
    pub fn primitives() -> &'static [BlockType] {
        use BlockType::*;
        &[
            And,
            Or,
            Not,
            Nand,
            Nor,
            Xor,
            Xnor,
            Buffer,
            Switch,
            Button,
            Led,
            ConstantHigh,
            ConstantLow,
            Clock,
        ]
    }

    /// The unrotated footprint and port positions for this block type.
    pub fn layout(&self, chips: &ChipLibrary) -> PortLayout {
        match self {
            // Two-input gates.
            BlockType::And
            | BlockType::Or
            | BlockType::Nand
            | BlockType::Nor
            | BlockType::Xor
            | BlockType::Xnor => PortLayout {
                size: Vec2f::new(2.0, 2.0),
                inputs: vec![Vec2f::new(0.0, 0.5), Vec2f::new(0.0, 1.5)],
                outputs: vec![Vec2f::new(2.0, 1.0)],
            },
            // Single-input gates.
            BlockType::Not | BlockType::Buffer => PortLayout {
                size: Vec2f::new(2.0, 2.0),
                inputs: vec![Vec2f::new(0.0, 1.0)],
                outputs: vec![Vec2f::new(2.0, 1.0)],
            },
            // Sources (no inputs).
            BlockType::Switch
            | BlockType::Button
            | BlockType::ConstantHigh
            | BlockType::ConstantLow
            | BlockType::Clock => PortLayout {
                size: Vec2f::new(2.0, 2.0),
                inputs: vec![],
                outputs: vec![Vec2f::new(2.0, 1.0)],
            },
            // Sink (no outputs).
            BlockType::Led => PortLayout {
                size: Vec2f::new(2.0, 2.0),
                inputs: vec![Vec2f::new(0.0, 1.0)],
                outputs: vec![],
            },
            // Chip instance — port count comes from the referenced definition.
            BlockType::Chip(id) => {
                let (nin, nout) = chips.io_counts(*id).unwrap_or((0, 0));
                let rows = nin.max(nout).max(1);
                let h = rows as f32 + 1.0;
                let w = 3.0;
                let spread = |n: usize| -> Vec<f32> {
                    (0..n)
                        .map(|i| h * (i as f32 + 1.0) / (n as f32 + 1.0))
                        .collect()
                };
                PortLayout {
                    size: Vec2f::new(w, h),
                    inputs: spread(nin)
                        .into_iter()
                        .map(|y| Vec2f::new(0.0, y))
                        .collect(),
                    outputs: spread(nout).into_iter().map(|y| Vec2f::new(w, y)).collect(),
                }
            }
        }
    }
}

/// The footprint and port positions of a block type, in unrotated local (cell) space.
#[derive(Debug, Clone, PartialEq)]
pub struct PortLayout {
    pub size: Vec2f,
    pub inputs: Vec<Vec2f>,
    pub outputs: Vec<Vec2f>,
}

/// A placed block.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Block {
    pub id: BlockId,
    pub ty: BlockType,
    /// Top-left cell of the (rotated) footprint.
    pub pos: Pos,
    #[serde(default)]
    pub orientation: Orientation,
    /// Authored default level for `Switch`/`Button`. Top-level switches are overridden
    /// by live UI state during simulation.
    #[serde(default)]
    pub state: bool,
    /// An optional user-given name, shown above the type on the block.
    #[serde(default)]
    pub label: Option<String>,
    /// Optional custom body color (sRGB). `None` uses the theme default.
    #[serde(default)]
    pub color: Option<[u8; 3]>,
    /// Per-`Clock` frequency in Hz. `None` uses the default (1 Hz).
    #[serde(default)]
    pub freq_hz: Option<f32>,
}

impl Block {
    pub fn new(id: BlockId, ty: BlockType, pos: Pos) -> Self {
        Self {
            id,
            ty,
            pos,
            orientation: Orientation::default(),
            state: false,
            label: None,
            color: None,
            freq_hz: None,
        }
    }

    pub fn input_count(&self, chips: &ChipLibrary) -> usize {
        self.ty.layout(chips).inputs.len()
    }

    pub fn output_count(&self, chips: &ChipLibrary) -> usize {
        self.ty.layout(chips).outputs.len()
    }

    /// The block's bounding footprint in world cells, after orientation.
    pub fn footprint(&self, chips: &ChipLibrary) -> Vec2f {
        self.orientation
            .transformed_size(self.ty.layout(chips).size)
    }

    /// World-space (cell) position of a port, or `None` if the index is out of range.
    pub fn port_position(&self, port: Port, chips: &ChipLibrary) -> Option<Vec2f> {
        let layout = self.ty.layout(chips);
        let locals = match port.kind {
            PortKind::Input => &layout.inputs,
            PortKind::Output => &layout.outputs,
        };
        let local = locals.get(port.index as usize)?;
        Some(self.pos.as_vec() + self.orientation.transform(*local, layout.size))
    }
}
