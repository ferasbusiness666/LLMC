//! Geometry primitives shared by the model and the renderer.
//!
//! Two coordinate spaces exist:
//! * **grid space** — integer [`Pos`] cells where blocks live.
//! * **local space** — continuous [`Vec2f`] offsets inside a block's footprint,
//!   measured in cells relative to the block's (unrotated) top-left corner.
//!
//! [`Orientation`] maps a local point into the *placed* block's frame, accounting
//! for rotation and flip. The exact chirality of the rotation is unimportant as long
//! as the glyph and its ports are transformed by the *same* function — they always
//! stay aligned.

use serde::{Deserialize, Serialize};

/// A continuous 2D point/offset in cell units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2f {
    pub x: f32,
    pub y: f32,
}

impl Vec2f {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };

    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

impl std::ops::Add for Vec2f {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        Self::new(self.x + o.x, self.y + o.y)
    }
}

impl std::ops::Sub for Vec2f {
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        Self::new(self.x - o.x, self.y - o.y)
    }
}

impl std::ops::Mul<f32> for Vec2f {
    type Output = Self;
    fn mul(self, s: f32) -> Self {
        Self::new(self.x * s, self.y * s)
    }
}

/// An integer grid position (one unit == one grid cell).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Pos {
    pub x: i32,
    pub y: i32,
}

impl Pos {
    pub const fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }

    pub fn as_vec(self) -> Vec2f {
        Vec2f::new(self.x as f32, self.y as f32)
    }
}

impl std::ops::Add for Pos {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        Self::new(self.x + o.x, self.y + o.y)
    }
}

impl std::ops::Sub for Pos {
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        Self::new(self.x - o.x, self.y - o.y)
    }
}

/// Rotation in 90° increments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum Rotation {
    #[default]
    R0,
    R90,
    R180,
    R270,
}

impl Rotation {
    /// Rotate one quarter-turn clockwise.
    pub fn cw(self) -> Self {
        match self {
            Rotation::R0 => Rotation::R90,
            Rotation::R90 => Rotation::R180,
            Rotation::R180 => Rotation::R270,
            Rotation::R270 => Rotation::R0,
        }
    }

    /// Rotate one quarter-turn counter-clockwise.
    pub fn ccw(self) -> Self {
        match self {
            Rotation::R0 => Rotation::R270,
            Rotation::R90 => Rotation::R0,
            Rotation::R180 => Rotation::R90,
            Rotation::R270 => Rotation::R180,
        }
    }
}

/// A block's placement orientation: a rotation plus an optional horizontal mirror.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct Orientation {
    pub rotation: Rotation,
    pub flipped: bool,
}

impl Orientation {
    pub fn cw(self) -> Self {
        Self {
            rotation: self.rotation.cw(),
            ..self
        }
    }

    pub fn ccw(self) -> Self {
        Self {
            rotation: self.rotation.ccw(),
            ..self
        }
    }

    pub fn flip(self) -> Self {
        Self {
            flipped: !self.flipped,
            ..self
        }
    }

    /// The footprint size after this orientation is applied.
    pub fn transformed_size(self, size: Vec2f) -> Vec2f {
        match self.rotation {
            Rotation::R0 | Rotation::R180 => size,
            Rotation::R90 | Rotation::R270 => Vec2f::new(size.y, size.x),
        }
    }

    /// Map a local point (relative to the unrotated footprint's top-left) into the
    /// placed block's frame (relative to the rotated footprint's top-left). Add the
    /// block's grid position to obtain a world-space cell coordinate.
    pub fn transform(self, local: Vec2f, size: Vec2f) -> Vec2f {
        let mut x = local.x;
        let y = local.y;
        if self.flipped {
            x = size.x - x;
        }
        match self.rotation {
            Rotation::R0 => Vec2f::new(x, y),
            Rotation::R90 => Vec2f::new(size.y - y, x),
            Rotation::R180 => Vec2f::new(size.x - x, size.y - y),
            Rotation::R270 => Vec2f::new(y, size.x - x),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotation_cycles() {
        assert_eq!(Rotation::R0.cw().cw().cw().cw(), Rotation::R0);
        assert_eq!(Rotation::R0.ccw(), Rotation::R270);
    }

    #[test]
    fn r0_is_identity() {
        let o = Orientation::default();
        let p = o.transform(Vec2f::new(0.0, 0.6), Vec2f::new(2.0, 2.0));
        assert_eq!(p, Vec2f::new(0.0, 0.6));
        assert_eq!(
            o.transformed_size(Vec2f::new(2.0, 1.0)),
            Vec2f::new(2.0, 1.0)
        );
    }

    #[test]
    fn r90_swaps_footprint() {
        let o = Orientation {
            rotation: Rotation::R90,
            flipped: false,
        };
        assert_eq!(
            o.transformed_size(Vec2f::new(2.0, 1.0)),
            Vec2f::new(1.0, 2.0)
        );
    }

    #[test]
    fn flip_mirrors_x() {
        let o = Orientation {
            rotation: Rotation::R0,
            flipped: true,
        };
        let p = o.transform(Vec2f::new(0.0, 1.0), Vec2f::new(2.0, 2.0));
        assert_eq!(p, Vec2f::new(2.0, 1.0));
    }
}
