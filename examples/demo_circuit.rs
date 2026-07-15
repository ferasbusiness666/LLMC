//! Generates a small showcase `.llmc` file. Usage: `cargo run --example demo_circuit -- out.llmc`
//! (also used to produce screenshots). Not part of the app itself.

use llmc::backend::CircuitManager;
use llmc::io::{CameraState, Project};
use llmc::model::{BlockId, BlockType, ChipDef, Port, Pos};

fn wire(m: &mut CircuitManager, from: BlockId, fo: u16, to: BlockId, ti: u16) {
    m.connect(Port::output(from, fo), Port::input(to, ti));
}

fn main() {
    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "demo.llmc".to_string());

    // A half-adder chip: switches A,B -> inputs; LEDs Sum,Carry -> outputs.
    let mut ha = CircuitManager::new();
    let a = ha.add_block(BlockType::Switch, Pos::new(0, 0));
    let b = ha.add_block(BlockType::Switch, Pos::new(0, 3));
    let hx = ha.add_block(BlockType::Xor, Pos::new(5, 0));
    let hn = ha.add_block(BlockType::And, Pos::new(5, 4));
    let hs = ha.add_block(BlockType::Led, Pos::new(10, 0));
    let hc = ha.add_block(BlockType::Led, Pos::new(10, 4));
    wire(&mut ha, a, 0, hx, 0);
    wire(&mut ha, b, 0, hx, 1);
    wire(&mut ha, a, 0, hn, 0);
    wire(&mut ha, b, 0, hn, 1);
    wire(&mut ha, hx, 0, hs, 0);
    wire(&mut ha, hn, 0, hc, 0);

    let mut m = CircuitManager::new();
    let chip_id = m.chips.allocate_id();
    m.chips.insert(ChipDef::from_circuit(
        chip_id,
        "HalfAdd",
        ha.circuit.clone(),
    ));

    // Row 1: gate showcase driven by two switches (A=1, B=0) so several LEDs light.
    let sa = m.add_block(BlockType::Switch, Pos::new(0, 0));
    let sb = m.add_block(BlockType::Switch, Pos::new(0, 3));
    m.set_state(sa, true);
    for (i, (ty, gy)) in [
        (BlockType::Or, 0),
        (BlockType::And, 3),
        (BlockType::Xor, 6),
        (BlockType::Nand, 9),
    ]
    .into_iter()
    .enumerate()
    {
        let g = m.add_block(ty, Pos::new(6, gy));
        let led = m.add_block(BlockType::Led, Pos::new(11, gy));
        wire(&mut m, sa, 0, g, 0);
        wire(&mut m, sb, 0, g, 1);
        wire(&mut m, g, 0, led, 0);
        let _ = i;
    }

    // Row 2: a half-adder chip instance with X=1, Y=1 (Sum off, Carry on).
    let x = m.add_block(BlockType::Switch, Pos::new(0, 15));
    let y = m.add_block(BlockType::Switch, Pos::new(0, 18));
    m.set_state(x, true);
    m.set_state(y, true);
    let chip = m.add_block(BlockType::Chip(chip_id), Pos::new(6, 15));
    let sum = m.add_block(BlockType::Led, Pos::new(12, 15));
    let carry = m.add_block(BlockType::Led, Pos::new(12, 18));
    wire(&mut m, x, 0, chip, 0);
    wire(&mut m, y, 0, chip, 1);
    wire(&mut m, chip, 0, sum, 0);
    wire(&mut m, chip, 1, carry, 0);

    // Row 3: a clock into a NOT into an LED.
    let clk = m.add_block(BlockType::Clock, Pos::new(0, 24));
    let inv = m.add_block(BlockType::Not, Pos::new(6, 24));
    let led = m.add_block(BlockType::Led, Pos::new(11, 24));
    wire(&mut m, clk, 0, inv, 0);
    wire(&mut m, inv, 0, led, 0);

    let project = Project::new(
        m.circuit.clone(),
        m.chips.clone(),
        CameraState {
            pan_x: 130.0,
            pan_y: 90.0,
            zoom: 30.0,
        },
    );
    project.save(&out).expect("save demo");
    println!("wrote {out}");
}
