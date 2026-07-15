//! End-to-end tests for the logic engine. These run headlessly (no GUI) and are the
//! primary correctness proof for the simulator, including chip flattening.

use llmc::backend::{CircuitManager, Simulation};
use llmc::model::{BlockId, BlockType, ChipDef, Port, Pos};

fn wire(m: &mut CircuitManager, from: BlockId, fo: u16, to: BlockId, ti: u16) {
    m.connect(Port::output(from, fo), Port::input(to, ti));
}

fn settle(m: &CircuitManager, switches: &[(BlockId, bool)]) -> Simulation {
    let mut sim = Simulation::build(&m.circuit, &m.chips);
    for &(id, v) in switches {
        sim.set_switch(id, v);
    }
    sim.step(0.0);
    sim
}

/// Evaluate a 2-input gate for all input combinations.
fn two_input_gate(ty: BlockType) -> [bool; 4] {
    let mut m = CircuitManager::new();
    let a = m.add_block(BlockType::Switch, Pos::new(0, 0));
    let b = m.add_block(BlockType::Switch, Pos::new(0, 2));
    let g = m.add_block(ty, Pos::new(4, 0));
    let led = m.add_block(BlockType::Led, Pos::new(8, 0));
    wire(&mut m, a, 0, g, 0);
    wire(&mut m, b, 0, g, 1);
    wire(&mut m, g, 0, led, 0);

    let mut out = [false; 4];
    for (i, (av, bv)) in [(false, false), (false, true), (true, false), (true, true)]
        .iter()
        .enumerate()
    {
        let sim = settle(&m, &[(a, *av), (b, *bv)]);
        out[i] = sim.led_value(led).unwrap();
    }
    out
}

#[test]
fn two_input_gate_truth_tables() {
    assert_eq!(two_input_gate(BlockType::And), [false, false, false, true]);
    assert_eq!(two_input_gate(BlockType::Or), [false, true, true, true]);
    assert_eq!(two_input_gate(BlockType::Nand), [true, true, true, false]);
    assert_eq!(two_input_gate(BlockType::Nor), [true, false, false, false]);
    assert_eq!(two_input_gate(BlockType::Xor), [false, true, true, false]);
    assert_eq!(two_input_gate(BlockType::Xnor), [true, false, false, true]);
}

#[test]
fn not_and_buffer() {
    for (ty, expect) in [
        (BlockType::Not, [true, false]),
        (BlockType::Buffer, [false, true]),
    ] {
        let mut m = CircuitManager::new();
        let a = m.add_block(BlockType::Switch, Pos::new(0, 0));
        let g = m.add_block(ty, Pos::new(4, 0));
        let led = m.add_block(BlockType::Led, Pos::new(8, 0));
        wire(&mut m, a, 0, g, 0);
        wire(&mut m, g, 0, led, 0);
        for (i, v) in [false, true].iter().enumerate() {
            let sim = settle(&m, &[(a, *v)]);
            assert_eq!(sim.led_value(led).unwrap(), expect[i], "{ty:?} input={v}");
        }
    }
}

#[test]
fn constants_and_unconnected_inputs() {
    let mut m = CircuitManager::new();
    let hi = m.add_block(BlockType::ConstantHigh, Pos::new(0, 0));
    let lo = m.add_block(BlockType::ConstantLow, Pos::new(0, 2));
    let led_hi = m.add_block(BlockType::Led, Pos::new(4, 0));
    let led_lo = m.add_block(BlockType::Led, Pos::new(4, 2));
    let led_floating = m.add_block(BlockType::Led, Pos::new(4, 4));
    wire(&mut m, hi, 0, led_hi, 0);
    wire(&mut m, lo, 0, led_lo, 0);
    let sim = settle(&m, &[]);
    assert!(sim.led_value(led_hi).unwrap());
    assert!(!sim.led_value(led_lo).unwrap());
    // An unconnected input reads low (ground).
    assert!(!sim.led_value(led_floating).unwrap());
}

/// Build a half adder and return (sum, carry) for the given inputs.
fn half_adder(a_val: bool, b_val: bool) -> (bool, bool) {
    let mut m = CircuitManager::new();
    let a = m.add_block(BlockType::Switch, Pos::new(0, 0));
    let b = m.add_block(BlockType::Switch, Pos::new(0, 2));
    let xor = m.add_block(BlockType::Xor, Pos::new(4, 0));
    let and = m.add_block(BlockType::And, Pos::new(4, 3));
    let sum = m.add_block(BlockType::Led, Pos::new(8, 0));
    let carry = m.add_block(BlockType::Led, Pos::new(8, 3));
    wire(&mut m, a, 0, xor, 0);
    wire(&mut m, b, 0, xor, 1);
    wire(&mut m, a, 0, and, 0);
    wire(&mut m, b, 0, and, 1);
    wire(&mut m, xor, 0, sum, 0);
    wire(&mut m, and, 0, carry, 0);
    let sim = settle(&m, &[(a, a_val), (b, b_val)]);
    (sim.led_value(sum).unwrap(), sim.led_value(carry).unwrap())
}

#[test]
fn half_adder_truth_table() {
    assert_eq!(half_adder(false, false), (false, false));
    assert_eq!(half_adder(false, true), (true, false));
    assert_eq!(half_adder(true, false), (true, false));
    assert_eq!(half_adder(true, true), (false, true));
}

/// A full adder as a reusable circuit. Switch pins in y-order = [A, B, Cin];
/// Led pins in y-order = [Sum, Cout].
fn build_full_adder() -> CircuitManager {
    let mut m = CircuitManager::new();
    let a = m.add_block(BlockType::Switch, Pos::new(0, 0));
    let b = m.add_block(BlockType::Switch, Pos::new(0, 2));
    let cin = m.add_block(BlockType::Switch, Pos::new(0, 4));
    let xor1 = m.add_block(BlockType::Xor, Pos::new(4, 0));
    let xor2 = m.add_block(BlockType::Xor, Pos::new(8, 0));
    let and1 = m.add_block(BlockType::And, Pos::new(4, 4));
    let and2 = m.add_block(BlockType::And, Pos::new(8, 4));
    let or1 = m.add_block(BlockType::Or, Pos::new(12, 4));
    let sum = m.add_block(BlockType::Led, Pos::new(16, 0));
    let cout = m.add_block(BlockType::Led, Pos::new(16, 4));

    wire(&mut m, a, 0, xor1, 0);
    wire(&mut m, b, 0, xor1, 1);
    wire(&mut m, xor1, 0, xor2, 0);
    wire(&mut m, cin, 0, xor2, 1);
    wire(&mut m, xor2, 0, sum, 0);

    wire(&mut m, a, 0, and1, 0);
    wire(&mut m, b, 0, and1, 1);
    wire(&mut m, xor1, 0, and2, 0);
    wire(&mut m, cin, 0, and2, 1);
    wire(&mut m, and1, 0, or1, 0);
    wire(&mut m, and2, 0, or1, 1);
    wire(&mut m, or1, 0, cout, 0);
    m
}

#[test]
fn full_adder_truth_table() {
    let m = build_full_adder();
    // The three switches (added first, ids 0,1,2) are A, B, Cin in order.
    let switches: Vec<BlockId> = m
        .circuit
        .iter_blocks()
        .filter(|b| matches!(b.ty, BlockType::Switch))
        .map(|b| b.id)
        .collect();
    let (a, b, cin) = (switches[0], switches[1], switches[2]);
    let leds: Vec<BlockId> = m
        .circuit
        .iter_blocks()
        .filter(|b| matches!(b.ty, BlockType::Led))
        .map(|b| b.id)
        .collect();
    let (sum, cout) = (leds[0], leds[1]);

    for x in 0..8 {
        let av = x & 1 != 0;
        let bv = x & 2 != 0;
        let cv = x & 4 != 0;
        let sim = settle(&m, &[(a, av), (b, bv), (cin, cv)]);
        let total = av as u8 + bv as u8 + cv as u8;
        assert_eq!(sim.led_value(sum).unwrap(), total & 1 != 0, "sum for {x}");
        assert_eq!(sim.led_value(cout).unwrap(), total >= 2, "carry for {x}");
    }
}

#[test]
#[allow(clippy::needless_range_loop)]
fn four_bit_adder_from_chips() {
    // Register the full adder as a chip.
    let fa = build_full_adder();
    let mut m = CircuitManager::new();
    let chip_id = m.chips.allocate_id();
    m.chips.insert(ChipDef::from_circuit(
        chip_id,
        "FullAdder",
        fa.circuit.clone(),
    ));

    // Eight data switches (a0..a3, b0..b3) plus a carry-in switch.
    let a: Vec<BlockId> = (0..4)
        .map(|i| m.add_block(BlockType::Switch, Pos::new(0, i * 2)))
        .collect();
    let b: Vec<BlockId> = (0..4)
        .map(|i| m.add_block(BlockType::Switch, Pos::new(0, 8 + i * 2)))
        .collect();
    let cin0 = m.add_block(BlockType::Switch, Pos::new(0, 16));

    // Four full-adder instances, carries chained.
    let adders: Vec<BlockId> = (0..4)
        .map(|i| m.add_block(BlockType::Chip(chip_id), Pos::new(6, i * 6)))
        .collect();
    let sum_leds: Vec<BlockId> = (0..4)
        .map(|i| m.add_block(BlockType::Led, Pos::new(14, i * 6)))
        .collect();
    let carry_out = m.add_block(BlockType::Led, Pos::new(14, 24));

    for i in 0..4 {
        // Chip inputs: 0=A, 1=B, 2=Cin ; outputs: 0=Sum, 1=Cout.
        wire(&mut m, a[i], 0, adders[i], 0);
        wire(&mut m, b[i], 0, adders[i], 1);
        if i == 0 {
            wire(&mut m, cin0, 0, adders[i], 2);
        } else {
            wire(&mut m, adders[i - 1], 1, adders[i], 2);
        }
        wire(&mut m, adders[i], 0, sum_leds[i], 0);
    }
    wire(&mut m, adders[3], 1, carry_out, 0);

    // Check a handful of additions.
    for (x, y) in [(0u8, 0u8), (1, 1), (5, 10), (15, 15), (7, 9), (13, 2)] {
        let mut switches = Vec::new();
        for i in 0..4 {
            switches.push((a[i], x & (1 << i) != 0));
            switches.push((b[i], y & (1 << i) != 0));
        }
        switches.push((cin0, false));
        let sim = settle(&m, &switches);

        let mut result = 0u16;
        for i in 0..4 {
            if sim.led_value(sum_leds[i]).unwrap() {
                result |= 1 << i;
            }
        }
        if sim.led_value(carry_out).unwrap() {
            result |= 1 << 4;
        }
        assert_eq!(result, x as u16 + y as u16, "{x} + {y}");
    }
}

#[test]
fn mux_2to1() {
    // out = sel ? b : a  == (a AND NOT sel) OR (b AND sel)
    let mut m = CircuitManager::new();
    let a = m.add_block(BlockType::Switch, Pos::new(0, 0));
    let b = m.add_block(BlockType::Switch, Pos::new(0, 2));
    let sel = m.add_block(BlockType::Switch, Pos::new(0, 4));
    let notsel = m.add_block(BlockType::Not, Pos::new(4, 4));
    let and_a = m.add_block(BlockType::And, Pos::new(8, 0));
    let and_b = m.add_block(BlockType::And, Pos::new(8, 3));
    let or = m.add_block(BlockType::Or, Pos::new(12, 1));
    let led = m.add_block(BlockType::Led, Pos::new(16, 1));
    wire(&mut m, sel, 0, notsel, 0);
    wire(&mut m, a, 0, and_a, 0);
    wire(&mut m, notsel, 0, and_a, 1);
    wire(&mut m, b, 0, and_b, 0);
    wire(&mut m, sel, 0, and_b, 1);
    wire(&mut m, and_a, 0, or, 0);
    wire(&mut m, and_b, 0, or, 1);
    wire(&mut m, or, 0, led, 0);

    for av in [false, true] {
        for bv in [false, true] {
            for sv in [false, true] {
                let sim = settle(&m, &[(a, av), (b, bv), (sel, sv)]);
                let expect = if sv { bv } else { av };
                assert_eq!(
                    sim.led_value(led).unwrap(),
                    expect,
                    "a={av} b={bv} sel={sv}"
                );
            }
        }
    }
}

#[test]
fn sr_latch_holds_state() {
    // Cross-coupled NOR latch: q = NOR(r, qn), qn = NOR(s, q).
    let mut m = CircuitManager::new();
    let s = m.add_block(BlockType::Switch, Pos::new(0, 0));
    let r = m.add_block(BlockType::Switch, Pos::new(0, 4));
    let nor_q = m.add_block(BlockType::Nor, Pos::new(4, 0));
    let nor_qn = m.add_block(BlockType::Nor, Pos::new(4, 4));
    let q_led = m.add_block(BlockType::Led, Pos::new(8, 0));
    wire(&mut m, r, 0, nor_q, 0);
    wire(&mut m, nor_qn, 0, nor_q, 1);
    wire(&mut m, s, 0, nor_qn, 0);
    wire(&mut m, nor_q, 0, nor_qn, 1);
    wire(&mut m, nor_q, 0, q_led, 0);

    let mut sim = Simulation::build(&m.circuit, &m.chips);

    // Set: S=1, R=0 -> Q=1
    sim.set_switch(s, true);
    sim.set_switch(r, false);
    sim.step(0.0);
    assert!(sim.led_value(q_led).unwrap(), "after set");

    // Hold: S=0, R=0 -> Q stays 1
    sim.set_switch(s, false);
    sim.step(0.0);
    assert!(sim.led_value(q_led).unwrap(), "hold high");

    // Reset: R=1 -> Q=0
    sim.set_switch(r, true);
    sim.step(0.0);
    assert!(!sim.led_value(q_led).unwrap(), "after reset");

    // Hold: R=0 -> Q stays 0
    sim.set_switch(r, false);
    sim.step(0.0);
    assert!(!sim.led_value(q_led).unwrap(), "hold low");
}
