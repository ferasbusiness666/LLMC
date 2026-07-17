# LLMC

A clean, fast **digital-logic builder and simulator** for Windows (and Linux/macOS),
written in Rust with [egui](https://github.com/emilk/egui). Place gates, wire them up
with the mouse, and watch the logic run live. Package any circuit into a reusable chip
and drop it back onto the canvas.

> Architecturally inspired by the
> [Connection Machine](https://github.com/Martian-Technologies/Connection-Machine)
> project — a layered `Environment → Backend → CircuitManager` design — but a fresh,
> independent implementation in Rust.

## Features

- **Interactive canvas** — infinite grid, smooth pan/zoom around the cursor, precise
  hover/hit detection for blocks, ports, and wires.
- **Mouse-first wiring** — drag from a port to draw a wire; it snaps to the nearest
  valid target and shows a live preview. Outputs fan out to many inputs.
- **Live simulation** — toggle switches and watch LEDs and wires light up in real time.
  Combinational logic settles instantly; sequential logic (latches, registers) and
  clocks run step-by-step.
- **Reusable chips** — package the current circuit into a chip (switches become input
  pins, LEDs become output pins) and instance it like any other block, with full nesting
  (e.g. build a full adder once, drop four to make a 4-bit adder).
- **Full editing** — select (click / shift-click / box select), move, rotate, flip,
  duplicate, copy/paste, delete, and unlimited undo/redo.
- **Save/Load** — versioned `.llmc` JSON project files; persistent app settings.

### Components

Gates: `AND OR NOT NAND NOR XOR XNOR BUFFER` · I/O: `Switch Button LED Constant-1
Constant-0 Clock` · plus your own **chips**.

## Building & running

Requires a recent stable Rust toolchain.

```sh
cargo run --release
```

A prebuilt Windows binary (`llmc.exe`) is produced by CI on every push — see the
**Actions** tab → latest run → **Artifacts → llmc-windows**.

## Controls

| Action | Input |
| --- | --- |
| Place a component | pick it in the left palette, then click the canvas |
| Cancel the held component | right-click, or `Esc` |
| Select | click a block/wire · Shift-click to add · drag on empty space to box-select (grabs blocks **and** wires) |
| Move | drag a selected block (grid-snapped; one undo step) |
| Wire | drag from a port to another port — **or** click a port, then click the target port |
| Cancel a wire in progress | right-click, or `Esc` |
| Context menu | right-click the canvas → properties, copy, paste-here, duplicate, delete, rotate, flip |
| Block properties | right-click a block → **Properties…** (rename, custom color, clock frequency, and gate input-pin count) |
| Manage a chip | right-click it in the palette → place, edit, rename, delete |
| Toggle a switch | click it |
| Pan | drag with middle or right mouse button, or hold `Space` and drag |
| Zoom | scroll wheel (zooms around the cursor) |
| Rotate / flip | `R` / `Shift+R` · `F` |
| Delete | `Delete` / `Backspace` |
| Undo / redo | `Ctrl+Z` / `Ctrl+Y` (or `Ctrl+Shift+Z`) |
| Copy / paste / duplicate | `Ctrl+C` / `Ctrl+V` / `Ctrl+D` |
| Select all | `Ctrl+A` |
| Run / stop simulation | toolbar **Run** / **Stop** |

**Editing a chip:** right-click a chip in the palette and choose **Edit** to open its
internal circuit; the toolbar shows **Update chip** / **Cancel**. Updating re-derives the
chip from its switches (inputs) and LEDs (outputs) and returns you to your circuit.

## Architecture

The project is split into a GUI-free **library** (unit-tested headlessly) and a thin
**binary** that hosts the egui GUI.

```
src/
  model/      geometry, block types + port layouts, connections, circuits, chips
  backend/    EditCommand (single mutation path) + undo/redo, CircuitManager, Simulation
  io/         versioned .llmc project files, persistent app config
  ui/         theme, block glyphs, and the interactive canvas (binary only)
tests/
  simulation.rs   truth tables, half/full adder, 4-bit adder from chips, mux, SR latch
```

Every edit — from a mouse drag to (in future) an AI action — flows through a single
`EditCommand` batch that is atomic and invertible, which is what powers undo/redo.

The simulator flattens a circuit (expanding chip instances recursively via a union-find
over nets) into primitive gates, then iterates to a fixpoint each step.

## Roadmap

- **AI panel** (planned): a toolbar button opens a right-side panel where an AI assistant
  can build and *edit* circuits for you via a structured, validated command API, with a
  diff preview before anything is applied. Multiple providers (Groq, OpenRouter, Google
  AI Studio, Zen), model selection, and locally-stored keys.

## License

MIT — see [LICENSE](LICENSE).
