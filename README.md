# LLMC

A fast, professional **digital-logic builder and simulator** for Windows (and Linux/macOS),
written in Rust with [egui](https://github.com/emilk/egui) — with a built-in **AI assistant**
that designs, wires, and *tests* circuits for you.

Place gates, wire them with the mouse, and watch signals flow live. Package any circuit into a
reusable chip and build bigger things from smaller verified ones — adders, registers, RAM,
whole CPUs.

## Features

- **Canvas** — infinite dotted grid, smooth zoom around the cursor, minimap, zoom controls,
  *Fit everything* (`Home`) and *Zoom to selection* (`Shift+2`).
- **Real gate symbols** — standard AND/OR/XOR/NOT shapes with inversion bubbles (or labeled
  boxes, under *View → Gates*). Live signals light wires, pins, and LEDs as they change.
- **Wiring** — drag from a port (or click a port, then the target). Curved or right-angle wires
  (*View → Wires*). An input fed by several wires sees their OR.
- **Simulation** — Run/Pause (`F5`), Step to the next clock edge (`F10`), speed control,
  reset. Event-driven and fast; fast clocks never skip an edge; oscillating circuits are
  flagged instead of freezing the app. Registers, latches, and RAM **keep their contents
  while you edit**.
- **Chips** — package the canvas or a selection as a chip (switches → input pins, LEDs → output
  pins); chips show their pin names on the canvas. Double-click a chip to edit it in its own
  tab; *Update chip* applies the change to every placed copy, and wires follow their pins even
  if you move or rename them. A chip can never contain itself.
- **Properties panel** (`Ctrl+I`) — name, color, input count, clock frequency, default level,
  position/rotation, and live input/output values for whatever is selected.
- **Editing** — box select, move, nudge with the arrow keys, rotate/flip the selection as a
  group, duplicate, cut/copy/paste (at the cursor, between tabs, even between two LLMC windows;
  chips come along), and undo/redo for *everything* — including property edits and chip
  changes.
- **Projects** — tabs, recent files, drag-and-drop `.llmc` files onto the window, crash-safe
  saving, unsaved-changes prompts, and the assistant conversation saved with the project
  (optional).
- **Light & dark themes** — one design system (Inter, JetBrains Mono, Remix icons) across the
  whole app.

## The AI assistant

Open it with **Assistant** (`Ctrl+L`) and connect a provider under *File → AI providers*: Groq,
OpenRouter, Google AI Studio, Cerebras, Mistral, Ollama Cloud, opencode Zen — or **Custom**, any
OpenAI-compatible endpoint, including keyless local servers (Ollama, LM Studio, llama.cpp).
Keys are stored in the OS keychain (Windows Credential Manager) and never leave your machine.

In **Agent** mode the assistant works like an engineer: it plans, defines small chips that are
truth-table-verified the moment they're created, composes them, then reads an automatic
*auto-check* of the real circuit after every step and keeps fixing until it matches your
request — for as long as it needs. Sequential designs are verified with scripted tests that
set inputs, pulse clocks, and let time pass. Everything it does is one undo step per reply, and
**Stop** halts it instantly.

## Building & running

Requires a recent stable Rust toolchain.

```sh
cargo run --release
```

A prebuilt Windows binary (`llmc.exe`) is produced by CI on every push — see the **Actions**
tab → latest run → **Artifacts → llmc-windows**.

## Keyboard shortcuts

Press **F1** in the app for the full list. Highlights:

| | |
| --- | --- |
| `Ctrl+N` / `Ctrl+O` / `Ctrl+S` | New / open / save (in a chip tab: update the chip) |
| `Ctrl+W`, `Ctrl+Tab` | Close tab, next tab |
| `F5` / `F10` | Run–pause / step to the next clock edge |
| `Ctrl+Z` / `Ctrl+Y` | Undo / redo |
| `Ctrl+X` / `C` / `V` / `D` | Cut / copy / paste at cursor / duplicate |
| `R`, `Shift+R`, `F` | Rotate right / left, flip (the selection as a group) |
| Arrows (`Shift` = 5) | Nudge the selection |
| `Home`, `Shift+2`, `Ctrl+0` | Fit everything, zoom to selection, 100% |
| `Space` + drag, middle/right drag | Pan |
| `Ctrl+B`, `Ctrl+I`, `Ctrl+L` | Toggle sidebar, Properties, Assistant |
| Double-click | Properties (or edit a chip) |
| Shift+click a switch | Select it without toggling |

## Architecture

A GUI-free **library** (tested headlessly on Linux and Windows CI) and a thin **binary** that
hosts the egui GUI.

```
src/
  model/      geometry, block types + port layouts, connections, circuits, chips (pin maps, cycle checks)
  backend/    EditCommand (the single, undoable mutation path) + undo/redo, CircuitManager, Simulation
  ai/         reply parsing, applying commands as one undo step, prompt/context/auto-check
  io/         versioned .llmc project files, app config, crash-safe writes
  ui/         design system, canvas, document, shell, inspector, palette, assistant (binary only)
tests/
  simulation.rs   truth tables, adders built from chips, mux, SR latch
assets/fonts/     Inter, JetBrains Mono (SIL OFL), Remix Icon subset (Apache-2.0)
```

Every edit — a drag, a property change, a chip redefinition, an AI build — is an `EditCommand`
batch that is atomic and invertible, which is what makes undo/redo complete.

The simulator flattens a circuit (expanding chips via a union-find over nets) into primitive
gates and evaluates them event-driven: only gates whose inputs changed are re-evaluated.

## License

MIT. Bundled fonts and icons keep their own licenses (see `assets/fonts/`).
