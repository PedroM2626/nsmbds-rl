# nsmb_planner — Rust geometry planner

`src/main.rs` ports the four tilemap queries the real-time controller actually
calls (`pipe_ahead`, `surface_ahead`, `wall_ahead`, `is_narrow_corridor`) out of
`src/tilemap_planner.py`, and serves them to Python over a line protocol.

## Build (rustc only, no cargo)

```bash
bash build.sh          # rustc -O src/main.rs -o nsmb_planner.exe
```

Smart App Control / Device Guard on the development machine blocks `cargo.exe`
(error 4551, in every install location, unaffected by `Unblock-File`), so this
crate deliberately uses **no dependencies outside `std`** and is compiled
directly with `rustc`. Locally built binaries run fine. Do not add crates.io
dependencies unless cargo becomes available.

## Regenerate the geometry, then check parity

```bash
python ../tools/export_geometry.py    # ROM -> out/nsmb_geometry.bin
python ../tools/parity_planner.py     # every query must match tilemap_planner.py
```

Parity is the gate: `--planner rust` in `src/eval_pinn_realtime.py` is only
meaningful while the closed-loop trajectory stays identical. The current legacy reflex contract is 245 steps and total reward
1663.087402343757 against `evals/pinn_realtime_eval.json`;
`tools/parity_reflex.py` checks that reproduction separately from the planner
query parity gate. The earlier 220-step measurement was a historical run.

## What this is NOT

Not a speedup, measured on this machine:

| | per controller step |
| --- | --- |
| `tilemap_planner.py` in-process | 0.002 ms |
| `nsmb_planner.exe` over the pipe | 0.075 ms |
| whole step (emulator included) | ~43 ms |

The queries are dictionary lookups; moving them out of process only adds IPC.
The controller is ~0.01% of wall clock and DeSmuME is ~99.99%, so this port
buys a verified second implementation and a place to put real search, not FPS.
Rust starts to pay if the planner becomes an actual search over the tilemap
(the "full solid-tile 2D platformer planner" named in the README) rather than
the per-column lookups it is today.
