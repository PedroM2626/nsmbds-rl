"""The water probe: a swim-state oracle, cell by cell.

The wall probe (tools/probe_walls.py, 4r) mapped the 6-1 stall: col 120
blocks rows 22..26, col 123 blocks rows 14..19, and the rows 20..21
corridor between them is free at every probed column. Whether that
corridor is crossable is a question the solidity oracle cannot answer,
because NSMB DS water is region-based: what matters is not the tile
under Mario's feet but whether his body is inside the water rectangle,
and the parse's claim that "rows 6..20 are wet, the corridor is not" has
never been measured. This tool measures it.

Per (col, row) cell: teleport, pin until the write engages (same write
loop as the wall probe, with a wider tolerance because the float fights
the pin inside water), release, then two phases:

  1. FLOAT (120 frames, no input): in water Mario floats up toward the
     measured row-6 hover; on dry ground his y does not move; over dry
     air he falls. dy_f >= +8 px classifies the cell WET -- this is the
     region test, the actual oracle.
  2. STROKE (one clean A press, 72 frames): measures the A response at
     the cell. Water: the fixed ~67 px stroke (4p). Dry ground: the
     standstill jump (~63 px, 4b). The dry-pocket bob: ~10 px, the
     signature measured at the stall cell in 4r.

Classification, every branch carrying its number:
  died          lives dropped during a phase
  water         float rose (region test positive); apex reported
  dry-air       float fell; the A response over open air is nothing
  dry-jump      float flat, A press leaps >= 40 px -- dry ground
  bob           float flat, A press bobs 4..40 px -- the pocket class
  no-response   float flat, A does < 4 px (note it; it was not seen yet)

The question this run answers for F01_1: is the corridor wet anywhere,
and does any column west of the wall let Mario float out of the dry
pocket up into the water region -- i.e. is the 6-1 stall a route
problem the reflex could solve, or terrain the action set cannot cross.

One process per course (a second DeSmuME init in one process
access-violates). Output: out/water_probe/<id>.json.

Usage:
    python tools/probe_water.py --state data/state_6-1.ds1 \
        --course course/F01_1.bin --cols 114:126 --rows 18:27 \
        --out out/water_probe/F01_1.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from ram_env import MarioRamEnv  # noqa: E402
from ram_state import MARIO_BASE, OFF_A, CAM_ADDRS  # noqa: E402

FIXED = 4096
PIN_FRAMES = 150        # hold position until the y-write engages
ENGAGE_TOL = 16         # px tolerance; wider than the wall probe because
                        # the water float fights the pin between write+poll
FLOAT_FRAMES = 120      # no-input phase: float / stand / fall
STROKE_FRAMES = 72      # sampling window after the single A press
FLOAT_MIN = 8.0         # px of rise that classifies a cell WET
FALL_MIN = 8.0          # px of fall that classifies a cell dry-air
JUMP_MIN = 40.0         # A response that counts as stroke/jump, not bob


def probe_cell(env, col, row):
    emu = env.emu
    mb = env.ram.mario_base
    x = (col * 16 + 8) * FIXED
    y = -(row * 16) * FIXED
    cam = int(max(0, (col * 16 - 128) * FIXED)) & 0xFFFFFFFF
    engaged = 0
    for _ in range(PIN_FRAMES):
        emu.memory.write_long(mb + 0x5C, x & 0xFFFFFFFF)
        emu.memory.write_long(mb + OFF_A, y & 0xFFFFFFFF)
        emu.memory.write_long(mb + 0x74, 0)
        emu.memory.write_long(mb + 0x78, 0)
        emu.memory.write_long(CAM_ADDRS[0], cam)
        emu.memory.write_long(CAM_ADDRS[1], cam)
        emu.cycle()
        ma = env.ram.poll()["mario_A"]
        if ma is not None and abs(ma - y) < ENGAGE_TOL * FIXED:
            engaged += 1
            if engaged >= 6:
                break
        else:
            engaged = 0
    if engaged < 6:
        return {"col": col, "row": row, "kind": "pin-never-engaged",
                "note": "write did not hold (solid, or float defeated the pin)"}

    # mario_A sign convention (wall probe + 4p): more negative = lower row;
    # a RISE on screen is a positive delta of mario_A.
    ma0 = env.ram.poll()["mario_A"] / FIXED
    lives0 = env.ram.poll()["lives"]

    # Phase 1: float -- the region test.
    for _ in range(FLOAT_FRAMES):
        emu.cycle()
    ma1 = env.ram.poll()["mario_A"] / FIXED
    lives1 = env.ram.poll()["lives"]
    if lives1 < lives0:
        return {"col": col, "row": row, "kind": "died",
                "note": "died during float phase", "y0": round(ma0, 1)}
    dy_f = ma1 - ma0

    # Phase 2: one clean A press, then sample the response.
    from desmume.controls import Keys, keymask
    emu.input.keypad_add_key(keymask(Keys.KEY_A))
    emu.cycle()
    emu.cycle()
    emu.input.keypad_rm_key(keymask(Keys.KEY_A))
    apex = 0.0
    for _ in range(STROKE_FRAMES):
        emu.cycle()
        ma = env.ram.poll()["mario_A"]
        if ma is not None:
            apex = max(apex, ma / FIXED - ma1)
    lives2 = env.ram.poll()["lives"]
    if lives2 < lives1:
        return {"col": col, "row": row, "kind": "died",
                "note": "died during stroke phase",
                "float_dy": round(dy_f, 1), "y0": round(ma0, 1)}

    if dy_f >= FLOAT_MIN:
        kind = "water"
    elif dy_f <= -FALL_MIN:
        kind = "dry-air"
    elif apex >= JUMP_MIN:
        kind = "dry-jump"
    elif apex >= 4.0:
        kind = "bob"
    else:
        kind = "no-response"

    return {"col": col, "row": row, "kind": kind,
            "y0": round(ma0, 1), "row0": round(-ma0 / 16.0, 2),
            "float_dy": round(dy_f, 1),
            "row_after_float": round(-ma1 / 16.0, 2),
            "a_apex_px": round(apex, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rom", default=os.path.join(
        ROOT, "data", "0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds"))
    ap.add_argument("--state", required=True)
    ap.add_argument("--course", required=True, help="label only; not opened")
    ap.add_argument("--cols", required=True, help="lo:hi inclusive, course columns")
    ap.add_argument("--rows", required=True, help="lo:hi inclusive, tile rows")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    c_lo, c_hi = (int(v) for v in args.cols.split(":"))
    r_lo, r_hi = (int(v) for v in args.rows.split(":"))
    cells = [(c, r) for r in range(r_lo, r_hi + 1) for c in range(c_lo, c_hi + 1)]
    print(f"probing {len(cells)} cells of {args.course} "
          f"(cols {c_lo}..{c_hi}, rows {r_lo}..{r_hi})", flush=True)

    env = MarioRamEnv(rom_path=args.rom, state_path=args.state, max_steps=1000000)
    if not env.has_emulator:
        raise SystemExit("emulator failed to open")
    env.reset()

    marks = {"water": "~", "dry-jump": "J", "dry-air": "v", "bob": "b",
             "no-response": ".", "died": "X", "pin-never-engaged": "?"}
    out = []
    grid = {}
    # reset the savestate between cells so state never leaks across cells
    for (c, r) in cells:
        env.reset()
        res = probe_cell(env, c, r)
        out.append(res)
        grid[(r, c)] = marks.get(res["kind"], "?")
        detail = res.get("note") or (
            f"float {res['float_dy']:+.0f}px -> row {res['row_after_float']}, "
            f"A apex {res['a_apex_px']:.0f}px")
        print(f"  col {c:3d} row {r:2d} {marks.get(res['kind'], '?')} {detail}", flush=True)
    env.close()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"course": args.course, "state": args.state,
                   "cols": [c_lo, c_hi], "rows": [r_lo, r_hi],
                   "cells": out}, f, indent=1)

    kinds = {}
    for cell in out:
        kinds[cell["kind"]] = kinds.get(cell["kind"], 0) + 1
    print("\ngrid (rows top=low row number; ~ water, J dry-jump, v dry-air,"
          " b bob, . none, X died, ? pin):", flush=True)
    for r in range(r_lo, r_hi + 1):
        print(f"  row {r:2d} " + "".join(grid.get((r, c), " ") for c in range(c_lo, c_hi + 1)), flush=True)
    print(f"{kinds} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
