"""The wall probe: a 2D solidity oracle, cell by cell.

The drop probe (tools/probe_floor.py) answers "where can Mario land?" This
tool answers "can Mario's body occupy this cell and move east through it?": teleport
to each (col, row) of a target region, pin until the write engages,
release, hold RIGHT for ~1.5 s of game time, and measure the x advance.

A frozen x means the cell is solid to Mario at that height -- sprite
geometry, tileset-local classes, whatever the parser misses. A large x
advance means the cell is free even if the parser says nothing about it.
The 2-1 and 4-1 "sprite-solid pocket" hypotheses were retracted by the
measured wall maps (PROJECT_STATUS 4r/4v); their frontiers are enemy/contact
problems. The 6-1 wall at cols 120/123 is confirmed. Keep this oracle general
and use it to measure a named region, not to presume a blocker is geometric.
A fall (y dropping away) does not invalidate the x reading: the 4i trace
measured x frozen even airborne with right held.

One process per course (a second DeSmuME init in one process
access-violates). Output: out/wall_probe/<id>.json.

Usage:
    python tools/probe_walls.py --state data/state_2-1.ds1 \
        --course course/B01_1.bin --cols 35:43 --rows 17:28
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
ENGAGE_TOL = 8          # px tolerance for "the pin took"
HOLD_CYCLES = 90         # 1.5 s of game time holding RIGHT
ADVANCE_FREE = 4.0       # px of x advance that counts as "free"


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
        emu.cycle()
        ma = env.ram.poll()["mario_A"]
        if ma is not None and abs(ma - y) < ENGAGE_TOL * FIXED:
            engaged += 1
            if engaged >= 6:
                break
        else:
            engaged = 0
    if engaged < 6:
        return {"col": col, "row": row, "note": "pin-never-engaged (solid?)"}

    st0 = env.ram.poll()
    x0 = (st0["mario_B"] or 0) / FIXED
    lives0 = st0["lives"]
    from desmume.controls import Keys, keymask
    emu.input.keypad_add_key(keymask(Keys.KEY_RIGHT))
    fell = False
    for _ in range(HOLD_CYCLES):
        emu.cycle()
    emu.input.keypad_rm_key(keymask(Keys.KEY_RIGHT))
    for _ in range(10):
        emu.cycle()
    st1 = env.ram.poll()
    x1 = (st1["mario_B"] or 0) / FIXED
    y1 = (st1["mario_A"] or 0) / FIXED
    if st1["died"] or st1["lives"] < lives0:
        return {"col": col, "row": row, "note": "died during hold",
                "x0": round(x0, 1)}
    fell = y1 < y0 - 24 if False else (y1 < (env.ram.poll()["mario_A"] or 0) / FIXED)
    return {"col": col, "row": row,
            "x0": round(x0, 1), "x1": round(x1, 1),
            "advance": round(x1 - x0, 1), "y": round(y1, 1),
            "blocked": (x1 - x0) < ADVANCE_FREE}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rom", default=os.path.join(
        ROOT, "data", "0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds"))
    ap.add_argument("--state", required=True)
    ap.add_argument("--course", required=True)
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

    out = []
    # reset the savestate between cells so state never leaks across cells
    for (c, r) in cells:
        env.reset()
        res = probe_cell(env, c, r)
        out.append(res)
        mark = "#" if res.get("blocked") else ("." if "advance" in res else "?")
        detail = res.get("note") or ("advance %spx" % res["advance"])
        print(f"  col {c:3d} row {r:2d} {mark} {detail}", flush=True)
    env.close()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"course": args.course, "state": args.state,
                   "cols": [c_lo, c_hi], "rows": [r_lo, r_hi],
                   "cells": out}, f, indent=1)
    blocked = sum(1 for c in out if c.get("blocked"))
    print(f"{blocked}/{len(out)} cells block eastward movement -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
