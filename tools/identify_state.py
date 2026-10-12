"""Identify which ROM course a savestate is standing in, by measurement.

The level a savestate holds is not written anywhere in the .dsX file that this
pipeline can read, and the ROM's course file names do not spell out (world,
level): 4j measured that the letter is the world and the digits are the level
slot, but no identification here rests on that rule alone -- the rule is what
gets tested, not what gets trusted. What identifies the course is the emulator,
with the two formulas calibrated on `A01_1` and re-validated on `A02_1` (4j):

    mario_B / 4096 == entrance.x_px + 8      (half a sprite, sprites are 16 px)
    mario_A / 4096 == -(entrance.y_px + 16)  (one tile row below the entrance)

--match prints every course of the 191 whose any entrance satisfies both. That
is often not unique: a dozen-plus courses share the left-edge entrance x=48, so
--trace then holds RIGHT for a few seconds and scores the surviving candidates
by the floor rows Mario actually walks on (`Course.floor_row`), counting only
grounded samples (y stable between consecutive samples while x advances). That
is how the 8-1 savestate was told apart from its entrance-tied decoy: `H01_1`
matched 4/4 grounded samples where `F07_2` managed 1/4 (4j).

The trace needs no stylus and no clear: it usually ends in a fall, because
RIGHT-only does not jump; the samples before the fall are the measurement.
Deaths read as (0, 0) in RAM and end the trace early.

Usage
-----
    python tools/identify_state.py --state "data/0479 - ....ds2" --trace \
        --shot media/nsmb_w5_1_spawn.png

Outputs the candidate list, the trace agreement score per candidate, and the
frame the state booted in (before any input) when --shot is given.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import cv2  # noqa: E402
from desmume.controls import Keys, keymask  # noqa: E402

from course import Course  # noqa: E402
from ram_env import MarioRamEnv  # noqa: E402

LEVELS_JL = os.path.join(ROOT, "out", "dataset", "levels.jsonl")


def entrance_table(rom_path):
    """(x, y) of every entrance of all parsed courses, one ROM open.

    Mirrors Course's own unpacking of Blocks[5] (20-byte pixel records) without
    paying 191 separate ndspy ROM loads.
    """
    import ndspy.rom
    rom = ndspy.rom.NintendoDSRom.fromFile(rom_path)
    names = [json.loads(l)["level"] for l in open(LEVELS_JL, encoding="utf-8")]
    table = {}
    for name in names:
        raw = bytes(rom.getFileByName(name))
        blocks = [struct.unpack_from("<II", raw, i * 8) for i in range(14)]
        off, ln = blocks[5]
        table[name] = [struct.unpack_from("<HH", raw, off + i * 20)
                       for i in range(ln // 20)]
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True, help="the savestate to identify")
    ap.add_argument("--rom", default=os.path.join(
        ROOT, "data", "0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds"))
    ap.add_argument("--trace", action="store_true",
                    help="after the entrance match, hold RIGHT and score the "
                         "candidates by the floor rows actually walked on")
    ap.add_argument("--cycles", type=int, default=900,
                    help="emulated cycles for the trace (default 900 = 15 s)")
    ap.add_argument("--shot", default="", help="PNG path for the booted frame")
    args = ap.parse_args()

    env = MarioRamEnv(rom_path=args.rom, state_path=args.state, max_steps=4000)
    if not env.has_emulator:
        raise SystemExit(f"the emulator refused to load {args.state}")
    emu = env.emu
    env.reset()

    if args.shot:
        buf = np.array(emu.display_buffer_as_rgbx(), dtype=np.uint8)
        frame = buf.reshape(384, 256, 4)[:, :, :3].copy()
        os.makedirs(os.path.dirname(args.shot) or ".", exist_ok=True)
        cv2.imwrite(args.shot, cv2.resize(frame, (512, 768),
                                          interpolation=cv2.INTER_NEAREST))
        print(f"frame: {args.shot}")

    _, st = env._observe()
    sx, sy = st["mario_B"] / 4096.0, st["mario_A"] / 4096.0
    print(f"spawn: x={sx:.1f} px  y={sy:.1f} px  lives={st['lives']}")

    table = entrance_table(args.rom)
    cands = []
    for name, ents in table.items():
        for i, (ex, ey) in enumerate(ents):
            if abs(ex + 8.0 - sx) < 1.0 and abs(-(ey + 16.0) - sy) < 1.0:
                cands.append((name, i, ex, ey))
                print(f"  entrance match: {name}  entrance[{i}] ({ex}, {ey})")
    if not cands:
        raise SystemExit("no course entrance matches this spawn")
    if len(cands) == 1:
        print(f"IDENTIFIED (unique entrance): {cands[0][0]}")
    if not args.trace:
        env.close()
        return

    names = [c[0] for c in cands]
    courses = {n: Course(name=n, rom_path=args.rom) for n in set(names)}
    emu.input.keypad_add_key(keymask(Keys.KEY_RIGHT))
    samples, prev, prev_x = [], None, None
    for _ in range(max(1, args.cycles // 30)):
        for _ in range(30):
            emu.cycle()
        _, s = env._observe()
        x, y = s["mario_B"] / 4096.0, s["mario_A"] / 4096.0
        if x == 0.0 and y == 0.0:
            print("trace ended in a death (RAM zeroed); grounded samples so far kept")
            break
        if prev is not None and abs(y - prev) < 0.5 and x > prev_x:
            samples.append((x, y))
        prev, prev_x = y, x
    emu.input.keypad_rm_key(keymask(Keys.KEY_RIGHT))

    print(f"grounded samples: {len(samples)}")
    best, best_score = None, -1
    for n in dict.fromkeys(names):
        ok = sum(1 for x, y in samples
                 if courses[n].floor_row(int(x // 16)) is not None
                 and abs(courses[n].floor_row(int(x // 16)) - (-y / 16.0)) < 1.0)
        print(f"  {n}: floor-row agreement {ok}/{len(samples)}")
        if ok > best_score:
            best, best_score = n, ok
    if best:
        print(f"IDENTIFIED (walk-right trace): {best}")
    env.close()


if __name__ == "__main__":
    main()
