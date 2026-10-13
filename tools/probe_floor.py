"""The drop probe: Mario as a solidity oracle, column by column.

Why
---
The static parser classifies bgdat objects as ground with a tileset-local list
(GROUND_OBJS) that is measured-wrong outside the grass family, and the game's
own solid-tile map is not reachable without ARM9 RE the decomp does not cover
yet. But the game's physics is the definition of "solid": if Mario is placed
above a column and released, where he lands IS that column's surface, and a
fall to the bottom IS a pit. py-desmume exposes memory.write, so the probe is
a teleport-drop per column.

How
--
For each column c:
  1. teleport Mario to (c*16 + 8 px, row DROP_ROW) and pin him there while the
     level streams in around the moved camera (~PIN_FRAMES), zeroing vy each
     pin so he does not pre-fall onto unloaded terrain;
  2. release and let him fall, polling every 2 frames;
  3. landing = mario_A stable for STABLE_FRAMES  -> surface row = -y/16;
  4. pit = he passes PIT_ROW without landing    -> next column immediately;
  5. a real death (lives drop) reloads the savestate and retries the column.

Output: per-column measured surface rows + the parser's floor_row for the same
column, so every disagreement is a measured falsehood of the static parser.

Usage
-----
    python tools/probe_floor.py --state data/....ds1 --course course/A01_1.bin \
        --out out/floor_probe/A01_1.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from course import Course  # noqa: E402
from ram_env import MarioRamEnv  # noqa: E402
from ram_state import MARIO_BASE, OFF_A, OFF_B, OFF_TYPE, CAM_ADDRS  # noqa: E402

FIXED = 4096  # 20.12 fixed point, 1.0 px
DROP_ROW = 6  # drop from near the sky so the highest platform can catch him
PIT_ROW = 62  # below any plausible floor -> give up, call it a pit
MAX_PIN_FRAMES = 120  # frames to hold position while terrain streams in
STABLE_FRAMES = 12  # consecutive unchanged reads that count as a landing
MAX_FALL_FRAMES = 150
WALK_BAND = 3  # rows below the dominant floor that still count as walkable


def _spawn_b_of(env):
    """The state's own spawn x, read from a clean load."""
    env.reset()
    for _ in range(240):
        env.emu.cycle()
    return env.ram.poll()["mario_B"]


def _pin_test(env, base, spawn_b):
    """True if writing x authority (base+0x5C) moves the object's synced B."""
    emu = env.emu
    try:
        test = spawn_b + 32 * FIXED & 0xFFFFFFFF
        emu.memory.write_long(base + 0x5C, test)
        for _ in range(4):
            emu.cycle()
        moved = env.ram._rs32(base + OFF_B)
        return abs(moved - (spawn_b + 32 * FIXED)) < 8 * FIXED
    except Exception:
        return False


def _ensure_mario(env, spawn_b):
    """After a mid-run reset (a death), a second type-0x1C object can sit at
    the old address while the live player is re-allocated elsewhere -- reading
    spawn coordinates cannot tell them apart, so test which object's x
    authority actually moves its synced B, and pin that one.
    """
    if env.ram.mario_base is not None and \
            abs((env.ram.poll()["mario_B"] or 0) - spawn_b) < 100 * FIXED and \
            _pin_test(env, env.ram.mario_base, spawn_b):
        return
    lo, hi = 0x021C0000, 0x021E0000
    raw = bytes(env.emu.memory.read(lo, hi, 1, False))
    for p in range(0, len(raw) - 0x46, 2):
        if raw[p] == 0x1C and raw[p + 1] == 0x00:  # u16 type 0x1C at p
            cand = lo + p - OFF_TYPE
            try:
                if abs(env.ram._rs32(cand + OFF_B) - spawn_b) > 100 * FIXED:
                    continue
            except Exception:
                continue
            if _pin_test(env, cand, spawn_b):
                env.ram.mario_base = cand
                return


def probe_columns(env, course, cols):
    emu = env.emu
    spawn_b = _spawn_b_of(env)
    out = {}
    for c in cols:
        result = None
        for attempt in range(3):
            result = _drop_once(env, c, course)
            if result in ("reload", "stuck", "unloaded"):
                env.reset()
                _ensure_mario(env, spawn_b)
                continue
            break
        if isinstance(result, str):
            out[c] = {"surface": None, "note": result}
            continue
        out[c] = result
    return out


def _drop_once(env, col, course=None):
    emu = env.emu
    mb = env.ram.mario_base
    if mb is None:
        return "reload"
    x = (col * 16 + 8) * FIXED
    y = -(DROP_ROW * 16) * FIXED
    cam_x = max(0, (col * 16 - 128) * FIXED) & 0xFFFFFFFF
    # bail out of the fall before the game's kill-line can turn the landing
    # into a death: 10 rows below the expected floor is past any walkable
    # surface, and a death would break the teleport authority (measured on
    # .ds2: after one death the +0x5C write stops moving the player)
    bail_row = PIT_ROW
    if course is not None:
        near = [course.floor_row(cc) for cc in range(col - 2, col + 3)]
        floors = [f for f in near if f is not None]
        if floors:
            bail_row = min(floors) + 10

    def pin():
        # x authority is player+0x5C (the synced actor B at +0x68 follows it);
        # y authority IS the A field. The camera is dragged along so the level
        # streams in around the drop point -- without it, far columns stay
        # unloaded, the game holds the player at y=0 and nothing is measured.
        emu.memory.write_long(mb + 0x5C, x & 0xFFFFFFFF)
        emu.memory.write_long(mb + OFF_A, y & 0xFFFFFFFF)
        emu.memory.write_long(mb + 0x74, 0)  # vx
        emu.memory.write_long(mb + 0x78, 0)  # vy
        emu.memory.write_long(CAM_ADDRS[0], cam_x)
        emu.memory.write_long(CAM_ADDRS[1], cam_x)

    # stream the terrain in: hold him at the drop point while the camera moves,
    # and only release once the y pin actually engages (in unloaded space the
    # game holds the player at y=0 and the pin never takes)
    engaged = 0
    for _ in range(MAX_PIN_FRAMES):
        pin()
        emu.cycle()
        emu.cycle()
        ma = env.ram.poll()["mario_A"]
        if ma is not None and abs(ma - y) < 8 * FIXED:
            engaged += 1
            if engaged >= 3:
                break
        else:
            engaged = 0
    if engaged < 3:
        return "unloaded"

    # positioning check: the synced B must have followed +0x5C, or the
    # measurement would silently be about the spawn column
    if abs((env.ram.poll()["mario_B"] or 0) - x) > 8 * FIXED:
        return "stuck"

    stable = 0
    prev = None
    for f in range(MAX_FALL_FRAMES):
        emu.cycle()
        emu.cycle()
        st = env.ram.poll()
        ma = st["mario_A"]
        if ma is None:
            return "reload"
        cur = ma / FIXED
        if st["died"]:
            return "reload"
        if cur <= -(bail_row * 16):
            return {"surface": None, "note": "pit"}
        # death-hover: dying inside a pit pops the body up to ~the drop height
        # and holds it there through the jingle -- that is a pit, not a landing
        if stable >= 2 and abs(cur + DROP_ROW * 16) < 1.0 and prev is not None \
                and prev < -(32 * 16):
            return {"surface": None, "note": "pit"}
        if prev is not None and abs(cur - prev) < 0.01:
            stable += 1
            if stable >= STABLE_FRAMES // 2:  # polled every 2 frames
                row = round(-cur / 16.0, 2)
                if row < DROP_ROW + 2:
                    # never fell: standing on something at the drop height
                    # (a cave roof or high structure) -- real support
                    return {"surface": DROP_ROW, "note": "hover"}
                # landed, whatever the row: the row itself is the truth;
                # whether it is *walkable* is decided later, relative to the
                # course's dominant floor (a pit's inner shelf is 5+ rows
                # below it)
                return {"surface": row, "note": "landed"}
        else:
            stable = 0
        prev = cur
    return {"surface": None, "note": "timeout"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rom", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds"))
    ap.add_argument("--state", required=True)
    ap.add_argument("--course", default="course/A01_1.bin")
    ap.add_argument("--out", required=True)
    ap.add_argument("--step", type=int, default=1, help="probe every Nth column")
    ap.add_argument("--max-col", type=int, default=0, help="0 = course's own span")
    args = ap.parse_args()

    course = Course(name=args.course, rom_path=args.rom)
    xs = [r["tx"] + r["w"] for r in course.rects] + [r["tx"] for r in course.rects]
    if xs:
        lo = max(min(xs) + 1, 1)
        hi = args.max_col or max(xs)
    else:
        lo = min(course.ground_cover) + 1
        hi = args.max_col or max(course.ground_cover)
    if (course.exit_found and course.ground_coverage >= 0.30
            and int(course.flag_x_px // 16) + 1 >= lo):
        # Beyond the flagpole the clear cutscene owns Mario and the drop
        # measurement would only record the cutscene. The coverage guard
        # matters: a "flagpole" on a course whose tileset the parser
        # cannot read (C01_1: 1.6% coverage, a claimed flagpole at 272 px
        # = col 18) is a false sprite, and capping on it probed 18 columns
        # of a level whose rects run to col 280. When the parse cannot
        # validate its own flagpole, the cap is skipped; --max-col bounds
        # the run instead.
        hi = min(hi, int(course.flag_x_px // 16) + 1)
    cols = list(range(lo, hi + 1, args.step))
    print(f"probing {len(cols)} columns of {args.course} "
          f"(cols {lo}..{hi} step {args.step})", flush=True)

    env = MarioRamEnv(rom_path=args.rom, state_path=args.state, max_steps=100000)
    env.reset()
    measured = probe_columns(env, course, cols)
    env.close()

    # the course's dominant landing row is its walkable floor level; landings
    # far below it (a pit's inner shelf, e.g. row 35.5 under 1-1's row-30
    # floor) are collision, but not ground the level is walked on
    from collections import Counter
    landed_rows = [m["surface"] for m in measured.values()
                   if m.get("note") == "landed" and m["surface"] is not None]
    dominant = (Counter(round(r) for r in landed_rows).most_common(1)[0][0]
                if landed_rows else 30)

    agree = miss = extra = unmeasured = 0
    rows = []
    for c, m in sorted(measured.items(), key=lambda kv: int(kv[0])):
        fr = course.floor_row(int(c))
        surf = m["surface"]
        walk = surf is not None and surf <= dominant + WALK_BAND
        rows.append({"col": int(c), "measured": surf, "parser": fr,
                     "walkable": walk, "floor_row_dominant": dominant,
                     "note": m["note"]})
        if m["note"] in ("unloaded", "stuck", "reload"):
            unmeasured += 1  # the probe failed, not the parser
        elif not walk and fr is None:
            agree += 1  # both call it a pit (or an unreachable shelf)
        elif not walk and fr is not None:
            extra += 1  # parser claims floor where Mario falls: false floor
        elif walk and fr is None:
            miss += 1  # parser misses a floor the game has
        elif abs(surf - fr) <= 1.0:
            agree += 1
        else:
            extra += 1  # both see something, rows disagree by >1 tile
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"course": args.course, "state": args.state, "step": args.step,
                   "summary": {"agree": agree, "parser_false_floor": extra,
                               "parser_missed_floor": miss,
                               "unmeasured": unmeasured, "columns": len(rows)},
                   "columns": rows}, f, indent=1)
    print(f"agree={agree}  parser false floors={extra}  parser missed floors={miss}  "
          f"unmeasured={unmeasured}  -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
