"""Enter any course headless, bypassing the stylus-only world row.

The route that made every other world's savestates need a GUI session is
gone. What the map's A-press actually commits is not the Stage group/id
globals (writing those through the fade still loaded 1-1) but the **running
course's area index**: a u16 at 0x020980D4 (with a copy at 0x0209B59E) whose
value is the course file's position among the 191, calibrated on four
savestates against the ROM's own file order --

    A01_1 = 0, A02_1 = 3, E01_1 = 88, H01_1 = 152

-- and reproduced by sorting the course/ files by ROM id
(tools/fs_manifest.json holds the mapping). Stage::stageID/stageGroup/
stageAreaID (0x02098054/58/8C, decomp AAA.hpp) are written too, so the intro
plate reads the right world; their calibration on the same four states is
(nivel-1, mundo-1, area-1): 1-2 -> (1, 0, 0), 5-1 -> (0, 4, 0), 8-1 -> (0, 7, 0).

Mechanics: the map ignores the d-pad for its first ~30 s (4i), so the tool
settles first; then A on the current node starts CourseIn, and the course id
plus the Stage globals are re-written after **every** cycle for the whole
transition -- the map writes its own values early, and the load reads
whatever stands when the stage scene creates, so the last writer wins and
the tool must be the last writer.

The state this produces is a savestate mid-level, exactly like the GUI ones:
identify it with tools/identify_state.py (entrance match + walk-right
trace), floor-probe it with tools/probe_floor.py, then run the harness on it.

Usage
-----
    python tools/enter_course.py --course B01_1 --out data/state_2-1.ds1

--course is the ROM course id (letter = world, digits = level slot, suffix
= area -- the 4j naming rule); the area index and the Stage globals are
derived from it. --map-state selects the World 1 map to start from
(data/map_world1.ds1 by default; any map state works -- the tool overwrites
the course id regardless of which node the cursor sits on).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from ram_env import MarioRamEnv  # noqa: E402

ROM = os.path.join(ROOT, "data", "0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds")
MAP_STATE = os.path.join(ROOT, "data", "map_world1.ds1")
MANIFEST = os.path.join(ROOT, "tools", "fs_manifest.json")

AREA_ID = 0x020980D4      # u16 running course index among the 191
AREA_ID2 = 0x0209B59E      # the copy the load reads
STAGE_ID = 0x02098054      # u8 Stage::stageID    (level slot, 0-based)
STAGE_GROUP = 0x02098058   # u8 Stage::stageGroup (world, 0-based)
STAGE_AREA = 0x0209808C    # u8 Stage::stageAreaID (area, 0-based)

SETTLE_CYCLES = 1800       # the map drops inputs for its first ~30 s (4i)
HOLD_CYCLES = 1500         # the fade, scene change and intro plate


def course_area_index(course_id):
    """The course's position among the 191 course files, by ROM id order."""
    m = json.load(open(MANIFEST, encoding="utf-8"))
    courses = sorted((e for e in m
                      if e["path"].startswith("course/")
                      and e["path"].endswith(".bin")
                      and not e["path"].endswith("_bgdat.bin")),
                     key=lambda e: e["id"])
    ids = [e["path"].replace("course/", "").replace(".bin", "") for e in courses]
    if course_id not in ids:
        raise SystemExit(f"{course_id} is not one of the 191 course ids")
    return ids.index(course_id)


def stage_globals_of(course_id):
    """(stageID, stageGroup, stageAreaID) from the 4j naming rule."""
    letter, slot, area = course_id[0], course_id[1:3], course_id[4]
    world = ord(letter) - ord("A")
    return int(slot) - 1, world, int(area) - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--course", required=True, help="e.g. B01_1 (world 2, level 1, area 1)")
    ap.add_argument("--out", required=True, help="where to write the savestate")
    ap.add_argument("--map-state", default=MAP_STATE)
    args = ap.parse_args()

    area = course_area_index(args.course)
    stage_id, group, stage_area = stage_globals_of(args.course)
    print(f"{args.course}: area index {area}, stage (id={stage_id}, group={group}, "
          f"area={stage_area})", flush=True)

    env = MarioRamEnv(rom_path=ROM, state_path=args.map_state, max_steps=100000)
    if not env.has_emulator:
        raise SystemExit("emulator failed to open the map state")
    env.reset()
    emu = env.emu
    for _ in range(SETTLE_CYCLES):
        emu.cycle()

    from desmume.controls import Keys, keymask
    emu.input.keypad_add_key(keymask(Keys.KEY_A))
    for _ in range(6):
        emu.cycle()
    emu.input.keypad_rm_key(keymask(Keys.KEY_A))

    for f in range(HOLD_CYCLES):
        emu.cycle()
        emu.memory.write_short(AREA_ID, area)
        emu.memory.write_short(AREA_ID2, area)
        emu.memory.write_byte(STAGE_ID, stage_id)
        emu.memory.write_byte(STAGE_GROUP, group)
        emu.memory.write_byte(STAGE_AREA, stage_area)
        if f % 300 == 0:
            kept = emu.memory.read(AREA_ID, AREA_ID + 1, 2, False)
            print(f"  hold frame {f}: area id in RAM = {kept}", flush=True)

    emu.savestate.save_file(args.out)
    print(f"saved {out_hint(args.out)} -- identify with tools/identify_state.py", flush=True)
    env.close()


def out_hint(path):
    return path if os.path.isabs(path) else os.path.join(".", path)


if __name__ == "__main__":
    main()
