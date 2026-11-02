"""Savestates rebuilt from the ROM alone: a scripted cold boot (PROJECT_STATUS 4ba).

The repository used to require hand-made DeSmuME savestates that cannot be
distributed. A headless boot of the bare ROM -- no savestate, no battery save
-- reaches every screen the game needs when it is driven with the right
inputs. This module holds those input scripts and replays them.

Two traps, both measured:
  * DeSmuME starts with keys held: without releasing every key first, the file
    select picks Luigi (the L+R gesture) instead of Mario.
  * "Select a Game" lists "Mario Vs. Luigi" second; that mode waits forever on
    "Searching for Luigi..." (the stall 4g attributed to headless rendering).

The ROM is copied into a fresh temporary directory first, so DeSmuME finds no
battery save next to it and every boot starts from three "New" files.

Generation needs its own emulator, and only one DeSmuME may exist per process
(AGENTS.md), so callers that already own an emulator -- or will -- must use
ensure_state(), which runs tools/make_states.py in a subprocess.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time

import paths

ROM_SHA1 = "b0f7886720e842ed922569e1bb5c39c0fa090331"

# One script per state. Steps (frame counts are emulator cycles):
#   ("wait", n)                 run n frames
#   ("key", NAME)               hold a key 5 frames, release, run 12
#   ("touch", x, y)             tap the bottom screen (6 held, 20 after)
#   ("poke", addr, size, v)     write a u8/u16/u32 into main RAM
#   ("hold", n, writes)         run n frames, re-writing [(addr, size, v)] after
#                               every frame (enter_course.py's last-writer rule)
#   ("save",)                   write the state
#   ("series", n, writes)       calibration only: save each of the next n
#                               frames as f<k>.ds1, holding `writes`
_TO_WORLD1_MAP = [
    ("wait", 1200),           # boot animations end on "Select a Game"
    ("touch", 130, 55),       # "Mario Game" (not "Mario Vs. Luigi")
    ("wait", 150),
    ("key", "A"),             # File 1 (New)
    ("wait", 150),
    ("key", "A"),             # confirm; the opening cutscene plays
    ("wait", 3300),           # ... and hands over to the World 1 map
    ("key", "RIGHT"),         # start node -> 1-1 node
    ("wait", 150),
]
# The global frame counter, kept at three addresses (4ba). The reference
# states were made after longer boot histories, so their counter is ahead of
# a cold boot's; policies that read the RAM (and the background animation)
# follow it, so each course writes the reference value back.
FRAME_COUNTER = (0x0209AE9C, 0x0209AF40, 0x0209DF00)


def _counter(value):
    return [("poke", a, 4, value) for a in FRAME_COUNTER]


AREA_ID, AREA_ID2 = 0x020980D4, 0x0209B59E          # enter_course.py
STAGE_ID, STAGE_GROUP, STAGE_AREA = 0x02098054, 0x02098058, 0x0209808C


def course_writes(area_index, stage_id, group, stage_area):
    """The RAM writes that make the map's A-press load another course (4m)."""
    return [(AREA_ID, 2, area_index), (AREA_ID2, 2, area_index), (STAGE_ID, 1, stage_id),
            (STAGE_GROUP, 1, group), (STAGE_AREA, 1, stage_area)]


COURSES = {
    "1-1": {
        "out": paths.STATE_REL,
        "script": _TO_WORLD1_MAP + [
            ("key", "A"),         # enter 1-1
            # 342 frames after entry the stage timer equals the reference
            # state's to the last bit (obs[6] identical, 4ba); with the frame
            # counter written as well, the genetic 206-step clear and the RAM
            # arms' sampled episodes reproduce exactly (4ba).
            ("wait", 342),
            *_counter(11856),
            ("save",),
        ],
    },
}

LIVES = 0x0209DC00


def _entered(area_index, stage_id, group, stage_area, frame, counter, extra=(), progress=()):
    """Map A-press with the course-id writes held (tools/enter_course.py), saved
    `frame` cycles later. `counter`, `extra` (held through the transition) and
    `progress` (written once before saving) are the reference state's values,
    read by tools/calibrate_state.py (4ba)."""
    writes = course_writes(area_index, stage_id, group, stage_area) + list(extra)
    return _TO_WORLD1_MAP + [("key", "A"), ("hold", frame, writes), *_counter(counter),
                             *[("poke", *w) for w in extra],
                             *[("poke", addr, 4, v) for addr, v in progress], ("save",)]


# Player progress the reference states carried (coins, score and their copies,
# and three course-record words): the map-made states inherit a 1-1 clear
# (19 coins, 21000 points), the 5-1/8-1 GUI states a completed save. Values
# read from the references by tools/calibrate_state.py (4ba).
_FROM_11_CLEAR = ((0x02098134, 19), (0x02098138, 21000), (0x0209813C, 21000),
                  (0x0209DC18, 19), (0x0209DC20, 21000), (0x0209B250, 19),
                  (0x0209B254, 21000), (0x0209B274, 21000), (0x020980E0, 360))
_COMPLETED_SAVE = ((0x02098134, 99), (0x02098138, 1331960), (0x0209813C, 31960),
                   (0x0209DC18, 99), (0x0209DC20, 1331960), (0x0209B250, 99),
                   (0x0209B254, 1331960), (0x0209B274, 31960))


# course: (canonical file, area index, stage globals, save frame, counter, held writes,
#          progress writes).
# Frames are where the reset observation equals the reference's exactly
# (tools/calibrate_state.py); the enter_course.py states all sit at 1489. The
# 5-1 and 8-1 references were saved during the intro plate with 99 lives, where
# 75 frames share one observation; 280 (5-1) and 287 (8-1) are the frames whose
# replay of the published controller run matched it (4ba).
_ENTERED = {
    "1-2": ("data/state_1-2.ds1", 3, (1, 0, 0), 2409, 20338, (),
            _FROM_11_CLEAR + ((0x020980DC, 2), (0x0209B268, 2), (0x0209B26C, 2))),
    "2-1": ("data/state_2-1.ds1", 21, (0, 1, 0), 1489, 18584, (), _FROM_11_CLEAR),
    "3-1": ("data/state_3-1.ds1", 44, (0, 2, 0), 1489, 18585, (), _FROM_11_CLEAR),
    "3-1_main": ("data/state_3-1_main.ds1", 45, (0, 2, 1), 1489, 18575, (), _FROM_11_CLEAR),
    "4-1": ("data/state_4-1.ds1", 66, (0, 3, 0), 1489, 18577, (), _FROM_11_CLEAR),
    "5-1": ("data/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).ds2", 88, (0, 4, 0),
            280, 1977, ((LIVES, 1, 99),), _COMPLETED_SAVE),
    "6-1": ("data/state_6-1.ds1", 107, (0, 5, 0), 1489, 18577, (), _FROM_11_CLEAR),
    "7-1": ("data/state_7-1.ds1", 130, (0, 6, 0), 1489, 18581, (), _FROM_11_CLEAR),
    "8-1": ("data/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).ds3", 152, (0, 7, 0),
            287, 3523, ((LIVES, 1, 99),), _COMPLETED_SAVE),
}
for _c, (_out, _area, _sg, _frame, _ctr, _extra, _prog) in _ENTERED.items():
    COURSES[_c] = {"out": _out, "script": _entered(_area, *_sg, _frame, _ctr, _extra, _prog)}


def state_course(state_path):
    """The course a canonical state path belongs to, or None."""
    rel = paths.display(paths.resolve(state_path))
    for course, spec in COURSES.items():
        if rel == spec["out"]:
            return course
    return None


def run_script(steps, out_path=None, rom_path=None, series_dir=None):
    """Boot the ROM in this process and execute `steps` (see the table above)."""
    from desmume.emulator import DeSmuME
    from desmume.controls import Keys, keymask

    rom_path = paths.resolve(rom_path or paths.ROM_REL)
    work = tempfile.mkdtemp(prefix="nsmb_coldboot_")
    try:
        rom = os.path.join(work, "game.nds")   # no game.sav beside it: a fresh cartridge
        shutil.copyfile(rom_path, rom)
        emu = DeSmuME()
        emu.open(rom)
        for name in dir(Keys):       # DeSmuME starts with keys held (4ba)
            if name.startswith("KEY_") and name != "KEY_NONE":
                emu.input.keypad_rm_key(keymask(getattr(Keys, name)))

        def poke(addr, size, value):
            for i in range(size):
                emu.memory.unsigned[addr + i] = (value >> (8 * i)) & 0xFF

        def run(n, writes=()):
            for _ in range(n):
                emu.cycle()
                for w in writes:
                    poke(*w)

        def save(path):
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            emu.savestate.save_file(path + ".tmp")
            os.replace(path + ".tmp", path)

        for step in steps:
            kind = step[0]
            if kind == "wait":
                run(step[1])
            elif kind == "key":
                k = keymask(getattr(Keys, "KEY_" + step[1]))
                emu.input.keypad_add_key(k)
                run(5)
                emu.input.keypad_rm_key(k)
                run(12)
            elif kind == "touch":
                emu.input.touch_set_pos(step[1], step[2])
                run(6)
                emu.input.touch_release()
                run(20)
            elif kind == "poke":
                poke(*step[1:])
            elif kind == "hold":
                run(step[1], step[2])
            elif kind == "save":
                save(out_path)
            elif kind == "series":
                for k in range(step[1]):
                    save(os.path.join(series_dir, f"f{k}.ds1"))
                    run(1, step[2])
            else:
                raise ValueError(f"unknown step {step!r}")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return out_path


def generate(course, out_path, rom_path=None):
    """Boot the ROM (in this process) and save the course's state to out_path."""
    return run_script(COURSES[course]["script"], out_path, rom_path)


def ensure_state(state_path, rom_path=None, timeout_s=900):
    """Make sure a canonical savestate exists, building it from the ROM if not.

    Runs tools/make_states.py in a subprocess (one DeSmuME per process). A lock
    file serialises parallel callers such as SubprocVecEnv workers. Paths that
    are not a known course are left alone; the caller's own missing-file error
    then applies.
    """
    state_path = paths.resolve(state_path)
    if os.path.exists(state_path):
        return state_path
    course = state_course(state_path)
    if course is None:
        return state_path
    lock = state_path + ".lock"
    deadline = time.time() + timeout_s
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            if os.path.exists(state_path) or time.time() > deadline:
                return state_path
            time.sleep(2)
    try:
        if not os.path.exists(state_path):
            print(f"[cold_boot] {paths.display(state_path)} missing: building it from the ROM "
                  f"(tools/make_states.py --course {course})", file=sys.stderr, flush=True)
            cmd = [sys.executable, os.path.join(paths.ROOT, "tools", "make_states.py"),
                   "--course", course]
            if rom_path:
                cmd += ["--rom", paths.resolve(rom_path)]
            subprocess.run(cmd, check=True, cwd=paths.ROOT)
    finally:
        os.close(fd)
        os.remove(lock)
    return state_path
