"""Reach the World 1 map headlessly, and turn any unlocked map node into a savestate.

Why this exists
---------------
Every experiment in this repo until now started from one hand-made savestate
inside World 1-1, which made "does the geometry layer generalise?" unanswerable:
there was no second course to load. The obvious routes were tried and failed --
the title screen ignores buttons and stylus, and the pause menu's "Return to map"
is greyed out from a state that was dropped into a level directly.

What works is the game's own sequence: **clearing a level returns it to the world
map by itself.** After the flagpole the level plays its clear animation, walks
Mario to the castle, fades to black and lands on the World 1 map with the next
node unlocked. No menu, no stylus, no save editing. Measured here at 540 cycles
(9 s of game time) after the flagpole; the fade-to-black frame is the marker, and
START presses are not needed -- they only stall the sequence by opening the pause
menu, which is what made an earlier attempt look frozen for 240 s. The map then
needs ~30 s of its own before it accepts the d-pad: it slides in and drops input,
so a RIGHT pressed too early is lost and A replays the level just cleared.

Usage
-----
    python tools/reach_map.py --workdir <dir with game.nds/.sav/.ds1 copies> [--node 1]

--workdir must hold a *copy* of the ROM together with its .sav and the 1-1 .ds1,
because the emulator writes the save next to whatever ROM it opens. The originals
under data/ are never opened by this tool.

Outputs, in --workdir:
    map_w1.ds1        the World 1 map, cursor on the node reached after clearing 1-1
    map_node<N>.ds1   the map after N steps east along the path
    level_node<N>.ds1 the level entered from node N  <- feed this to the harness
                      with --state and its course with --course

Which course a node holds is NOT read off the file name: the ROM's course names
do not spell out (world, level) in any order that survives inspection. Identify
the loaded course by matching the emulator's own measurements against the parser
-- entrance x (mario_B/4096 == entrances[0].x_px + 8 px) and the surface rows
along a walk-right trace. Anchors measured this way: course/A01_1.bin is the
level the HUD calls W1-1, course/A02_1.bin is W1-2.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import cv2  # noqa: E402
from desmume.controls import Keys, keymask  # noqa: E402

import eval_pinn_realtime as ev  # noqa: E402
from ram_env import MarioRamEnv  # noqa: E402
from tilemap_planner import TilemapAStar  # noqa: E402

ALL_KEYS = [Keys.KEY_A, Keys.KEY_B, Keys.KEY_X, Keys.KEY_Y, Keys.KEY_START,
            Keys.KEY_SELECT, Keys.KEY_UP, Keys.KEY_DOWN, Keys.KEY_LEFT,
            Keys.KEY_RIGHT, Keys.KEY_L, Keys.KEY_R]
# The clear sequence reached the map 420 cycles after the flagpole; 1800 is the
# cap, and the fade is what we stop on rather than a fixed count.
FADE_MEAN = 30.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True,
                    help="directory holding game.nds, game.sav and game.ds1")
    ap.add_argument("--rom-name", default="game.nds")
    ap.add_argument("--state-name", default="game.ds1")
    ap.add_argument("--node", type=int, default=1,
                    help="how many nodes east of 1-2 to enter (0 = stop on the map)")
    ap.add_argument("--shots", default="", help="directory for the PNGs")
    args = ap.parse_args()

    rom = os.path.join(args.workdir, args.rom_name)
    state = os.path.join(args.workdir, args.state_name)
    if not (os.path.exists(rom) and os.path.exists(state)):
        raise SystemExit(f"copy the ROM and the 1-1 savestate into {args.workdir} first")
    shots = args.shots or os.path.join(args.workdir, "reach_map")
    os.makedirs(shots, exist_ok=True)

    env = MarioRamEnv(rom_path=rom, state_path=state, geo=True, max_steps=2000)
    agent = ev.ReactiveAgent(planner=TilemapAStar(rom_path=rom))
    emu = env.emu
    obs, _ = env.reset()
    agent.reset()
    _, _st0 = env._observe()
    start_spawn = _st0["mario_B"] / 4096.0

    def cycles(n):
        for _ in range(n):
            emu.cycle()

    def release():
        for k in ALL_KEYS:
            emu.input.keypad_rm_key(keymask(k))

    def tap(key, hold=5, after=12):
        emu.input.keypad_add_key(keymask(key))
        cycles(hold)
        release()
        cycles(after)

    def frame():
        buf = np.array(emu.display_buffer_as_rgbx(), dtype=np.uint8)
        return buf.reshape(384, 256, 4)[:, :, :3].copy()

    def save_shot(tag):
        f = frame()
        cv2.imwrite(os.path.join(shots, tag + ".png"),
                    cv2.resize(f, (512, 768), interpolation=cv2.INTER_NEAREST))
        return f

    cleared = False
    for i in range(400):
        obs, r, done, trunc, info = env.step(int(agent.act(obs)))
        if done:
            cleared = True
            print(f"cleared 1-1 at env step {i + 1}", flush=True)
            break
    if not cleared:
        raise SystemExit("the published reactive agent did not clear 1-1; "
                         "nothing downstream is meaningful")
    release()

    # wait for the game's own fade to the map
    used = 0
    while used < 1800:
        cycles(30)
        used += 30
        g = cv2.cvtColor(frame(), cv2.COLOR_BGR2GRAY)
        if float(g.mean()) < FADE_MEAN:
            print(f"fade to map after {used} cycles ({used / 60:.1f}s)", flush=True)
            break
    # The map slides in and ignores the d-pad while it does: a RIGHT pressed too
    # early is dropped, and A then replays the level already cleared. 1800 cycles
    # (30 s) is what the first successful step east needed.
    cycles(1800)
    save_shot("00_map")
    emu.savestate.save_file(os.path.join(args.workdir, "map_w1.ds1"))

    for n in range(1, max(args.node, 0) + 1):
        tap(Keys.KEY_RIGHT, hold=6, after=18)
        cycles(600)
        save_shot(f"10_node{n}")
        emu.savestate.save_file(os.path.join(args.workdir, f"map_node{n}.ds1"))
        print(f"node {n}: plate visible in {shots}/10_node{n}.png", flush=True)

    if args.node >= 0:
        tap(Keys.KEY_A, hold=6, after=24)
        cycles(900)          # the load, the title card and the intro
        release()
        save_shot("20_entered")
        emu.savestate.save_file(os.path.join(args.workdir, "level_node%d.ds1" % max(args.node, 0)))
        o, st = env._observe()
        got = st["mario_B"] / 4096.0
        print("entered level: mario abs x=%.1f px  y=%.1f px  lives=%d" % (
            got, st["mario_A"] / 4096.0, st["lives"]), flush=True)
        if abs(got - start_spawn) < 0.5:
            print("WARNING: that is the same spawn as the level we started from, "
                  "so the map cursor never moved -- raise the wait before the first "
                  "RIGHT, or the node is locked.", file=sys.stderr, flush=True)
    env.close()


if __name__ == "__main__":
    main()
