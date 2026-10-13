"""Behavioral hostility oracle: touch a RAM actor type and measure the answer.

Why
---
HOSTILE_ENEMY_TYPES drives the reflex's avoidance, and the set was measured
on World 1's actor ids only -- the transfers to 5-1/8-1 die with unknown types
(172/184/312/247/...) in the death windows. Species names cannot be read: the
decomp's classes for those slots are stubs, and the community sprite lists are
not on this machine. But hostility does not need a name -- it needs the game's
answer to contact, and py-desmume can place Mario anywhere.

Three contact modes, because each can lie on its own:
  pin     hold Mario on the actor's moving position; a death or the actor
          vanishing on contact is hostile. A trampoline masked as inert -- the
          continuous pin suppresses the bounce response -- so a negative here
          is not conclusive.
  drop    release Mario from ~28 px above the actor and let physics resolve:
          death / bounce / pass-through.
  walk    far actors only link into the entity list near the camera, which
          does not follow teleports -- so walk there with the reactive agent
          first, then pin. This is how 247 was reached (it kills on the first
          contact frame).

Measured with this tool (2026-10-12):
  hostile: 172, 184, 312 (H01_1, all stomp-vanish), 247 (E01_1, kills in 1
           contact frame)
  not hostile: 180 (140 pinned frames + drops, harmless), 258 (contact frames
           without death), 291/307 (static foam at the level edge)

Usage
-----
    python tools/probe_hostility.py --state data/....ds3 --types 172 184 312
    python tools/probe_hostility.py --state data/....ds2 --types 258 \
        --mode walk --course course/E01_1.bin
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from ram_env import MarioRamEnv  # noqa: E402
from ram_state import OFF_A  # noqa: E402

ROM = os.path.join(ROOT, "data", "0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds")
FIXED = 4096


def _nearest(env, t):
    ens = [e for e in env.ram.enemies() if e["type"] == t]
    if not ens:
        return None
    return min(ens, key=lambda e: abs(e["dx"] or 1e9))


def _pin_onto(env, e, above=8.0):
    mb = env.ram.mario_base
    ex, ey = e["B"] / FIXED, e["A"] / FIXED
    env.emu.memory.write_long(mb + 0x5C, int(ex) & 0xFFFFFFFF)
    env.emu.memory.write_long(mb + OFF_A, int(ey - above) & 0xFFFFFFFF)
    env.emu.memory.write_long(mb + 0x74, 0)
    env.emu.memory.write_long(mb + 0x78, 0)


def probe_pin(env, t, frames=140):
    lives0 = env.ram.poll()["lives"]
    touched = 0
    for _ in range(frames):
        e = _nearest(env, t)
        if e is not None:
            _pin_onto(env, e)
            touched += 1
        env.emu.cycle()
        env.emu.cycle()
        st = env.ram.poll()
        if st["died"] or st["lives"] < lives0:
            return "HOSTILE (death)"
        if e is not None and _nearest(env, t) is None and touched > 2:
            return "HOSTILE (stomped/vanished on contact)"
    return f"no harmful contact ({touched} contact frames)"


def probe_drop(env, t, drops=10):
    for _ in range(drops):
        e = _nearest(env, t)
        if e is None:
            env.emu.cycle()
            env.emu.cycle()
            continue
        lives0 = env.ram.poll()["lives"]
        for _ in range(60):
            e2 = _nearest(env, t)
            if e2 is None:
                if env.ram.poll()["lives"] < lives0:
                    return "HOSTILE (death during fall)"
                return "CONTACT: actor vanished without death (stomped?)"
            _pin_onto(env, e2, above=28.0)  # follow its x, fall freely
            env.emu.cycle()
            env.emu.cycle()
            st = env.ram.poll()
            if st["died"] or st["lives"] < lives0:
                return "HOSTILE (death on contact)"
    return "no harmful contact in falls"


def probe_walk(env, t, course_name, goal_px=None, max_steps=80):
    import eval_pinn_realtime as ev
    from tilemap_planner import TilemapAStar
    from level_profile import LevelProfile
    ev.BaseReflexController.reflex_mode = "profile"
    ev.BaseReflexController.profile = LevelProfile(ROM, course_name, exit_px=goal_px)
    agent = ev.ReactiveAgent(planner=TilemapAStar(rom_path=ROM, name=course_name,
                                                  exit_known=goal_px is None))
    obs, _ = env.reset()
    agent.reset()
    for _ in range(max_steps):
        if _nearest(env, t) is not None:
            return probe_pin(env, t)
        obs, r, done, trunc, _ = env.step(int(agent.act(obs)))
        if done or trunc:
            return "never linked before the episode ended"
    return "never linked"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--types", type=int, nargs="+", required=True)
    ap.add_argument("--mode", choices=("pin", "drop", "walk"), default="pin")
    ap.add_argument("--course", default="course/A01_1.bin",
                    help="needed for --mode walk (the agent needs its planner)")
    ap.add_argument("--goal-px", type=int, default=0,
                    help="override the course goal for --mode walk on courses "
                         "whose exit is not a flagpole sprite (0 = from the ROM)")
    args = ap.parse_args()

    env = MarioRamEnv(rom_path=ROM, state_path=args.state, max_steps=100000,
                      geo=True, course_name=args.course)
    goal = args.goal_px or None
    for t in args.types:
        env.reset()
        for _ in range(120):
            env.emu.cycle()
        if args.mode == "pin":
            verdict = probe_pin(env, t)
        elif args.mode == "drop":
            verdict = probe_drop(env, t)
        else:
            verdict = probe_walk(env, t, args.course, goal_px=goal)
        print(f"RAM type {t}: {verdict}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
