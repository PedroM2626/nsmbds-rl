"""The contact oracle: where around an enemy is the stomp zone, and where
is the damage zone.

Why
---
The 4v traces put both remaining pocket deaths on enemy contact: 2-1's
Pokey sits below the +-55 px threat window and hits Mario at the frozen
ledge edge; 4-1's Koopa body is the wall itself, and the reflex's answer
-- leap at a close hostile -- bounces vertically against it for ten
steps until contact kills. The tenth candidate (a point-blank counter
-> retreat-then-leap) was vetoed by parity: the PUBLISHED 1-1 clear
contains its own 16-step point-blank run (steps 219-234, a goomba at
dx ~15) that it SURVIVED -- presumably by resolving, likely stomping --
longer than 4-1's fatal bounce. No counter separates them. What would
separate them is the contact geometry itself: did the 1-1 bounces land
on the goomba's head while the 4-1 arcs pass beside the Koopa? This
tool measures exactly that, per type:

Per (dx, dy) offset: reset, warm the entity list, find the nearest
actor of the type, teleport Mario to enemy_x + dx / enemy_y - dy (dy
positive = above), pin until the write takes, then RELEASE and let
physics resolve a pure vertical fall, polling every frame. The verdict
per cell:

  stomp         the actor vanished or changed type (Koopa -> shell)
                and Mario lives
  damage-death  Mario's lives dropped
  damage-hit    a hit-hop vy spike with the actor unchanged
                (Mario is Super in these savestates: two hits to die)
  miss          the fall resolved with no interaction

The actor walks during the fall (~15-30 px), deterministically, so
every cell also records the CONTACT-frame dx -- the true geometry the
classification should be read in.

Usage:
    python tools/probe_contact.py --state data/....ds1 --type 160 \\
        --dxs -32:32:8 --dys 16:48:16 --out out/contact_probe/A01_1_160.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from ram_env import MarioRamEnv, YVEL_ADDR  # noqa: E402
from ram_state import OFF_A, MARIO_BASE  # noqa: E402

ROM = os.path.join(ROOT, "data", "0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds")
FIXED = 4096
PIN_FRAMES = 12       # hold the teleport write until it takes
FALL_FRAMES = 48      # release and let the vertical fall resolve
HIT_HOP_VY = 0.4      # px/frame: the damage/stomp bounce threshold


def _nearest(env, t):
    ens = [e for e in env.ram.enemies() if e["type"] == t]
    if not ens:
        return None
    return min(ens, key=lambda e: abs(e["dx"] or 1e9))


def _walk_until_linked(env, t, mode="plain", course_name="course/A01_1.bin",
                       goal_px=None, max_cycles=1200, max_steps=90):
    """The entity list links actors near the CAMERA, which does not follow
    teleports (the hostility oracle's note) -- so drive east until the
    target type links, then freeze that moment as the working savestate
    every cell reloads.

    plain: hold RIGHT and tap A every ~30 cycles -- enough on flat starts
           (1-1). 4-1's opening step pins a plain walker at x 105.
    agent: drive the reactive reflex itself -- the proven walker (it
           reaches 4-1's Koopa in 43 steps); stop at the FIRST link,
           well before the contact the frontier dies on."""
    from desmume.controls import Keys, keymask
    emu = env.emu
    if mode == "plain":
        emu.input.keypad_add_key(keymask(Keys.KEY_RIGHT))
        try:
            for _ in range(max_cycles):
                emu.cycle()
                if _ % 16 == 0 and _nearest(env, t) is not None:
                    return True
        finally:
            emu.input.keypad_rm_key(keymask(Keys.KEY_RIGHT))
        return _nearest(env, t) is not None
    # agent walk: stop only at a CLEAN link -- the target type linked AND
    # no other hostile within 120 px (the first-link moment on D01_1 sits
    # ~100 px west of the Koopa with a trailing hostile in the area, and
    # every cell then measured that other enemy's hit instead: damage at
    # contact dx -50..-116. The in-run death trace shows the followers
    # (135/311) behind Mario by the contact, so a clean frame exists.)
    import eval_pinn_realtime as ev
    from tilemap_planner import TilemapAStar
    agent = ev.ReactiveAgent(planner=TilemapAStar(
        rom_path=ROM, name=course_name, exit_known=goal_px is None))
    agent.reset()
    obs, _ = env._observe()
    linked_at = None
    for _ in range(max_steps):
        obs, r, done, trunc, info = env.step(int(agent.act(obs)))
        if done or trunc:
            break
        if _nearest(env, t) is not None:
            if linked_at is None:
                linked_at = _
            others = [e for e in env.ram.enemies()
                      if e["type"] != t and ev.is_hostile_enemy(e["type"])
                      and abs(e["dx"] or 1e9) <= 120.0]
            if not others:
                return True
    return _nearest(env, t) is not None


def probe_cell(env, t, dx, dy):
    """One (dx, dy) fall. Returns the verdict dict."""
    emu = env.emu
    mb = env.ram.mario_base
    e0 = _nearest(env, t)
    if e0 is None:
        return {"dx": dx, "dy": dy, "kind": "no-actor"}
    ex, ey = e0["B"] / FIXED, e0["A"] / FIXED
    # mario_A convention (4o/4t): MORE NEGATIVE = LOWER on screen, so
    # "dy above the actor" is ey + dy. The first version wrote ey - dy
    # and teleported Mario INTO the ground under the actor -- every
    # dy-16 cell ejected at the pin and every dy-32/48 cell fell through
    # the floor to a sub-floor five rows down (the "miss" rows).
    mx, my = int((ex + dx) * FIXED), int((ey + dy) * FIXED)
    lives0 = env.ram.poll()["lives"]

    # pin: hold the write until it engages (same loop as the wall probe)
    engaged = 0
    for _ in range(PIN_FRAMES * 8):
        emu.memory.write_long(mb + 0x5C, mx & 0xFFFFFFFF)
        emu.memory.write_long(mb + OFF_A, my & 0xFFFFFFFF)
        emu.memory.write_long(mb + 0x74, 0)
        emu.memory.write_long(mb + 0x78, 0)
        emu.cycle()
        ma = env.ram.poll()["mario_A"]
        if ma is not None and abs(ma - my) < 8 * FIXED:
            engaged += 1
            if engaged >= 4:
                break
        else:
            engaged = 0
    if engaged < 4:
        return {"dx": dx, "dy": dy, "kind": "pin-never-engaged"}

    # release: pure vertical fall, poll every frame
    last_dx = None
    vy_spiked = False
    for _ in range(FALL_FRAMES):
        emu.cycle()
        st = env.ram.poll()
        vy = env._s32read(YVEL_ADDR) / 4096.0 / 4.0
        if vy >= HIT_HOP_VY:
            vy_spiked = True
        e = _nearest(env, t)
        if e is not None and abs(e["B"] / FIXED - ex) < 120.0:
            last_dx = round(e["B"] / FIXED - (st["mario_B"] or 0) / FIXED, 1)
        if st["died"] or st["lives"] < lives0:
            return {"dx": dx, "dy": dy, "kind": "damage-death",
                    "contact_dx": last_dx}
        if e is None:
            # the actor vanished while Mario lives: the stomp signature
            return {"dx": dx, "dy": dy, "kind": "stomp", "contact_dx": last_dx}
        if e["type"] != t:
            # type change (Koopa -> shell): same signature
            return {"dx": dx, "dy": dy, "kind": "stomp",
                    "contact_dx": last_dx, "after_type": e["type"]}
        # A Pokey does not vanish on a stomp -- its TOP SEGMENT pops and
        # the entity's center drops with it. An A shift while a vy spike
        # happened is the segment-pop stomp, not a damage hit.
        if vy_spiked and abs(e["A"] / FIXED - ey) >= 8.0:
            return {"dx": dx, "dy": dy, "kind": "stomp (segment pop)",
                    "contact_dx": last_dx}
    if vy_spiked:
        return {"dx": dx, "dy": dy, "kind": "damage-hit", "contact_dx": last_dx}
    return {"dx": dx, "dy": dy, "kind": "miss", "contact_dx": last_dx}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--type", type=int, required=True)
    ap.add_argument("--dxs", default="-32:32:8",
                    help="lo:hi:step -- horizontal offsets from the actor, px")
    ap.add_argument("--dys", default="16:48:16",
                    help="lo:hi:step -- fall heights above the actor, px")
    ap.add_argument("--course", default="course/A01_1.bin")
    ap.add_argument("--goal-px", type=int, default=0,
                    help="course goal override for --walk agent on courses "
                         "whose exit is not a flagpole sprite (0 = from the ROM)")
    ap.add_argument("--walk", choices=("plain", "agent"), default="plain",
                    help="pre-phase driver that brings the target into the "
                         "entity list (agent = the reactive reflex, for "
                         "courses whose opening pins a plain walker)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dxs = parse_range(args.dxs)
    dys = parse_range(args.dys)
    cells = [(dx, dy) for dy in dys for dx in dxs]
    print(f"contact-probing type {args.type}: {len(cells)} cells "
          f"(dx {min(dxs)}..{max(dxs)}, dy {min(dys)}..{max(dys)})", flush=True)

    env = MarioRamEnv(rom_path=ROM, state_path=args.state, max_steps=1000000,
                      geo=True, course_name=args.course)
    if not env.has_emulator:
        raise SystemExit("emulator failed to open")

    # working state: warm, walk east until the type links, save that moment
    env.reset()
    for _ in range(120):
        env.emu.cycle()
    if _nearest(env, args.type) is None:
        ok = _walk_until_linked(env, args.type, mode=args.walk,
                                course_name=args.course,
                                goal_px=(args.goal_px or None))
        if not ok:
            env.close()
            raise SystemExit(f"type {args.type} never linked into the entity list")
    work_state = os.path.join(os.path.dirname(args.out) or ".",
                              "_workstate.ds1")
    os.makedirs(os.path.dirname(work_state) or ".", exist_ok=True)
    env.emu.savestate.save_file(work_state)
    print(f"working state saved (type {args.type} linked at "
          f"{_nearest(env, args.type)['B'] / FIXED:.0f} px)", flush=True)

    out = []
    marks = {"stomp": "S", "stomp (segment pop)": "S", "damage-death": "X",
             "damage-hit": "!", "miss": ".", "no-actor": "?",
             "pin-never-engaged": "?"}
    for dy in dys:
        for dx in dxs:
            env.emu.savestate.load_file(work_state)
            for _ in range(16):         # settle cycles after the load
                env.emu.cycle()
            res = probe_cell(env, t=args.type, dx=dx, dy=dy)
            out.append(res)
            print(f"  dy {dy:3d} dx {dx:+4d} {marks.get(res['kind'], '?')} "
                  f"{res['kind']}"
                  + (f" (contact dx {res.get('contact_dx')})" if res.get("contact_dx") is not None else ""),
                  flush=True)
    env.close()
    try:
        os.remove(work_state)
    except OSError:
        pass

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    kinds = {}
    for c in out:
        kinds[c["kind"]] = kinds.get(c["kind"], 0) + 1
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"state": args.state, "type": args.type,
                   "dxs": [min(dxs), max(dxs)], "dys": [min(dys), max(dys)],
                   "kinds": kinds, "cells": out}, f, indent=1)
    print(f"{kinds} -> {args.out}", flush=True)


def parse_range(spec):
    lo, hi, step = (int(v) for v in spec.split(":"))
    return list(range(lo, hi + 1, step))


if __name__ == "__main__":
    main()
