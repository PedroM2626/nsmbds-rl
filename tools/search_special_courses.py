"""Search real emulator action routes through 3-1/7-1, including pipe inputs.

This is an emulator-assisted search, not a learned policy or PINN rollout.
One native emulator is reused; each branch restores both its savestate and
Python-side observation/reward bookkeeping. No teleport writes are used.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eval_provenance import collect
from multi_area_env import MultiAreaMarioEnv

CONFIG = {
    "3-1": ("C01_1", "data/state_3-1.ds1"),
    "7-1": ("G01_1", "data/state_7-1.ds1"),
}
ALLOWED_COURSES = {"3-1": {"C01_1", "C01_2"}, "7-1": {"G01_1"}}
WATER = {"course/C01_2.bin": True, "course/G01_1.bin": False}


def route_score(world, info):
    x, y = info["mario_px"], info["mario_y_px"]
    if x is None or y is None:
        return -float("inf")
    if world == "3-1":
        if info["course"] == "course/C01_2.bin":
            return 1000.0 + x
        if info["course"] == "course/C01_1.bin" and x < 600:
            return 6000.0 + min(x, 272.0)
        return x
    # The second, lower route is reached through a pipe in the same course.
    if y < -950:
        return 10000.0 + min(x, 2848.0) - abs(x - 2848.0) * 0.1
    if y < -790 and x < 1800:
        return -float("inf")
    return min(x, 2800.0) + 0.15 * y


def candidate_actions(world, info):
    if world == "3-1":
        if info["course"] == "course/C01_2.bin":
            return (0, 1, 2, 5, 7, 9, 10) if info["mario_px"] > 3600 else (1, 2, 4, 7)
        return (1, 3, 4) if info["mario_px"] < 600 else (1, 2, 3, 4, 8)
    return (0, 2, 3, 4, 5, 9)


class Branches:
    def __init__(self, env, folder):
        self.env, self.folder, self.serial = env, folder, 0

    def save(self, actions, obs, info):
        self.serial += 1
        path = Path(self.folder) / f"{self.serial}.ds1"
        self.env.emu.savestate.save_file(str(path))
        excluded = {"emu", "ram", "course", "_courses", "action_space", "observation_space"}
        py = {k: copy.deepcopy(v) for k, v in self.env.__dict__.items() if k not in excluded}
        ram = {k: copy.deepcopy(v) for k, v in self.env.ram.__dict__.items() if k != "emu"}
        return dict(path=path, py=py, ram=ram, actions=list(actions),
                    obs=obs.copy(), info=dict(info))

    def load(self, node):
        self.env.emu.savestate.load_file(str(node["path"]))
        self.env.__dict__.update(copy.deepcopy(node["py"]))
        self.env.ram.__dict__.update(copy.deepcopy(node["ram"]))
        self.env.course = self.env._courses[self.env.active_course]
        self.env._apply_water_override()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--world", choices=CONFIG, required=True)
    ap.add_argument("--width", type=int, default=6)
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--macro", type=int, default=8)
    ap.add_argument("--prefix", help="JSON containing an actions list, replayed from the initial state")
    ap.add_argument("--state", help="Optional explicit savestate; must match --course")
    ap.add_argument("--course", help="Explicit ROM course ID when using another starting area")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if min(args.width, args.rounds, args.macro) < 1:
        ap.error("width, rounds and macro must be positive")
    cid, state = CONFIG[args.world]
    cid, state = args.course or cid, args.state or state
    if cid not in ALLOWED_COURSES[args.world]:
        ap.error("Starting course does not belong to the requested world/level")
    state = str(ROOT / state)
    env = MultiAreaMarioEnv(state_path=state, course_name=f"course/{cid}.bin",
                            max_steps=10000, water_overrides=WATER)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    provenance = collect(env.rom_path, state, sources=[
        __file__, ROOT / "src/multi_area_env.py", ROOT / "src/ram_env.py"],
        extra={"seed": 0, "world": args.world})
    best = -float("inf")
    started = time.monotonic()
    try:
        obs, info = env.reset(seed=0)
        actions = []
        if args.prefix:
            actions = json.loads(Path(args.prefix).read_text())["actions"]
            for action in actions:
                obs, _, done, trunc, info = env.step(action)
                if done or trunc:
                    raise ValueError("Prefix terminates before search starts")
        else:
            # Populate absolute telemetry without advancing the game.
            st = env.ram.poll()
            info.update(mario_px=st["mario_B"] / 4096.0,
                        mario_y_px=st["mario_A"] / 4096.0)
        with tempfile.TemporaryDirectory(prefix="nsmb-route-") as folder:
            branches = Branches(env, folder)
            beam = [branches.save(actions, obs, info)]
            for generation in range(args.rounds):
                candidates = []
                solved = False
                for node in beam:
                    for action in candidate_actions(args.world, node["info"]):
                        branches.load(node)
                        actions = list(node["actions"])
                        for _ in range(args.macro):
                            obs, _, done, trunc, info = env.step(action)
                            actions.append(action)
                            if done or trunc:
                                break
                        score = route_score(args.world, info)
                        if info["level_cleared"]:
                            score = float("inf")
                            solved = True
                        if info["died"] or info["timeout"] or score == -float("inf"):
                            continue
                        key = (info["course"], int(info["mario_px"] // 24),
                               int(info["mario_y_px"] // 24), int(obs[2] * 3),
                               int(obs[3] * 3), int(obs[4]))
                        candidates.append((score, key, branches.save(actions, obs, info)))
                        if solved:
                            break
                    if solved:
                        break
                if not candidates:
                    print(f"Search exhausted at round {generation}", flush=True)
                    return 1
                candidates.sort(key=lambda item: item[0], reverse=True)
                top = candidates[0]
                if top[0] > best:
                    best = top[0]
                    record = dict(method="emulator_beam_search", world=args.world,
                                  start=dict(course=cid, state=provenance["files"]["state"]["path"]),
                                  cleared=solved, round=generation, actions=top[2]["actions"],
                                  final=top[2]["info"], elapsed_s=time.monotonic() - started,
                                  search=dict(width=args.width, macro=args.macro, rounds=args.rounds),
                                  provenance=provenance)
                    output.write_text(json.dumps(record, indent=2), encoding="utf-8")
                print(f"round={generation} x={top[2]['info']['mario_px']:.1f} "
                      f"y={top[2]['info']['mario_y_px']:.1f} clear={solved}", flush=True)
                seen, next_beam = set(), []
                for _, key, node in candidates:
                    if key not in seen and len(next_beam) < args.width:
                        seen.add(key)
                        next_beam.append(node)
                    else:
                        node["path"].unlink()
                for node in beam:
                    node["path"].unlink()
                beam = next_beam
                if solved:
                    return 0
        return 1
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
