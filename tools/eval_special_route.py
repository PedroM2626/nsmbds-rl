"""Replay a searched route in real DeSmuME and record strict clear evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eval_provenance import collect, sha1_file
from multi_area_env import MultiAreaMarioEnv
from search_special_courses import CONFIG, WATER, ALLOWED_COURSES


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--world", choices=CONFIG, required=True)
    ap.add_argument("--route", required=True)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--state")
    ap.add_argument("--course")
    args = ap.parse_args()
    if args.episodes < 1:
        ap.error("episodes must be positive")
    route = json.loads(Path(args.route).read_text(encoding="utf-8"))
    actions = route["actions"]
    if not actions or any(type(a) is not int or not 0 <= a < 13 for a in actions):
        ap.error("route must contain integer actions in 0..12")
    cid, state = CONFIG[args.world]
    cid = args.course or route.get("start", {}).get("course", cid)
    state = args.state or route.get("start", {}).get("state", state)
    if cid not in ALLOWED_COURSES[args.world]:
        ap.error("Starting course does not belong to the requested world/level")
    env = MultiAreaMarioEnv(state_path=str(ROOT / state),
                            course_name=f"course/{cid}.bin", max_steps=len(actions) + 120,
                            water_overrides=WATER)
    result = dict(method="emulator_searched_action_route", world=args.world,
                  provenance=collect(env.rom_path, env.state_path, sources=[
                      __file__, ROOT / "src/multi_area_env.py", ROOT / "src/ram_env.py",
                      ROOT / "src/eval_provenance.py"], extra={
                      "seed": 0, "route_sha1": sha1_file(args.route),
                      "course_index": env.ram.course_index(),
                      "water_overrides": WATER}), episodes=[])
    try:
        for episode in range(args.episodes):
            obs, info = env.reset(seed=episode)
            trajectory, total = [], 0.0
            for step, action in enumerate(actions + [0] * 120, 1):
                obs, reward, done, trunc, info = env.step(action)
                total += reward
                trajectory.append(dict(step=step, action=action, reward=reward, **info))
                if done or trunc:
                    break
            reason = ("goal" if info["level_cleared"] else "death" if info["died"]
                      else "timeout" if info["timeout"] else "route_exhausted")
            record = dict(episode=episode, seed=episode, steps=step, reward=total,
                          termination=reason, cleared=info["level_cleared"],
                          final=info, trajectory=trajectory)
            result["episodes"].append(record)
            print(f"episode={episode} steps={step} x={info['mario_px']} "
                  f"y={info['mario_y_px']} termination={reason}", flush=True)
    finally:
        env.close()
    result["clear_rate"] = sum(e["cleared"] for e in result["episodes"]) / args.episodes
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0 if result["clear_rate"] == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
