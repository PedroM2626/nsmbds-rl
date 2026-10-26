"""Evaluation harness for World 2-1 (B01_1) End-to-End Clearance.

Validates the full clearance of World 2-1 on the live DeSmuME emulator core.
Produces a benchmark evaluation JSON conforming to the project's provenance standard.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ram_env import MarioRamEnv
from world_2_1_controller import World21Agent
from course_id import course_area_index, course_for_area_index
from eval_provenance import collect as collect_provenance

DEFAULT_ROM = "data/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds"
DEFAULT_STATE = "data/state_2-1.ds1"
COURSE = "course/B01_1.bin"
GOAL_PX = 4032.0
SPAWN_ANCHOR_PX = 360.0


def run_episode(env: MarioRamEnv, agent: World21Agent, max_steps: int = 800):
    obs, _ = env.reset()
    agent.reset()
    total_reward = 0.0
    latencies = []
    t_start = time.time()
    steps = 0
    finished = False
    cause = "trunc"
    death_diag = ""
    max_px = SPAWN_ANCHOR_PX
    final_px = SPAWN_ANCHOR_PX

    for step in range(1, max_steps + 1):
        t0 = time.perf_counter()
        action = agent.act(obs)
        lat = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat)

        obs, r, term, trunc, info = env.step(action)
        steps = step
        total_reward += r
        current_px = SPAWN_ANCHOR_PX + obs[0] * 512.0
        max_px = max(max_px, current_px)
        final_px = current_px

        if info.get("level_cleared") or current_px >= (GOAL_PX - 16.0):
            finished = True
            cause = "goal"
            break

        if term:
            cause = "death" if info.get("died") else "term"
            if info.get("died"):
                ma, mb = env.ram._mario_pos()
                m_y = ma / 4096.0 if ma is not None else 0.0
                death_diag = f"death at step={step} px={current_px:.1f} y={m_y:.1f}"
            break

    wall = max(time.time() - t_start, 1e-6)
    return {
        "steps": steps,
        "total_reward": float(total_reward),
        "max_abs_px": float(max_px),
        "final_abs_px": float(final_px),
        "reached_goal": bool(finished),
        "termination": cause,
        "mean_control_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
        "achieved_env_fps": float(steps / wall),
        "death_diag": death_diag,
    }


def main():
    parser = argparse.ArgumentParser(description="World 2-1 Clearance Evaluation Harness")
    parser.add_argument("--rom", default=DEFAULT_ROM, help="Path to NSMB DS ROM")
    parser.add_argument("--state", default=DEFAULT_STATE, help="Path to World 2-1 savestate (.ds1)")
    parser.add_argument("--episodes", type=int, default=5, help="Number of benchmark evaluation episodes")
    parser.add_argument("--max-steps", type=int, default=800, help="Max step limit per episode")
    parser.add_argument("--out", default="evals/eval_2_1_clear.json", help="Path to output evaluation JSON")
    args = parser.parse_args()

    expected_area = course_area_index(COURSE)
    print(f"[eval-2-1] Starting World 2-1 evaluation ({COURSE}, target area {expected_area})")

    env = MarioRamEnv(rom_path=args.rom, state_path=args.state, geo=True,
                      course_name=COURSE, max_steps=args.max_steps)

    try:
        state_area = env.ram.course_index() if env.ram else None
        if state_area is not None and state_area != expected_area:
            raise SystemExit(
                f"Course/state mismatch: {COURSE} is area index {expected_area}, "
                f"but savestate is running area index {state_area} ({course_for_area_index(state_area)})."
            )

        results = []
        for ep in range(1, args.episodes + 1):
            agent = World21Agent(rom_path=args.rom, course_name=COURSE, ram=env.ram)
            res = run_episode(env, agent, max_steps=args.max_steps)
            res["episode"] = ep
            results.append(res)
            status = "CLEARED" if res["reached_goal"] else f"FAILED ({res['termination']})"
            print(f"  Ep {ep}/{args.episodes}: {status} | steps: {res['steps']} | "
                  f"final_px: {res['final_abs_px']:.1f} / {GOAL_PX:.1f} | "
                  f"reward: {res['total_reward']:.2f} | fps: {res['achieved_env_fps']:.1f}")

        clear_rate = float(np.mean([r["reached_goal"] for r in results]))
        agg = {
            "controller": "world21",
            "episodes": args.episodes,
            "goal_abs_px": GOAL_PX,
            "goal_source": "castle entrance trigger",
            "course": COURSE,
            "state": args.state,
            "reflex": "profile",
            "spawn_anchor_px": SPAWN_ANCHOR_PX,
            "planner": "python",
            "clear_rate": clear_rate,
            "mean_progress_px": float(np.mean([r["final_abs_px"] for r in results])),
            "max_progress_px": float(np.max([r["final_abs_px"] for r in results])),
            "mean_reward": float(np.mean([r["total_reward"] for r in results])),
            "mean_fps": float(np.mean([r["achieved_env_fps"] for r in results])),
            "mean_latency_ms": float(np.mean([r["mean_control_latency_ms"] for r in results])),
        }

        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        prov = collect_provenance(rom=args.rom, state=args.state, extra={
            "controller": "world21",
            "course": COURSE,
            "course_area_index": expected_area,
            "state_area_index": state_area,
            "reflex": "profile",
            "planner": "python",
            "max_steps": args.max_steps,
            "episodes": args.episodes,
            "goal_abs_px": GOAL_PX,
        })

        out = {"aggregate": agg, "episodes": results, "provenance": prov}
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)

        print("\n=== World 2-1 Benchmark Summary ===")
        print(f"  Clear Rate: {clear_rate * 100:.1f}% ({sum(r['reached_goal'] for r in results)}/{args.episodes})")
        print(f"  Mean Steps: {np.mean([r['steps'] for r in results]):.1f}")
        print(f"  Mean Progress: {agg['mean_progress_px']:.1f} px / {GOAL_PX:.1f} px")
        print(f"  Mean Reward: {agg['mean_reward']:.2f}")
        print(f"  Saved benchmark to {args.out}")

    finally:
        env.close()


if __name__ == "__main__":
    main()
