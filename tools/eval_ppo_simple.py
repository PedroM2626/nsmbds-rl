"""Evaluate trained PPO (simple reward) policy on World 1-1 with full provenance."""
from __future__ import annotations

import argparse
import json
import os
import sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from ram_env import MarioRamEnv
from stable_baselines3 import PPO
from eval_provenance import collect
import paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/ppo_simple_policy.zip")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=450)
    parser.add_argument("--out", default="evals/eval_ppo_simple_w11.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # 1. Initialize environment FIRST (DeSmuME native process constraint)
    env = MarioRamEnv(geo=True)

    print(f"[eval-ppo-simple] Loading PPO model from {args.model}...")
    model = PPO.load(args.model, device=args.device)

    results = []
    print(f"[eval-ppo-simple] Running {args.episodes} evaluation episodes on World 1-1...")

    for ep in range(args.episodes):
        obs, _ = env.reset()
        max_x = 0.0
        final_x = 0.0
        tot_reward = 0.0
        cleared = False
        died = False

        for step in range(args.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)

            obs, reward, done, trunc, info = env.step(action)
            tot_reward += float(reward)

            mario_px = float(info.get("mario_px", 0.0))
            if mario_px > max_x:
                max_x = mario_px
            final_x = mario_px

            if info.get("level_cleared", False) or info.get("cleared", False):
                cleared = True
                break
            if info.get("died", False) or done or trunc:
                died = True
                break

        res = {
            "episode": ep + 1,
            "max_x_px": max_x,
            "final_x_px": final_x,
            "steps": step + 1,
            "total_reward": tot_reward,
            "cleared": cleared,
            "died": died,
            "termination": "goal" if cleared else ("death" if died else "timeout")
        }
        results.append(res)
        print(f"  Ep {ep+1:2d} | max_x: {max_x:7.1f} px | steps: {step+1:3d} | term: {res['termination']}")

    max_xs = [r["max_x_px"] for r in results]
    steps_list = [r["steps"] for r in results]
    clear_count = sum(1 for r in results if r["cleared"])

    payload = {
        "model": args.model,
        "n_episodes": args.episodes,
        "mean_max_x_px": float(np.mean(max_xs)),
        "max_max_x_px": float(np.max(max_xs)),
        "mean_steps": float(np.mean(steps_list)),
        "clear_rate": float(clear_count / args.episodes),
        "episodes": results,
        "provenance": collect(rom=paths.ROM_REL, state=paths.STATE_REL)
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\n=======================================================")
    print(f"  PPO (SIMPLE REWARD) EVALUATION SUMMARY (World 1-1)")
    print(f"=======================================================")
    print(f"Mean Max Distance: {payload['mean_max_x_px']:.1f} px")
    print(f"Absolute Peak X:   {payload['max_max_x_px']:.1f} px")
    print(f"Clear Rate:        {payload['clear_rate']*100:.1f}% ({clear_count}/{args.episodes})")
    print(f"Saved evaluation with provenance to {args.out}")


if __name__ == "__main__":
    main()
