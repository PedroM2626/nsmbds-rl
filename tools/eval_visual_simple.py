"""Evaluate trained Visual PPO policy (raw pixels) on World 1-1 with full provenance."""
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

from dual_obs_env import DualObsEnv
from train_visual_simple import PixelObservationWrapper
from stable_baselines3 import PPO
from eval_provenance import collect
import paths


def main():
    parser = argparse.ArgumentParser(description="Evaluate visual PPO on raw pixels")
    parser.add_argument("--model", default="models/visual_simple_best.zip")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=450)
    parser.add_argument("--out", default="evals/eval_visual_simple_w11.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--stochastic", action="store_true", help="Run with stochastic action sampling instead of argmax")
    parser.add_argument("--stall-limit", type=int, default=30, help="Stall step threshold")
    args = parser.parse_args()

    deterministic = not args.stochastic

    # 1. Initialize environment FIRST (DeSmuME native process constraint)
    dual_env = DualObsEnv(rom_path=paths.ROM_REL, state_path=paths.STATE_REL, pixel_size=84)
    env = PixelObservationWrapper(dual_env)

    print(f"[eval-visual] Loading visual model from {args.model}...")
    model = PPO.load(args.model, device=args.device)

    results = []
    print(f"[eval-visual] Running {args.episodes} evaluation episodes (deterministic={deterministic}) on raw pixels...")

    for ep in range(args.episodes):
        obs, _ = env.reset()
        max_x = 0.0
        final_x = 0.0
        tot_reward = 0.0
        cleared = False
        died = False
        recent_x = []

        for step in range(args.max_steps):
            action, _ = model.predict(obs, deterministic=deterministic)
            action = int(action)

            obs, reward, done, trunc, info = env.step(action)
            tot_reward += float(reward)

            mario_px = float(info.get("mario_px", 0.0))
            if mario_px > max_x:
                max_x = mario_px
            final_x = mario_px

            recent_x.append(mario_px)
            if len(recent_x) > args.stall_limit:
                recent_x.pop(0)
                if max(recent_x) - min(recent_x) < 2.5:
                    trunc = True

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
        "deterministic": deterministic,
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
    print(f"  VISUAL PPO EVALUATION SUMMARY (Raw Pixels)")
    print(f"=======================================================")
    print(f"Mean Max Distance: {payload['mean_max_x_px']:.1f} px")
    print(f"Absolute Peak X:   {payload['max_max_x_px']:.1f} px")
    print(f"Clear Rate:        {payload['clear_rate']*100:.1f}% ({clear_count}/{args.episodes})")
    print(f"Saved evaluation with provenance to {args.out}")


if __name__ == "__main__":
    main()
