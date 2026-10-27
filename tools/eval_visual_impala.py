"""Evaluate a Visual Impala PPO policy on raw pixels with cryptographic provenance.

Runs 5 live evaluation episodes on DeSmuME.
Observation: Raw top-screen (84, 84, 1) uint8 grayscale pixels. Zero RAM features.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import numpy as np
import torch
import gymnasium as gym

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from dual_obs_env import DualObsEnv
from eval_provenance import collect as collect_provenance
from stable_baselines3 import PPO
from train_visual_impala import ImpalaFeaturesExtractor, PixelObservationWrapper
import paths


def main():
    parser = argparse.ArgumentParser(description="Evaluate Visual Impala PPO Policy")
    parser.add_argument("--model", default="models/visual_impala_best.zip", help="Path to trained model")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes")
    parser.add_argument("--max-steps", type=int, default=450, help="Max steps per episode")
    parser.add_argument("--stall-limit", type=int, default=30, help="Stall limit before truncating")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic action sampling")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--out", default="evals/eval_visual_impala_w11.json", help="Output JSON path")
    args = parser.parse_args()

    print(f"[eval-visual-impala] Loading visual model from {args.model}...")
    model = PPO.load(args.model, map_location="cpu")

    base_env = DualObsEnv(pixel_size=84)
    env = PixelObservationWrapper(base_env)

    results = []
    det_str = "False (stochastic)" if args.stochastic else "True (deterministic)"
    print(f"[eval-visual-impala] Running {args.episodes} evaluation episodes (deterministic={not args.stochastic}) on raw pixels...")

    for ep in range(1, args.episodes + 1):
        obs, info = env.reset(seed=args.seed + ep)
        max_x = float(info.get("mario_px", 88.0))
        final_x = max_x
        steps = 0
        total_reward = 0.0
        cleared = False
        died = False
        cause = "timeout"
        recent_x = []

        for step in range(1, args.max_steps + 1):
            action, _ = model.predict(obs, deterministic=not args.stochastic)
            obs, r, done, trunc, info = env.step(int(action))
            steps = step
            total_reward += r

            mario_px = float(info.get("mario_px", 0.0))
            if mario_px > max_x:
                max_x = mario_px
            final_x = mario_px

            recent_x.append(mario_px)
            if len(recent_x) > args.stall_limit:
                recent_x.pop(0)
                if max(recent_x) - min(recent_x) < 2.5:
                    cause = "stalled"
                    break

            if info.get("level_cleared", False) or info.get("cleared", False):
                cleared = True
                cause = "clear"
                break
            elif info.get("died", False) or done:
                died = True
                cause = "death"
                break

        results.append({
            "episode": ep,
            "max_x_px": float(max_x),
            "final_x_px": float(final_x),
            "steps": steps,
            "total_reward": float(total_reward),
            "cleared": cleared,
            "died": died,
            "termination": cause
        })

        print(f"  Ep {ep:2d} | max_x: {max_x:7.1f} px | steps: {steps:3d} | term: {cause}")

    env.close()

    mean_max_x = float(np.mean([r["max_x_px"] for r in results]))
    max_max_x = float(np.max([r["max_x_px"] for r in results]))
    clear_rate = float(np.mean([r["cleared"] for r in results]))
    mean_steps = float(np.mean([r["steps"] for r in results]))

    print("\n=======================================================")
    print("  VISUAL IMPALA EVALUATION SUMMARY (Raw Pixels)")
    print("=======================================================")
    print(f"Mean Max Distance: {mean_max_x:.1f} px")
    print(f"Absolute Peak X:   {max_max_x:.1f} px")
    print(f"Clear Rate:        {clear_rate * 100:.1f}% ({int(clear_rate * args.episodes)}/{args.episodes})")

    # Collect provenance block
    prov = collect_provenance(
        rom=paths.ROM_REL,
        state=paths.STATE_REL,
        extra={"seed": args.seed}
    )

    out_data = {
        "model": args.model,
        "n_episodes": args.episodes,
        "deterministic": not args.stochastic,
        "mean_max_x_px": mean_max_x,
        "max_max_x_px": max_max_x,
        "mean_steps": mean_steps,
        "clear_rate": clear_rate,
        "episodes": results,
        "provenance": prov
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out_data, f, indent=2)

    print(f"Saved evaluation with provenance to {args.out}")


if __name__ == "__main__":
    main()
