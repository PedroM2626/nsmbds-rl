"""Train PPO with simple potential maximum reward on NSMB DS.

Replaces the legacy complex reward with:
    r_t = max(0, x_t - max_x_so_far) / 16.0 + 10.0 * cleared - 1.0 * died

Zero synthetic data: every transition executes live on DeSmuME.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import gymnasium as gym
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from ram_env import MarioRamEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


class SimpleRewardEnv(gym.Wrapper):
    """Wraps MarioRamEnv to compute potential-based running maximum progress reward."""

    def __init__(self, env: MarioRamEnv, stall_limit: int = 25):
        super().__init__(env)
        self.stall_limit = stall_limit
        self.max_x_so_far = 0.0
        self.recent_x = []
        self.ep_reward = 0.0
        self.ep_steps = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.max_x_so_far = 0.0
        self.recent_x = []
        self.ep_reward = 0.0
        self.ep_steps = 0
        return obs, info

    def step(self, action):
        obs, orig_r, done, trunc, info = self.env.step(action)
        self.ep_steps += 1
        mario_px = float(info.get("mario_px", 0.0))

        # Potential-based maximum progress
        delta_max = max(0.0, mario_px - self.max_x_so_far)
        if mario_px > self.max_x_so_far:
            self.max_x_so_far = mario_px

        r_step = delta_max / 16.0

        if info.get("level_cleared", False) or info.get("cleared", False):
            r_step += 10.0
            done = True
        elif info.get("died", False):
            r_step -= 1.0
            done = True

        self.recent_x.append(mario_px)
        if len(self.recent_x) > self.stall_limit:
            self.recent_x.pop(0)
            if max(self.recent_x) - min(self.recent_x) < 2.5:
                trunc = True

        self.ep_reward += r_step
        info["simple_reward"] = r_step
        info["max_x_so_far"] = self.max_x_so_far
        info["ep_reward"] = self.ep_reward
        info["ep_steps"] = self.ep_steps

        return obs, float(r_step), done, trunc, info


class PPOProgressLogger(BaseCallback):
    """Logs episode progress and tracks peak max_x in real time."""

    def __init__(self, out_log: str, verbose: int = 1):
        super().__init__(verbose)
        self.out_log = out_log
        self.best_max_x = 0.0
        self.history = []
        self.episode_count = 0
        self.t0 = time.time()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for info, done in zip(infos, dones):
            if done:
                self.episode_count += 1
                max_x = float(info.get("max_x_so_far", 0.0))
                ep_r = float(info.get("ep_reward", 0.0))
                ep_s = int(info.get("ep_steps", 0))
                cleared = bool(info.get("level_cleared", False) or info.get("cleared", False))

                if max_x > self.best_max_x:
                    self.best_max_x = max_x

                elapsed_min = (time.time() - self.t0) / 60.0
                record = {
                    "episode": self.episode_count,
                    "total_steps": int(self.num_timesteps),
                    "ep_steps": ep_s,
                    "max_x_px": max_x,
                    "best_max_x_px": self.best_max_x,
                    "ep_reward": ep_r,
                    "cleared": cleared,
                    "elapsed_min": elapsed_min
                }
                self.history.append(record)

                print(
                    f"Ep {self.episode_count:03d} | Steps: {self.num_timesteps:5d} | "
                    f"Max X: {max_x:7.1f} px (Best: {self.best_max_x:7.1f} px) | "
                    f"Reward: {ep_r:6.2f} | Cleared: {cleared} | Time: {elapsed_min:.1f}m",
                    flush=True
                )

                if self.episode_count % 5 == 0:
                    os.makedirs(os.path.dirname(self.out_log) or ".", exist_ok=True)
                    with open(self.out_log, "w") as f:
                        json.dump({
                            "total_steps": int(self.num_timesteps),
                            "episodes": self.episode_count,
                            "best_max_x": self.best_max_x,
                            "history": self.history
                        }, f, indent=2)

        return True


def main():
    parser = argparse.ArgumentParser(description="Train PPO with simple potential reward")
    parser.add_argument("--total-steps", type=int, default=20000, help="Total environment steps")
    parser.add_argument("--n-steps", type=int, default=1024, help="PPO rollout steps per update")
    parser.add_argument("--batch-size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--n-epochs", type=int, default=4, help="Epochs per update")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--out-model", default="models/ppo_simple_policy.zip", help="Path to save PPO policy")
    parser.add_argument("--out-log", default="evals/ppo_simple_training_log.json", help="Path to save training history")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print("=================================================================")
    print("  PPO WITH SIMPLE POTENTIAL REWARD (Live DeSmuME Emulator)")
    print("=================================================================")
    print(f"Device: {args.device} | Total steps: {args.total_steps} | Rollout n_steps: {args.n_steps}")
    print(f"Reward: r_t = max(0, x_t - max_x_so_far) / 16.0 (Pure Potential Maximum)")

    # 1. Initialize environment FIRST (satisfies DeSmuME process constraint)
    raw_env = MarioRamEnv(geo=True)
    env = SimpleRewardEnv(raw_env)

    # 2. Initialize PPO
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=args.ent_coef,
        verbose=0,
        device=args.device
    )

    callback = PPOProgressLogger(out_log=args.out_log)

    t0 = time.time()
    model.learn(total_timesteps=args.total_steps, callback=callback)
    total_time = time.time() - t0

    print("=================================================================")
    print(f"PPO Training Complete in {total_time/60.0:.2f} min ({args.total_steps} steps).")
    print(f"Overall Best Max X: {callback.best_max_x:.1f} px")

    # Save model
    os.makedirs(os.path.dirname(args.out_model) or ".", exist_ok=True)
    model.save(args.out_model)
    print(f"Saved PPO model to {args.out_model}")

    # Save final log
    os.makedirs(os.path.dirname(args.out_log) or ".", exist_ok=True)
    with open(args.out_log, "w") as f:
        json.dump({
            "total_steps": args.total_steps,
            "episodes": callback.episode_count,
            "best_max_x": callback.best_max_x,
            "total_time_min": total_time / 60.0,
            "history": callback.history
        }, f, indent=2)
    print(f"Saved PPO log to {args.out_log}")


if __name__ == "__main__":
    main()
