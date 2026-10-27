"""Train a purely visual PPO policy with IMPALA CNN backbone and simple potential reward.

Backbone: DeepMind IMPALA CNN (Espeholt et al., 2018) with 3 residual stages (16, 32, 32 channels).
Observation: (84, 84, 1) uint8 grayscale image from DeSmuME top screen (Zero RAM features in observation).
Reward: r_t = max(0, x_t - max_x_so_far) / 16.0 + 10.0 * cleared - 1.0 * died.

Zero synthetic data: every frame rendered live by DeSmuME display buffer.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from dual_obs_env import DualObsEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import paths


class ResidualBlock(nn.Module):
    """Classic pre-activation / post-activation residual block."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv0 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv1(torch.relu(self.conv0(torch.relu(x))))


class ConvSequence(nn.Module):
    """One sequence in the IMPALA architecture: Conv -> MaxPool -> 2x ResBlocks."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res0 = ResidualBlock(out_channels)
        self.res1 = ResidualBlock(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res1(self.res0(self.max_pool(self.conv(x))))


class ImpalaFeaturesExtractor(BaseFeaturesExtractor):
    """DeepMind IMPALA CNN feature extractor for Stable-Baselines3."""

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256, depths=(16, 32, 32)):
        super().__init__(observation_space, features_dim)
        in_channels = observation_space.shape[2] if observation_space.shape[2] in [1, 3] else observation_space.shape[0]
        layers = []
        c_in = in_channels
        for c_out in depths:
            layers.append(ConvSequence(c_in, c_out))
            c_in = c_out
        self.conv_seqs = nn.Sequential(*layers)

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 84, 84)
            n_flat = self.conv_seqs(dummy).flatten(1).shape[1]
        self.fc = nn.Linear(n_flat, features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv_seqs(observations))
        return torch.relu(self.fc(x.flatten(1)))


class PixelObservationWrapper(gym.ObservationWrapper):
    """Provides ONLY raw (84, 84, 1) pixels to policy, stripping RAM state."""

    def __init__(self, env: DualObsEnv):
        super().__init__(env)
        self.observation_space = env.observation_space["pixels"]

    def observation(self, obs):
        return obs["pixels"]


class SimpleRewardPixelEnv(gym.Wrapper):
    """Wraps environment to provide the pure potential-based running maximum progress reward."""

    def __init__(self, env: gym.Env, stall_limit: int = 30):
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
        info["max_x_so_far"] = self.max_x_so_far
        info["ep_reward"] = self.ep_reward
        info["ep_steps"] = self.ep_steps

        return obs, float(r_step), done, trunc, info


class ImpalaProgressLogger(BaseCallback):
    """Logs episode progress and atomically checkpoints best model."""

    def __init__(self, out_log: str, best_model_path: str = "", verbose: int = 1):
        super().__init__(verbose)
        self.out_log = out_log
        self.best_model_path = best_model_path
        self.best_max_x = 0.0
        self.history = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for i, (done, info) in enumerate(zip(dones, infos)):
            if done and "max_x_so_far" in info:
                ep_max_x = float(info["max_x_so_far"])
                ep_steps = int(info.get("ep_steps", 0))
                ep_reward = float(info.get("ep_reward", 0.0))
                cleared = bool(info.get("level_cleared", False) or info.get("cleared", False))
                ep_num = len(self.history) + 1

                is_new_best = ep_max_x > self.best_max_x
                if is_new_best:
                    self.best_max_x = ep_max_x
                    if self.best_model_path:
                        self.model.save(self.best_model_path)
                        print(f"  >>> [IMPALA CHECKPOINT] New best distance {ep_max_x:.1f} px saved to {self.best_model_path}!")

                entry = {
                    "episode": ep_num,
                    "step": self.num_timesteps,
                    "max_x_px": ep_max_x,
                    "ep_steps": ep_steps,
                    "reward": ep_reward,
                    "cleared": cleared,
                    "best_max_x": self.best_max_x
                }
                self.history.append(entry)

                print(
                    f"Ep {ep_num:03d} | Step {self.num_timesteps:5d} | "
                    f"max_x: {ep_max_x:6.1f} px | best: {self.best_max_x:6.1f} px | "
                    f"steps: {ep_steps:3d} | rew: {ep_reward:6.1f} | clear: {cleared}"
                )

                if ep_num % 10 == 0 or is_new_best:
                    os.makedirs(os.path.dirname(self.out_log) or ".", exist_ok=True)
                    with open(self.out_log, "w") as f:
                        json.dump({
                            "best_max_x_px": self.best_max_x,
                            "history": self.history
                        }, f, indent=2)

        return True


def main():
    parser = argparse.ArgumentParser(description="Train Visual PPO with IMPALA CNN on Raw Pixels")
    parser.add_argument("--steps", type=int, default=10000, help="Total training timesteps")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--n-steps", type=int, default=512, help="Rollout steps per update")
    parser.add_argument("--batch-size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--lr", type=float, default=2.5e-4, help="Learning rate")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--out-model", default="models/visual_impala_best.zip", help="Output model path")
    parser.add_argument("--out-log", default="evals/visual_impala_training_log.json", help="Output log path")
    args = parser.parse_args()

    print("=================================================================")
    print("  VISUAL PPO TRAINING WITH IMPALA CNN (Raw Pixels on Live DeSmuME)")
    print("=================================================================")
    print(f"Budget: {args.steps} steps | Policy: PPO + IMPALA Residual CNN (1.13M params)")
    print(f"Observation: Box(0, 255, (84, 84, 1), uint8) | Reward: Simple Potential Progress")

    # Construct live DeSmuME env
    base_env = DualObsEnv(pixel_size=84)
    pixel_env = PixelObservationWrapper(base_env)
    env = SimpleRewardPixelEnv(pixel_env, stall_limit=30)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    policy_kwargs = {
        "features_extractor_class": ImpalaFeaturesExtractor,
        "features_extractor_kwargs": {"features_dim": 256, "depths": (16, 32, 32)},
        "net_arch": dict(pi=[128], vf=[128])
    }

    model = PPO(
        "CnnPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=args.ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=0,
        seed=args.seed,
        device=device
    )

    logger = ImpalaProgressLogger(out_log=args.out_log, best_model_path=args.out_model)

    t0 = time.time()
    try:
        model.learn(total_timesteps=args.steps, callback=logger)
    finally:
        env.close()

    total_time = time.time() - t0
    print("=================================================================")
    print(f"Training finished in {total_time:.1f}s ({total_time/60:.2f} min).")
    print(f"Peak Progress Reached: {logger.best_max_x:.1f} px")

    # Final log save
    os.makedirs(os.path.dirname(args.out_log) or ".", exist_ok=True)
    with open(args.out_log, "w") as f:
        json.dump({
            "steps": args.steps,
            "best_max_x_px": logger.best_max_x,
            "total_time_s": total_time,
            "history": logger.history
        }, f, indent=2)
    print(f"Saved training log to {args.out_log}")


if __name__ == "__main__":
    main()
