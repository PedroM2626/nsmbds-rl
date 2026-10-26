"""Train Implicit Quantile Networks (IQN) on Super Mario Bros DS via simple potential reward.

Reward:
    r_t = max(0, x_t - max_x_so_far) / 16.0 + 10.0 * cleared - 1.0 * died

Zero synthetic data: every transition is collected live on the DeSmuME emulator.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from ram_env import MarioRamEnv
from iqn import IQNAgent, ReplayBuffer


def main():
    parser = argparse.ArgumentParser(description="Train IQN on NSMB DS")
    parser.add_argument("--total-steps", type=int, default=15000, help="Total environment interaction steps")
    parser.add_argument("--batch-size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--buffer-size", type=int, default=50000, help="Replay buffer capacity")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--start-steps", type=int, default=1000, help="Steps of random actions before training")
    parser.add_argument("--eps-decay-steps", type=int, default=8000, help="Steps to decay epsilon to 0.05")
    parser.add_argument("--stall-limit", type=int, default=25, help="Stall step threshold")
    parser.add_argument("--max-ep-steps", type=int, default=350, help="Max steps per episode")
    parser.add_argument("--out-model", default="models/iqn_policy.pt", help="Path to save best policy")
    parser.add_argument("--out-log", default="evals/iqn_training_log.json", help="Path to save training history")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=================================================================")
    print("  IMPLICIT QUANTILE NETWORKS (IQN) TRAINING (Live DeSmuME)")
    print("=================================================================")
    print(f"Device: {args.device} | Budget: {args.total_steps} steps | Batch: {args.batch_size}")
    print(f"Reward: r_t = max(0, x_t - max_x_so_far) / 16.0 (Pure Potential Maximum)")

    # 1. Initialize environment FIRST (satisfies DeSmuME process constraint)
    env = MarioRamEnv(geo=True)

    # 2. Initialize IQN Agent and Replay Buffer
    agent = IQNAgent(
        obs_dim=23,
        n_actions=8,
        hidden_dim=128,
        lr=args.lr,
        gamma=args.gamma,
        device=args.device
    )
    buffer = ReplayBuffer(capacity=args.buffer_size)

    total_steps = 0
    episode_idx = 0
    best_max_x = 0.0
    history = []
    t0 = time.time()

    while total_steps < args.total_steps:
        episode_idx += 1
        obs, _ = env.reset()
        ep_steps = 0
        ep_reward = 0.0
        max_x_so_far = 0.0
        recent_x = []
        ep_loss = []
        cleared = False
        died = False
        stalled = False

        for step in range(args.max_ep_steps):
            # Epsilon schedule
            if total_steps < args.start_steps:
                eps = 1.0
            else:
                progress = min(1.0, (total_steps - args.start_steps) / float(args.eps_decay_steps))
                eps = 1.0 - progress * (1.0 - 0.05)

            action = agent.select_action(obs, epsilon=eps)
            next_obs, env_reward, done, trunc, info = env.step(action)
            total_steps += 1
            ep_steps += 1

            mario_px = float(info.get("mario_px", 0.0))

            # Potential-based maximum reward
            delta_max = max(0.0, mario_px - max_x_so_far)
            if mario_px > max_x_so_far:
                max_x_so_far = mario_px

            r_step = delta_max / 16.0
            done_signal = False

            if info.get("level_cleared", False) or info.get("cleared", False):
                cleared = True
                done_signal = True
                r_step += 10.0

            if info.get("died", False) or done:
                died = True
                done_signal = True
                r_step -= 1.0

            recent_x.append(mario_px)
            if len(recent_x) > args.stall_limit:
                recent_x.pop(0)
                if max(recent_x) - min(recent_x) < 2.5:
                    stalled = True
                    done_signal = True

            ep_reward += r_step
            buffer.push(obs, action, r_step, next_obs, done_signal)
            obs = next_obs

            # Train network once we have enough transitions
            if total_steps >= args.start_steps and len(buffer) >= args.batch_size:
                loss = agent.update(buffer, batch_size=args.batch_size)
                agent.update_target_network(tau=0.005)
                ep_loss.append(loss)

            if done_signal or trunc or total_steps >= args.total_steps:
                break

        if max_x_so_far > best_max_x:
            best_max_x = max_x_so_far
            os.makedirs(os.path.dirname(args.out_model) or ".", exist_ok=True)
            torch.save({
                "model_state": agent.online_net.state_dict(),
                "best_max_x": best_max_x,
                "episode": episode_idx,
                "total_steps": total_steps
            }, args.out_model)

        mean_loss = float(np.mean(ep_loss)) if ep_loss else 0.0
        elapsed_min = (time.time() - t0) / 60.0

        ep_stat = {
            "episode": episode_idx,
            "total_steps": total_steps,
            "ep_steps": ep_steps,
            "max_x_px": max_x_so_far,
            "best_max_x_px": best_max_x,
            "ep_reward": float(ep_reward),
            "epsilon": float(eps),
            "loss": mean_loss,
            "cleared": cleared,
            "elapsed_min": elapsed_min
        }
        history.append(ep_stat)

        print(
            f"Ep {episode_idx:03d} | Steps: {total_steps:5d}/{args.total_steps:5d} | "
            f"Max X: {max_x_so_far:7.1f} px (Best: {best_max_x:7.1f} px) | "
            f"Eps: {eps:.2f} | Loss: {mean_loss:6.4f} | "
            f"Cleared: {cleared} | Time: {elapsed_min:.1f}m",
            flush=True
        )

        # Periodically save training log to disk
        if episode_idx % 5 == 0:
            os.makedirs(os.path.dirname(args.out_log) or ".", exist_ok=True)
            with open(args.out_log, "w") as f:
                json.dump({
                    "total_steps": total_steps,
                    "episodes": episode_idx,
                    "best_max_x": best_max_x,
                    "total_time_min": elapsed_min,
                    "history": history
                }, f, indent=2)

    total_time = time.time() - t0
    print("=================================================================")
    print(f"IQN Training Complete in {total_time/60.0:.2f} min ({total_steps} steps).")
    print(f"Overall Best Max X: {best_max_x:.1f} px")

    # Final log save
    os.makedirs(os.path.dirname(args.out_log) or ".", exist_ok=True)
    with open(args.out_log, "w") as f:
        json.dump({
            "total_steps": total_steps,
            "episodes": episode_idx,
            "best_max_x": best_max_x,
            "total_time_min": total_time / 60.0,
            "history": history
        }, f, indent=2)
    print(f"Saved IQN log to {args.out_log}")


if __name__ == "__main__":
    main()
