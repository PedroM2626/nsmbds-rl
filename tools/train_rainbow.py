"""Train Rainbow DQN on Super Mario Bros DS via simple potential maximum reward.

Zero synthetic data: every transition executes live on DeSmuME.
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
from rainbow_dqn import RainbowAgent, PrioritizedReplayBuffer, NStepBuffer


def main():
    parser = argparse.ArgumentParser(description="Train Rainbow DQN on NSMB DS")
    parser.add_argument("--total-steps", type=int, default=20000, help="Total environment steps")
    parser.add_argument("--batch-size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--buffer-size", type=int, default=50000, help="PER buffer capacity")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--n-step", type=int, default=3, help="Multi-step return horizon")
    parser.add_argument("--start-steps", type=int, default=1000, help="Warmup random exploration steps")
    parser.add_argument("--beta-start", type=float, default=0.4, help="PER initial beta")
    parser.add_argument("--stall-limit", type=int, default=25, help="Stall step threshold")
    parser.add_argument("--max-ep-steps", type=int, default=350, help="Max steps per episode")
    parser.add_argument("--out-model", default="models/rainbow_policy.pt", help="Path to save best policy")
    parser.add_argument("--out-log", default="evals/rainbow_training_log.json", help="Path to save training history")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=================================================================")
    print("  RAINBOW DQN TRAINING (Live DeSmuME Emulator)")
    print("=================================================================")
    print(f"Device: {args.device} | Budget: {args.total_steps} steps | Batch: {args.batch_size}")
    print(f"Extensions: Double-Q + PER + Dueling + N-step ({args.n_step}) + C51 (51 atoms) + NoisyNet")
    print(f"Reward: r_t = max(0, x_t - max_x_so_far) / 16.0 (Pure Potential Maximum)")

    # 1. Initialize environment FIRST (DeSmuME process constraint)
    env = MarioRamEnv(geo=True)

    # 2. Initialize Rainbow Agent, PER Buffer and N-Step Buffer
    agent = RainbowAgent(
        obs_dim=23,
        n_actions=8,
        hidden_dim=128,
        num_atoms=51,
        v_min=-2.0,
        v_max=16.0,
        lr=args.lr,
        gamma=args.gamma,
        n_step=args.n_step,
        device=args.device
    )
    buffer = PrioritizedReplayBuffer(capacity=args.buffer_size, alpha=0.6)
    n_step_buf = NStepBuffer(n_step=args.n_step, gamma=args.gamma)

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
            # Beta annealing for PER
            beta = min(1.0, args.beta_start + total_steps * (1.0 - args.beta_start) / float(args.total_steps))

            if total_steps < args.start_steps:
                action = np.random.randint(0, 8)
            else:
                action = agent.select_action(obs)

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

            # N-step buffer accumulation
            trans = n_step_buf.append(obs, action, r_step, next_obs, done_signal)
            if trans is not None:
                s0, a0, R_n, sn, dn = trans
                buffer.push(s0, a0, R_n, sn, dn)

            obs = next_obs

            # Train network
            if total_steps >= args.start_steps and len(buffer) >= args.batch_size:
                loss = agent.update(buffer, batch_size=args.batch_size, beta=beta)
                agent.update_target_network(tau=0.005)
                ep_loss.append(loss)

            if done_signal or trunc or total_steps >= args.total_steps:
                break

        # Flush remaining transitions in n-step buffer
        for s0, a0, R_n, sn, dn in n_step_buf.flush():
            buffer.push(s0, a0, R_n, sn, dn)

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
            "loss": mean_loss,
            "cleared": cleared,
            "elapsed_min": elapsed_min
        }
        history.append(ep_stat)

        print(
            f"Ep {episode_idx:03d} | Steps: {total_steps:5d}/{args.total_steps:5d} | "
            f"Max X: {max_x_so_far:7.1f} px (Best: {best_max_x:7.1f} px) | "
            f"Loss: {mean_loss:6.4f} | Cleared: {cleared} | Time: {elapsed_min:.1f}m",
            flush=True
        )

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
    print(f"Rainbow DQN Training Complete in {total_time/60.0:.2f} min ({total_steps} steps).")
    print(f"Overall Best Max X: {best_max_x:.1f} px")

    os.makedirs(os.path.dirname(args.out_log) or ".", exist_ok=True)
    with open(args.out_log, "w") as f:
        json.dump({
            "total_steps": total_steps,
            "episodes": episode_idx,
            "best_max_x": best_max_x,
            "total_time_min": total_time / 60.0,
            "history": history
        }, f, indent=2)
    print(f"Saved Rainbow log to {args.out_log}")


if __name__ == "__main__":
    main()
