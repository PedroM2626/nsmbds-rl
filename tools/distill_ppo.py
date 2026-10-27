"""Policy Distillation for PPO: Sharpens stochastic PPO policy into a deterministic controller.

Resolves the action bimodal ambiguity at obstacles by distilling winning stochastic rollouts
into a decisive deterministic policy network.

Zero synthetic data: every demonstration transition is generated live on DeSmuME.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from ram_env import MarioRamEnv
from stable_baselines3 import PPO
from eval_provenance import collect
import paths


class DistilledMlpPolicy(nn.Module):
    """Compact 2-layer MLP for sharp deterministic control."""

    def __init__(self, obs_dim: int = 23, n_actions: int = 8, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def act(self, obs: np.ndarray) -> int:
        with torch.no_grad():
            t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            logits = self.forward(t)
            return int(logits.argmax(dim=-1).item())


def collect_winning_demonstrations(env: MarioRamEnv, teacher_model: PPO, min_dist: float = 1000.0, max_episodes: int = 15):
    """Collects transitions only from successful stochastic teacher rollouts."""
    dataset_states = []
    dataset_actions = []
    collected_eps = 0

    print(f"[distill] Collecting demonstrations (filtering episodes reaching >= {min_dist:.1f} px)...")

    for ep in range(max_episodes):
        obs, _ = env.reset()
        ep_states = []
        ep_actions = []
        max_x = 0.0

        for step in range(450):
            action, _ = teacher_model.predict(obs, deterministic=False)
            action = int(action)

            ep_states.append(np.array(obs, dtype=np.float32))
            ep_actions.append(action)

            obs, reward, done, trunc, info = env.step(action)
            px = float(info.get("mario_px", 0.0))
            if px > max_x:
                max_x = px

            if done or trunc or info.get("died") or info.get("cleared"):
                break

        print(f"  Rollout {ep+1:2d}: max_x = {max_x:7.1f} px | steps = {step+1:3d} | accepted: {max_x >= min_dist}")

        if max_x >= min_dist:
            dataset_states.extend(ep_states)
            dataset_actions.extend(ep_actions)
            collected_eps += 1

    print(f"[distill] Collected {len(dataset_states)} transitions from {collected_eps} winning rollouts.")
    return np.array(dataset_states, dtype=np.float32), np.array(dataset_actions, dtype=np.int64)


def train_distilled_policy(states: np.ndarray, actions: np.ndarray, epochs: int = 250, batch_size: int = 64, lr: float = 1e-3) -> DistilledMlpPolicy:
    student = DistilledMlpPolicy(obs_dim=23, n_actions=8, hidden_dim=64)
    optimizer = optim.Adam(student.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    X = torch.as_tensor(states, dtype=torch.float32)
    Y = torch.as_tensor(actions, dtype=torch.int64)
    N = len(X)

    print(f"[distill] Training student network ({sum(p.numel() for p in student.parameters())} params) for {epochs} epochs...")

    for epoch in range(epochs):
        indices = torch.randperm(N)
        total_loss = 0.0
        batches = 0

        for start in range(0, N, batch_size):
            batch_idx = indices[start:start + batch_size]
            bx, by = X[batch_idx], Y[batch_idx]

            optimizer.zero_grad()
            logits = student(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1:03d}/{epochs:03d} | Loss: {total_loss/batches:.4f}")

    return student


def evaluate_deterministic(env: MarioRamEnv, policy: DistilledMlpPolicy, episodes: int = 5, max_steps: int = 450) -> dict:
    results = []
    print(f"[distill] Evaluating distilled policy deterministically over {episodes} episodes...")

    for ep in range(episodes):
        obs, _ = env.reset()
        max_x = 0.0
        final_x = 0.0
        tot_reward = 0.0
        cleared = False
        died = False

        for step in range(max_steps):
            action = policy.act(obs)
            obs, reward, done, trunc, info = env.step(action)
            tot_reward += float(reward)

            px = float(info.get("mario_px", 0.0))
            if px > max_x:
                max_x = px
            final_x = px

            if info.get("level_cleared") or info.get("cleared"):
                cleared = True
                break
            if info.get("died") or done or trunc:
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

    return {
        "n_episodes": episodes,
        "mean_max_x_px": float(np.mean(max_xs)),
        "max_max_x_px": float(np.max(max_xs)),
        "mean_steps": float(np.mean(steps_list)),
        "clear_rate": float(clear_count / episodes),
        "episodes": results,
        "provenance": collect(rom=paths.ROM_REL, state=paths.STATE_REL)
    }


def main():
    parser = argparse.ArgumentParser(description="Distill stochastic PPO into a sharp deterministic policy")
    parser.add_argument("--teacher", default="models/ppo_simple_best.zip", help="Path to teacher PPO model")
    parser.add_argument("--out-model", default="models/ppo_distilled_policy.pt", help="Path to save distilled policy")
    parser.add_argument("--out-eval", default="evals/eval_ppo_distilled_w11.json", help="Path to save evaluation JSON")
    parser.add_argument("--min-dist", type=float, default=1000.0, help="Min distance for teacher demonstration")
    parser.add_argument("--max-eps", type=int, default=12, help="Max teacher demonstration rollouts")
    parser.add_argument("--epochs", type=int, default=200, help="Distillation training epochs")
    args = parser.parse_args()

    # 1. Initialize environment FIRST
    env = MarioRamEnv(geo=True)

    # 2. Load teacher PPO model
    print(f"[distill] Loading teacher model from {args.teacher}...")
    teacher = PPO.load(args.teacher, device="cpu")

    # 3. Collect demonstrations from teacher
    states, actions = collect_winning_demonstrations(env, teacher, min_dist=args.min_dist, max_episodes=args.max_eps)

    if len(states) == 0:
        print("[distill] Warning: No demonstrations met the filter threshold. Lowering threshold to 500 px...")
        states, actions = collect_winning_demonstrations(env, teacher, min_dist=500.0, max_episodes=8)

    # 4. Train distilled student policy
    student = train_distilled_policy(states, actions, epochs=args.epochs)

    # 5. Save distilled policy
    os.makedirs(os.path.dirname(args.out_model) or ".", exist_ok=True)
    torch.save({
        "model_state": student.state_dict(),
        "obs_dim": 23,
        "n_actions": 8,
        "hidden_dim": 64
    }, args.out_model)
    print(f"[distill] Saved distilled policy to {args.out_model}")

    # 6. Evaluate deterministically over 5 episodes
    eval_payload = evaluate_deterministic(env, student, episodes=5)

    os.makedirs(os.path.dirname(args.out_eval) or ".", exist_ok=True)
    with open(args.out_eval, "w") as f:
        json.dump(eval_payload, f, indent=2)

    print("\n=======================================================")
    print("  DISTILLED PPO DETERMINISTIC EVALUATION SUMMARY")
    print("=======================================================")
    print(f"Mean Max Distance: {eval_payload['mean_max_x_px']:.1f} px")
    print(f"Absolute Peak X:   {eval_payload['max_max_x_px']:.1f} px")
    print(f"Clear Rate:        {eval_payload['clear_rate']*100:.1f}%")
    print(f"Saved evaluation with provenance to {args.out_eval}")


if __name__ == "__main__":
    main()
