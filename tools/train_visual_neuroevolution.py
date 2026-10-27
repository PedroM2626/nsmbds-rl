"""Visual Neuroevolution (Genetic Algorithm) for NSMB DS on Raw Pixels.

Optimizes a compact Convolutional Neural Network (CNN) policy (~8.6k parameters)
directly from raw top-screen grayscale pixels (84x84x1) using parallel DeSmuME workers.

Observation: Box(0, 255, (84, 84, 1), uint8) — zero RAM features in observation.
Fitness: f = max(mario_px) + 1000 * cleared + 2 * (max_steps - steps).

Zero gradient descent, zero synthetic data: every evaluation executes live on DeSmuME.
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
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import gymnasium as gym

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from dual_obs_env import DualObsEnv
import paths


class PixelObservationWrapper(gym.ObservationWrapper):
    """Provides ONLY raw (84, 84, 1) pixels to policy, stripping RAM state."""

    def __init__(self, env: DualObsEnv):
        super().__init__(env)
        self.observation_space = env.observation_space["pixels"]

    def observation(self, obs):
        return obs["pixels"]


class CompactVisualCNN(nn.Module):
    """Lightweight CNN policy for Visual Neuroevolution (~8,600 parameters)."""

    def __init__(self, n_actions: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, stride=2, padding=2)   # 84x84 -> 42x42
        self.pool1 = nn.MaxPool2d(2)                                        # 42x42 -> 21x21
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)  # 21x21 -> 11x11
        self.pool2 = nn.MaxPool2d(2)                                        # 11x11 -> 5x5
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1) # 5x5 -> 3x3
        self.fc1 = nn.Linear(16 * 3 * 3, 32)
        self.fc2 = nn.Linear(32, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.relu(self.conv3(x))
        x = x.flatten(1)
        x = torch.tanh(self.fc1(x))
        return self.fc2(x)


class VisualIndividual:
    """An individual in the population defined by a flat parameter vector."""

    def __init__(self, weights: np.ndarray | None = None, model: CompactVisualCNN | None = None, seed: int | None = None):
        self.model = CompactVisualCNN()
        self.model.eval()
        if weights is not None:
            self.theta = weights.astype(np.float32)
        elif model is not None:
            self.theta = parameters_to_vector(model.parameters()).detach().cpu().numpy().astype(np.float32)
        else:
            rng = np.random.default_rng(seed)
            # Small initialization
            p_vec = parameters_to_vector(self.model.parameters()).detach().cpu().numpy()
            scale = 0.05
            self.theta = rng.normal(0.0, scale, size=len(p_vec)).astype(np.float32)

        vector_to_parameters(torch.from_numpy(self.theta), self.model.parameters())

    def act_batch(self, obs_batch: np.ndarray) -> list[int]:
        """Runs forward pass on a batch of pixel observations (B, 84, 84, 1)."""
        with torch.no_grad():
            t = torch.from_numpy(obs_batch).float().permute(0, 3, 1, 2) / 255.0
            logits = self.model(t)
            actions = torch.argmax(logits, dim=-1).cpu().tolist()
            return [int(a) for a in actions]

    def act(self, obs: np.ndarray) -> int:
        """Runs forward pass on a single pixel observation (84, 84, 1)."""
        with torch.no_grad():
            t = torch.from_numpy(obs).float().unsqueeze(0).permute(0, 3, 1, 2) / 255.0
            logits = self.model(t)
            return int(torch.argmax(logits, dim=-1).item())


def make_visual_env_fn():
    """Factory function for top-level SubprocVecEnv worker instantiation."""
    def _thunk():
        env = DualObsEnv(pixel_size=84)
        return PixelObservationWrapper(env)
    return _thunk


class VisualGeneticTrainer:
    """Evolutionary trainer for visual policies."""

    def __init__(
        self,
        pop_size: int = 16,
        elite_count: int = 2,
        mutation_rate: float = 0.20,
        mutation_scale: float = 0.08,
        seed: int = 42
    ):
        self.pop_size = pop_size
        self.elite_count = elite_count
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.rng = np.random.default_rng(seed)

        self.population = [VisualIndividual(seed=int(self.rng.integers(0, 1000000))) for _ in range(pop_size)]
        self.generation = 0
        self.best_overall_fitness = 0.0
        self.best_overall_policy: VisualIndividual | None = None
        self.history = []

    def evolve_generation_parallel(self, venv, n_workers: int, max_steps: int = 350, stall_limit: int = 25) -> dict:
        self.generation += 1
        eval_results = []
        gen_start = time.time()

        for chunk_start in range(0, self.pop_size, n_workers):
            chunk_end = min(chunk_start + n_workers, self.pop_size)
            active_inds = self.population[chunk_start:chunk_end]
            n_active = len(active_inds)

            obs = venv.reset()
            dones_tracked = [False] * n_workers
            max_xs = [0.0] * n_workers
            recent_xs = [[] for _ in range(n_workers)]
            cleared_flags = [False] * n_workers
            stalled_flags = [False] * n_workers
            died_flags = [False] * n_workers
            steps_taken = [0] * n_workers

            for step in range(max_steps):
                actions = []
                for i in range(n_workers):
                    if i < n_active and not dones_tracked[i]:
                        actions.append(active_inds[i].act(obs[i]))
                    else:
                        actions.append(0)

                obs, rewards, dones, infos = venv.step(actions)

                for i in range(n_active):
                    if not dones_tracked[i]:
                        steps_taken[i] = step + 1
                        mario_px = float(infos[i].get("mario_px", 0.0))
                        if mario_px > max_xs[i]:
                            max_xs[i] = mario_px

                        recent_xs[i].append(mario_px)
                        if len(recent_xs[i]) > stall_limit:
                            recent_xs[i].pop(0)
                            if max(recent_xs[i]) - min(recent_xs[i]) < 2.5:
                                stalled_flags[i] = True
                                dones_tracked[i] = True

                        if infos[i].get("level_cleared", False) or infos[i].get("cleared", False):
                            cleared_flags[i] = True
                            dones_tracked[i] = True
                        elif infos[i].get("died", False) or dones[i]:
                            died_flags[i] = True
                            dones_tracked[i] = True

                if all(dones_tracked[:n_active]):
                    break

            for i in range(n_active):
                fit = max_xs[i]
                if cleared_flags[i]:
                    fit += 1000.0 + (max_steps - steps_taken[i]) * 2.0

                res = {
                    "ind_idx": chunk_start + i,
                    "fitness": float(fit),
                    "max_x": float(max_xs[i]),
                    "steps": steps_taken[i],
                    "cleared": cleared_flags[i],
                    "stalled": stalled_flags[i],
                    "died": died_flags[i]
                }
                eval_results.append(res)

        eval_results.sort(key=lambda r: r["fitness"], reverse=True)
        best_res = eval_results[0]
        mean_fit = float(np.mean([r["fitness"] for r in eval_results]))
        max_fit = float(best_res["fitness"])
        max_x = float(best_res["max_x"])

        if max_fit > self.best_overall_fitness:
            self.best_overall_fitness = max_fit
            self.best_overall_policy = VisualIndividual(weights=self.population[best_res["ind_idx"]].theta.copy())

        # Elites
        new_population = []
        for i in range(self.elite_count):
            elite_idx = eval_results[i]["ind_idx"]
            new_population.append(VisualIndividual(weights=self.population[elite_idx].theta.copy()))

        # Mating pool
        parent_pool = [self.population[eval_results[i]["ind_idx"]] for i in range(self.pop_size // 2)]

        while len(new_population) < self.pop_size - 1:
            p1, p2 = self.rng.choice(parent_pool, size=2, replace=True)
            # Uniform crossover
            mask = self.rng.binomial(1, 0.5, size=len(p1.theta)).astype(bool)
            child_theta = np.where(mask, p1.theta, p2.theta)

            # Gaussian mutation
            mut_mask = self.rng.binomial(1, self.mutation_rate, size=len(child_theta)).astype(bool)
            noise = self.rng.normal(0.0, self.mutation_scale, size=len(child_theta)).astype(np.float32)
            child_theta += mut_mask * noise

            new_population.append(VisualIndividual(weights=child_theta))

        # Random immigrant
        new_population.append(VisualIndividual(seed=int(self.rng.integers(0, 1000000))))

        self.population = new_population
        elapsed = time.time() - gen_start

        gen_stat = {
            "generation": self.generation,
            "max_fitness": max_fit,
            "mean_fitness": mean_fit,
            "best_max_x_px": max_x,
            "best_steps": best_res["steps"],
            "best_cleared": best_res["cleared"],
            "time_s": elapsed
        }
        self.history.append(gen_stat)
        return gen_stat


def main():
    parser = argparse.ArgumentParser(description="Visual Neuroevolution for NSMB DS")
    parser.add_argument("--generations", type=int, default=12, help="Number of evolutionary generations")
    parser.add_argument("--pop-size", type=int, default=16, help="Population size")
    parser.add_argument("--elite", type=int, default=2, help="Elites count")
    parser.add_argument("--workers", type=int, default=4, help="Parallel DeSmuME workers")
    parser.add_argument("--max-steps", type=int, default=350, help="Max steps per individual")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--out-model", default="models/visual_genetic_best.pt", help="Output model path")
    parser.add_argument("--out-log", default="evals/visual_genetic_training_log.json", help="Output log path")
    args = parser.parse_args()

    print("=================================================================")
    print("  VISUAL NEUROEVOLUTION TRAINING (Raw Pixels on Live DeSmuME)")
    print("=================================================================")
    print(f"Pop size: {args.pop_size} | Elites: {args.elite} | Generations: {args.generations} | Workers: {args.workers}")
    print(f"Observation: Box(0, 255, (84, 84, 1), uint8) | Model: CompactVisualCNN (8,600 params)")

    trainer = VisualGeneticTrainer(
        pop_size=args.pop_size,
        elite_count=args.elite,
        mutation_rate=0.20,
        mutation_scale=0.08,
        seed=args.seed
    )

    from stable_baselines3.common.vec_env import SubprocVecEnv
    print(f"Launching {args.workers} isolated DeSmuME visual workers via SubprocVecEnv...")
    venv = SubprocVecEnv([make_visual_env_fn() for _ in range(args.workers)], start_method="spawn")

    t0 = time.time()
    try:
        for g in range(args.generations):
            stat = trainer.evolve_generation_parallel(venv, n_workers=args.workers, max_steps=args.max_steps)
            print(
                f"Gen {stat['generation']:02d}/{args.generations:02d} | "
                f"Best Max X: {stat['best_max_x_px']:7.1f} px | "
                f"Mean Fit: {stat['mean_fitness']:7.1f} | "
                f"Steps: {stat['best_steps']:3d} | "
                f"Cleared: {stat['best_cleared']} | "
                f"Time: {stat['time_s']:.1f}s",
                flush=True
            )
    finally:
        venv.close()

    total_time = time.time() - t0
    print("=================================================================")
    print(f"Visual Evolution finished in {total_time:.1f}s ({total_time/60:.2f} min).")
    print(f"Overall Best Fitness / Max X: {trainer.best_overall_fitness:.1f} px")

    # Save best policy weights
    os.makedirs(os.path.dirname(args.out_model) or ".", exist_ok=True)
    if trainer.best_overall_policy is not None:
        torch.save({
            "theta": trainer.best_overall_policy.theta,
            "state_dict": trainer.best_overall_policy.model.state_dict(),
            "best_fitness": trainer.best_overall_fitness,
            "history": trainer.history
        }, args.out_model)
        print(f"Saved best visual neuroevolutionary policy to {args.out_model}")

    # Save log
    os.makedirs(os.path.dirname(args.out_log) or ".", exist_ok=True)
    with open(args.out_log, "w") as f:
        json.dump({
            "generations": args.generations,
            "pop_size": args.pop_size,
            "elite": args.elite,
            "best_fitness": trainer.best_overall_fitness,
            "total_time_s": total_time,
            "history": trainer.history
        }, f, indent=2)
    print(f"Saved visual training log to {args.out_log}")


if __name__ == "__main__":
    main()
