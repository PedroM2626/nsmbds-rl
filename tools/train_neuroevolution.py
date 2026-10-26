"""Neuroevolution / Genetic Algorithm for Nintendo DS Super Mario Bros.

Optimizes a compact neural policy mapping RAM telemetry (23-dim) to discrete actions
using an elitist evolutionary strategy with fitness = max(mario_px).

Zero gradient descent, zero synthetic data: every evaluation executes live on DeSmuME.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from ram_env import MarioRamEnv

OBS_DIM = 23
N_ACTIONS = 8
HIDDEN_DIM = 32

# Flat parameter dimension: W1 (32x23), b1 (32), W2 (8x32), b2 (8)
PARAM_DIM = (HIDDEN_DIM * OBS_DIM) + HIDDEN_DIM + (N_ACTIONS * HIDDEN_DIM) + N_ACTIONS


class CompactPolicy:
    """Compact 2-layer MLP without PyTorch overhead for fast CPU execution."""

    def __init__(self, weights: np.ndarray | None = None, seed: int | None = None):
        if weights is not None:
            self.theta = weights.astype(np.float32)
        else:
            rng = np.random.default_rng(seed)
            # Xavier-like initialization
            w1 = rng.normal(0, np.sqrt(2.0 / (OBS_DIM + HIDDEN_DIM)), size=(HIDDEN_DIM, OBS_DIM))
            b1 = np.zeros(HIDDEN_DIM, dtype=np.float32)
            w2 = rng.normal(0, np.sqrt(2.0 / (HIDDEN_DIM + N_ACTIONS)), size=(N_ACTIONS, HIDDEN_DIM))
            b2 = np.zeros(N_ACTIONS, dtype=np.float32)
            self.theta = np.concatenate([w1.flatten(), b1, w2.flatten(), b2]).astype(np.float32)

        self._unpack()

    def _unpack(self):
        idx = 0
        w1_size = HIDDEN_DIM * OBS_DIM
        self.w1 = self.theta[idx:idx + w1_size].reshape(HIDDEN_DIM, OBS_DIM)
        idx += w1_size

        self.b1 = self.theta[idx:idx + HIDDEN_DIM]
        idx += HIDDEN_DIM

        w2_size = N_ACTIONS * HIDDEN_DIM
        self.w2 = self.theta[idx:idx + w2_size].reshape(N_ACTIONS, HIDDEN_DIM)
        idx += w2_size

        self.b2 = self.theta[idx:idx + N_ACTIONS]

    def act(self, obs: np.ndarray) -> int:
        """Forward pass: returns discrete action index 0..7."""
        # Layer 1
        h = np.tanh(self.w1 @ obs + self.b1)
        # Layer 2 (logits)
        logits = self.w2 @ h + self.b2
        return int(np.argmax(logits))


def evaluate_individual(policy: CompactPolicy, env: MarioRamEnv, max_steps: int = 400, stall_limit: int = 25) -> dict:
    """Evaluates one individual on a live DeSmuME emulator instance.

    Returns dict with fitness (max_x), steps, cleared status, and termination reason.
    """
    obs, _ = env.reset()
    max_x = 0.0
    recent_x = []
    stalled = False
    cleared = False
    died = False

    for step in range(max_steps):
        action = policy.act(obs)
        obs, reward, done, trunc, info = env.step(action)

        # True absolute pixel position from RAM
        mario_px = float(info.get("mario_px", 0.0))
        if mario_px > max_x:
            max_x = mario_px

        recent_x.append(mario_px)
        if len(recent_x) > stall_limit:
            recent_x.pop(0)
            # If Mario has not advanced at least 2.5 px over stall_limit steps
            if max(recent_x) - min(recent_x) < 2.5:
                stalled = True
                break

        if info.get("level_cleared", False) or info.get("cleared", False):
            cleared = True
            break

        if info.get("died", False) or done or trunc:
            died = True
            break

    # Fitness: max horizontal coordinate reached
    fitness = max_x
    if cleared:
        # Clear bonus + speed incentive
        fitness += 1000.0 + (max_steps - step) * 2.0

    return {
        "fitness": float(fitness),
        "max_x": float(max_x),
        "steps": step + 1,
        "cleared": cleared,
        "stalled": stalled,
        "died": died
    }


class GeneticTrainer:
    """Elitist genetic algorithm optimizer for MarioRamEnv policies."""

    def __init__(
        self,
        pop_size: int = 24,
        elite_count: int = 4,
        mutation_rate: float = 0.2,
        mutation_scale: float = 0.08,
        seed: int = 42
    ):
        self.pop_size = pop_size
        self.elite_count = elite_count
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.rng = np.random.default_rng(seed)

        # Initialize population
        self.population = [CompactPolicy(seed=self.rng.integers(0, 1000000)) for _ in range(pop_size)]
        self.generation = 0
        self.best_overall_fitness = -1.0
        self.best_overall_policy: CompactPolicy | None = None
        self.history: list[dict] = []

    def evolve_generation(self, env: MarioRamEnv, max_steps: int = 350) -> dict:
        self.generation += 1
        eval_results = []
        gen_start = time.time()

        for idx, ind in enumerate(self.population):
            res = evaluate_individual(ind, env, max_steps=max_steps)
            res["ind_idx"] = idx
            eval_results.append(res)

        # Sort by fitness descending
        eval_results.sort(key=lambda r: r["fitness"], reverse=True)

        best_res = eval_results[0]
        mean_fit = float(np.mean([r["fitness"] for r in eval_results]))
        max_fit = float(best_res["fitness"])
        max_x = float(best_res["max_x"])

        if max_fit > self.best_overall_fitness:
            self.best_overall_fitness = max_fit
            self.best_overall_policy = CompactPolicy(weights=self.population[best_res["ind_idx"]].theta.copy())

        # Select elites
        new_population = []
        for i in range(self.elite_count):
            elite_idx = eval_results[i]["ind_idx"]
            new_population.append(CompactPolicy(weights=self.population[elite_idx].theta.copy()))

        # Parent selection pool (top 50%)
        top_indices = [r["ind_idx"] for r in eval_results[:self.pop_size // 2]]

        # Generate offspring
        while len(new_population) < self.pop_size - 1:
            p1_idx = self.rng.choice(top_indices)
            p2_idx = self.rng.choice(top_indices)
            p1 = self.population[p1_idx]
            p2 = self.population[p2_idx]

            # Uniform crossover
            mask = self.rng.binomial(1, 0.5, size=PARAM_DIM).astype(bool)
            child_theta = np.where(mask, p1.theta, p2.theta)

            # Mutation
            mut_mask = self.rng.binomial(1, self.mutation_rate, size=PARAM_DIM).astype(bool)
            mut_noise = self.rng.normal(0, self.mutation_scale, size=PARAM_DIM).astype(np.float32)
            child_theta = child_theta + mut_mask * mut_noise

            new_population.append(CompactPolicy(weights=child_theta))

        # Add 1 immigrant individual for genetic diversity
        new_population.append(CompactPolicy(seed=self.rng.integers(0, 1000000)))

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
    parser = argparse.ArgumentParser(description="Neuroevolution for NSMB DS")
    parser.add_argument("--generations", type=int, default=15, help="Number of evolutionary generations")
    parser.add_argument("--pop-size", type=int, default=24, help="Population size")
    parser.add_argument("--elite", type=int, default=4, help="Elites count")
    parser.add_argument("--max-steps", type=int, default=350, help="Max steps per individual")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--out-model", default="models/genetic_policy.npz", help="Output model path")
    parser.add_argument("--out-log", default="evals/genetic_training_log.json", help="Output log path")
    args = parser.parse_args()

    print("=================================================================")
    print("  NEUROEVOLUTION AGENT TRAINING (Live DeSmuME Emulator)")
    print("=================================================================")
    print(f"Pop size: {args.pop_size} | Elites: {args.elite} | Generations: {args.generations}")
    print(f"Fitness function: f = max(mario_px) | Model: Compact MLP (1,032 params)")

    env = MarioRamEnv(geo=True)
    trainer = GeneticTrainer(
        pop_size=args.pop_size,
        elite_count=args.elite,
        mutation_rate=0.20,
        mutation_scale=0.08,
        seed=args.seed
    )

    t0 = time.time()
    for g in range(args.generations):
        stat = trainer.evolve_generation(env, max_steps=args.max_steps)
        print(
            f"Gen {stat['generation']:02d}/{args.generations:02d} | "
            f"Best Max X: {stat['best_max_x_px']:7.1f} px | "
            f"Mean Fit: {stat['mean_fitness']:7.1f} | "
            f"Steps: {stat['best_steps']:3d} | "
            f"Cleared: {stat['best_cleared']} | "
            f"Time: {stat['time_s']:.1f}s",
            flush=True
        )

    total_time = time.time() - t0
    print("=================================================================")
    print(f"Evolution finished in {total_time:.1f}s ({total_time/60:.2f} min).")
    print(f"Overall Best Fitness / Max X: {trainer.best_overall_fitness:.1f} px")

    # Save best policy weights
    os.makedirs(os.path.dirname(args.out_model) or ".", exist_ok=True)
    if trainer.best_overall_policy is not None:
        np.savez(
            args.out_model,
            theta=trainer.best_overall_policy.theta,
            w1=trainer.best_overall_policy.w1,
            b1=trainer.best_overall_policy.b1,
            w2=trainer.best_overall_policy.w2,
            b2=trainer.best_overall_policy.b2,
            best_fitness=trainer.best_overall_fitness,
            history=trainer.history
        )
        print(f"Saved best neuroevolutionary policy to {args.out_model}")

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
    print(f"Saved training log to {args.out_log}")


if __name__ == "__main__":
    main()
