"""Neuroevolution / Genetic Algorithm for NSMB DS across Worlds 2-1, 4-1, 5-1, 6-1, 8-1.

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
from eval_provenance import collect as collect_provenance
from course_id import course_area_index, course_for_area_index
import paths

OBS_DIM = 23
N_ACTIONS = 8
HIDDEN_DIM = 32

WORLD_CONFIGS = {
    "2-1": {
        "course": "course/B01_1.bin",
        "state": "data/state_2-1.ds1",
        "goal_px": 4032.0,
        "spawn_px": 360.0,
        "max_steps": 400,
    },
    "4-1": {
        "course": "course/D01_1.bin",
        "state": "data/state_4-1.ds1",
        "goal_px": 4344.0,
        "spawn_px": 40.0,
        "max_steps": 450,
    },
    "5-1": {
        "course": "course/E01_1.bin",
        "state": "data/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).ds2",
        "goal_px": 4552.0,
        "spawn_px": 40.0,
        "max_steps": 500,
    },
    "6-1": {
        "course": "course/F01_1.bin",
        "state": "data/state_6-1.ds1",
        "goal_px": 4080.0,
        "spawn_px": 40.0,
        "max_steps": 450,
    },
    "8-1": {
        "course": "course/H01_1.bin",
        "state": "data/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).ds3",
        "goal_px": 3920.0,
        "spawn_px": 56.0,
        "max_steps": 450,
    },
}


class CompactPolicy:
    """Compact 2-layer MLP without PyTorch overhead for fast CPU execution."""

    def __init__(self, weights: np.ndarray | None = None, seed: int | None = None):
        if weights is not None:
            self.theta = weights.astype(np.float32)
        else:
            rng = np.random.default_rng(seed)
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
        h = np.tanh(self.w1 @ obs + self.b1)
        logits = self.w2 @ h + self.b2
        return int(np.argmax(logits))


def make_world_env_fn(course_path: str, state_path: str):
    """Factory function for top-level SubprocVecEnv worker instantiation."""
    def _thunk():
        return MarioRamEnv(state_path=state_path, course_name=course_path, geo=True)
    return _thunk


class MultiWorldGeneticTrainer:
    """Genetic algorithm trainer across any target world."""

    def __init__(
        self,
        pop_size: int = 24,
        elite_count: int = 4,
        mutation_rate: float = 0.20,
        mutation_scale: float = 0.08,
        seed: int = 42
    ):
        self.pop_size = pop_size
        self.elite_count = elite_count
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.rng = np.random.default_rng(seed)

        self.population = [CompactPolicy(seed=int(self.rng.integers(0, 1000000))) for _ in range(pop_size)]
        self.generation = 0
        self.best_overall_fitness = 0.0
        self.best_overall_policy: CompactPolicy | None = None
        self.history = []

    def evolve_generation_parallel(self, venv, n_workers: int, max_steps: int = 400, stall_limit: int = 25) -> dict:
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
            self.best_overall_policy = CompactPolicy(weights=self.population[best_res["ind_idx"]].theta.copy())

        # Select elites
        new_population = []
        for i in range(self.elite_count):
            elite_idx = eval_results[i]["ind_idx"]
            new_population.append(CompactPolicy(weights=self.population[elite_idx].theta.copy()))

        parent_pool = [self.population[eval_results[i]["ind_idx"]] for i in range(self.pop_size // 2)]

        while len(new_population) < self.pop_size - 1:
            p1, p2 = self.rng.choice(parent_pool, size=2, replace=True)
            mask = self.rng.binomial(1, 0.5, size=len(p1.theta)).astype(bool)
            child_theta = np.where(mask, p1.theta, p2.theta)

            mut_mask = self.rng.binomial(1, self.mutation_rate, size=len(child_theta)).astype(bool)
            noise = self.rng.normal(0.0, self.mutation_scale, size=len(child_theta)).astype(np.float32)
            child_theta += mut_mask * noise
            new_population.append(CompactPolicy(weights=child_theta))

        new_population.append(CompactPolicy(seed=int(self.rng.integers(0, 1000000))))

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


def evaluate_policy(policy: CompactPolicy, course: str, state: str, episodes: int = 5, max_steps: int = 500, seed: int = 42) -> dict:
    """Evaluates champion policy on target world with cryptographic provenance."""
    env = MarioRamEnv(state_path=state, course_name=course, geo=True, max_steps=max_steps)
    results = []

    for ep in range(1, episodes + 1):
        obs, _ = env.reset(seed=seed + ep)
        max_x = 0.0
        final_x = 0.0
        steps = 0
        total_reward = 0.0
        cleared = False
        died = False
        cause = "timeout"
        recent_x = []

        for step in range(1, max_steps + 1):
            action = policy.act(obs)
            obs, r, done, trunc, info = env.step(action)
            steps = step
            total_reward += r

            mario_px = float(info.get("mario_px", 0.0))
            if mario_px > max_x:
                max_x = mario_px
            final_x = mario_px

            recent_x.append(mario_px)
            if len(recent_x) > 30:
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

    env.close()

    mean_max_x = float(np.mean([r["max_x_px"] for r in results]))
    max_max_x = float(np.max([r["max_x_px"] for r in results]))
    clear_rate = float(np.mean([r["cleared"] for r in results]))
    mean_steps = float(np.mean([r["steps"] for r in results]))

    prov = collect_provenance(
        rom=paths.ROM_REL,
        state=state,
        extra={"seed": seed}
    )

    return {
        "n_episodes": episodes,
        "mean_max_x_px": mean_max_x,
        "max_max_x_px": max_max_x,
        "mean_steps": mean_steps,
        "clear_rate": clear_rate,
        "episodes": results,
        "provenance": prov
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-World Neuroevolution Trainer")
    parser.add_argument("--world", required=True, choices=list(WORLD_CONFIGS.keys()), help="Target world (e.g. 2-1, 4-1, 5-1, 6-1, 8-1)")
    parser.add_argument("--generations", type=int, default=15, help="Number of evolutionary generations")
    parser.add_argument("--pop-size", type=int, default=24, help="Population size")
    parser.add_argument("--elite", type=int, default=4, help="Elites count")
    parser.add_argument("--workers", type=int, default=4, help="Parallel DeSmuME workers")
    parser.add_argument("--max-steps", type=int, default=0, help="Max steps per individual (0 = use world default)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--out-model", default="", help="Output model path (npz)")
    parser.add_argument("--out-log", default="", help="Output log path (json)")
    parser.add_argument("--out-eval", default="", help="Output eval path (json)")
    args = parser.parse_args()

    cfg = WORLD_CONFIGS[args.world]
    max_steps = args.max_steps if args.max_steps > 0 else cfg["max_steps"]
    out_model = args.out_model or f"models/genetic_policy_w{args.world.replace('-', '')}.npz"
    out_log = args.out_log or f"evals/genetic_training_log_w{args.world.replace('-', '')}.json"
    out_eval = args.out_eval or f"evals/eval_genetic_w{args.world.replace('-', '')}.json"

    print("=================================================================")
    print(f"  MULTI-WORLD NEUROEVOLUTION TRAINING — WORLD {args.world}")
    print("=================================================================")
    print(f"Course: {cfg['course']} | Savestate: {cfg['state']}")
    print(f"Goal: {cfg['goal_px']:.1f} px | Spawn: {cfg['spawn_px']:.1f} px | Max Steps: {max_steps}")
    print(f"Pop size: {args.pop_size} | Elites: {args.elite} | Generations: {args.generations} | Workers: {args.workers}")

    trainer = MultiWorldGeneticTrainer(
        pop_size=args.pop_size,
        elite_count=args.elite,
        mutation_rate=0.20,
        mutation_scale=0.08,
        seed=args.seed
    )

    from stable_baselines3.common.vec_env import SubprocVecEnv
    print(f"Launching {args.workers} isolated DeSmuME workers via SubprocVecEnv...")
    venv = SubprocVecEnv([make_world_env_fn(cfg["course"], cfg["state"]) for _ in range(args.workers)], start_method="spawn")

    t0 = time.time()
    try:
        for g in range(args.generations):
            stat = trainer.evolve_generation_parallel(venv, n_workers=args.workers, max_steps=max_steps)
            print(
                f"Gen {stat['generation']:02d}/{args.generations:02d} | "
                f"Best Max X: {stat['best_max_x_px']:7.1f} px / {cfg['goal_px']:.1f} px | "
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
    print(f"Evolution finished in {total_time:.1f}s ({total_time/60:.2f} min).")
    print(f"Overall Best Fitness / Max X: {trainer.best_overall_fitness:.1f} px")

    # Save model
    os.makedirs(os.path.dirname(out_model) or ".", exist_ok=True)
    if trainer.best_overall_policy is not None:
        np.savez(
            out_model,
            theta=trainer.best_overall_policy.theta,
            w1=trainer.best_overall_policy.w1,
            b1=trainer.best_overall_policy.b1,
            w2=trainer.best_overall_policy.w2,
            b2=trainer.best_overall_policy.b2,
            best_fitness=trainer.best_overall_fitness,
            history=trainer.history
        )
        print(f"Saved best neuroevolutionary policy to {out_model}")

    # Save training log
    os.makedirs(os.path.dirname(out_log) or ".", exist_ok=True)
    with open(out_log, "w") as f:
        json.dump({
            "world": args.world,
            "course": cfg["course"],
            "state": cfg["state"],
            "generations": args.generations,
            "pop_size": args.pop_size,
            "elite": args.elite,
            "best_fitness": trainer.best_overall_fitness,
            "total_time_s": total_time,
            "history": trainer.history
        }, f, indent=2)
    print(f"Saved training log to {out_log}")

    # Run formal 5-episode evaluation with provenance
    if trainer.best_overall_policy is not None:
        print(f"\nRunning 5-episode benchmark evaluation on World {args.world}...")
        eval_data = evaluate_policy(
            trainer.best_overall_policy,
            course=cfg["course"],
            state=cfg["state"],
            episodes=5,
            max_steps=max_steps,
            seed=args.seed
        )
        eval_data["world"] = args.world
        eval_data["model"] = out_model

        os.makedirs(os.path.dirname(out_eval) or ".", exist_ok=True)
        with open(out_eval, "w") as f:
            json.dump(eval_data, f, indent=2)

        print(f"Evaluation Summary for World {args.world}:")
        print(f"  Mean Max X: {eval_data['mean_max_x_px']:.1f} px")
        print(f"  Peak Max X: {eval_data['max_max_x_px']:.1f} px / {cfg['goal_px']:.1f} px")
        print(f"  Clear Rate: {eval_data['clear_rate']*100:.1f}%")
        print(f"Saved evaluation with provenance to {out_eval}")


if __name__ == "__main__":
    main()
