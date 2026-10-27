"""Closed-loop imagination planning benchmark, one DeSmuME per subprocess.

The CEM planner sees only the real 23-dimensional observation and imagines
candidate futures in the learned PINN. There is no reflex/controller fallback
and no emulator savestate branching inside planning. Pipe actions remain out
of scope for the existing eight-action world-model checkpoint.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from collect_multiworld_pool import WORLDS
from eval_provenance import collect, sha1_file

COURSES = {w["name"]: w for w in WORLDS}
COURSES.update({
    "3-1": dict(course="course/C01_2.bin", state="data/state_3-1_main.ds1"),
    "7-1": dict(course="course/G01_1.bin", state="data/state_7-1.ds1"),
})


def worker(args):
    import numpy as np
    from multi_area_env import MultiAreaMarioEnv
    cfg = COURSES[args.world]
    env = MultiAreaMarioEnv(course_name=cfg["course"], state_path=str(ROOT / cfg["state"]),
                            geo=True, max_steps=args.max_steps,
                            water_overrides={"course/C01_2.bin": True,
                                             "course/G01_1.bin": False})
    try:
        # Both construction and identity validation precede every model load.
        env.reset(seed=args.seed)
        import torch
        from pinn_world_model import MarioPINNWorldModel, CEMMPController
        torch.set_num_threads(1)
        torch.manual_seed(args.seed)
        checkpoint = torch.load(args.model, map_location=args.device, weights_only=False)
        model = MarioPINNWorldModel(obs_dim=checkpoint.get("obs_dim", 23),
                                    n_actions=checkpoint.get("n_actions", 8),
                                    hid=checkpoint.get("hid", 128),
                                    layers=checkpoint.get("layers", 2),
                                    k=checkpoint.get("k", 1 / 16)).to(args.device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        if model.obs_dim != env.observation_space.shape[0] or model.n_actions != 8:
            raise ValueError("Checkpoint must match the 23-observation/eight-action protocol")
        provenance = collect(env.rom_path, env.state_path,
                             sources=[__file__, ROOT / "src/pinn_world_model.py",
                                      ROOT / "src/multi_area_env.py", ROOT / "src/ram_env.py"],
                             extra=dict(seed=args.seed, course_index=env.ram.course_index(),
                                        model_sha1=sha1_file(args.model)))
        episodes = []
        for ep in range(args.episodes):
            seed = args.seed + ep
            obs, _ = env.reset(seed=seed)
            controller = CEMMPController(model, horizon=args.horizon,
                                          n_candidates=args.candidates, n_elite=args.elites,
                                          n_iters=args.iterations, device=args.device, seed=seed)
            st = env.ram.poll()
            max_x = st["mario_B"] / 4096.0
            progress_anchor = max_x
            last_gain, total, previous = 0, 0.0, None
            trajectory, squared_errors = [], []
            started = time.monotonic()
            reason = "step_cap"
            for step in range(1, args.max_steps + 1):
                action, probabilities = controller.plan(obs, best_prev=previous)
                with torch.no_grad():
                    predicted = model.step_batch(
                        torch.as_tensor(obs[None], device=args.device),
                        torch.tensor([action], device=args.device))[0][0].cpu().numpy()
                next_obs, reward, done, trunc, info = env.step(action)
                squared_errors.append(float(np.sum(((predicted[:2] - next_obs[:2]) * 512) ** 2)))
                total += reward
                trajectory.append(dict(step=step, action=action, reward=reward,
                                       mario_px=info["mario_px"], mario_y_px=info["mario_y_px"],
                                       course=info["course"], died=info["died"],
                                       cleared=info["level_cleared"]))
                x = info["mario_px"]
                if x is not None:
                    max_x = max(max_x, x)
                    if x > progress_anchor + 1:
                        progress_anchor, last_gain = x, step
                obs = next_obs
                # Shift the previous distribution by the one executed action.
                previous = torch.cat((probabilities[1:],
                                      torch.full_like(probabilities[:1], 1 / 8)), dim=0)
                if done:
                    reason = ("goal" if info["level_cleared"] else
                              "death" if info["died"] else "timeout")
                    break
                if step - last_gain >= args.stall_steps:
                    reason = "stalled"
                    break
                if trunc:
                    break
            record = dict(seed=seed, steps=step, max_x_px=max_x, final=info,
                          reward=total, cleared=info["level_cleared"], termination=reason,
                          elapsed_s=time.monotonic() - started,
                          one_step_position_rmse_px=float(np.sqrt(np.mean(squared_errors))),
                          trajectory=trajectory)
            episodes.append(record)
            print(f"{args.world} seed={seed} steps={step} max_x={max_x:.1f} {reason}", flush=True)
        result = dict(world=args.world, method="PINN_CEM_MPC", reflex=False,
                      action_count=8, observation_dim=model.obs_dim,
                      config=dict(horizon=args.horizon, candidates=args.candidates,
                                  elites=args.elites, iterations=args.iterations,
                                  max_steps=args.max_steps, stall_steps=args.stall_steps),
                      checkpoint_metrics=checkpoint.get("metrics"), provenance=provenance,
                      episodes=episodes,
                      clear_rate=sum(e["cleared"] for e in episodes) / len(episodes))
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    finally:
        env.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--world", choices=COURSES, help="Worker mode: evaluate just this course")
    ap.add_argument("--worlds", nargs="+", choices=COURSES, default=list(COURSES))
    ap.add_argument("--model", default=str(ROOT / "models/pinn_nsmb.pt"))
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=450)
    ap.add_argument("--stall-steps", type=int, default=100)
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--candidates", type=int, default=96)
    ap.add_argument("--elites", type=int, default=12)
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--out", default=str(ROOT / "evals/pinn_imagination_multiworld.json"))
    args = ap.parse_args()
    if min(args.episodes, args.max_steps, args.stall_steps, args.horizon,
           args.candidates, args.elites, args.iterations) < 1 or args.elites > args.candidates:
        ap.error("Budgets must be positive and elites cannot exceed candidates")
    if args.world:
        worker(args)
        return
    output = Path(args.out)
    worlds = {}
    for world in args.worlds:
        destination = output.with_name(f"{output.stem}_{world.replace('-', '_')}.json")
        cmd = [sys.executable, __file__, "--world", world, "--out", str(destination)]
        for name in ("model", "episodes", "max_steps", "stall_steps", "horizon", "candidates",
                     "elites", "iterations", "seed", "device"):
            cmd.extend(("--" + name.replace("_", "-"), str(getattr(args, name))))
        subprocess.run(cmd, cwd=ROOT, check=True)
        record = json.loads(destination.read_text(encoding="utf-8"))
        worlds[world] = dict(file=str(destination.relative_to(ROOT)) if destination.is_relative_to(ROOT)
                             else str(destination), clear_rate=record["clear_rate"],
                             episodes=[{k: ep[k] for k in ("seed", "steps", "max_x_px", "termination",
                                                          "one_step_position_rmse_px")}
                                       for ep in record["episodes"]])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(method="PINN_CEM_MPC", reflex=False, worlds=worlds,
                                     provenance=collect(extra=dict(seed=args.seed,
                                                                  model_sha1=sha1_file(args.model)))),
                                 indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
