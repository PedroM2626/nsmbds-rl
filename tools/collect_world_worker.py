"""Worker process to collect real NSMB-DS RAM transitions for a single world.

Runs strictly ONE DeSmuME instance per process to respect the DeSmuME C-library
lifecycle contract (AGENTS.md hard constraint).

Usage:
    python tools/collect_world_worker.py --course course/A01_1.bin --state ... --out out/pool_1_1.npz
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from ram_env import MarioRamEnv
from pinn_world_model import N_ACTIONS
import eval_pinn_realtime as ev

ROM = os.path.join(ROOT, "data", "0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--course", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--controller", default="forward")
    ap.add_argument("--target-steps", type=int, default=1800)
    ap.add_argument("--start-ep", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    # Forward biased distribution for high-speed running and jumping
    p_forward = np.array([0.05, 0.20, 0.13, 0.20, 0.30, 0.03, 0.03, 0.03])
    p_forward = p_forward / p_forward.sum()

    env = MarioRamEnv(rom_path=ROM, state_path=args.state, geo=True,
                      course_name=args.course, max_steps=1000)
    if not env.has_emulator:
        sys.exit(f"Failed to initialize emulator for {args.course}")

    agent = None
    if args.controller == "world41":
        from world_4_1_controller import World41Agent
        agent = World41Agent(rom_path=ROM, course_name=args.course, ram=env.ram)
    elif args.controller == "world51":
        from world_5_1_controller import World51Agent
        agent = World51Agent(rom_path=ROM, course_name=cfg["course"] if "cfg" in locals() else args.course, ram=env.ram)
    elif args.controller == "world81":
        from world_8_1_controller import World81Agent
        agent = World81Agent(rom_path=ROM, course_name=args.course, ram=env.ram)
    elif args.controller == "reactive":
        agent = ev.ReactiveAgent(planner=None, enabled=True, ram=env.ram)

    S, A, S2, R, D, EP = [], [], [], [], [], []
    ep_id = args.start_ep
    use_controller = True
    obs, _ = env.reset()
    if agent is not None:
        agent.reset()

    t0 = time.time()
    print(f"[{args.course}] Collecting {args.target_steps} steps...", flush=True)

    while len(S) < args.target_steps:
        if agent is not None and use_controller and rng.random() > 0.30:
            try:
                a = int(agent.act(obs))
            except Exception:
                a = int(rng.choice(N_ACTIONS, p=p_forward))
        else:
            a = int(rng.choice(N_ACTIONS, p=p_forward))

        obs2, r, term, trunc, _ = env.step(a)
        S.append(obs.astype(np.float32))
        A.append(a)
        S2.append(obs2.astype(np.float32))
        R.append(np.float32(r))
        D.append(np.float32(1.0 if term else 0.0))
        EP.append(ep_id)
        obs = obs2

        if term or trunc:
            obs, _ = env.reset()
            if agent is not None:
                agent.reset()
            ep_id += 1
            use_controller = not use_controller

    env.close()
    elapsed = time.time() - t0
    rate = len(S) / max(elapsed, 1e-6)
    print(f"[{args.course}] Collected {len(S)} transitions in {elapsed:.1f}s ({rate:.0f} steps/s), episodes={ep_id - args.start_ep}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(
        args.out,
        S=np.stack(S),
        A=np.array(A, np.int64),
        S2=np.stack(S2),
        R=np.array(R, np.float32),
        D=np.array(D, np.float32),
        EP=np.array(EP, np.int64),
        next_ep=np.int64(ep_id + 1)
    )
    print(f"[{args.course}] Saved to {args.out}", flush=True)


if __name__ == "__main__":
    main()
