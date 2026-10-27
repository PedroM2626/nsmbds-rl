"""Multi-World Transition Pool Orchestrator.

Invokes tools/collect_world_worker.py in a dedicated subprocess for each world,
guaranteeing exactly ONE DeSmuME instance per process (AGENTS.md hard constraint).
Then merges and validates the real transition pool into models/pinn_pool.npz.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

WORLDS = [
    {
        "name": "1-1",
        "course": "course/A01_1.bin",
        "state": os.path.join(ROOT, "data", "0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).ds1"),
        "controller": "reactive",
        "target_steps": 2500,
    },
    {
        "name": "1-2",
        "course": "course/A02_1.bin",
        "state": os.path.join(ROOT, "data", "state_1-2.ds1"),
        "controller": "reactive",
        "target_steps": 1800,
    },
    {
        "name": "2-1",
        "course": "course/B01_1.bin",
        "state": os.path.join(ROOT, "data", "state_2-1.ds1"),
        "controller": "reactive",
        "target_steps": 1800,
    },
    {
        "name": "4-1",
        "course": "course/D01_1.bin",
        "state": os.path.join(ROOT, "data", "state_4-1.ds1"),
        "controller": "world41",
        "target_steps": 1800,
    },
    {
        "name": "5-1",
        "course": "course/E01_1.bin",
        "state": os.path.join(ROOT, "data", "0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).ds2"),
        "controller": "world51",
        "target_steps": 1800,
    },
    {
        "name": "6-1",
        "course": "course/F01_1.bin",
        "state": os.path.join(ROOT, "data", "state_6-1.ds1"),
        "controller": "reactive",
        "target_steps": 1800,
    },
    {
        "name": "8-1",
        "course": "course/H01_1.bin",
        "state": os.path.join(ROOT, "data", "0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).ds3"),
        "controller": "world81",
        "target_steps": 1800,
    },
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "models", "pinn_pool.npz"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    worker_script = os.path.join(HERE, "collect_world_worker.py")
    out_dir = os.path.join(ROOT, "out", "world_pools")
    os.makedirs(out_dir, exist_ok=True)

    print("===============================================================")
    print("  MULTI-WORLD REAL TRANSITION COLLECTION ORCHESTRATOR")
    print(f"  Target courses: {[c['name'] for c in WORLDS]}")
    print("===============================================================", flush=True)

    t_start = time.time()
    curr_ep = 0
    worker_files = []

    for cfg in WORLDS:
        out_npz = os.path.join(out_dir, f"pool_{cfg['name']}.npz")
        cmd = [
            sys.executable, worker_script,
            "--course", cfg["course"],
            "--state", cfg["state"],
            "--controller", cfg["controller"],
            "--target-steps", str(cfg["target_steps"]),
            "--start-ep", str(curr_ep),
            "--seed", str(args.seed + curr_ep),
            "--out", out_npz
        ]
        print(f"\n>>> Spawning isolated DeSmuME worker for World {cfg['name']}...", flush=True)
        res = subprocess.run(cmd, check=True)
        if res.returncode != 0:
            sys.exit(f"Worker for {cfg['name']} failed with code {res.returncode}")

        p = np.load(out_npz)
        curr_ep = int(p["next_ep"])
        worker_files.append(out_npz)
        print(f">>> World {cfg['name']} worker completed successfully ({len(p['S'])} transitions).", flush=True)

    print("\n===============================================================")
    print("  CONCATENATING MULTI-WORLD TRANSITION POOL")
    print("===============================================================", flush=True)

    all_S, all_A, all_S2, all_R, all_D, all_EP, all_WORLD = [], [], [], [], [], [], []
    for cfg, f in zip(WORLDS, worker_files):
        p = np.load(f)
        if len(p["S"]) != cfg["target_steps"]:
            raise ValueError(f"Unexpected transition count for {cfg['name']}")
        all_S.append(p["S"])
        all_A.append(p["A"])
        all_S2.append(p["S2"])
        all_R.append(p["R"])
        all_D.append(p["D"])
        all_EP.append(p["EP"])
        all_WORLD.append(np.full(len(p["EP"]), cfg["name"]))

    S_cat = np.concatenate(all_S, axis=0)
    A_cat = np.concatenate(all_A, axis=0)
    S2_cat = np.concatenate(all_S2, axis=0)
    R_cat = np.concatenate(all_R, axis=0)
    D_cat = np.concatenate(all_D, axis=0)
    EP_cat = np.concatenate(all_EP, axis=0)
    WORLD_cat = np.concatenate(all_WORLD, axis=0)

    total_time = time.time() - t_start
    print(f"Total transitions collected: {len(S_cat)} across {curr_ep} episodes in {total_time:.1f}s")
    print(f"Data matrices: S={S_cat.shape}, A={A_cat.shape}, S2={S2_cat.shape}, R={R_cat.shape}, D={D_cat.shape}")
    print(f"Saving merged pool to: {args.out}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, S=S_cat, A=A_cat, S2=S2_cat, R=R_cat, D=D_cat,
                        EP=EP_cat, WORLD=WORLD_cat)
    print("Successfully committed multi-world pool to disk.", flush=True)


if __name__ == "__main__":
    main()
