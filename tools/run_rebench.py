"""Queue runner for the v2 re-benchmark (PROJECT_STATUS 4az).

Runs every arm as: train -> eval best -> eval final, at most --parallel arms
at a time, each job in its own process (one DeSmuME per process). Logs go to
out/rebench/logs/<arm>.log and a status line per finished step to
out/rebench/logs/_status.jsonl. Launch detached (AGENTS.md: long runs):

  Start-Process -WindowStyle Hidden .venv\\Scripts\\python.exe "tools\\run_rebench.py --parallel 6"
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable
LOGS = os.path.join(ROOT, "out", "rebench", "logs")

# Heaviest first so the long poles start early.
SB3_ARMS = ["rtrf_impala_spr", "rppo_impala_curl", "rppo_hybrid", "rppo_curl", "rppo_spr",
            "rppo_icm_ae", "rppo_icm", "rppo_drq", "rppo_ae", "rppo_pure",
            "ppo_impala", "ppo_naturecnn", "ppo_gnn", "ppo_ram23", "ppo_ram17"]
WM_ARMS = ["wm_ga", "wm_cma"]


def jobs_for(arm, timesteps, num_envs):
    if arm.startswith("wm_"):
        train = [PY, "tools/train_rebench_worldmodels.py", "--variant", arm[3:],
                 "--timesteps", str(timesteps), "--workers", str(num_envs)]
        return [("train", train), ("eval_best", [PY, "tools/eval_rebench.py", "--arm", arm])]
    train = [PY, "tools/train_rebench.py", "--arm", arm, "--timesteps", str(timesteps),
             "--num-envs", str(num_envs)]
    return [("train", train),
            ("eval_best", [PY, "tools/eval_rebench.py", "--arm", arm, "--which", "best"]),
            ("eval_final", [PY, "tools/eval_rebench.py", "--arm", arm, "--which", "final"])]


def status(rec):
    with open(os.path.join(LOGS, "_status.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")


def run_arm(arm, timesteps, num_envs):
    with open(os.path.join(LOGS, f"{arm}.log"), "a", buffering=1) as log:
        for stage, cmd in jobs_for(arm, timesteps, num_envs):
            t0 = time.time()
            status({"arm": arm, "stage": stage, "event": "start", "t": time.time()})
            rc = subprocess.call(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                 env={**os.environ, "PYTHONIOENCODING": "utf-8"})
            status({"arm": arm, "stage": stage, "event": "end", "exit": rc,
                    "min": (time.time() - t0) / 60.0, "t": time.time()})
            if rc != 0:
                return


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--timesteps", type=int, default=100_000)
    ap.add_argument("--num-envs", type=int, default=4)
    ap.add_argument("--arms", default=",".join(SB3_ARMS + WM_ARMS))
    args = ap.parse_args()
    os.makedirs(LOGS, exist_ok=True)
    arms = [a for a in args.arms.split(",") if a]
    status({"event": "runner_start", "arms": arms, "timesteps": args.timesteps,
            "num_envs": args.num_envs, "parallel": args.parallel, "t": time.time()})
    with ThreadPoolExecutor(args.parallel) as ex:
        for a in arms:
            ex.submit(run_arm, a, args.timesteps, args.num_envs)
            time.sleep(20)   # stagger emulator start-up
    status({"event": "runner_end", "t": time.time()})


if __name__ == "__main__":
    main()
