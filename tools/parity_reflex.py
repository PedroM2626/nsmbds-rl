"""Parity control for the reflex generalization.

--reflex legacy must reproduce the published World 1-1 clear byte for byte, or
the generalization changed behaviour it claimed not to change. --reflex profile
is then reported, not asserted: the point of the arm is to find out how much of
the 1-1 clear came from geometry and how much from seven hand-placed pixel bands.

Usage:
    python tools/parity_reflex.py [--episodes 1]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable
PUBLISHED = os.path.join(ROOT, "evals", "pinn_realtime_eval.json")


def run(mode, episodes, extra):
    out = os.path.join(ROOT, "out", f"reflex_{mode}.json")
    cmd = [PY, os.path.join(ROOT, "src", "eval_pinn_realtime.py"),
           "--controller", "reactive", "--headless", "--episodes", str(episodes),
           "--reflex", mode, "--out", out] + extra
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:], file=sys.stderr)
        sys.exit(f"arm {mode} failed")
    return json.load(open(out, encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--planner", default="python", choices=["python", "rust"])
    args = ap.parse_args()
    os.makedirs(os.path.join(ROOT, "out"), exist_ok=True)

    if not os.path.exists(PUBLISHED):
        sys.exit(f"missing published eval {PUBLISHED}")
    pub = json.load(open(PUBLISHED, encoding="utf-8"))["episodes"][0]

    legacy = run("legacy", args.episodes, ["--planner", args.planner])
    l_ep = legacy["episodes"][0]
    keys = ("steps", "total_reward", "max_progress_px", "final_abs_px", "termination")
    bad = [k for k in keys if l_ep[k] != pub[k]]
    print("published evals/pinn_realtime_eval.json episode 1 vs legacy arm:")
    for k in keys:
        flag = "FAIL" if k in bad else "ok  "
        print(f"  {flag} {k}: legacy={l_ep[k]!r} published={pub[k]!r}")
    legacy_ok = not bad

    prof = run("profile", args.episodes, ["--planner", args.planner])
    p_ep = prof["episodes"][0]
    print("\nprofile arm (thresholds derived from the ROM, no 1-1 pixel bands):")
    for k in keys:
        same = "same" if p_ep[k] == l_ep[k] else "DIFFERS"
        print(f"  {k}: {p_ep[k]!r}  [{same}]")
    print(f"  cleared: {p_ep['termination'] == 'goal'}   legacy cleared: "
          f"{l_ep['termination'] == 'goal'}")

    if not legacy_ok:
        print("\nFAIL: legacy arm does not reproduce the published numbers", file=sys.stderr)
        return 1
    print("\nPASS: legacy reproduces the published clear exactly.")
    if p_ep["termination"] != l_ep["termination"]:
        print("NOTE: the profile arm reaches a different outcome -- the pixel bands "
              "were load-bearing, which is the number to report, not hide.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
