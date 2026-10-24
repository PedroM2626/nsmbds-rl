"""One command for the full pre-commit gate set.

Usage:
    python tools/gates.py            # numbers + planner + pytest (fast set)
    python tools/gates.py --all      # + the reflex parity gate (emulator, ~45 s)

Exit code is non-zero when any gate fails; each gate runs regardless of the
others so one summary shows everything that broke.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable


def run(name, cmd, cwd=ROOT):
    label = "PASS" if True else "FAIL"
    print(f"\n=== {name} ===", flush=True)
    r = subprocess.run(cmd, cwd=cwd)
    ok = r.returncode == 0
    verdict = "PASS" if ok else "FAIL"
    print(f"=== {name}: {verdict} ===", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true",
                    help="include the emulator parity gate (~45 s)")
    args = ap.parse_args()
    gates = [
        ("numbers", [PY, os.path.join("tools", "check_readme_numbers.py")]),
        ("planner", [PY, os.path.join("tools", "parity_planner.py")]),
        ("tests",   [PY, "-m", "pytest", "tests", "-q"]),
    ]
    if args.all:
        gates.insert(1, ("parity", [PY, os.path.join("tools", "parity_reflex.py")]))
    results = {}
    for name, cmd in gates:
        results[name] = run(name, cmd)
    print("\n" + "=" * 48)
    for name, ok in results.items():
        verdict = "PASS" if ok else "FAIL"
        print(f"  {verdict}  {name}")
    print("=" * 48)
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
