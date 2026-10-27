"""Campaign runner for Multi-World Neuroevolution across Worlds 4-1, 5-1, 6-1, 8-1.

Runs parallel neuroevolution (GA) on each remaining target world in sequence,
saving the champion model, training log, and 5-episode benchmark evaluation with provenance.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

WORLDS_DEFAULT = ["4-1", "5-1", "6-1", "8-1"]


def main():
    parser = argparse.ArgumentParser(description="Multi-World Neuroevolution Campaign Runner")
    parser.add_argument("--worlds", nargs="+", default=WORLDS_DEFAULT, help="Worlds to train on")
    parser.add_argument("--generations", type=int, default=15, help="Generations per world")
    parser.add_argument("--pop-size", type=int, default=20, help="Population size")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    args = parser.parse_args()

    print("=================================================================")
    print("  MULTI-WORLD NEUROEVOLUTION CAMPAIGN")
    print("=================================================================")
    print(f"Target Worlds: {args.worlds}")
    print(f"Generations per world: {args.generations} | Pop size: {args.pop_size} | Workers: {args.workers}")

    t_start = time.time()
    for w in args.worlds:
        print(f"\n>>> Starting Campaign for World {w} at {time.strftime('%X')}...")
        cmd = [
            sys.executable, "tools/train_neuroevolution_multiworld.py",
            "--world", w,
            "--generations", str(args.generations),
            "--pop-size", str(args.pop_size),
            "--workers", str(args.workers)
        ]
        t0 = time.time()
        ret = subprocess.run(cmd)
        elapsed = time.time() - t0
        print(f">>> World {w} finished in {elapsed:.1f}s ({elapsed/60:.2f} min) with return code {ret.returncode}.")
        if ret.returncode != 0:
            print(f"ERROR: World {w} failed with code {ret.returncode}. Aborting campaign.")
            sys.exit(ret.returncode)

    total_time = time.time() - t_start
    print("\n=================================================================")
    print(f"Campaign finished successfully in {total_time:.1f}s ({total_time/60:.2f} min).")


if __name__ == "__main__":
    main()
