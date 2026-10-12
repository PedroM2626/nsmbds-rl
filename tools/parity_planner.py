"""Parity control for the Rust planner: nsmb_planner.exe must agree with
src/tilemap_planner.py on every query, state by state.

A port that is faster but differs is not a port, it is a new agent -- so this
script is the gate for --planner rust in src/rust_planner.py.

Usage:
    python tools/parity_planner.py [--n-random 5000] [--seed 7]
"""
from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from tilemap_planner import TilemapAStar, TILE  # noqa: E402

GEOM = os.path.join(ROOT, "out", "nsmb_geometry.bin")
EXE = os.path.join(ROOT, "rust", "nsmb_planner.exe")
DEFAULT_ROM = os.path.join(ROOT, "data",
                           "0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds")


def fmt_opt(v):
    return "NA" if v is None else str(int(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-random", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--rom", default=DEFAULT_ROM)
    args = ap.parse_args()

    if not os.path.exists(EXE):
        print(f"missing {EXE}: run rust/build.sh", file=sys.stderr)
        return 2
    if not os.path.exists(GEOM):
        print(f"missing {GEOM}: run tools/export_geometry.py", file=sys.stderr)
        return 2

    p = TilemapAStar(rom_path=args.rom)
    goal = p.goal_px()

    # Sweep every pixel column of the level plus random fractional positions:
    # the integer sweep catches off-by-one in the col mapping, the random draws
    # catch truncation differences between int(px/16) and Rust's as-i32.
    rnd = random.Random(args.seed)
    xs = [float(x) for x in range(0, int(goal) + 64)]
    xs += [rnd.uniform(0.0, float(goal) + 64.0) for _ in range(args.n_random)]

    queries, expected = [], []
    queries.append("G")
    expected.append(f"G {p.min_col} {p.max_col} {p.reach} {str(p.feasible).lower()} {int(goal)}")
    for x in xs:
        cur, prof = p.surface_ahead(x, look_tiles=5)
        queries.append(f"S {x:.6f} 5")
        expected.append("S " + fmt_opt(cur) + " " + " ".join(
            f"{c}:{fmt_opt(r)}" for (c, r) in prof))

        queries.append(f"W {x:.6f} 6 2")
        expected.append(f"W {p.wall_ahead(x, look_tiles=6, min_wall_tiles=2)}")

        d = p.pipe_ahead(x, look_tiles=5, min_height_tiles=2)
        queries.append(f"P {x:.6f} 5 2")
        expected.append("P NA" if d is None else f"P {float(d):.6f}")

        queries.append(f"N {x:.6f}")
        expected.append(f"N {1 if p.is_narrow_corridor(x) else 0}")

    qpath = os.path.join(ROOT, "out", "parity_queries.txt")
    with open(qpath, "w") as f:
        f.write("\n".join(queries) + "\n")

    got = subprocess.run([EXE, GEOM, "batch", qpath], capture_output=True, text=True)
    if got.returncode != 0:
        print(got.stdout, got.stderr, file=sys.stderr)
        return 2
    lines = got.stdout.strip().splitlines()
    if len(lines) != len(expected):
        print(f"FAIL: rust answered {len(lines)} of {len(expected)} queries",
              file=sys.stderr)
        return 2

    mism = []
    for i, (e, g) in enumerate(zip(expected, lines)):
        if e != g:
            mism.append((i, e, g))

    n_states = len(xs)
    print(f"states compared: {n_states} (sweep 0..{int(goal)}px + {args.n_random} random)")
    print(f"queries compared: {len(expected)}")
    if mism:
        print(f"FAIL: {len(mism)} mismatches, first 10:")
        for i, e, g in mism[:10]:
            print(f"  expected [{e}]\n  rust     [{g}]")
        return 1
    print("PASS: Rust output is identical to tilemap_planner.py on every query")
    return 0


if __name__ == "__main__":
    sys.exit(main())
