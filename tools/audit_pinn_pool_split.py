"""Audit the committed PINN pool's episode split and course composition.

The pool has no course column. Its collection script concatenates courses in
WORLDS order, with exactly target_steps transitions per course. Check lengths
and episode boundaries before reporting the inferred per-course splits. The
pool itself cannot prove course identity without the original worker pools.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from collect_multiworld_pool import WORLDS
from train_enhanced_pinn import split_multiworld

ROOT = Path(__file__).resolve().parents[1]


def sha1(path):
    return hashlib.sha1(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=ROOT / "models/pinn_pool.npz")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    with np.load(args.pool, allow_pickle=False) as pool:
        data = tuple(pool[k] for k in ("S", "A", "S2", "R", "D", "EP"))
    episode = data[-1]
    expected = sum(w["target_steps"] for w in WORLDS)
    if len(episode) != expected or any(len(x) != expected for x in data):
        raise ValueError("Pool does not match the documented course collection sizes")
    parts = split_multiworld(data)
    split_ids = [set(map(int, np.unique(p[-1]))) for p in parts]
    courses = {}
    seen_ids = set()
    offset = 0
    for world in WORLDS:
        n = world["target_steps"]
        ids = set(map(int, np.unique(episode[offset:offset + n])))
        if ids & seen_ids:
            raise ValueError("An episode crosses course boundaries")
        seen_ids.update(ids)
        courses[world["name"]] = {
            "transitions": n,
            "episodes": len(ids),
            "train_episodes": len(ids & split_ids[0]),
            "validation_episodes": len(ids & split_ids[1]),
            "test_episodes": len(ids & split_ids[2]),
        }
        offset += n
    result = {
        "pool_sha1": sha1(args.pool),
        "collector_sha1": sha1(ROOT / "tools/collect_multiworld_pool.py"),
        "trainer_sha1": sha1(ROOT / "tools/train_enhanced_pinn.py"),
        "layout_assumption": "WORLDS order and target_steps; no course column in pool",
        "transitions": len(episode),
        "episodes": len(np.unique(episode)),
        "train_transitions": len(parts[0][-1]),
        "validation_transitions": len(parts[1][-1]),
        "test_transitions": len(parts[2][-1]),
        "courses": courses,
    }
    rendered = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
