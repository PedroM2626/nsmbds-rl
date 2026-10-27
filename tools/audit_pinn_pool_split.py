"""Audit the current course-stratified PINN split without training a model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from train_enhanced_pinn import LEGACY_AUDIT, load_pool, split_multiworld, split_summary
from eval_provenance import sha1_file

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=ROOT / "models/pinn_pool.npz")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.out and args.out.resolve() == Path(LEGACY_AUDIT).resolve():
        parser.error("Refusing to overwrite the historical hash-bound pool audit")
    data = load_pool(args.pool)
    result = {
        "pool_sha1": sha1_file(args.pool),
        "seed": args.seed,
        "split_source_sha1": sha1_file(ROOT / "tools/train_enhanced_pinn.py"),
        "course_labels": "WORLD array, or hash-bound historical audit for legacy pool",
        "split": split_summary(split_multiworld(data, seed=args.seed)),
    }
    rendered = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
