"""Audit evals/*.json for provenance blocks.

New evaluation artifacts are required to carry provenance (commit, ROM/state
hashes, versions and seed). Some historical training logs and metrics lack it,
including files written after the requirement was introduced. This audit only
checks whether a block exists; it does not infer an artifact's date or verify
the fields. Missing provenance must not be backfilled after the fact.
"""
from __future__ import annotations

import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    paths = sorted(glob.glob(os.path.join(ROOT, "evals", "*.json")))
    modern, legacy = [], []
    for p in paths:
        try:
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        bucket = modern if isinstance(d.get("provenance"), dict) else legacy
        bucket.append(os.path.relpath(p, ROOT))
    print(f"evals with provenance: {len(modern)}")
    for p in modern:
        print(f"  {p}")
    print(f"\nwithout provenance (date not inferred): {len(legacy)}")
    for p in legacy:
        print(f"  {p}")


if __name__ == "__main__":
    main()
