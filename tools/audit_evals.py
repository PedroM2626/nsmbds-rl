"""Audit evals/*.json for provenance blocks.

Every eval JSON written since 2026-10-23 carries a "provenance" block
(git commit, ROM/state SHA-1, package versions, seed). The committed
artifacts that predate the mechanism are reported as legacy -- not
annotated after the fact, because a back-filled block would lie about
what produced them.
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
    print(f"\nlegacy (pre-2026-10-23, no provenance): {len(legacy)}")
    for p in legacy:
        print(f"  {p}")


if __name__ == "__main__":
    main()
