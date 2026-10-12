"""Export TilemapAStar level geometry to a compact binary for the Rust planner.

The Rust side (rust/nsmb_planner) deliberately has no crates.io dependencies:
this machine's Smart App Control blocks cargo.exe, so the crate is built with
rustc alone and therefore parses a format it can read with std only.

Usage:
    python tools/export_geometry.py [--rom ...] [--out out/nsmb_geometry.bin]
"""
from __future__ import annotations

import argparse
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from tilemap_planner import TilemapAStar  # noqa: E402

MAGIC = b"NSMBGEO1"
DEFAULT_ROM = os.path.join(ROOT, "data",
                           "0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds")


def write_pairs(f, pairs):
    f.write(struct.pack("<I", len(pairs)))
    for col, val in pairs:
        f.write(struct.pack("<ii", col, val))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rom", default=DEFAULT_ROM)
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "nsmb_geometry.bin"))
    args = ap.parse_args()

    p = TilemapAStar(rom_path=args.rom)

    with open(args.out, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", 1))                       # version
        f.write(struct.pack("<iiii", p.min_col, p.max_col, p.reach,
                            1 if p.feasible else 0))
        write_pairs(f, sorted(p.surface.items()))           # col -> top walkable row
        write_pairs(f, sorted(p.obstacle_h.items()))        # col -> tiles above floor
        f.write(struct.pack("<I", len(p.solid)))
        for col in sorted(p.solid):
            f.write(struct.pack("<i", col))
        f.write(struct.pack("<I", len(p.pits)))
        for (a, b) in p.pits:
            f.write(struct.pack("<ii", a, b))
        f.write(struct.pack("<I", len(p.takeoffs)))
        for tk in sorted(p.takeoffs):
            a, b, land, w = p.takeoffs[tk]
            f.write(struct.pack("<iiiii", tk, a, b, land, w))

    size = os.path.getsize(args.out)
    print(f"wrote {args.out} ({size} bytes)")
    print(f"cols=[{p.min_col},{p.max_col}] reach={p.reach} feasible={p.feasible} "
          f"pits={len(p.pits)} takeoffs={len(p.takeoffs)} "
          f"surface={len(p.surface)} obstacles={len(p.obstacle_h)}")
    print(f"goal_px={p.goal_px()}")


if __name__ == "__main__":
    main()
