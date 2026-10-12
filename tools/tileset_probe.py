"""Can ground detection be derived from geometry instead of a global class list?

GROUND_OBJS is a flat set of tile-object numbers, but those numbers are local to
each course's tileset: A01_1's floor is class 0x0A/0x07/0x09 (ts 16) while A08_1's
terrain is class 0x16/0x17/0x18 in its own ts 16, and 0x16 never appears in
A01_1 at all. The authoritative solidity data is BG_chk/*MainUnitChangeData.bin,
which docs/REVERSE_ENGINEERING_REPORT.md lists as undecoded.

Candidate rule: a rect is floor-bearing where its bottom edge sits in the level's
floor band (row >= FLOOR_BAND). This probe compares that against the class list on
A01_1 (where the rendered frames give ground truth) and reports what it would do
to A08_1. Nothing here is written back into src/ yet.
"""
import struct
import sys
import collections

import ndspy.rom

sys.path.insert(0, r"D:/mario-ds/src")
from course import GROUND_OBJS, TILE  # noqa: E402

ROM = r"D:/mario-ds/data/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds"
FLOOR_BAND = 28
rom = ndspy.rom.NintendoDSRom(open(ROM, "rb").read())


def rects(name):
    bg = bytes(rom.getFileByName(name.replace(".bin", "_bgdat.bin")))
    i, out = 0, []
    while i + 10 <= len(bg):
        obj, ts, x, y, w, h = struct.unpack_from("<BBHHHH", bg, i)
        if obj == 0xFF:
            break
        out.append({"obj": obj, "ts": ts, "tx": x, "ty": y, "w": w, "h": h})
        i += 10
    return out


def cover(R, predicate):
    cov = {}
    for r in R:
        if not predicate(r):
            continue
        for xx in range(r["tx"], r["tx"] + r["w"]):
            for yy in range(r["ty"], r["ty"] + r["h"]):
                if yy >= 18:
                    cov.setdefault(xx, set()).add(yy)
    return cov


def bottom_in_band(r):
    return r["ty"] + r["h"] - 1 >= FLOOR_BAND


def profile(cov):
    """floor = top of the bottom contiguous run, same rule as Course.floor_row."""
    floors = {}
    for c, rows in cov.items():
        r = max(rows)
        while (r - 1) in rows:
            r -= 1
        floors[c] = r
    return floors


def pits(floors):
    if not floors:
        return []
    lo, hi = min(floors), max(floors)
    out, c = [], lo
    while c <= hi:
        if c not in floors:
            a = c
            while c <= hi and c not in floors:
                c += 1
            out.append(c - a)
        c += 1
    return out


for name in ("course/A01_1.bin", "course/A08_1.bin", "course/A02_1.bin", "course/A05_1.bin"):
    R = rects(name)
    by_class = cover(R, lambda r: r["obj"] in GROUND_OBJS)
    by_geom = cover(R, bottom_in_band)
    f_cls, f_geom = profile(by_class), profile(by_geom)
    span = max((r["tx"] + r["w"] for r in R), default=0)
    print(f"== {name}: {len(R)} rects, tile x extent 0..{span} ({span*TILE} px)")
    print(f"   class list : {len(f_cls):3d} floor cols, {sum(pits(f_cls))} pit cols, "
          f"pits {sorted(collections.Counter(pits(f_cls)).items())}")
    print(f"   bottom>=28 : {len(f_geom):3d} floor cols, {sum(pits(f_geom))} pit cols, "
          f"pits {sorted(collections.Counter(pits(f_geom)).items())}")
    agree = sum(1 for c in set(f_cls) | set(f_geom) if f_cls.get(c) == f_geom.get(c))
    total = len(set(f_cls) | set(f_geom))
    print(f"   agreement  : {agree}/{total} columns identical ({100*agree/max(1,total):.1f}%)")
    cls_only = sorted({r["obj"] for r in R if r["obj"] in GROUND_OBJS})
    geom_only = sorted({r["obj"] for r in R if bottom_in_band(r) and r["obj"] not in GROUND_OBJS})
    dropped = sorted({r["obj"] for r in R if r["obj"] in GROUND_OBJS and not bottom_in_band(r)})
    print(f"   classes: ground-list={len(cls_only)} geometry-only={len(geom_only)} "
          f"({[hex(o) for o in geom_only]}) list-only-never-reaching-floor={len(dropped)} "
          f"({[hex(o) for o in dropped]})")
