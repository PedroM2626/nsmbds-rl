"""Measured course identity: the ROM course index and its reverse mapping.

The running course's index among the 191 course/*.bin files is a u16 in RAM
(0x020980D4, copied at 0x0209B59E). This was measured in 4m while building
savestates; the index is the course file's position when the ROM filesystem
entries are sorted by file id. It is a direct identity oracle:

    A01_1=0  A02_1=3  B01_1=21  C01_1=44  D01_1=66
    E01_1=88 F01_1=107 G01_1=130 H01_1=152

Measured on every committed savestate on 2026-10-23 (including the .ds2/.ds3
states, E01_1=88 and H01_1=152). This is stronger than the entrance/floor
matcher in tools/identify_state.py, which shares entrances across courses and
scored the 1-2 state ambiguously (4ad/4ae); the HUD screenshot happened to
agree with the index. Prefer this read, and treat the matcher as a fallback.

The manifest is committed (tools/fs_manifest.json), so this module needs no
ROM and can be unit-tested offline.
"""
from __future__ import annotations

import functools
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "tools", "fs_manifest.json")

AREA_ID_ADDR = 0x020980D4   # u16, running course index among the 191
AREA_ID2_ADDR = 0x0209B59E  # u16 copy read by the level load


def _course_id(name):
    name = os.path.basename(str(name))
    return name[:-4] if name.endswith(".bin") else name


@functools.lru_cache(maxsize=1)
def course_ids(manifest=MANIFEST):
    """The 191 course ids in ROM file-id order (the area-index order)."""
    with open(manifest, encoding="utf-8") as f:
        entries = json.load(f)
    courses = sorted(
        (e for e in entries
         if e["path"].startswith("course/")
         and e["path"].endswith(".bin")
         and not e["path"].endswith("_bgdat.bin")),
        key=lambda e: e["id"])
    ids = tuple(_course_id(e["path"]) for e in courses)
    if len(ids) != 191:
        raise ValueError(f"{manifest}: expected 191 course files, found {len(ids)}")
    return ids


def course_area_index(course, manifest=MANIFEST):
    """The area index of a course id or path (A02_1 or course/A02_1.bin)."""
    cid = _course_id(course)
    ids = course_ids(manifest)
    try:
        return ids.index(cid)
    except ValueError:
        raise KeyError(f"{cid} is not one of the {len(ids)} parsed course ids") from None


def course_for_area_index(index, manifest=MANIFEST):
    """The course id at an area index, or None when out of range."""
    ids = course_ids(manifest)
    return ids[index] if 0 <= index < len(ids) else None
