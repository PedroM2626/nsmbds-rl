"""Identify the course -> tileset table binding by fitting GROUPS, not courses.

Per-course fitting is degenerate: six of the 48 MainUnitChangeData tables score
J=1.000 against A01_1 alone, because 1-1 only uses a handful of tile classes. The
header carries a small u16 that behaves like a tileset index (A01_1 -> 8, A04_1 -> 0,
A05_1 -> 33, A08_1 -> 1), so group every course by that field and score each table on
the whole group. A table that explains ten unrelated courses at once is not a
coincidence, and the winner's margin over the runner-up says whether the field is real.

Validation that does not depend on the fit itself: the winner for A01_1's group must
reproduce 1-1's ground set, which was independently confirmed against rendered frames.
"""
import collections
import struct
import sys

import ndspy.rom

sys.path.insert(0, r"D:/mario-ds/src")
from course import GROUND_OBJS  # noqa: E402

ROM = r"D:/mario-ds/data/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds"
rom = ndspy.rom.NintendoDSRom(open(ROM, "rb").read())
bg = rom.filenames.subfolder('BG_chk')
MUCD = [n for n in bg.files if n.endswith('MainUnitChangeData.bin')]
TERRAIN = {}
for n in MUCD:
    u = struct.unpack('<1536H', bytes(rom.getFileByName('BG_chk/' + n)))
    TERRAIN[n] = frozenset(k for k in range(256) if any(u[k * 6:k * 6 + 6]))

COURSE = [n for n in rom.filenames.subfolder('course').files
          if n.endswith('.bin') and not n.endswith('_bgdat.bin')]


def rects(name):
    b = bytes(rom.getFileByName('course/' + name.replace('.bin', '_bgdat.bin')))
    i, out = 0, []
    while i + 10 <= len(b):
        obj, ts, x, y, w, h = struct.unpack_from("<BBHHHH", b, i)
        if obj == 0xFF:
            break
        out.append((obj, ts, x, y, w, h))
        i += 10
    return out


def header_u16s(name, count=16):
    raw = bytes(rom.getFileByName('course/' + name))
    return struct.unpack_from('<%dH' % count, raw, 112) if len(raw) >= 144 else None


DATA = {}
for nm in COURSE:
    R = rects(nm)
    h = header_u16s(nm)
    if not R or h is None:
        continue
    span = max((x + w for _, _, x, _, w, _ in R), default=1)
    cls = collections.defaultdict(set)
    for obj, ts, x, y, w, h_ in R:
        cls[obj].update(range(x, x + w))
    DATA[nm] = (cls, span, h)
print(f"{len(DATA)} courses with parsed geometry")

for field in (2, 3, 6, 7):        # u16 index into the header block
    groups = collections.defaultdict(list)
    for nm, (cls, span, h) in DATA.items():
        groups[h[field]].append(nm)
    big = {v: g for v, g in groups.items() if len(g) >= 6}
    if not big:
        continue
    margins, hits = [], 0
    for val, g in sorted(big.items()):
        score = []
        for n in MUCD:
            T = TERRAIN[n]
            tot = inter = 0
            for nm in g:
                cls, span, _ = DATA[nm]
                cols = set()
                for o, c in cls.items():
                    if o in T:
                        cols |= c
                tot += span
                inter += min(len(cols), span)
            score.append((inter / tot, n))
        score.sort(reverse=True)
        margins.append(score[0][0] - score[1][0])
        if val == 8:
            print(f"   field u16[{field}] group {val}: winner {score[0][1][:18]} "
                  f"cov={score[0][0]:.3f} runner-up cov={score[1][0]:.3f}")
    print(f"field u16[{field}]: {len(big)} groups >=6 courses, "
          f"median winner margin {sorted(margins)[len(margins)//2]:.4f}")
