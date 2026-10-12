"""Identify the course -> MainUnitChangeData table using positive AND negative evidence.

Coverage alone saturated (median winner margin 0.0008) because any table that happens
to contain a course's ground classes wins, regardless of what else it claims. The
tables are actually distinguishable -- 37 distinct terrain-class sets across 48 files,
median pairwise difference of 25 classes -- so the objective was the problem.

Per course, from geometry:
  POS = classes with a rect whose bottom edge sits on the course's modal floor row
        (these must be terrain).
  NEG = classes whose rects never come within 4 rows of that floor (floating blocks,
        ceiling art -- these must NOT be terrain).
Score a table on a group of courses sharing one header field value by how much of POS
it contains and how much of NEG it excludes. A01_1 is the held-out check: its group's
winner must reproduce the ground set already confirmed against rendered frames.
"""
import collections
import struct
import sys

import ndspy.rom

sys.path.insert(0, r"D:/mario-ds/src")

ROM = r"D:/mario-ds/data/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds"
rom = ndspy.rom.NintendoDSRom(open(ROM, "rb").read())
bg = rom.filenames.subfolder('BG_chk')
MUCD = [n for n in bg.files if n.endswith('MainUnitChangeData.bin')]
TERR = {}
for n in MUCD:
    u = struct.unpack('<1536H', bytes(rom.getFileByName('BG_chk/' + n)))
    TERR[n] = frozenset(k for k in range(256) if any(u[k * 6:k * 6 + 6]))

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


def pos_neg(R):
    if not R:
        return set(), set()
    bottoms = collections.Counter()
    for obj, ts, x, y, w, h in R:
        b = y + h - 1
        if b <= 31:
            bottoms[b] += w
    if not bottoms:
        return set(), set()
    floor = bottoms.most_common(1)[0][0]
    pos, neg = set(), set()
    per = collections.defaultdict(lambda: [None, None])  # obj -> [max bottom, min top]
    for obj, ts, x, y, w, h in R:
        b = y + h - 1
        slot = per[obj]
        slot[0] = b if slot[0] is None else max(slot[0], b)
        slot[1] = y if slot[1] is None else min(slot[1], y)
    for obj, (maxb, mintop) in per.items():
        if maxb is not None and maxb >= floor - 1:
            pos.add(obj)
        elif maxb is not None and maxb <= floor - 5:
            neg.add(obj)
    return pos, neg


rows = []
for nm in COURSE:
    R = rects(nm)
    raw = bytes(rom.getFileByName('course/' + nm))
    if not R or len(raw) < 144:
        continue
    h = struct.unpack_from('<16H', raw, 112)
    pos, neg = pos_neg(R)
    rows.append((nm, h, pos, neg))
print(f"{len(rows)} courses labelled")

for field in (2, 3, 6, 7):
    groups = collections.defaultdict(list)
    for nm, h, pos, neg in rows:
        groups[h[field]].append((pos, neg))
    big = {v: g for v, g in groups.items() if len(g) >= 5}
    margins = []
    winner8 = None
    for val, g in sorted(big.items()):
        sc = []
        for n in MUCD:
            T = TERR[n]
            hit = sum(1 for pos, neg in g if pos and pos <= T)
            clean = sum(1 for pos, neg in g if not (T & neg))
            sc.append(((hit + clean) / (2 * len(g)), n))
        sc.sort(reverse=True)
        margins.append(sc[0][0] - sc[1][0])
        if val == 8:
            winner8 = sc[:3]
    if margins:
        margins.sort(reverse=True)
        print(f"field u16[{field}]: {len(big)} groups, best margins "
              f"{[round(m,3) for m in margins[:5]]}, median {round(margins[len(margins)//2],4)}")
        if winner8:
            print(f"   group value 8 (A01_1's field): {[(round(s,3), n[:16]) for s, n in winner8]}")
