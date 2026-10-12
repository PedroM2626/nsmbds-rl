"""Recover the course -> tileset binding from data, without disassembling ARM9.

BG_chk/*MainUnitChangeData.bin is 1536 u16 flags laid out as 256 object slots x 6
(verified: 'any of the 6 slots set' separates A01_1's validated ground classes at
94.9% against a 64.1% baseline), and the flags mean "this object number is a
terrain unit in THIS tileset" -- ShiRo marks 0x06..0x0A solid, GaKe does not.

So the missing link is which of the 48 files a course uses. The course header has
no tileset name string, so score every file against each course's own geometric
floor signature (the bottom contiguous run of tiles at the course's modal floor
row, which is what the rendered frames confirmed for A01_1) and keep the best.
A sharp winner means the binding is recovered; a flat score means it is not.
"""
import collections
import struct
import sys

import ndspy.rom

sys.path.insert(0, r"D:/mario-ds/src")

ROM = r"D:/mario-ds/data/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds"
rom = ndspy.rom.NintendoDSRom(open(ROM, "rb").read())
MUCD = [n for n in rom.filenames.subfolder('BG_chk').files
        if n.endswith('MainUnitChangeData.bin')]


def rects_of(course_name):
    bg = bytes(rom.getFileByName(course_name.replace(".bin", "_bgdat.bin")))
    i, out = 0, []
    while i + 10 <= len(bg):
        obj, ts, x, y, w, h = struct.unpack_from("<BBHHHH", bg, i)
        if obj == 0xFF:
            break
        out.append((obj, ts, x, y, w, h))
        i += 10
    return out


def flags_of(name):
    u = struct.unpack('<1536H', bytes(rom.getFileByName('BG_chk/' + name)))
    return {o: max(u[o * 6:o * 6 + 6]) for o in range(256)}


def geom_floor_cols(R):
    """Columns whose bottom-most tile bottom edge sits at the course's modal floor."""
    per_col = {}
    for obj, ts, x, y, w, h in R:
        for xx in range(x, x + w):
            bot = y + h - 1
            if bot <= 31 and (xx not in per_col or bot > per_col[xx]):
                per_col[xx] = bot
    if not per_col:
        return set(), 30
    mode = collections.Counter(per_col.values()).most_common(1)[0][0]
    return {c for c, b in per_col.items() if b == mode}, mode


def solid_cols(R, F):
    cols = set()
    for obj, ts, x, y, w, h in R:
        if F[obj]:
            cols.update(range(x, x + w))
    return cols


def f1(a, b):
    if not a or not b:
        return 0.0
    inter = len(a & b)
    p = inter / len(b)
    r = inter / len(a)
    return 2 * p * r / (p + r) if (p + r) else 0.0


print(f"{len(MUCD)} MUCD files")
for name in ("course/A01_1.bin", "course/A02_1.bin", "course/A05_1.bin",
             "course/A08_1.bin", "course/A04_1.bin"):
    R = rects_of(name)
    truth, mode = geom_floor_cols(R)
    scored = []
    for m in MUCD:
        F = flags_of(m)
        scored.append((f1(truth, solid_cols(R, F)), m))
    scored.sort(reverse=True)
    top = scored[:3]
    spread = top[0][0] - scored[-1][0]
    print(f"\n{name}: modal floor row {mode}, {len(truth)} floor columns")
    for s, m in top:
        print(f"   F1={s:.3f}  {m.replace('MainUnitChangeData.bin','')}")
    print(f"   worst={scored[-1][0]:.3f}  spread={spread:.3f}  "
          f"{'SHARP winner' if spread > 0.25 else 'flat: binding not recovered'}")
