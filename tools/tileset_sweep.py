"""Systematic sweep: which header field, read as an index into the BG_chk file
order, makes every course's ground coverage come out plausible?

Precomputes coverage of all 191 courses under all 48 MainUnitChangeData tables
once, then scores each u16 slot of the course header as a tileset index. A correct
binding should give high coverage for nearly every course; a wrong one leaves a
long tail of levels with almost no ground (which is exactly what the global
GROUND_OBJS list does today: A05_1 11%, A08_1 4%).
"""
import struct
import sys

import ndspy.rom

sys.path.insert(0, r"D:/mario-ds/src")

ROM = r"D:/mario-ds/data/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds"
rom = ndspy.rom.NintendoDSRom(open(ROM, "rb").read())
bg = rom.filenames.subfolder('BG_chk')
MUCD = sorted([n for n in bg.files if n.endswith('MainUnitChangeData.bin')],
              key=lambda n: bg.idOf(n))
TERR = []
for n in MUCD:
    u = struct.unpack('<1536H', bytes(rom.getFileByName('BG_chk/' + n)))
    TERR.append(frozenset(k for k in range(256) if any(u[k * 6:k * 6 + 6])))

COURSE = [n for n in rom.filenames.subfolder('course').files
          if n.endswith('.bin') and not n.endswith('_bgdat.bin')]

cov = []      # per course: list of 48 coverage fractions, span, header u16s
for nm in COURSE:
    b = bytes(rom.getFileByName('course/' + nm.replace('.bin', '_bgdat.bin')))
    raw = bytes(rom.getFileByName('course/' + nm))
    if len(raw) < 144:
        continue
    i, cls_cols = 0, {}
    span = 1
    while i + 10 <= len(b):
        obj, ts, x, y, w, h = struct.unpack_from("<BBHHHH", b, i)
        if obj == 0xFF:
            break
        if x + w > span:
            span = x + w
        cls_cols.setdefault(obj, set()).update(range(x, x + w))
        i += 10
    if not cls_cols:
        continue
    h = struct.unpack_from('<16H', raw, 112)
    per = [len(set().union(*[cls_cols[o] for o in cls_cols if o in T])) / span if T else 0.0
           for T in TERR]
    cov.append((nm, per, h))
print(f"{len(cov)} courses scored against {len(MUCD)} tables")

best = None
for field in range(16):
    good = 0
    for nm, per, h in cov:
        idx = h[field]
        if idx < len(per) and per[idx] >= 0.70:
            good += 1
    frac = good / len(cov)
    if best is None or frac > best[1]:
        best = (field, frac)
    if frac > 0.5:
        print(f"  header u16[{field}] as index: {good}/{len(cov)} courses at >=70% coverage "
              f"({100*frac:.1f}%)")
print(f"\nbest field: u16[{best[0]}] at {100*best[1]:.1f}%")

# and the ceiling: what would a per-course oracle achieve?
oracle = sum(1 for nm, per, h in cov if max(per) >= 0.70)
print(f"oracle (best table per course): {oracle}/{len(cov)} courses could reach 70%")
