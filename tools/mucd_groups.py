"""Learn the course -> MUCD table binding from the course header, by group vote.

The header's u16 at offset 112+6 looks like a tileset index (A01_1 -> 8, A04_1 -> 0,
A05_1 -> 33, A08_1 -> 1) and the ROM's file IDs (262..302) do not appear there, so
the ordering is internal to the game. Test it empirically: for every course, find
the MUCD table whose solidity flags best fit that course's own floor geometry, then
check whether courses sharing a header index agree on the winner. Consistent groups
mean the index is real and the binding can be read off the header; scattered groups
mean the fit was coincidence.
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
FLAGS = {}
for n in MUCD:
    u = struct.unpack('<1536H', bytes(rom.getFileByName('BG_chk/' + n)))
    FLAGS[n] = {o: bool(max(u[o * 6:o * 6 + 6])) for o in range(256)}


def rects_of(name):
    bg = bytes(rom.getFileByName(name.replace(".bin", "_bgdat.bin")))
    i, out = 0, []
    while i + 10 <= len(bg):
        obj, ts, x, y, w, h = struct.unpack_from("<BBHHHH", bg, i)
        if obj == 0xFF:
            break
        out.append((obj, ts, x, y, w, h))
        i += 10
    return out


def f1(a, b):
    if not a or not b:
        return 0.0
    inter = len(a & b)
    p, r = inter / len(b), inter / len(a)
    return 2 * p * r / (p + r) if p + r else 0.0


def truth_cols(R):
    per = {}
    for obj, ts, x, y, w, h in R:
        for xx in range(x, x + w):
            bot = y + h - 1
            if bot <= 31 and (xx not in per or bot > per[xx]):
                per[xx] = bot
    if not per:
        return set()
    mode = collections.Counter(per.values()).most_common(1)[0][0]
    return {c for c, b in per.items() if b == mode}


groups = collections.defaultdict(list)
rows = []
for name in rom.filenames.subfolder('course').files:
    if not name.endswith('.bin') or name.endswith('_bgdat.bin'):
        continue
    raw = bytes(rom.getFileByName('course/' + name))
    if len(raw) < 144:
        continue
    idx = struct.unpack_from('<H', raw, 118)[0]
    R = rects_of('course/' + name)
    truth = truth_cols(R)
    if len(truth) < 20:
        continue
    scored = sorted(((f1(truth, {xx for o, ts, x, y, w, h in R if FLAGS[m][o]
                                 for xx in range(x, x + w)}), m) for m in MUCD), reverse=True)
    best_score, best = scored[0]
    runner = scored[1][0]
    rows.append((name, idx, best, best_score, runner))
    groups[idx].append((best, best_score))

print(f"{len(rows)} courses scored")
sharp = sum(1 for _, _, _, s, r in rows if s - r > 0.05)
print(f"courses with a unique winner (margin > 0.05): {sharp}/{len(rows)}")
print("\nheader index -> winner table (showing groups with >=3 courses):")
for idx, lst in sorted(groups.items()):
    if len(lst) < 3:
        continue
    votes = collections.Counter(t for t, _ in lst)
    top, n = votes.most_common(1)[0]
    mean = sum(s for _, s in lst) / len(lst)
    print(f"  idx {idx:3d}: {len(lst):3d} courses, modal winner "
          f"{top.replace('MainUnitChangeData.bin',''):12s} {n}/{len(lst)} agree, mean F1={mean:.3f}")
