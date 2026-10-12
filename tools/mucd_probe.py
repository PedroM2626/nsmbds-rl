"""What does the 0/1 flag in BG_chk/*MainUnitChangeData.bin index?

Each of the 48 real MUCD files is 3072 bytes = 1536 u16 values, and the values
are only ever 0 or 1, so this is a boolean table, not an obj->model mapping
(the six 512-byte files in that folder are ScrollControl_*, unrelated).

Search the plausible indexings for one whose flag separates the tile classes that
A01_1 uses as walkable ground from the ones it uses as non-ground, using A01_1 as
the labelled example (its ground classes were validated against rendered frames).
"""
import struct
import sys

import ndspy.rom

sys.path.insert(0, r"D:/mario-ds/src")
from course import Course, GROUND_OBJS  # noqa: E402

ROM = r"D:/mario-ds/data/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds"
rom = ndspy.rom.NintendoDSRom(open(ROM, "rb").read())
files = [(i if isinstance(i, str) else i[0])
         for i in rom.filenames.subfolder('BG_chk').files]
files = [n for n in files if n.endswith('MainUnitChangeData.bin')]
assert files, "no MUCD files listed"

# classes actually used in A01_1, labelled by the validated ground list
c = Course(rom_path=ROM)
used = sorted({r["obj"] for r in c.rects})
label = {o: (1 if o in GROUND_OBJS else 0) for o in used}
print(f"A01_1 uses {len(used)} classes; ground-labelled {sum(label.values())}, "
      f"other {len(used)-sum(label.values())}")

best = []
for n in files:
    data = bytes(rom.getFileByName('BG_chk/' + n))
    u16 = struct.unpack('<1536H', data)
    u8 = list(data)
    for scheme, get in (
            ("u16[obj]", lambda o: u16[o] if o < 1536 else None),
            ("u8[obj]", lambda o: u8[o] if o < 3072 else None),
            ("u16[obj*6+k0..5] any", lambda o: max(u16[o * 6:o * 6 + 6]) if o * 6 + 6 <= 1536 else None),
            ("u16[k*256+obj] any", lambda o: max([u16[k * 256 + o] for k in range(6)
                                                  if k * 256 + o < 1536]) if o < 256 else None),
            ("u8[k*256+obj] any", lambda o: max([u8[k * 256 + o] for k in range(12)
                                                 if k * 256 + o < 3072]) if o < 256 else None),
    ):
        vals = [(label[o], get(o)) for o in used]
        if any(v is None for _, v in vals):
            continue
        agree = sum(1 for lab, v in vals if (1 if v else 0) == lab)
        best.append((agree / len(vals), n, scheme))

best.sort(reverse=True)
print("\ntop separations (fraction of A01_1 classes whose flag equals the ground label):")
for score, n, scheme in best[:10]:
    print(f"  {score*100:5.1f}%  {scheme:24s} {n}")
print(f"\n(random guess on this mix would score {100*max(sum(label.values()), len(used)-sum(label.values()))/len(used):.1f}%)")
