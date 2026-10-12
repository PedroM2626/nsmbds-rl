"""Build an NSMB DS save with the worlds unlocked, from the user's own save.

Why derive from the user's save
-------------------------------
A downloaded "100%" save arrives with someone else's progress in it, and it has
to be trusted until checked. The one checked here -- after the fact -- did
check out (same `Mario2d`/`9000` format, all ten checksums clean; see below),
and it earned a role: the source of the 16 unlock bytes this tool writes. The
file it *emits* still derives from the user's own `data/…sav`, which already
parses on this ROM, so every field we do not touch stays real.

The layout comes from newluigidev's NSMB DS save editor (C#, single Form1.cs).
Its *field* offsets check out against the user's own save: lives 0x116, coins
0x11A, score 0x11E, star coins 0x122, powerup 0x13A, per file slot at
0x100 / 0x380 / 0x600. The three slots in `data/…sav` differ in ~31 bytes and
those differences land exactly on the score region, which is what a
per-file-progress field should do.

**Its "unlock all worlds" offset was right; this file's own earlier retraction
of it was wrong, and a verified download is what settled it.** The first
revision argued 0x16A..0x17A "does not gate the world row" because those bytes
are identical across all three local slots. Identical, yes -- but all three
slots here were played to the same early point, which the reference exposes:
its two beaten slots read `ff 03 ff 03 ff 03 ff 03 ff 03 ff 07 ff 03 df 07`
at 0x16A..0x179 (8 x u16, one per world), while its untouched slot 3 reads the
same `43 00` x8 that every local slot reads. So the field is real. What the
editor gets wrong is the value -- 0xFFFF per world, where the verified save
carries 0x03FF (0x07FF/0x07DF where a world has an 11th node) -- and the
length: the 17th byte it writes, +0x7A, is the slot sentinel (d7/d0 in every
valid save seen) and the 0xFF run clobbers it.

The reference is `new-super-mario-bros.16742.dst`, an ARDS container: a
500-byte header (magic `ARDS`, game title in UTF-16) wrapping exactly this
8 KB save format. Extracted (bytes 500..) and run through this port's verify,
all ten sections checksum clean, so its unlock bytes are copied, not invented.

Its "unlock all levels" stays refused: 0x241..~0x325 is 0xC0 fill in the local
save (the download instead carries 00-records with d0 sentinels every 11 bytes
there, so the region is live state with unknown semantics), and 0x326..0x355
is structured data (`05 01 01 01 09 01 07 01 02 00 0b 00 ...`) that the
editor's fixed 277-byte run destroys. `--levels` needs `--force`.

An earlier revision also fingered the +0x2A..+0x66 stride-4 range (values
00/05/06/0B/13) as the likely progress field. The download kills that lead
too: its beaten slots read 00 there while the local early slots read
05/05/0B/13, so those bytes do not track unlocks either.

What the tool guarantees: the 16-bit checksum port is validated against the
untouched input before anything is written -- all ten sections of a save the
game already accepts reproduce exactly (and the reference passes 10/10 too)
-- the write touches only the 16 measured bytes, mirrors them into both 4 KB
blocks, and rewrites every checksum. What it cannot check from here is the
tap: the world row is stylus-only, so "all worlds visible" is confirmed in
the GUI session below.

Usage
-----
    python tools/make_save.py --out data/save_unlocked.sav --slot 1

Then, in a GUI DeSmuME (not headless -- the world row on the map is stylus-only
and the stylus does nothing in py-desmume): boot the ROM with this save, press
START on the title, pick the file, tap the world you want, enter the level and
save a state. Hand the .dss/.ds1 back and the whole repo runs on it:

    python src/eval_pinn_realtime.py --controller reactive --headless \
        --state <that state> --course <the course file> --reflex profile
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_IN = os.path.join(ROOT, "data",
                          "0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).sav")

MAGIC = b"Mario2d"
SLOT1_BASE = 0x100         # three file slots, each with its own checksum + magic
SLOT_STRIDE = 0x280
WORLD_OFF = 0x16A          # per slot 1; +0x280 per further slot
WORLD_LEN = 0x10           # 8 x u16, one per world; the editor's 17th byte is
                           # the slot sentinel at +0x7A (d7/d0), not data
# What a beaten world reads in the verified 100% reference (see docstring):
WORLD_UNLOCKED = (0x03FF, 0x03FF, 0x03FF, 0x03FF, 0x03FF, 0x07FF, 0x03FF, 0x07DF)
LEVEL_OFF = 0x241
LEVEL_LEN = 0x115          # i in 0..0x114 inclusive
# (section offset, section length) inside each 4 KB block; the stored checksum is
# the first two bytes and the covered data starts ten bytes in.
SECTIONS = [(0x000, 0x0F4), (0x100, 0x248), (0x380, 0x248), (0x600, 0x248), (0x880, 0x014)]
BLOCKS = [0x0000, 0x1000]


def nsmb_checksum(data: bytearray, length: int, pos: int) -> int:
    """The editor's nsmbChecksum: a 16-bit rolling rotate-through-carry XOR."""
    c = 654
    for i in range(length):
        c = data[pos + i] ^ (((2 * c) & 0xFFFE) | ((c >> 15) & 1))
    return c & 0xFFFF


def section_positions(base: int, off: int) -> tuple:
    """(checksum position, data start) for one section of one block."""
    return base + off, base + off + 10


def stored(data: bytearray, cpos: int) -> int:
    return data[cpos] | (data[cpos + 1] << 8)


def recompute(data: bytearray) -> list:
    """Write the correct checksum into every section; return (label, was, now)."""
    report = []
    for base in BLOCKS:
        for off, ln in SECTIONS:
            cpos, dpos = section_positions(base, off)
            want = nsmb_checksum(data, ln, dpos)
            report.append((f"0x{cpos:04X}", stored(data, cpos), want))
            data[cpos] = want & 0xFF
            data[cpos + 1] = want >> 8
    return report


def verify(data: bytearray) -> list:
    """Check without writing: (label, stored, expected, ok)."""
    out = []
    for base in BLOCKS:
        for off, ln in SECTIONS:
            cpos, dpos = section_positions(base, off)
            want = nsmb_checksum(data, ln, dpos)
            got = stored(data, cpos)
            out.append((f"0x{cpos:04X}", got, want, got == want))
    return out


def do_diff(path_a: str, path_b: str) -> int:
    """Group the differing bytes of two saves, so a reference save can be read.

    A save with a later world unlocked differs from this one in the field that
    gates the world row. Printing runs rather than a hex wall, and labelling each
    run with the three-slot test (does this offset already vary between the files
    in the local save?), is what makes that field findable by eye.
    """
    a = bytearray(open(path_a, "rb").read())
    b = bytearray(open(path_b, "rb").read())
    if len(a) != len(b):
        print(f"different sizes: {path_a}={len(a)} {path_b}={len(b)}")
        return 1
    local = bytearray(open(DEFAULT_IN, "rb").read())
    offs = [i for i in range(len(a)) if a[i] != b[i]]
    print(f"{path_a} vs {path_b}: {len(offs)} differing bytes")
    runs = []
    for i in offs:
        if runs and i == runs[-1][1] + 1:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    for lo, hi in runs:
        slot = next((n for n in (1, 2, 3)
                     if SLOT1_BASE + (n - 1) * SLOT_STRIDE <= lo
                     < SLOT1_BASE + (n - 1) * SLOT_STRIDE + 0x248), 0)
        varies = ""
        if slot:
            rel = lo - (SLOT1_BASE + (slot - 1) * SLOT_STRIDE)
            trio = {local[SLOT1_BASE + k * SLOT_STRIDE + rel] for k in range(3)}
            varies = "  local-slots-vary" if len(trio) > 1 else "  local-slots-constant"
        print(f"  0x{lo:04X}..0x{hi:04X} ({hi - lo + 1:3d}B) slot{slot} "
              f"{bytes(a[lo:hi + 1]).hex(' ')} -> {bytes(b[lo:hi + 1]).hex(' ')}"
              f"{varies}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=DEFAULT_IN)
    ap.add_argument("--out", default="", help="where to write the patched save")
    ap.add_argument("--slot", type=int, default=1, choices=(1, 2, 3),
                    help="which of the three file slots to unlock")
    ap.add_argument("--worlds", action="store_true", default=True,
                    help="unlock the world row (the default)")
    ap.add_argument("--no-worlds", dest="worlds", action="store_false")
    ap.add_argument("--levels", action="store_true",
                    help="also run the editor's 277-byte 0xC0 write -- overwrites "
                         "structured data at 0x326..0x355, needs --force")
    ap.add_argument("--force", action="store_true",
                    help="allow the --levels write despite the measured damage")
    ap.add_argument("--no-mirror", dest="mirror", action="store_false",
                    help="patch only the first 4 KB block")
    ap.add_argument("--diff", default="",
                    help="compare two 8 KB saves byte by byte instead of writing: "
                         "grouped runs plus which of them sit in a slot that varies. "
                         "This is how a reference save turns into an unlock recipe.")
    ap.add_argument("--against", default="", help="second save for --diff")
    args = ap.parse_args()

    if args.diff:
        return do_diff(args.diff, args.against or args.inp)
    if not args.out:
        raise SystemExit("--out is required unless --diff is used")

    data = bytearray(open(args.inp, "rb").read())
    if len(data) != 8192:
        raise SystemExit(f"{args.inp} is {len(data)} bytes, expected 8192")
    if MAGIC not in data[:16]:
        raise SystemExit(f"no {MAGIC!r} signature in the first 16 bytes -- wrong game")
    print(f"input {os.path.basename(args.inp)}  magic ok  "
          f"version field {data[10:14]!r}", flush=True)

    # Positive control before any write: the port has to reproduce the checksums
    # of a save the game already accepts.
    checks = verify(data)
    good = [c for c in checks if c[3]]
    print(f"checksum port: {len(good)}/{len(checks)} sections already correct", flush=True)
    for label, got, want, ok in checks:
        if not ok:
            print(f"   {label}: stored 0x{got:04X} != computed 0x{want:04X}", flush=True)
    if not good:
        raise SystemExit("the checksum port reproduces none of the stored values; "
                         "refusing to write a file the game would reject")

    if args.mirror:
        same = data[BLOCKS[0]:BLOCKS[0] + 0x900] == data[BLOCKS[1]:BLOCKS[1] + 0x900]
        print(f"blocks 0x0000/0x1000 identical over 0x900 bytes: {same}", flush=True)
    targets = BLOCKS if args.mirror else BLOCKS[:1]

    if args.levels and not args.force:
        raise SystemExit("--levels runs the editor's fixed 0xC0 run over 0x241..0x355, "
                         "and 0x326..0x355 is real per-level data in this save. "
                         "Pass --force if that is what you want.")
    base_off = (args.slot - 1) * SLOT_STRIDE
    if args.worlds:
        p = targets[0] + base_off + WORLD_OFF
        after = " ".join(f"{v & 0xFF:02x} {v >> 8:02x}" for v in WORLD_UNLOCKED)
        print(f"slot {args.slot} +0x6A: {bytes(data[p:p + WORLD_LEN]).hex(' ')} -> "
              f"{after}  (the 16 bytes the verified 100% save carries)", flush=True)
        for b in targets:
            for i, v in enumerate(WORLD_UNLOCKED):
                data[b + base_off + WORLD_OFF + 2 * i] = v & 0xFF
                data[b + base_off + WORLD_OFF + 2 * i + 1] = v >> 8
    if args.levels:
        for b in targets:
            for i in range(LEVEL_LEN):
                data[b + LEVEL_OFF + i] = 0xC0

    report = recompute(data)
    print("checksums rewritten:", ", ".join(f"{l} 0x{w:04X}->0x{n:04X}"
                                            for l, w, n in report if w != n) or "none",
          flush=True)

    after = verify(data)
    if not all(c[3] for c in after):
        raise SystemExit("post-write verification failed; not writing")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(bytes(data))
    print(f"wrote {args.out}  slot {args.slot}  worlds={args.worlds} levels={args.levels} "
          f"mirror={args.mirror}  (input was opened read-only)", flush=True)


if __name__ == "__main__":
    main()
