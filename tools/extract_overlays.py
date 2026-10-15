"""Extract the ARM9 and every ARM9 overlay from the NDS ROM as raw
binaries with their RAM base addresses -- the input files for the
decompiler pass (rizin/Ghidra) the 4aa static hunt named.

Prints the overlay table (id, ram base, size, file id) and writes:
    out/re/ov<id>.bin        the loaded content, base 0x<ram>

Usage:
    python tools/extract_overlays.py
"""
from __future__ import annotations

import os
import struct
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROM = os.path.join(ROOT, "data", "0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds")
OUT = os.path.join(ROOT, "out", "re")


def main():
    raw = open(ROM, "rb").read()
    arm9_off = struct.unpack_from("<I", raw, 0x20)[0]
    arm9_size = struct.unpack_from("<I", raw, 0x2C)[0]
    arm9_ram = struct.unpack_from("<I", raw, 0x28)[0]
    ov_off = struct.unpack_from("<I", raw, 0x50)[0]
    ov_size = struct.unpack_from("<I", raw, 0x54)[0]
    fat_off = struct.unpack_from("<I", raw, 0x48)[0]

    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "arm9.bin"), "wb").write(raw[arm9_off:arm9_off + arm9_size])
    print(f"arm9: 0x{arm9_size:X} bytes at ram 0x{arm9_ram:08X} -> out/re/arm9.bin")

    table = []
    for i in range(ov_size // 0x20):
        b = ov_off + 0x20 * i
        oid, ram, rsize, bss, a0, a1, fid, crc = struct.unpack_from("<8I", raw, b)
        s, e = struct.unpack_from("<II", raw, fat_off + 8 * fid)
        blob = raw[s:e]
        path = os.path.join(OUT, f"ov{oid}.bin")
        open(path, "wb").write(blob)
        table.append({"id": oid, "ram": ram, "file_size": len(blob), "ram_size": rsize,
                      "bss": bss, "file_id": fid})
        print(f"ov{oid:3d}: file {len(blob):6d}B  ram 0x{ram:08X}  "
              f"ram_size 0x{rsize:X}  bss 0x{bss:X}  fid {fid}")

    import json
    json.dump({"arm9_ram": arm9_ram, "arm9_size": arm9_size, "overlays": table},
              open(os.path.join(OUT, "overlay_table.json"), "w"), indent=1)
    print(f"{len(table)} overlays -> out/re/")


if __name__ == "__main__":
    sys.exit(main())
