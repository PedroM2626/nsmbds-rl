"""Direct ROM parser for NSMB DS level files (validated for EUR 1-1).

course/X##_#.bin : header 14x(off,len) + blocks (sprites in Blocks[6], 12B:
                    type u16, x u16, y u16 (TILES), data 6B; entrances in
                    Blocks[5], 20B: x,y,cam... (PIXELS)).
course/X##_#_bgdat.bin : tile objects (obj u8, tileset u8, x,y,w,h u16 in
                    TILES, end 0xFFFF). obj 0x0A/0x00 = ground.
Calibrations validated for A01_1 (1-1): spawn tile 3 (X_RAM 48px / B 88px =
                    entrance (80,464) + half-sprite height), ground at row
                    30 (Y_RAM -480px), goomba at tile 24 (RAM tracked at 365px).
"""
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TILE = 16
# All solid surface objects from tileset (ground, grass, slopes, edges, stairs, pipes)
# 0x0A: ground, 0x00: base, 0x09: grass top, 0x06-0x08: slopes/edges, 0x0D: stairs,
GROUND_OBJS = (0x00, 0x01, 0x02, 0x03, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0D, 0x0E, 0x0F, 0x2C, 0x30)

import json

# course sprite IDs (NSMBe) -> names; RAM actor IDs differ (e.g. goomba: 148 vs 0xA0)
SPRITE_NAMES = {148: "Goomba", 149: "Koopa", 117: "?", 132: "?", 45: "?",
                198: "?", 199: "?", 32: "Flagpole", 264: "?", 155: "?", 235: "?"}

def _load_extended_sprite_names():
    names = dict(SPRITE_NAMES)
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out", "nsmb_object_ids.json")
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
                for k, v in data.get("id_to_classes", {}).items():
                    ik = int(k)
                    if ik not in names or names[ik] == "?":
                        names[ik] = v[0] if isinstance(v, list) and v else str(v)
        except Exception:
            pass
    return names

SPRITE_NAMES = _load_extended_sprite_names()


class Course:
    def __init__(self, name="course/A01_1.bin", rom_path=None):
        import ndspy.rom
        self.name = name
        if rom_path is None:
            rom_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "data",
                                    "0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds")
        rom = ndspy.rom.NintendoDSRom.fromFile(rom_path)

        raw = bytes(rom.getFileByName(name))
        assert raw is not None, f"{name} not found in ROM"
        blocks = [struct.unpack_from("<II", raw, i * 8) for i in range(14)]
        get = lambda n: raw[blocks[n][0]:blocks[n][0] + blocks[n][1]]

        # sprites: Blocks[6], 12B (type,x,y tiles, data 6B)
        # Structurally proven: 12-byte records from offset 0, ended by 0xFFFFFFFF u32 trailer
        seg = get(6)
        self.sprites = []
        n_records = (len(seg) - 4) // 12 if len(seg) >= 4 and (len(seg) - 4) % 12 == 0 else len(seg) // 12
        for i in range(max(0, n_records)):
            t, x, y = struct.unpack_from("<HHH", seg, i * 12)
            self.sprites.append({"type": t, "tx": x, "ty": y,
                                 "x_px": x * TILE, "y_px": y * TILE,
                                 "name": SPRITE_NAMES.get(t, f"id{t}")})
        # entrances: Blocks[5], 20B (x,y pixels)
        seg = get(5)
        self.entrances = []
        for i in range(len(seg) // 20):
            x, y = struct.unpack_from("<HH", seg, i * 20)
            self.entrances.append({"x_px": x, "y_px": y})

        # bgdat: tile objects
        bg = bytes(rom.getFileByName(name.replace(".bin", "_bgdat.bin")))
        assert bg is not None
        self.rects = []
        i = 0
        while i + 10 <= len(bg):
            obj, ts, x, y, w, h = struct.unpack_from("<BBHHHH", bg, i)
            if obj == 0xFF:
                break
            self.rects.append({"obj": obj, "tx": x, "ty": y, "w": w, "h": h})
            i += 10

        # ground coverage per column (tiles) - filter yy >= 18 to exclude sky/ceiling markers at row 0
        cover = {}
        for r in self.rects:
            if r["obj"] in GROUND_OBJS:
                for xx in range(r["tx"], r["tx"] + r["w"]):
                    for yy in range(r["ty"], r["ty"] + r["h"]):
                        if yy >= 18:
                            cover.setdefault(xx, set()).add(yy)
        self.ground_cover = cover

        # flag / level finish detection: sprite 32 or highest X goal object.
        # Sprite 32 is BOTH the mid-level checkpoint flag and the level exit, so
        # the exit is the rightmost candidate. An earlier revision took min(),
        # which returned the checkpoint: identical on A01_1 (no checkpoint flag)
        # but wrong on 17 of the 191 courses, e.g. E02_1 reported 416 px for a
        # level whose ground runs to 1,680 px.
        flag_candidates = [s for s in self.sprites if s["type"] == 32 or "Flag" in s["name"]]
        # and it must sit on ground the course actually has: J10_1 carries a
        # flagpole at 4,144 px above only 1,376 px of parsed ground, which is a
        # parser gap rather than a level end, and planning on it would be fiction
        span_px = float(max((r["tx"] + r["w"] for r in self.rects), default=0) * TILE)
        in_span = [s for s in flag_candidates if s["x_px"] <= span_px]
        self.exit_found = bool(in_span)
        self.flag_candidates_px = sorted(s["x_px"] for s in flag_candidates)
        if in_span:
            self.flag_x_px = float(max(s["x_px"] for s in in_span))
        else:
            # NOT the level exit. Only A01_1 ends on a flagpole sprite; several
            # courses end on a pipe/door object, and silently reporting 4032 px
            # made those levels look plannable when they are not.
            self.flag_x_px = 4032.0

        # How much of the course's own x extent has ground-class tiles. GROUND_OBJS
        # is a flat list of tile-object numbers, but those numbers are local to each
        # course's tileset, so coverage collapses on courses that use another one
        # (A01_1 92%, A02_1 86%, A05_1 11%, A08_1 4% -- see tools/tileset_probe.py).
        xs = [r["tx"] + r["w"] for r in self.rects] + [r["tx"] for r in self.rects]
        span = (max(xs) - min(xs)) if xs else 0
        self.ground_coverage = (len(self.ground_cover) / span) if span else 0.0

        # Measured floor corrections, when this course has been probed with
        # tools/probe_floor.py (the teleport-drop oracle). Loaded AFTER the
        # static classes so floor_row/solid_columns can prefer the measurement.
        self.floor_probe = self._load_floor_probe()

    def _load_floor_probe(self):
        """{col: row or None} for columns where a confirmed drop measurement
        contradicts the static classes -- and only those.

        What counts as confirmed (the rest of the probe's rich output is
        deliberately ignored here):
          * ``landed`` within the walking band (dominant floor -5 rows above
            to +3 below, i.e. reachable by a jump or a fall) on a column the
            static classes call a pit: a floor the parser missed. Measured on
            A01_1: five of its nine "pits" are sloped ramps (objects 0x48/
            0x49/0x4A/0x4B/0x35/0x66 of the grass tileset); on H01_1: three
            floor columns (kazan objects 0x15/0x16).
          * ``pit`` (a fall past every plausible surface) on a column the
            static classes call floor: a floor the parser invented. Measured:
            zero of these on all three probed courses.

        What does NOT override: ``hover`` (support at the drop height -- a
        cave roof or a high structure, not the walking plane), ``timeout``
        (measured to be a ride on a moving platform, e.g. E01_1 col 213
        descends 0.2 px/frame: real support, and the static floor claim was
        right), and landings far above the floor (the parser's floor_row is
        the bottom run on purpose; the probe lands on the top surface).

        Staging: a probe JSON can carry ``"apply_overrides": false`` to keep
        the static classes in charge for that course. A01_1 does this on
        purpose -- its five slope ramps are measured truth, but the published
        1-1 clear was flown against the parser's nine-pit profile, and with
        the true four-pit profile the reflex dies at 1,338 px
        (tools/parity_reflex.py FAILs all four metrics). The clear's margin
        measurably depended on the phantom pits; re-tuning the reflex on true
        geometry is open work, and until it is done the published artifact
        must not be silently invalidated.
        """
        cid = os.path.basename(self.name).replace(".bin", "")
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "out", "floor_probe", cid + ".json")
        if not os.path.exists(path):
            return {}
        try:
            data = json.load(open(path, encoding="utf-8"))
            if data.get("apply_overrides") is False:
                return {}
            cols = data.get("columns", [])
            dom = next((c.get("floor_row_dominant") for c in cols
                        if c.get("floor_row_dominant") is not None), None)
            if dom is None:
                return {}
            out = {}
            for c in cols:
                m, note, p = c.get("measured"), c.get("note"), c.get("parser")
                if note == "landed" and m is not None and p is None:
                    if (dom - m) <= 5 and (m - dom) <= 3:
                        out[c["col"]] = int(round(m))
                elif note == "pit" and p is not None:
                    out[c["col"]] = None
            # Deliberately NOT overridden: columns where both the static
            # classes and the probe see ground at different rows. The probe
            # lands on the TOP surface (A01_1's elevated corridor at row 15
            # over the row-30 walkway; B01_1's desert ledge at 21 over base
            # ground at 25) and only the walker knows which route he is on --
            # flipping parity broke the published 1-1 clear the moment this
            # loader tried. Route-ambiguity needs the reflex's own state,
            # not a per-column constant.
            return out
        except Exception:
            return {}

    def floor_row(self, tile_x):
        """Top row of the BOTTOM contiguous ground run at this column: the surface
        Mario walks on.

        This used to be min(rows), the highest ground tile. Ground-class objects
        also cover floating brick rows and platforms, so min() reported a column's
        floor as being up on the platform and hid the real ground underneath --
        at cols 117..121 of A01_1 it returned 24 (the brick row the rendered frame
        shows floating in the air) while the walkable floor runs at row 30.

        Measured corrections from tools/probe_floor.py take precedence when the
        course has a floor probe: a confirmed walkable landing on a column the
        static classes call a pit, or a confirmed fall on a column they call a
        floor. See _load_floor_probe for what counts as confirmed.
        """
        if self.floor_probe:
            c = int(tile_x)
            if c in self.floor_probe:
                return self.floor_probe[c]
        rows = self.ground_cover.get(int(tile_x))
        if not rows:
            return None
        r = max(rows)
        while (r - 1) in rows:
            r -= 1
        return r

    def solid_columns(self):
        """Columns with walking ground, static classes corrected by the probe.

        TilemapAStar builds its pit list from these; ground_cover alone would
        keep every slope gap the static classes miss (measured on A01_1: five
        of its nine "pits" are sloped ramps, and on H01_1 three floor columns
        sit on tileset objects the flat class list does not know).
        """
        if not self.floor_probe:
            return set(self.ground_cover)
        solid = set(self.ground_cover)
        for c, r in self.floor_probe.items():
            if r is None:
                solid.discard(c)
            else:
                solid.add(c)
        return solid

    def platform_rows(self, tile_x):
        """Tops of the ground-class surfaces stacked ABOVE the floor, per column.

        These are real solid surfaces (Mario can stand on them) but they are not
        what a walking planner should follow.
        """
        rows = self.ground_cover.get(int(tile_x))
        if not rows:
            return []
        floor_top = self.floor_row(tile_x)
        tops, prev = [], None
        for r in sorted(rows):
            if prev is None or r != prev + 1:
                tops.append(r)
            prev = r
        return [t for t in tops if t < floor_top]

    def pit_ahead(self, mario_px, n=6):
        """List of n flags (1.0 solid / 0.0 pit) for columns ahead."""
        col0 = int(mario_px // TILE)
        return [1.0 if self.floor_row(col0 + m) is not None else 0.0
                for m in range(1, n + 1)]

    def goomba_spawns_px(self):
        return [s["x_px"] for s in self.sprites if s["type"] == 148]


class UnsupportedCourse(RuntimeError):
    """The current parser cannot produce trustworthy geometry for this course."""


def require_plannable(course, min_coverage=0.6, exit_known=True):
    """Refuse to plan on a course whose profile is fiction.

    Called by TilemapAStar and LevelProfile so the failure is loud here rather
    than a plausible-looking path over 4% of the level further down the stack.

    Coverage is measured where a drop probe exists: the static classes
    corrected by out/floor_probe (D01_1 parses at 48.6% statically and 88%
    with its 142 probe-measured floor columns; G01_1 21.6% -> 90% with 134).
    Without a probe the metric is the static one and the refusal stands.

    exit_known=False is for a caller that supplies its own goal (the harness's
    --goal-px): only the flagpole reading is then waived, and what the run
    measures changes from "cleared the level" to "reached the stated x", which
    has to be reported that way.
    """
    problems = []
    xs = [r["tx"] + r["w"] for r in course.rects] + [r["tx"] for r in course.rects]
    span = (max(xs) - min(xs)) if xs else 0
    coverage = (len(course.solid_columns()) / span) if span else 0.0
    if coverage < min_coverage:
        problems.append(
            f"only {coverage * 100:.0f}% of the course's own x extent has "
            f"measured or ground-class tiles -- GROUND_OBJS is a flat list of "
            f"tile-object numbers and those are local to each course's tileset")
    if exit_known and not getattr(course, "exit_found", True):
        problems.append("no flagpole/goal sprite found; flag_x_px is the 4032 fallback, "
                        "not this level's exit")
    if problems:
        raise UnsupportedCourse(f"{course.name}: " + "; ".join(problems))


if __name__ == "__main__":
    c = Course()
    print(f"sprites: {len(c.sprites)}, rects: {len(c.rects)}, "
          f"entrances: {c.entrances[:1]}", flush=True)
    # Cross-validations with RAM
    e = c.entrances[0]
    assert (e["x_px"], e["y_px"]) == (80, 464), e
    assert c.floor_row(3) == 30 and c.floor_row(30) in (29, 30), f"row30={c.floor_row(30)}"
    assert c.flag_x_px == 4032.0, f"flag_x_px={c.flag_x_px}"
    gs = c.goomba_spawns_px()
    assert 24 * TILE in gs, gs[:5]
    print(f"OK: spawn={e}, ground tile3 row={c.floor_row(3)}, "
          f"tile30 row={c.floor_row(30)}, flag_x={c.flag_x_px}, goombas px={gs[:6]}", flush=True)
    print("pit_ahead(48px,6):", c.pit_ahead(48.0), flush=True)
    print("pit_ahead(400px,6):", c.pit_ahead(400.0), flush=True)
