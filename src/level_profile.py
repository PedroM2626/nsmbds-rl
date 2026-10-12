"""Level-derived parameters for the reflex controller.

The published World 1-1 clear was driven by seven hand-placed pixel bands inside
BaseReflexController (3870/3920/4020 endgame, 1450..1485 corridor step-up,
2430..2630 floating-block skip, abs_px >= 3700 staircase hop, 3480..3520 pit-9
leap suppression, planner.is_narrow_corridor cols 91..97, SPAWN_ABS_PX = 88).
Those are coordinates of *one* level, so they do not transfer.

This module derives the equivalent quantities from the ROM course data instead,
so the same controller can be pointed at any course that Course can parse.
Every value here is computed, nothing is a constant copied from 1-1.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from course import Course, TILE  # noqa: E402

# Half a 16 px sprite: the entrance record is the sprite origin, Mario's RAM
# base sits half a tile higher (validated for A01_1: entrance x 80 -> 88).
HALF_SPRITE_PX = TILE / 2.0


class LevelProfile:
    """Geometry-derived stand-ins for the 1-1 pixel constants."""

    def __init__(self, rom_path, name="course/A01_1.bin"):
        self.course = Course(name=name, rom_path=rom_path)
        self.name = name

        ent = self.course.entrances[0] if self.course.entrances else {"x_px": 0}
        self.spawn_px = float(ent["x_px"]) + HALF_SPRITE_PX
        self.flag_px = float(self.course.flag_x_px)

        cols = sorted(self.course.ground_cover.keys())
        self.min_col, self.max_col = cols[0], cols[-1]

        # Walkable surface per column, same construction as TilemapAStar: the top
        # of the solid run contiguous with the floor.
        solid_rows = {}
        for r in self.course.rects:
            for cc in range(r["tx"], r["tx"] + r["w"]):
                s = solid_rows.setdefault(cc, set())
                for yy in range(r["ty"], r["ty"] + r["h"]):
                    s.add(yy)
        self.surface = {}
        for cc in range(self.min_col, self.max_col + 1):
            fr = self.course.floor_row(cc)
            if fr is None:
                continue
            top = fr
            rows = solid_rows.get(cc, set())
            while (top - 1) in rows:
                top -= 1
            self.surface[cc] = top

        # Pits as maximal runs of floor-less columns.
        self.pits = []
        c = self.min_col
        while c <= self.max_col:
            if c not in self.surface:
                a = c
                while c <= self.max_col and c not in self.surface:
                    c += 1
                self.pits.append((a, c - 1))
            c += 1
        self.takeoffs = {a - 1: (a, b) for (a, b) in self.pits if a - 1 in self.surface}

        # The end staircase: the last run of >= 3 columns before the flag whose
        # surface never drops as Mario advances (row non-increasing). It does NOT
        # touch the flagpole: in A01_1 the stairs end at col 242 and flat floor
        # runs from 243 to 250, with the flag base block sitting 10 rows up.
        flag_col = int(self.flag_px // TILE)
        scan_lo = max(self.min_col, flag_col - 40)
        run = None
        best = None
        for cc in range(scan_lo, flag_col + 1):
            cur = self.surface.get(cc)
            prev = self.surface.get(cc - 1)
            rising = cur is not None and prev is not None and prev >= cur
            if rising:
                run = (cc - 1, cc) if run is None else (run[0], cc)
                if run[1] - run[0] + 1 >= 3:
                    best = run
            else:
                run = None
        if best:
            self.ramp_top_col = best[1]
            self.ramp_start_px = float(best[0] * TILE)
            self.ramp_end_px = float((best[1] + 1) * TILE)
            # Take off from the last two columns of the stairs and land past the
            # lip; 12 px is Mario's half width, not a level coordinate.
            self.ramp_leap_from = self.ramp_end_px - 2 * TILE
            self.ramp_leap_to = self.ramp_end_px + 2 * TILE
            self.run_to_px = self.flag_px - 12.0
        else:
            self.ramp_top_col = None
            self.ramp_start_px = None
            self.ramp_end_px = None
            self.ramp_leap_from = None
            self.ramp_leap_to = None
            self.run_to_px = None

        # Isolated single-column rises are stacked ?-blocks you run under, not
        # walls: a real ledge/staircase keeps its height for >= 2 columns.
        # This replaces the hardcoded 2430..2630 skip.
        self.persistent_rise_cols = set()
        for cc in sorted(self.surface):
            if self.surface[cc] >= self.floor_row_at(cc):
                continue
            nxt = self.surface.get(cc + 1)
            if nxt is not None and nxt <= self.surface[cc]:
                self.persistent_rise_cols.add(cc)

        # Elevated narrow ledges: a 2-tile step up whose next column goes higher
        # still -- Mario must hop onto the lip, not leap over it. Replaces the
        # 1450..1485 band.
        self.ledge_cols = []
        for cc in sorted(self.surface):
            cur = self.surface.get(cc)
            up = self.surface.get(cc + 1)
            beyond = self.surface.get(cc + 2)
            if cur is None or up is None or beyond is None:
                continue
            if cur - up == 2 and up - beyond >= 1:
                self.ledge_cols.append(cc + 1)

    def floor_row_at(self, col):
        return self.course.floor_row(col)

    def abs_px(self, rel_x):
        """obs[0] is Mario's x relative to the episode spawn, in 512 px units."""
        return self.spawn_px + rel_x * 512.0

    def near_takeoff(self, abs_px, ahead_px=64.0):
        """True when a pit takeoff edge sits within ahead_px in front of Mario."""
        col = int(abs_px // TILE)
        for tk in self.takeoffs:
            d = tk * TILE - abs_px
            if 0.0 <= d <= ahead_px:
                return True
        return False

    def ledge_ahead(self, abs_px, look_tiles=3):
        col = int(abs_px // TILE)
        for d in range(1, look_tiles + 1):
            if (col + d) in self.ledge_cols:
                return col + d
        return None

    def on_end_ramp(self, abs_px):
        if self.ramp_start_px is None:
            return False
        return self.ramp_start_px <= abs_px < self.flag_px

    def summary(self):
        return (f"{self.name}: spawn={self.spawn_px:.0f}px flag={self.flag_px:.0f}px "
                f"cols[{self.min_col},{self.max_col}] pits={len(self.pits)} "
                f"ramp={self.ramp_start_px}..{self.ramp_end_px} "
                f"ledges={self.ledge_cols[:6]}")


if __name__ == "__main__":
    ROM = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data", "0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds")
    for nm in ("course/A01_1.bin", "course/A08_1.bin"):
        try:
            print(LevelProfile(ROM, nm).summary())
        except Exception as e:
            print(f"{nm}: FAILED {e}")
