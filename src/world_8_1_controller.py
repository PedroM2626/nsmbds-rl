"""World 8-1 (H01_1) Kinematic Actuation and Spatio-Temporal Controller.

This controller executes the end-to-end clearance of World 8-1 on the live
DeSmuME emulator core. It handles:
  1. Pit 1 (cols 26..28, px 416..464) safe launch and Splunkin enemy stomping.
  2. Dynamic Bullet Bill evasion and Bill Blaster (sprite 27) clearance.
  3. Bill Blaster staircase traversal (cols 202..213): launching a running leap
     from col 200 (px ~3205) sailing over the multi-tier blasters directly onto
     the Platform 3 roof (px 3360+).
  4. Endgame pit clearance (cols 222-223 and cols 234-239).
  5. Sprinted flagpole contact at px 3920.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tilemap_planner import TilemapAStar, dynamic_jump_reach_tiles
from level_profile import LevelProfile
import eval_pinn_realtime as ev


def setup_world_8_1_planner_and_profile(rom_path: str, course_name: str = "course/H01_1.bin"):
    """Prepares TilemapAStar and LevelProfile with Bill Blaster sprite solid semantics."""
    planner = TilemapAStar(rom_path=rom_path, name=course_name)
    # Register sprite type 27 (Bill Blasters) into surface & obstacle tables
    for s in planner.course.sprites:
        if s["type"] == 27:
            col = s["tx"]
            top = s["ty"] - 1
            if col in planner.surface:
                planner.surface[col] = min(planner.surface[col], top)
                planner.obstacle_h[col] = max(planner.obstacle_h.get(col, 0), 2)
            else:
                planner.solid.add(col)
                planner.surface[col] = top

    planner.pits = []
    c = planner.min_col
    while c <= planner.max_col:
        if c not in planner.solid:
            a = c
            while c <= planner.max_col and c not in planner.solid:
                c += 1
            planner.pits.append((a, c - 1))
        c += 1
    planner.takeoffs = {a - 1: (a, b, b + 1, b - a + 1) for (a, b) in planner.pits if (a - 1) in planner.solid}

    profile = LevelProfile(rom_path, course_name)
    for s in profile.course.sprites:
        if s["type"] == 27:
            col = s["tx"]
            top = s["ty"] - 1
            profile.surface[col] = min(profile.surface.get(col, top), top)
            profile.persistent_rise_cols.add(col)
    for cc in sorted(profile.surface):
        prev = profile.surface.get(cc - 1)
        cur = profile.surface.get(cc)
        if prev is not None and cur < prev:
            profile.persistent_rise_cols.add(cc)

    return planner, profile


class World81Agent(ev.BaseReflexController):
    """Specialized kinematic & reflexive agent for World 8-1."""
    name = "world81"

    def __init__(self, rom_path: str, course_name: str = "course/H01_1.bin", ram=None):
        planner, profile = setup_world_8_1_planner_and_profile(rom_path, course_name)
        super().__init__(planner=planner, enabled=True, ram=ram)
        self.reflex_mode = "profile"
        self.profile = profile
        self.wall_probe_course = planner.course
        ev.HOSTILE_ENEMY_TYPES.add(36)  # Bullet Bill

    def _threat_landing_unsafe(self, obs) -> bool:
        """Gated landing cone evaluation (lo..lo+1) to prevent false staircase alarms."""
        if self.planner is None:
            return False
        abs_px = self._abs_px(obs)
        col = int(abs_px // 16)
        reach = dynamic_jump_reach_tiles()
        lo, hi = col + reach - 1, col + reach + 1
        try:
            _, prof = self.planner.surface_ahead(abs_px, look_tiles=hi - col + 1)
        except Exception:
            return False
        return any(r is None and lo <= c <= hi for c, r in prof)

    def act(self, obs) -> int:
        spawn_px = self.profile.spawn_px
        px = spawn_px + obs[0] * 512.0
        on_ground = obs[4] > 0.5
        vy = obs[3]

        # Base reactive decision
        a = self.reflex_act(obs)
        if a is None:
            a = self.RUN

        has_immediate_pit = any(obs[17 + c] < 0.5 for c in range(2))

        # 1. Pit 1 & Splunkin kinematics
        if has_immediate_pit and on_ground:
            a = self.JUMP  # Action 4
        elif not has_immediate_pit and 220.0 <= px <= 265.0 and on_ground:
            a = self.IN_PLACE_HOP  # Action 7
        elif not has_immediate_pit and 340.0 <= px <= 364.0:
            a = self.HOP  # Action 2

        # 2. Mid-air descent enemy avoidance
        elif not on_ground and vy <= 0.0:
            for k in range(3):
                edx = obs[8 + 3 * k] * 256.0
                edy = obs[9 + 3 * k] * 256.0
                et = obs[10 + 3 * k] * 256.0
                if ev.is_hostile_enemy(et) and 0.0 <= edx <= 85.0 and abs(edy) < 45.0:
                    if 980.0 <= px <= 1160.0:
                        a = self.IN_PLACE_HOP
                    elif edx <= 38.0 and not (2740.0 <= px <= 2920.0 or 3260.0 <= px <= 3420.0):
                        a = self.IN_PLACE_HOP

        # 3. Bill Blaster 1 & 2 sequence:
        # Early leap at 2660..2685 over the col 173 Bullet Bill
        if 2660.0 <= px <= 2685.0 and on_ground:
            a = self.JUMP
        # Hold running leap (a = 4) continuously to float over the col 182 Bill Blaster!
        elif 2685.0 < px <= 2950.0:
            a = self.JUMP

        # 4. Staircase approach (col 197-200)
        elif 3160.0 <= px < 3195.0:
            a = self.RUN
        # Launch running leap at 3195..3215:
        elif 3195.0 <= px <= 3215.0 and on_ground:
            a = self.JUMP
        # Fly all the way across the staircase onto Platform 3 (roof):
        elif 3215.0 < px <= 3480.0:
            a = self.JUMP

        # 5. Roof sprint & Pit 1 leap
        elif 3480.0 < px < 3515.0:
            a = self.RUN
        elif 3515.0 <= px <= 3540.0 and on_ground:
            a = self.JUMP
        elif 3540.0 < px <= 3620.0:
            a = self.JUMP

        # 6. Intermediate blocks & Pit 2 leap
        elif 3620.0 < px < 3705.0:
            a = self.RUN
        elif 3705.0 <= px <= 3735.0 and on_ground:
            a = self.JUMP
        elif 3735.0 < px <= 3820.0:
            a = self.JUMP

        # 7. Final sprint into flagpole
        elif px > 3820.0:
            a = self.RUN

        if self._queue:
            self._queue.clear()

        # 8. Mid-air ascent pass under high enemies (172/312)
        if not on_ground and vy > 0.0 and not has_immediate_pit:
            for k in range(3):
                edx = obs[8 + 3 * k] * 256.0
                edy = obs[9 + 3 * k] * 256.0
                et = int(round(obs[10 + 3 * k] * 256.0))
                if et in (172, 312) and 0.0 <= edx <= 65.0 and edy > 6.0:
                    a = self.RUN

        return a
