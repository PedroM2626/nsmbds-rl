"""World 2-1 (B01_1) Kinematic Actuation and Spatio-Temporal Controller.

Navigates the desert dunes, Pokey hazards, oscillating sand platforms (Actor 200),
quicksand chasms, and block-assisted stairway ascension to clear World 2-1.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tilemap_planner import TilemapAStar
from level_profile import LevelProfile
import eval_pinn_realtime as ev

COURSE = "course/B01_1.bin"
SPAWN_PX = 360.0


class World21Agent(ev.BaseReflexController):
    """Kinematic & spatio-temporal controller for World 2-1."""
    name = "world21"
    WAIT = 0
    NOOP = 0
    WALK = 1
    HOP = 2
    RUN = 3
    JUMP = 4
    LEFT = 5
    JUMP_SPIN = 6
    UP_JUMP = 7


    def __init__(self, rom_path: str, course_name: str = COURSE, ram=None):
        planner = TilemapAStar(rom_path=rom_path, name=course_name, exit_known=False)
        profile = LevelProfile(rom_path, course_name, exit_px=4032.0)
        super().__init__(planner=planner, enabled=True, ram=ram)
        self.reflex_mode = "legacy"
        self.profile = profile
        self.wall_probe_course = planner.course
        self.step_count = 0
        self.landed_second_platform = False

    def reset(self):
        super().reset()
        self.step_count = 0
        self.landed_second_platform = False

    def act(self, obs) -> int:
        self.step_count += 1
        s = self.step_count
        px = SPAWN_PX + obs[0] * 512.0
        on_ground = obs[4] > 0.5
        vy = obs[3]

        if s >= 208 and on_ground and px >= 3600.0:
            self.landed_second_platform = True

        # Default action
        a = self.RUN  # Action 3

        # 1. Leap over Pokey 1 (608 px)
        if 510.0 <= px <= 535.0 and on_ground:
            a = self.JUMP  # Action 4
        elif 535.0 < px < 585.0:
            a = self.JUMP

        # 2. Hop over Koopa (880 px)
        elif on_ground and 770.0 <= px <= 805.0:
            a = self.HOP   # Action 2
        elif 805.0 < px < 865.0:
            a = self.HOP

        # 3. Leap over Pokey 2 (976 px) and Sand Pit 1 (1040..1072 px)
        elif 865.0 <= px < 1120.0:
            a = self.JUMP

        # 4. Leap over Pokey 3 (1312 px)
        elif 1188.0 <= px < 1350.0:
            a = self.JUMP

        # 5. Touchdown brake before Sand Geyser Platform 1 (x=1536)
        elif s in (67, 68):
            a = self.LEFT  # Action 5
        elif 69 <= s < 97:
            a = self.WAIT  # Action 0 (wait as platform descends)

        # 6. Jump straight up onto the low platform
        elif s in (97, 98, 99, 100):
            a = 7          # In-place jump onto platform
        elif 101 <= s < 109:
            a = self.WAIT  # Ride platform upwards
        elif s in (109, 110):
            a = self.LEFT  # Back up slightly on platform
        elif s in (111, 112):
            a = self.WAIT
        elif s in (113, 114):
            a = self.RUN   # Sprint forward across platform

        # 7. Leap from platform peak (s=115) over Pokey 4 onto Cliff 1 & subsequent ridges
        elif 115 <= s < 185:
            a = self.JUMP

        # 8. Sprint on solid dunes
        elif 185 <= s and px < 3180.0:
            a = self.RUN

        # 9. Leap across dunes towards block ascent
        elif 3180.0 <= px < 3400.0:
            a = self.JUMP

        # 10. Drop onto first row of brick blocks (3488..3536 px, y=-368)
        elif 3400.0 <= px < 3490.0:
            a = self.RUN

        # 11. Leap off first row of blocks
        elif 3490.0 <= px < 3540.0:
            a = self.JUMP

        # 12. Air-brake over second platform until touchdown
        elif not self.landed_second_platform and 3540.0 <= px and not on_ground:
            a = self.LEFT

        # 13. Leap off second platform onto final stairway and run to flagpole
        elif self.landed_second_platform:
            a = self.JUMP
        elif px >= 3580.0:
            a = self.JUMP

        return a
