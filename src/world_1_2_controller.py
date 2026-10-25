"""World 1-2 (A02_1) Kinematic Actuation and Spatio-Temporal Controller."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tilemap_planner import TilemapAStar
from level_profile import LevelProfile
import eval_pinn_realtime as ev

COURSE = "course/A02_1.bin"
SPAWN_PX = 40.0


class World12Agent(ev.BaseReflexController):
    """Specialized kinematic & reflexive agent for World 1-2."""
    name = "world12"

    def __init__(self, rom_path: str, course_name: str = COURSE, ram=None):
        planner = TilemapAStar(rom_path=rom_path, name=course_name, exit_known=False)
        profile = LevelProfile(rom_path, course_name, exit_px=4038.8)
        super().__init__(planner=planner, enabled=True, ram=ram)
        self.reflex_mode = "legacy"
        self.profile = profile
        self.wall_probe_course = planner.course

    def act(self, obs) -> int:
        px = SPAWN_PX + obs[0] * 512.0
        on_ground = obs[4] > 0.5
        vy = obs[3]

        # Base reactive decision
        a = self.reflex_act(obs)
        if a is None:
            a = self.RUN

        # 1. Pipe hop sequence (px 2980..3225)
        if 2985.0 <= px <= 3010.0 and on_ground:
            a = self.HOP  # Action 2: Hop onto ledge
        elif 3010.0 < px < 3075.0:
            a = self.RUN  # Action 3: Run across ledge
        elif 3075.0 <= px <= 3095.0 and on_ground:
            a = self.HOP  # Action 2: Hop onto pipe 1
        elif 3130.0 <= px <= 3160.0 and on_ground:
            a = self.HOP  # Action 2: Hop onto pipe 2
        elif 3190.0 <= px <= 3225.0 and on_ground:
            a = self.JUMP  # Action 4: Running leap onto col 205 ledge

        # 2. Elevated yellow brick run before purple seesaw lift (px 3520..3655)
        elif 3520.0 <= px < 3630.0:
            a = self.RUN  # Run to the edge of the yellow bricks
        elif 3630.0 <= px <= 3655.0 and on_ground:
            a = self.JUMP  # Leap from the edge onto purple seesaw lift
        elif 3655.0 < px <= 3760.0:
            a = self.JUMP  # Hold running leap onto lift
        elif 3760.0 < px < 3870.0:
            a = self.RUN  # Run across purple seesaw lift
        elif 3870.0 <= px <= 3910.0 and on_ground:
            a = self.JUMP  # Leap off far end of lift onto final platform
        elif 3910.0 < px <= 4050.0:
            a = self.JUMP  # Carry momentum directly into exit pipe

        if self._queue:
            self._queue.clear()

        return a
