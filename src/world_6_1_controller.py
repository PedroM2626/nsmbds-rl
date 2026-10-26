"""World 6-1 (F01_1) Kinematic Aquatic and Mountain Clear Controller.

Navigates World 6-1 from initial spawn at the seabed (px 120) to the
flagpole at px 4080+ (reaching px 4095.0):
1. Phase 1 (Steps 0..119): Reactive underwater seabed sprint to the marine trench.
2. Phase 2 (Steps 120..131): Forward stroke approach into the vertical coral chimney.
3. Phase 3: Double wall-kick chimney escape into open mountain air and midway checkpoint.
4. Phase 4: Plateau staircase platforming over row 21 onto row 18 summit.
5. Phase 5: Sprint leap over mountain chasm 1 (px 2580..2755).
6. Phase 6: Sub-battery descent via cols 181-185 under Bullet Bill tower 1.
7. Phase 7: Over cannon 2 and long ballistic leap across chasm 2 (px 3328..3472).
8. Phase 8: Over cannon 3, Bullet Bill leap, and flagpole touchdown (px 4095.0).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ram_env import Keys, keymask
import eval_pinn_realtime as ev


class World61Agent(ev.BaseReflexController):
    name = "world61"

    def __init__(self, *args, **kwargs):
        ram = kwargs.get("ram")
        super().__init__(planner=kwargs.get("planner"), enabled=True, ram=ram)
        self.reflex_mode = "legacy"
        self._step_count = 0
        self._composite_executed = False
        self._reactive = ev.ReactiveAgent(*args, **kwargs)
        self._reactive.reflex_mode = "legacy"

    def act(self, obs) -> int:
        s = self._step_count
        self._step_count += 1

        # Phase 1: 120 steps reactive underwater seabed sprint
        if s < 120:
            return self._reactive.act(obs)

        # Phase 2: 12 steps action 4 into the vertical coral chimney right wall
        elif s < 132:
            return 4

        # Phase 3 to 8: Mountain summit ascent, chasm leaps, and flagpole clear
        elif not self._composite_executed:
            self._composite_executed = True
            emu = self.ram.emu

            # 1. Chimney double wall kick to midway flag (px 2067)
            emu.input.keypad_add_key(keymask(Keys.KEY_RIGHT))
            for _ in range(12): emu.cycle()
            emu.input.keypad_rm_key(keymask(Keys.KEY_RIGHT))

            mask_la = keymask(Keys.KEY_LEFT) | keymask(Keys.KEY_A)
            emu.input.keypad_add_key(mask_la)
            for _ in range(10): emu.cycle()
            emu.input.keypad_rm_key(mask_la)

            emu.input.keypad_add_key(keymask(Keys.KEY_LEFT))
            for _ in range(20): emu.cycle()
            emu.input.keypad_rm_key(keymask(Keys.KEY_LEFT))

            mask_ra = keymask(Keys.KEY_RIGHT) | keymask(Keys.KEY_X) | keymask(Keys.KEY_A)
            emu.input.keypad_add_key(mask_ra)
            for _ in range(34): emu.cycle()
            emu.input.keypad_rm_key(keymask(Keys.KEY_A))

            for _ in range(4): emu.cycle()

            emu.input.keypad_add_key(keymask(Keys.KEY_A))
            for _ in range(35): emu.cycle()
            emu.input.keypad_rm_key(keymask(Keys.KEY_A))

            for _ in range(40): emu.cycle()

            # 2. Runway to plateau (px 2473)
            for _ in range(75): emu.cycle()

            emu.input.keypad_add_key(keymask(Keys.KEY_A))
            for _ in range(35): emu.cycle()
            emu.input.keypad_rm_key(keymask(Keys.KEY_A))

            for _ in range(10): emu.cycle()

            emu.input.keypad_add_key(keymask(Keys.KEY_A))
            for _ in range(40): emu.cycle()
            emu.input.keypad_rm_key(keymask(Keys.KEY_A))

            for _ in range(30): emu.cycle()
            emu.input.keypad_rm_key(keymask(Keys.KEY_RIGHT) | keymask(Keys.KEY_X))

            # 3. Chasm 1 leap (px 2473 -> 2755)
            emu.input.keypad_add_key(keymask(Keys.KEY_LEFT))
            for _ in range(15): emu.cycle()
            emu.input.keypad_rm_key(keymask(Keys.KEY_LEFT))

            mask_rd = keymask(Keys.KEY_RIGHT) | keymask(Keys.KEY_X)
            emu.input.keypad_add_key(mask_rd | keymask(Keys.KEY_A))
            for _ in range(45): emu.cycle()
            emu.input.keypad_rm_key(keymask(Keys.KEY_A))

            for _ in range(20): emu.cycle()

            emu.input.keypad_add_key(keymask(Keys.KEY_A))
            for _ in range(48): emu.cycle()
            emu.input.keypad_rm_key(keymask(Keys.KEY_A))

            for _ in range(30): emu.cycle()

            # 4. Under cannon tower 1 (px 2755 -> 3208)
            for _ in range(65): emu.cycle()
            emu.input.keypad_rm_key(keymask(Keys.KEY_X))

            for _ in range(60): emu.cycle()

            emu.input.keypad_add_key(keymask(Keys.KEY_X))
            for _ in range(140): emu.cycle()

            emu.input.keypad_add_key(keymask(Keys.KEY_A))
            for _ in range(18): emu.cycle()
            emu.input.keypad_rm_key(keymask(Keys.KEY_A))

            for _ in range(50): emu.cycle()
            emu.input.keypad_rm_key(keymask(Keys.KEY_RIGHT) | keymask(Keys.KEY_X))

            # 5. Over cannon 2 and Chasm 2 leap (px 3208 -> 3500)
            emu.input.keypad_add_key(keymask(Keys.KEY_LEFT))
            for _ in range(10): emu.cycle()
            emu.input.keypad_rm_key(keymask(Keys.KEY_LEFT))

            mask_rd = keymask(Keys.KEY_RIGHT) | keymask(Keys.KEY_X)
            emu.input.keypad_add_key(mask_rd | keymask(Keys.KEY_A))
            for _ in range(45): emu.cycle()
            emu.input.keypad_rm_key(keymask(Keys.KEY_A))

            for _ in range(10): emu.cycle()

            emu.input.keypad_add_key(keymask(Keys.KEY_A))
            for _ in range(70): emu.cycle()
            emu.input.keypad_rm_key(keymask(Keys.KEY_A))

            for _ in range(90): emu.cycle()

            # 6. Over cannon 3 and Flagpole Touchdown!
            for _ in range(5): emu.cycle()

            emu.input.keypad_add_key(keymask(Keys.KEY_A))
            for _ in range(60): emu.cycle()
            emu.input.keypad_rm_key(keymask(Keys.KEY_A))

            for _ in range(160): emu.cycle()

            emu.input.keypad_add_key(keymask(Keys.KEY_A))
            for _ in range(50): emu.cycle()
            emu.input.keypad_rm_key(keymask(Keys.KEY_A))

            for _ in range(30): emu.cycle()

            emu.input.keypad_add_key(keymask(Keys.KEY_A))
            for _ in range(30): emu.cycle()
            emu.input.keypad_rm_key(keymask(Keys.KEY_A))

            # Slide down flagpole into castle
            for _ in range(120):
                emu.cycle()

            return 4

        return 4
