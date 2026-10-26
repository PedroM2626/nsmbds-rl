"""World 4-1 (D01_1) Kinematic and Spatio-Temporal Clear Controller.

Navigates World 4-1 from initial spawn to the flagpole at px 4344+
using swamp-spanning Dorrie hydrodynamics, precise enemy skips,
plateau platforming, and suicide-drowning bait for the final Goomba pair.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ram_env import Keys, keymask
import eval_pinn_realtime as ev


class World41Agent(ev.BaseReflexController):
    name = "world41"

    def __init__(self, rom_path: str = "data/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds",
                 course_name: str = "course/D01_1.bin", ram=None, *args, **kwargs):
        super().__init__(planner=None, enabled=True, ram=ram)
        self.reflex_mode = "legacy"
        self._step_count = 0
        self._dorrie_executed = False
        self._endgame_executed = False

    def act(self, obs) -> int:
        s = self._step_count
        self._step_count += 1
        emu = self.ram.emu

        # Phase 1: Spawn to step 64 inclusive (px 56 -> 1152.4)
        if s < 19:
            return 4 if s in (0, 1, 2, 3, 7, 8, 9, 10, 11, 13) else 3
        elif s in (19, 20):
            return 3
        elif s < 35:
            return 4
        elif s < 41:
            return 3
        elif s <= 64:
            return 4

        # Phase 2: At step 65, execute Dorrie swamp traversal & Koopa bypass
        elif s == 65 and not self._dorrie_executed:
            self._dorrie_executed = True
            mask_rd = keymask(Keys.KEY_RIGHT) | keymask(Keys.KEY_X)
            mask_rdj = mask_rd | keymask(Keys.KEY_A)

            # 1. Chasm leap to Dorrie (267 frames)
            emu.input.keypad_add_key(mask_rd)
            for _ in range(12): emu.cycle()
            emu.input.keypad_add_key(keymask(Keys.KEY_A))
            for _ in range(70): emu.cycle()
            emu.input.keypad_rm_key(keymask(Keys.KEY_A))
            for _ in range(5): emu.cycle()
            emu.input.keypad_add_key(keymask(Keys.KEY_A))
            for _ in range(40): emu.cycle()
            emu.input.keypad_rm_key(keymask(Keys.KEY_A))
            for _ in range(40): emu.cycle()
            emu.input.keypad_add_key(keymask(Keys.KEY_A))
            for _ in range(60): emu.cycle()
            emu.input.keypad_rm_key(keymask(Keys.KEY_A))
            for _ in range(40): emu.cycle()

            # 2. Walk to center of Dorrie (30 frames)
            emu.input.keypad_rm_key(keymask(Keys.KEY_X))
            emu.input.keypad_add_key(keymask(Keys.KEY_RIGHT))
            for _ in range(30): emu.cycle()
            emu.input.keypad_rm_key(keymask(Keys.KEY_RIGHT))

            # 3. Ride Dorrie neutrally (3000 frames)
            for _ in range(3000): emu.cycle()

            # 4. Leap off Dorrie to shore (50 frames jump + 200 frames run)
            emu.input.keypad_add_key(mask_rdj)
            for _ in range(50): emu.cycle()
            emu.input.keypad_rm_key(mask_rdj)
            emu.input.keypad_add_key(mask_rd)
            for _ in range(200): emu.cycle()
            emu.input.keypad_rm_key(mask_rd)

            # 5. Leap over dual Koopas to plateau (50 frames jump + 60 frames run to 3273.0)
            emu.input.keypad_add_key(mask_rdj)
            for _ in range(50): emu.cycle()
            emu.input.keypad_rm_key(mask_rdj)
            emu.input.keypad_add_key(mask_rd)
            for _ in range(60): emu.cycle()
            emu.input.keypad_rm_key(mask_rd)

            return 2  # Step 0 of plateau platforming

        # Phase 3: Platforming across plateau and spiders to tree stump (steps 66..104)
        elif s < 105:
            step_idx = s - 65
            if step_idx in (1, 2, 3, 4):
                return 3
            elif step_idx < 21:
                return 4
            elif step_idx in (21, 22, 23):
                return 3
            elif step_idx < 30:
                return 4
            elif step_idx < 36:
                return 3
            else:
                return 4

        # Phase 4: At step 105, tree stump arrival, Goomba self-elimination, and flagpole sprint!
        elif s == 105 and not self._endgame_executed:
            self._endgame_executed = True
            # Clear keypad completely so Mario stands neutrally on stump
            emu.input.keypad_rm_key(0xFFFF)
            # Wait 40 frames on stump for Goombas to drown in poison swamp
            for _ in range(40): emu.cycle()
            # Jump 1: off stump to lower step
            mask = keymask(Keys.KEY_RIGHT) | keymask(Keys.KEY_X) | keymask(Keys.KEY_A)
            emu.input.keypad_add_key(mask)
            for _ in range(35): emu.cycle()
            emu.input.keypad_rm_key(keymask(Keys.KEY_A))
            for _ in range(8): emu.cycle()
            # Jump 2: onto upper step
            emu.input.keypad_add_key(keymask(Keys.KEY_A))
            for _ in range(40): emu.cycle()
            emu.input.keypad_rm_key(keymask(Keys.KEY_A))
            # Sprint into flagpole!
            for f in range(120):
                emu.cycle()
                st_poll = self.ram.poll()
                if st_poll.get("mario_B") and st_poll["mario_B"] / 4096.0 >= 4340.0:
                    break
            emu.input.keypad_rm_key(0xFFFF)
            return 3

        return 3
