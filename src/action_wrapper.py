"""Rising-Edge Action Wrappers for NSMB DS Gymnasium Environments.

Addresses the deterministic evaluation collapse in reinforcement learning agents
operating under discrete platformer action spaces.

Mechanics in NSMB DS:
1. Hardware Rising Edge:
   The Nintendo DS input register and NSMB engine state machine evaluate player
   jumps on the transition 0 -> 1 of KEY_A. If KEY_A is held continuously across
   ground contact, the engine does not register a new jump on native hardware.
2. Macro-Action Rising Edge & Duty Cycle:
   In discrete action tables (e.g. Actions V2), agents frequently saturate on
   Action 4 (RIGHT + DASH + JUMP). When evaluated deterministically (a = argmax),
   the policy emits Action 4 on every macro-step.
   - If the environment auto-pulses KEY_A on touchdown, this results in continuous
     bunny-hopping with a fixed spatial wavelength (~180 px), causing Mario to land
     inevitably in pit 2 (x = 1216-1248 px) at step 73.
   - If the environment does not pulse KEY_A, Mario never jumps again after landing.
   - If a macro-edge or grounded stride cooldown is enforced, the agent executes
     ground dashes between jumps, allowing traversal of platforms between obstacles.
"""
from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np

# Action table V2 mappings
# 0: noop
# 1: right
# 2: right + jump (A)
# 3: right + dash (X)
# 4: right + dash (X) + jump (A)
# 5: left
# 6: dash (X) + jump (A) in place
# 7: jump (A) in place

JUMP_ACTIONS = {2, 4, 6, 7}
GROUND_COUNTERPARTS = {
    2: 1,  # right + jump -> right
    4: 3,  # right + dash + jump -> right + dash
    6: 0,  # dash + jump in place -> noop
    7: 0,  # jump in place -> noop
}


class RisingEdgeActionWrapper(gym.ActionWrapper):
    """Wrapper that shapes discrete actions to enforce rising-edge jump semantics
    and/or ground stride cooldowns.

    Modes:
      - 'macro_edge': A jump action only emits jump on the rising edge
        (i.e. when transitioning from a non-jump action to a jump action).
        Consecutive repeated jump actions while grounded are mapped to their
        ground counterparts (e.g. 4 -> 3).
      - 'stride_cooldown': After landing from a jump, jump actions are mapped to
        their ground counterparts for `cooldown_steps` macro-steps, forcing Mario
        to run along solid ground before taking off again.
      - 'hardware_pulse': Operates at the macro level ensuring that jump actions
        pulse for 1 step and then release for 1 step if repeated.
    """

    def __init__(
        self,
        env: gym.Env,
        mode: str = "stride_cooldown",
        cooldown_steps: int = 2,
    ):
        super().__init__(env)
        if mode not in ("macro_edge", "stride_cooldown", "hardware_pulse"):
            raise ValueError(f"Unknown mode: {mode}")
        self.mode = mode
        self.cooldown_steps = cooldown_steps

        self.prev_action = 0
        self.ground_steps = 0
        self.airborne_steps = 0
        self.was_grounded = True
        self.just_landed = False

    def reset(self, **kwargs):
        self.prev_action = 0
        self.ground_steps = 0
        self.airborne_steps = 0
        self.was_grounded = True
        self.just_landed = False
        return self.env.reset(**kwargs)

    def _is_on_ground(self) -> bool:
        """Query underlying environment for ground contact state."""
        # Check unwrapped or inner env attributes
        env_cur = self.env
        while hasattr(env_cur, "env"):
            if hasattr(env_cur, "_last_on_ground"):
                return bool(env_cur._last_on_ground)
            if hasattr(env_cur, "ram") and env_cur.ram is not None:
                vy = env_cur._s32read(0x021C1908) if hasattr(env_cur, "_s32read") else 0
                return (vy == 0)
            env_cur = env_cur.env
        if hasattr(env_cur, "_last_on_ground"):
            return bool(env_cur._last_on_ground)
        return True

    def action(self, action: int) -> int:
        act = int(action)
        is_jump = act in JUMP_ACTIONS
        was_jump = self.prev_action in JUMP_ACTIONS
        on_ground = self._is_on_ground()

        if on_ground:
            self.ground_steps += 1
            self.airborne_steps = 0
        else:
            self.airborne_steps += 1
            self.ground_steps = 0

        target_act = act

        if self.mode == "macro_edge":
            # Jump is only active if transitioning from non-jump, or if in the air
            if is_jump and on_ground and was_jump:
                target_act = GROUND_COUNTERPARTS.get(act, act)

        elif self.mode == "stride_cooldown":
            # If on ground and hasn't satisfied cooldown steps, convert jump to ground counterpart
            if is_jump and on_ground and (self.ground_steps <= self.cooldown_steps) and was_jump:
                target_act = GROUND_COUNTERPARTS.get(act, act)

        elif self.mode == "hardware_pulse":
            # Alternating pulse if jump is held on ground
            if is_jump and on_ground and was_jump and (self.ground_steps % 2 == 1):
                target_act = GROUND_COUNTERPARTS.get(act, act)

        self.prev_action = act
        return target_act
