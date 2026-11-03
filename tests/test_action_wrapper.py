"""Unit tests for RisingEdgeActionWrapper (no-emulator)."""
import gymnasium as gym
import numpy as np
import pytest

from action_wrapper import (
    RisingEdgeActionWrapper,
    JUMP_ACTIONS,
    GROUND_COUNTERPARTS,
)


class MockEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Discrete(8)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        self._last_on_ground = True

    def reset(self, **kwargs):
        self._last_on_ground = True
        return np.zeros((4,), dtype=np.float32), {}

    def step(self, action):
        return np.zeros((4,), dtype=np.float32), 1.0, False, False, {}


def test_invalid_mode():
    env = MockEnv()
    with pytest.raises(ValueError, match="Unknown mode"):
        RisingEdgeActionWrapper(env, mode="invalid_mode")


def test_macro_edge_transition():
    env = MockEnv()
    wrapped = RisingEdgeActionWrapper(env, mode="macro_edge")
    wrapped.reset()

    # Step 1: Start with Action 1 (Right, non-jump)
    assert wrapped.action(1) == 1

    # Step 2: Rising edge: Transition to Action 4 (Right + Dash + Jump)
    # Since prev_action was 1 (non-jump), Jump is allowed!
    assert wrapped.action(4) == 4

    # Step 3: Repeated Action 4 while still on ground:
    # Must be mapped to ground counterpart (Action 3: Right + Dash)
    assert wrapped.action(4) == 3

    # Step 4: Mario becomes airborne
    env._last_on_ground = False
    # While airborne, Action 4 passes through so jump button stays held
    assert wrapped.action(4) == 4

    # Step 5: Mario lands back on ground
    env._last_on_ground = True
    # Still holding Action 4 on the ground -> mapped to 3
    assert wrapped.action(4) == 3

    # Step 6: Policy explicitly releases jump to Action 3
    assert wrapped.action(3) == 3

    # Step 7: Policy requests jump again (rising edge 3 -> 4) -> allowed!
    assert wrapped.action(4) == 4


def test_stride_cooldown():
    env = MockEnv()
    cooldown = 2
    wrapped = RisingEdgeActionWrapper(env, mode="stride_cooldown", cooldown_steps=cooldown)
    wrapped.reset()

    # Step 1: Jump initiated (first action)
    # prev_action = 0 (non-jump) -> allowed
    assert wrapped.action(4) == 4

    # Mario becomes airborne
    env._last_on_ground = False
    assert wrapped.action(4) == 4
    assert wrapped.action(4) == 4

    # Mario touches down
    env._last_on_ground = True

    # Ground step 1 (<= cooldown 2): action 4 mapped to 3
    assert wrapped.action(4) == 3

    # Ground step 2 (<= cooldown 2): action 4 mapped to 3
    assert wrapped.action(4) == 3

    # Ground step 3 (> cooldown 2): cooldown expired -> action 4 allowed to fire new jump!
    assert wrapped.action(4) == 4


def test_ground_counterparts():
    for act, expected in [(2, 1), (4, 3), (6, 0), (7, 0)]:
        assert GROUND_COUNTERPARTS[act] == expected
