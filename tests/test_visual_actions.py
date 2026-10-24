"""The corrected visual action tables (src/env.py) -- the 4af B/X fix.

The measurement: A and B produce the identical standing jump (peak
67.40625 px); RIGHT+B advances exactly like RIGHT (no dash); X/Y carry the
dash. So "legacy" (the June 6-action table) has no dash and two identical
jump actions, and "v2" mirrors MarioRamEnv exactly.
"""
import pytest

from env import ACTION_KEYS_V2, ACTION_KEYS_LEGACY, Keys


def test_table_sizes():
    assert len(ACTION_KEYS_V2) == 8
    assert len(ACTION_KEYS_LEGACY) == 6


def test_v2_dash_actions_use_x_not_b():
    for a in (3, 4, 6):  # dash, dash+jump, in-place dash+jump
        assert Keys.KEY_X in ACTION_KEYS_V2[a]
        assert Keys.KEY_B not in ACTION_KEYS_V2[a]


def test_v2_jump_actions_use_a():
    for a in (2, 4, 6, 7):
        assert Keys.KEY_A in ACTION_KEYS_V2[a]


def test_noop_and_left_are_untouched():
    assert ACTION_KEYS_V2[0] == []
    assert ACTION_KEYS_LEGACY[0] == []
    assert ACTION_KEYS_V2[5] == [Keys.KEY_LEFT]
    assert ACTION_KEYS_LEGACY[4] == [Keys.KEY_LEFT]


def test_legacy_preserves_the_june_table_verbatim():
    # the committed checkpoints were trained against exactly this table;
    # changing it would silently invalidate every n30 visual row
    assert ACTION_KEYS_LEGACY[2] == [Keys.KEY_RIGHT, Keys.KEY_B]
    assert ACTION_KEYS_LEGACY[3] == [Keys.KEY_RIGHT, Keys.KEY_B, Keys.KEY_A]
    flat = [k for keys in ACTION_KEYS_LEGACY.values() for k in keys]
    assert Keys.KEY_X not in flat  # the legacy arm had no dash at all


def test_v2_mirrors_ram_env():
    pytest.importorskip("desmume.controls")  # ram_env needs the real Keys
    from ram_env import MarioRamEnv
    env = MarioRamEnv(rom_path="missing.nds", state_path="missing.ds1")
    for a in range(8):
        assert set(ACTION_KEYS_V2[a]) == set(env._action_keys(a))