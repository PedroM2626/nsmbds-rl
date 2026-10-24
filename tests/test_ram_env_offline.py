"""The no-emulator stub of MarioRamEnv (the 4af fix).

Before 2026-10-23 the documented offline fallback crashed with
AttributeError: NoneType has no attribute "poll". These tests pin the
fixed behaviour: reset/step work, shapes match the declared space, and
the info dict carries the termination flags the eval harness reads.
"""
import numpy as np
import pytest

from ram_env import MarioRamEnv, OBS_DIM, GEO_DIM


@pytest.fixture(scope="module")
def stub():
    env = MarioRamEnv(rom_path="missing.nds", state_path="missing.ds1")
    assert env.has_emulator is False
    return env


def test_reset_returns_declared_shape(stub):
    obs, info = stub.reset()
    assert obs.shape == stub.observation_space.shape
    assert obs.dtype == np.float32
    assert np.all(obs == 0)


def test_step_matches_space_and_flags(stub):
    obs, reward, done, trunc, info = stub.step(3)
    assert obs.shape == stub.observation_space.shape
    assert reward == -0.05  # the flat per-step penalty
    assert done is False and trunc is False
    assert info["died"] is False and info["timeout"] is False
    assert info["level_cleared"] is False


def test_truncation_at_max_steps():
    env = MarioRamEnv(rom_path="missing.nds", state_path="missing.ds1",
                      max_steps=3)
    env.reset()
    trunc_flags = [env.step(0)[3] for _ in range(4)]
    assert trunc_flags == [False, False, True, True]


def test_geo_failure_shrinks_the_space_not_the_obs():
    # a course the parser refuses must drop the 6 pit dims BEFORE the
    # space is built -- the old code advertised 23 and returned 17 (a
    # silent SB3 check failure later). With the course load failed the
    # space IS 17, and the observation matches it.
    env = MarioRamEnv(rom_path="missing.nds", state_path="missing.ds1",
                      geo=True, course_name="course/A01_1.bin")
    assert env.geo is False
    assert env.observation_space.shape[0] == OBS_DIM  # 17, not 23
    obs, _ = env.reset()
    assert obs.shape == env.observation_space.shape


def test_all_eight_actions_are_stepable(stub):
    stub.reset()
    for a in range(8):
        obs, r, d, t, info = stub.step(a)  # must not raise
        assert obs.shape == stub.observation_space.shape