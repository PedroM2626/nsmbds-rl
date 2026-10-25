"""Unit tests for World 8-1 (H01_1) Kinematic Actuation & Clearance."""
import json
import os
import pytest
import numpy as np

from world_8_1_controller import setup_world_8_1_planner_and_profile, World81Agent

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROM = os.path.join(ROOT, "data", "0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds")


def test_world_8_1_planner_sprites_and_takeoffs():
    if not os.path.exists(ROM):
        pytest.skip("ROM file not present")
    planner, profile = setup_world_8_1_planner_and_profile(ROM, "course/H01_1.bin")

    # Verify Bill Blaster sprites (type 27) are mapped into surface
    blaster_cols = [s["tx"] for s in planner.course.sprites if s["type"] == 27]
    assert len(blaster_cols) > 0
    for col in blaster_cols:
        assert col in planner.surface
        assert col in planner.solid
        assert col in profile.persistent_rise_cols

    # Verify pits and takeoffs exist
    assert len(planner.pits) >= 3
    assert len(planner.takeoffs) >= 3


def test_world_8_1_agent_offline_action_decisions():
    if not os.path.exists(ROM):
        pytest.skip("ROM file not present")
    agent = World81Agent(rom_path=ROM, course_name="course/H01_1.bin")

    # Synthetic obs vector (dim 23):
    # obs[0]: rel_x, obs[1]: rel_y, obs[3]: vy, obs[4]: ground (1/0), obs[17:23]: pit distances
    obs = np.zeros(23, dtype=np.float32)
    obs[4] = 1.0  # on ground

    # 1. Immediate pit -> commit running leap (Action 4)
    obs[17] = 0.0  # immediate pit ahead
    assert agent.act(obs) == 4
    obs[17] = 1.0

    # 2. Bill Blaster early leap (col 173 Bullet Bill, px 2665)
    # spawn_px is 56.0. px = 56 + rel_x * 512 = 2670 => rel_x = (2670 - 56) / 512
    obs[0] = (2670.0 - 56.0) / 512.0
    assert agent.act(obs) == 4

    # 3. Floating over Bill Blaster 2 (px 2800)
    obs[0] = (2800.0 - 56.0) / 512.0
    assert agent.act(obs) == 4

    # 4. Staircase running leap (px 3205)
    obs[0] = (3205.0 - 56.0) / 512.0
    assert agent.act(obs) == 4

    # 5. Flagpole final sprint (px 3850)
    obs[0] = (3850.0 - 56.0) / 512.0
    assert agent.act(obs) == 3


def test_eval_8_1_clear_artifact():
    path = os.path.join(ROOT, "evals", "eval_8_1_clear.json")
    assert os.path.exists(path), "evals/eval_8_1_clear.json must exist"
    data = json.load(open(path, encoding="utf-8"))

    agg = data["aggregate"]
    assert agg["clear_rate"] == 1.0
    assert agg["episodes"] == 5
    assert agg["course"] == "course/H01_1.bin"
    assert agg["mean_progress_px"] >= 3900.0

    assert "provenance" in data
    prov = data["provenance"]
    assert prov["schema"] == 1
    assert "git" in prov and "commit" in prov["git"]
    assert prov["files"]["rom"]["sha1"] is not None

    episodes = data["episodes"]
    assert len(episodes) == 5
    for ep in episodes:
        assert ep["reached_goal"] is True
        assert ep["termination"] == "goal"
        assert ep["steps"] == 205
