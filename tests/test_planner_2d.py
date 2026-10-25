"""Unit tests for the Full 2D Spatio-Temporal A* Planner (Task 3)."""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from tilemap_planner import TilemapAStar, sample_ballistic_arc_px  # noqa: E402
from rust_planner import RustPlanner  # noqa: E402


def test_ballistic_arc_regimes():
    arc_hop = sample_ballistic_arc_px(100.0, 30.0, 1.8, "tap_hop", n_steps=5)
    arc_leap = sample_ballistic_arc_px(100.0, 30.0, 2.136, "run_leap", n_steps=5)
    assert len(arc_hop) == 5
    assert len(arc_leap) == 5
    # Running leap reaches higher (smaller row) and further in X than tap_hop
    min_row_hop = min(r for _, r, _ in arc_hop)
    min_row_leap = min(r for _, r, _ in arc_leap)
    assert min_row_leap < min_row_hop
    assert arc_leap[-1][0] > arc_hop[-1][0]


def test_2d_planner_on_1_1_and_8_1():
    p11 = TilemapAStar(name="course/A01_1.bin")
    assert len(p11.path_2d) > 200
    assert len(p11.cells_2d) > 400
    # Check ceiling_ahead parity with RustPlanner on A01_1
    rp = RustPlanner()
    try:
        for px in (100.0, 350.0, 1280.0, 1500.0, 2500.0):
            assert p11.ceiling_ahead(px) == rp.ceiling_ahead(px)
    finally:
        rp.close()

    p81 = TilemapAStar(name="course/H01_1.bin")
    assert len(p81.path_2d) > 180
    # In H01_1, cols 66..73 and 111..114 are volcano slopes (obj 19..23), not true vertical walls
    assert p81.obstacle_h.get(66, 0) >= 2
    assert p81.true_wall_h.get(66, 0) == 0
    # Stepping stones over wide pits in H01_1
    assert 26 in p81.stepping_platforms
    # Step-up detection at col 22 -> 23 (352..368 px)
    dist, rise = p81.step_up_ahead(356.0, look_tiles=2)
    assert dist == 1 and rise == 1
    # Spatio-temporal plan_2d at pit edge vs flat ground with approaching enemy
    plan_pit = p81.plan_2d(404.0, mario_row=45.0, vx_pf=2.1)
    assert plan_pit["best_regime"] in ("run_leap", "walk_jump", "tap_hop")
    assert plan_pit["recommended_action"] in (3, 4)
    plan_enemy = p81.plan_2d(
        530.0,
        mario_row=46.0,
        vx_pf=2.0,
        enemies=[{"dx": 28.0, "dy": 0.0, "vx": -0.8, "vy": 0.0}],
    )
    assert plan_enemy["recommended_action"] in (3, 4, 6)
