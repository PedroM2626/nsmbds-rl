"""Unit tests for automated tileset solidity and non-flagpole exit inference."""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from course import Course  # noqa: E402


def test_tileset_solidity_dataset_exists_and_valid():
    path = os.path.join(ROOT, "out", "dataset", "tileset_solidity.json")
    assert os.path.exists(path), f"Missing {path}"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    summary = data["summary"]
    assert summary["total_courses"] == 191
    assert summary["inferred_plannable_courses"] > summary["static_plannable_courses"]
    assert summary["inferred_plannable_courses"] >= 120
    assert "A01_1" in data["courses"]
    assert "H01_1" in data["courses"]


def test_default_course_contract_unchanged():
    """Default Course(..., use_inferred=False) must keep the exact numbers checked by check_readme_numbers.py."""
    c_11 = Course("course/A01_1.bin")
    c_12 = Course("course/A02_1.bin")
    c_15 = Course("course/A05_1.bin")
    c_18 = Course("course/A08_1.bin")
    assert round(c_11.ground_coverage * 100, 1) == 92.1
    assert round(c_12.ground_coverage * 100, 1) == 85.7
    assert round(c_15.ground_coverage * 100, 1) == 11.0
    assert round(c_18.ground_coverage * 100, 1) == 3.7
    assert c_11.exit_found is True
    assert c_12.exit_found is False


def test_inferred_course_solidity_and_exits():
    """Course(..., use_inferred=True) exposes inferred coverage, slopes, 2D cells, and non-flagpole exits."""
    c_18 = Course("course/A08_1.bin", use_inferred=True)
    assert c_18.ground_coverage >= 0.60
    assert c_18.exit_found is True
    assert c_18.flag_x_px == 4272.0

    c_12 = Course("course/A02_1.bin", use_inferred=True)
    assert c_12.exit_found is True
    assert c_12.inferred_exit_kind.startswith("warp_entrance")
    assert c_12.flag_x_px == 4640.0

    c_81 = Course("course/H01_1.bin")
    assert {19, 21, 22, 23}.issubset(c_81.inferred_slope_objs)
    cells_2d = c_81.solid_cells_2d()
    assert len(cells_2d) > 300
