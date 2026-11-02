"""No-emulator checks of the ROM-only savestate scripts (PROJECT_STATUS 4ba)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import cold_boot  # noqa: E402
import paths  # noqa: E402


def test_every_course_boots_mario_and_ends_with_a_save():
    for course, spec in cold_boot.COURSES.items():
        steps = spec["script"]
        assert steps[-1] == ("save",), course
        assert ("touch", 130, 55) in steps, course          # "Mario Game", not Mario Vs. Luigi
        counter = [s for s in steps if s[0] == "poke" and s[1] in cold_boot.FRAME_COUNTER]
        assert len(counter) == 3 and len({s[3] for s in counter}) == 1, course


def test_canonical_paths_map_back_to_their_course():
    assert cold_boot.state_course(paths.STATE_REL) == "1-1"
    assert cold_boot.state_course(paths.resolve("data/state_6-1.ds1")) == "6-1"
    assert cold_boot.state_course("scratch/elsewhere.ds1") is None


def test_entered_courses_hold_the_calibrated_area_index():
    from enter_course import course_area_index
    ids = {"1-2": "A02_1", "2-1": "B01_1", "3-1": "C01_1", "3-1_main": "C01_2", "4-1": "D01_1",
           "5-1": "E01_1", "6-1": "F01_1", "7-1": "G01_1", "8-1": "H01_1"}
    for course, cid in ids.items():
        hold = next(s for s in cold_boot.COURSES[course]["script"] if s[0] == "hold")
        assert (cold_boot.AREA_ID, 2, course_area_index(cid)) in hold[2], course


def test_ensure_state_leaves_existing_and_unknown_paths_alone(tmp_path):
    existing = tmp_path / "x.ds1"
    existing.write_bytes(b"")
    assert cold_boot.ensure_state(str(existing)) == str(existing)
    unknown = tmp_path / "missing.ds1"
    assert cold_boot.ensure_state(str(unknown)) == str(unknown)
    assert not unknown.exists()
