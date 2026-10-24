"""The area-index oracle (src/course_id.py) against the committed manifest.

These are the exact indices measured from every committed savestate on
2026-10-23 (PROJECT_STATUS 4af): the strongest identity check this repo has,
and it needs no ROM -- the manifest is committed.
"""
import pytest

import course_id


@pytest.mark.parametrize("course, index", [
    ("A01_1", 0),
    ("A02_1", 3),
    ("B01_1", 21),
    ("C01_1", 44),
    ("D01_1", 66),
    ("E01_1", 88),
    ("F01_1", 107),
    ("G01_1", 130),
    ("H01_1", 152),
])
def test_measured_area_indices(course, index):
    assert course_id.course_area_index(course) == index


def test_path_form_matches_id_form():
    assert course_id.course_area_index("course/A02_1.bin") == \
        course_id.course_area_index("A02_1")


def test_reverse_mapping():
    assert course_id.course_for_area_index(3) == "A02_1"
    assert course_id.course_for_area_index(0) == "A01_1"
    assert course_id.course_for_area_index(191) is None
    assert course_id.course_for_area_index(-1) is None


def test_the_191_courses_parse():
    ids = course_id.course_ids()
    assert len(ids) == 191
    assert len(set(ids)) == 191  # indices are unique identities


def test_unknown_course_raises():
    with pytest.raises(KeyError):
        course_id.course_area_index("ZZ99_9")