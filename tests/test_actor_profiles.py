"""Unit tests for complete actor profile decoding (385/385 classes)."""
import csv
import json
import os
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_PATH = os.path.join(ROOT, "out", "dataset", "object_ids.json")
RAW_JSON_PATH = os.path.join(ROOT, "out", "nsmb_object_ids.json")
ENTITIES_PATH = os.path.join(ROOT, "out", "dataset", "entities.csv")


def test_actor_profile_coverage_100_percent():
    assert os.path.exists(JSON_PATH), f"Missing {JSON_PATH}"
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    id_to_classes = data.get("id_to_classes", {})
    # Total valid actor slots in NSMB DS is 0..384 (385 slots)
    assert len(id_to_classes) == 385, f"Expected 385 profiles, got {len(id_to_classes)}"
    assert data.get("max_id") == 384, f"Expected max_id 384, got {data.get('max_id')}"

    # Verify no gap in 0..384
    missing = [i for i in range(385) if str(i) not in id_to_classes]
    assert len(missing) == 0, f"Gaps detected in profile IDs: {missing}"


def test_all_placed_entities_resolved():
    assert os.path.exists(ENTITIES_PATH), f"Missing {ENTITIES_PATH}"
    with open(ENTITIES_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        unresolved = [row for row in reader if not row["class"]]

    assert len(unresolved) == 0, f"Found {len(unresolved)} unresolved entities in entities.csv"


def test_profile_json_copies_are_in_sync():
    """course.py reads one copy; this suite reads the other, so both must match."""
    assert os.path.exists(RAW_JSON_PATH), f"Missing {RAW_JSON_PATH}"
    assert os.path.exists(JSON_PATH), f"Missing {JSON_PATH}"
    with open(RAW_JSON_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    assert raw == dataset, (
        "out/nsmb_object_ids.json and out/dataset/object_ids.json diverged; "
        "update both copies together"
    )
