"""Single writer for the actor-profile mapping and its generated dataset copy."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANONICAL = os.path.join(ROOT, "out", "nsmb_object_ids.json")
DATASET_COPY = os.path.join(ROOT, "out", "dataset", "object_ids.json")


def load_profiles(path=CANONICAL):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_profiles(data):
    text = json.dumps(data, indent=1, ensure_ascii=False)
    os.makedirs(os.path.dirname(DATASET_COPY), exist_ok=True)
    for path in (CANONICAL, DATASET_COPY):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)


def sync_dataset_copy():
    os.makedirs(os.path.dirname(DATASET_COPY), exist_ok=True)
    with open(CANONICAL, "rb") as src, open(DATASET_COPY, "wb") as dst:
        dst.write(src.read())
