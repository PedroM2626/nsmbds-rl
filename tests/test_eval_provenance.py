"""The provenance block (src/eval_provenance.py) -- the reproducibility fix.

Every new eval JSON must be able to say what produced it: git commit,
file hashes, versions, timestamp. These tests pin the mechanics without
touching the emulator.
"""
import os
import tempfile

import eval_provenance


def test_sha1_of_known_content():
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("mario")
        path = f.name
    try:
        # sha1("mario") -- stable, so a regression in the hashing shows
        assert eval_provenance.sha1_file(path) == \
            "addb47291ee169f330801ce73520b96f2eaf20ea"
    finally:
        os.remove(path)


def test_sha1_of_missing_file_is_none():
    assert eval_provenance.sha1_file("Z:/nope.bin") is None
    assert eval_provenance.sha1_file(None) is None


def test_git_state_shape():
    state = eval_provenance.git_state()
    assert set(state) == {"commit", "dirty"}
    assert isinstance(state["dirty"], bool) or state["dirty"] is None


def test_collect_block_is_json_serializable():
    import json
    block = eval_provenance.collect(rom=None, state=None)
    assert block["schema"] == 1
    assert "recorded_at_utc" in block and "packages" in block
    assert set(block["files"]) == {"rom", "state"}
    json.dumps(block)  # must round-trip


def test_missing_files_hash_to_none_but_do_not_raise():
    block = eval_provenance.collect(rom="missing.nds", state="missing.ds1")
    assert block["files"]["rom"]["sha1"] is None
    assert block["files"]["rom"]["bytes"] is None


def test_explicit_untracked_source_cannot_claim_clean_head(tmp_path):
    import subprocess
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    source = tmp_path / "new_evaluator.py"
    source.write_text("print('measurement')\n", encoding="utf-8")
    block = eval_provenance.collect(root=str(tmp_path), sources=[source])
    assert block["git"]["dirty"] is True
    assert block["sources"]["new_evaluator.py"] == {
        "sha1": eval_provenance.sha1_file(source), "tracked": False}
