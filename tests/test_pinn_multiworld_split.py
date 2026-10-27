"""Episode isolation and course coverage for the PINN training split."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from train_enhanced_pinn import eval_metrics, load_pool, split_multiworld, split_summary


def dataset(worlds=("1-1", "2-1"), episodes_per_world=6):
    ep = np.repeat(np.arange(len(worlds) * episodes_per_world), 2)
    world = np.repeat(np.asarray(worlds), episodes_per_world * 2)
    s = np.arange(len(ep))[:, None]
    return s, s[:, 0], s, s[:, 0], s[:, 0], ep, world


def test_each_course_in_each_split_and_no_episode_leakage():
    data = dataset()
    parts = split_multiworld(data, seed=19)
    repeated = split_multiworld(data, seed=19)
    assert all(np.array_equal(a[5], b[5]) for a, b in zip(parts, repeated))
    assert all(set(part[6]) == {"1-1", "2-1"} for part in parts)
    groups = [set(part[5]) for part in parts]
    assert all(groups[i].isdisjoint(groups[j]) for i in range(3) for j in range(i + 1, 3))
    assert set.union(*groups) == set(data[5])
    assert sum(len(part[0]) for part in parts) == len(data[0])


def test_rejects_too_few_episodes_or_inconsistent_world():
    with pytest.raises(ValueError, match="at least three"):
        split_multiworld(dataset(episodes_per_world=2))
    data = list(dataset())
    data[6] = data[6].copy()
    data[6][1] = "2-1"
    with pytest.raises(ValueError, match="span multiple courses"):
        split_multiworld(tuple(data))


def test_rejects_unlabelled_unverified_pool(tmp_path):
    data = dataset()
    path = tmp_path / "unknown.npz"
    np.savez(path, **dict(zip(("S", "A", "S2", "R", "D", "EP"), data)))
    with pytest.raises(ValueError, match="does not match"):
        load_pool(path)


def test_reads_explicit_labels_and_rejects_noncontiguous_episode(tmp_path):
    data = dataset()
    path = tmp_path / "labelled.npz"
    np.savez(path, **dict(zip(("S", "A", "S2", "R", "D", "EP", "WORLD"), data)))
    loaded = load_pool(path)
    assert np.array_equal(loaded[6], data[6])
    broken = list(loaded)
    broken[5] = broken[5].copy()
    broken[5][4] = broken[5][0]
    with pytest.raises(ValueError, match="non-contiguous"):
        split_multiworld(tuple(broken))


def test_committed_legacy_pool_has_all_courses_in_each_partition():
    path = Path(__file__).resolve().parents[1] / "models" / "pinn_pool.npz"
    data = load_pool(path)
    summary = split_summary(split_multiworld(data, seed=42))
    expected = {"1-1", "1-2", "2-1", "4-1", "5-1", "6-1", "8-1"}
    assert all(set(part["courses"]) == expected for part in summary.values())


def test_drift_reports_episodes_separately_from_windows():
    class IdentityModel(torch.nn.Module):
        def forward(self, s, action):
            return s, torch.zeros(len(s)), torch.zeros(len(s)), torch.zeros_like(s)

        def step_batch(self, s, action):
            return s, None, None, None

    state = np.zeros((24, 23), np.float32)
    episode = np.repeat([0, 1], 12)
    test = (state, np.zeros(24, np.int64), state, np.zeros(24),
            np.zeros(24), episode)
    result = eval_metrics(IdentityModel(), (None, None, test), "cpu")
    assert result["n_drift_episodes"] == 2
    assert result["n_drift_windows"] == 6
