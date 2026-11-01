"""No-emulator checks of the v2 re-benchmark plumbing (PROJECT_STATUS 4az)."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import rebench_common as rc  # noqa: E402
import eval_rebench as ev  # noqa: E402


def test_every_arm_resolves_a_policy():
    for arm, cfg in rc.ARMS.items():
        policy, kw = rc.policy_spec(arm)
        assert policy is not None, arm
        if cfg["algo"] == "rppo":
            assert kw["lstm_hidden_size"] == 256 and kw["enable_critic_lstm"], arm


def test_spr_transition_covers_the_v2_table():
    import torch
    from spr import SPR
    enc = torch.nn.Identity()
    spr = SPR(16, enc, enc, num_actions=8)
    out = spr.transition_model(torch.zeros(2, 16), torch.tensor([6, 7]))
    assert out.shape == (2, 16)


def test_drq_extractor_is_plain_cnn_in_eval_mode():
    import torch
    from gymnasium import spaces
    ext = rc.DrQNatureCNN(spaces.Box(0, 255, (1, 84, 84), np.uint8))
    x = torch.rand(2, 1, 84, 84)
    ext.eval()
    assert torch.equal(ext(x), ext.cnn(x))


def test_readme_rebench_table_matches_the_eval_jsons():
    import rebench_table as rt
    table, _ = rt.build()
    assert rt.readme_block() == table, "regenerate with tools/rebench_table.py"


def test_summarize_uses_sample_sd_and_t_interval():
    eps = [{"max_x_px": float(v), "cleared": v > 25, "return": 0.0, "steps": 1,
            "termination": "death"} for v in range(30)]
    s = ev.summarize(eps)
    xs = np.arange(30, dtype=float)
    assert s["sd_max_x_px"] == pytest.approx(xs.std(ddof=1))
    assert s["ci95_halfwidth_px"] == pytest.approx(2.045 * xs.std(ddof=1) / np.sqrt(30))
    assert s["clears"] == 4 and s["terminations"] == {"death": 30}
