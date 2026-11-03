"""The windowed transformer pass equals the rollout pass (PROJECT_STATUS 4bb)."""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from transformer_policy import CausalTransformerWrapper  # noqa: E402


def _rollout_vs_sequence(train_window, steps=40):
    torch.manual_seed(0)
    m = CausalTransformerWrapper(input_size=12, hidden_size=16, context_len=4, nhead=2,
                                 train_window=train_window, dropout=0.0).eval()
    x = torch.randn(steps, 3, 12)
    h0 = torch.randn(4, 3, 16)                 # a stored history, as SB3 keeps it
    state = (h0, torch.zeros(1))
    one = []
    for t in range(steps):                     # collection: one step at a time
        o, state = m(x[t:t + 1], state)
        one.append(o)
    one = torch.cat(one)
    seq, last = m(x, (h0, torch.zeros(1)))     # PPO update: whole sequence at once
    return one, seq, state[0], last[0]


def test_windowed_sequence_pass_reproduces_rollout_outputs():
    one, seq, h_roll, h_seq = _rollout_vs_sequence(True)
    assert torch.allclose(one, seq, atol=1e-5)
    assert torch.allclose(h_roll, h_seq, atol=1e-6)


def test_legacy_sequence_pass_does_not():
    one, seq, _, _ = _rollout_vs_sequence(False)
    assert not torch.allclose(one, seq, atol=1e-3)
