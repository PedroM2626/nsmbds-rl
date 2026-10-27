"""CEM's explicit seed must isolate it from global torch randomness."""
import numpy as np
import torch

from pinn_world_model import CEMMPController, OBS_DIM


class ToyModel(torch.nn.Module):
    def forward(self, state, action):
        reward = action[:, 3] + 0.3 * action[:, 4]
        return state, reward, torch.full_like(reward, 8), torch.zeros_like(state)


def test_controller_seed_ignores_unrelated_torch_draws():
    obs = np.zeros(OBS_DIM, np.float32)
    model = ToyModel()
    first = CEMMPController(model, horizon=3, n_candidates=8, n_elite=2,
                            n_iters=2, seed=123)
    action, probs = first.plan(obs)
    torch.rand(1000)
    second = CEMMPController(model, horizon=3, n_candidates=8, n_elite=2,
                             n_iters=2, seed=123)
    action2, probs2 = second.plan(obs)
    assert action == action2
    assert torch.equal(probs, probs2)
