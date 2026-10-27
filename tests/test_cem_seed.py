"""CEM's explicit seed must isolate it from global torch randomness."""
import numpy as np
import torch

from pinn_world_model import CEMMPController, OBS_DIM, shaped_step, ShapingConfig


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


def continuation(actor, dy, falling=True):
    state = torch.zeros((1, OBS_DIM))
    state[0, 3] = -0.5 if falling else 0
    state[0, 9] = dy / 256
    state[0, 10] = actor / 256
    return shaped_step(ToyModel(), state, torch.tensor([0]), ShapingConfig())[2].item()


def test_falling_contact_only_stomps_enemy_below_mario():
    assert continuation(160, -10) > 0.99
    assert continuation(160, 10) == 0


def test_unstompable_pokey_never_gets_the_stomp_exemption():
    assert continuation(232, -10) == 0


def test_ridden_rail_platform_is_not_a_lethal_enemy():
    assert continuation(221, -8, falling=False) > 0.99
