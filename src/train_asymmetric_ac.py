"""Asymmetric actor-critic PPO: the policy sees pixels, the critic may see RAM.

The A/B the dual env exists for (Pinto et al. 2017, "Asymmetric Actor
Critic"): the policy gradient's variance depends on the value baseline,
and a pixel-critic must first learn to SEE before it can estimate
value -- exactly where the June pixel PPO drowned (120.8 px @ 100k).
A RAM-critic knows the value surface from step one; the advantage
estimates stay low-variance while the CNN policy is still learning
what a goomba looks like. The trained ARTIFACT is still a pure CV
policy: at deployment the value head is dead weight.

Both arms run the SAME code, reward and budget; --value-input {pixels,ram}
is the only difference, so the measured delta IS the asymmetry's value.

Usage:
    python src/train_asymmetric_ac.py --value-input ram  --total-timesteps 5000   # smoke
    python src/train_asymmetric_ac.py --value-input ram  --total-timesteps 200000 --n-envs 12
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

ROM = os.path.join(os.path.dirname(HERE), "data",
                   "0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds")
STATE = os.path.join(os.path.dirname(HERE), "data",
                     "0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).ds1")

N_STACK = 1                     # 1 grayscale frame per step, June-baseline parity


def make_env(rank=0, state_path=STATE):
    from dual_obs_env import DualObsEnv
    env = DualObsEnv(rom_path=ROM, state_path=state_path, max_steps=1000)
    return env


class NatureCNN(nn.Module):
    """The standard small game CNN (SB3's NatureCNN shape)."""

    def __init__(self, w=84, h=84, ch=1, feat=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            out = self.net(torch.zeros(1, ch, h, w)).shape[1]
        self.fc = nn.Sequential(nn.Linear(out, feat), nn.ReLU())

    def forward(self, x):
        # accept both (B,84,84,1) and SB3's pre-transposed (B,1,84,84)
        if x.dim() == 4 and x.shape[1] != 1 and x.shape[-1] == 1:
            x = x.permute(0, 3, 1, 2)
        x = x.float() / 255.0
        return self.fc(self.net(x))


class RamMLP(nn.Module):
    def __init__(self, dim, feat=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, feat), nn.ReLU(),
                                 nn.Linear(feat, feat), nn.ReLU())

    def forward(self, x):
        return self.net(x)


class AsymmetricPolicy:
    """The value-input choice, wired by the trainer into the PPO policy."""

    def __init__(self, value_input="ram"):
        self.value_input = value_input


# -- we build the PPO loop on SB3 with a custom ActorCriticPolicy ---------
def build_policy_class(value_input):
    from stable_baselines3.common.policies import ActorCriticPolicy

    class DualActorCriticPolicy(ActorCriticPolicy):
        """Policy head: NatureCNN(pixels). Value head: pixels or ram."""

        def _build(self, lr_schedule):
            self.pixel_cnn = NatureCNN()
            self.ram_mlp = None
            ram_dim = self.observation_space["ram"].shape[0]
            if value_input == "ram":
                self.ram_mlp = RamMLP(ram_dim)
            self.action_net = nn.Linear(512, self.action_space.n)
            self.value_net = nn.Linear(512 if value_input == "pixels" else 256, 1)
            self.vfeat_cache = None
            # register modules so .to(device)/save/load work
            self.add_module("pixel_cnn", self.pixel_cnn)
            if self.ram_mlp is not None:
                self.add_module("ram_mlp", self.ram_mlp)
            self.add_module("action_net", self.action_net)
            self.add_module("value_net", self.value_net)
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_schedule(1),
                                              eps=1e-5)

        def extract_features(self, obs):
            return obs  # keep the dict; heads pick their input

        def forward(self, obs, deterministic=False):
            feats = self.pixel_cnn(obs["pixels"])
            logits = self.action_net(feats)
            vfeats = self.ram_mlp(obs["ram"]) if self.ram_mlp is not None else feats
            values = self.value_net(vfeats)
            dist = self.action_dist.proba_distribution(action_logits=logits)
            actions = dist.get_actions(deterministic)
            log_prob = dist.log_prob(actions)
            return actions, values, log_prob

        def _predict(self, obs, deterministic=False):
            feats = self.pixel_cnn(obs["pixels"])
            logits = self.action_net(feats)
            dist = self.action_dist.proba_distribution(action_logits=logits)
            return dist.get_actions(deterministic)

        def evaluate_actions(self, obs, actions):
            feats = self.pixel_cnn(obs["pixels"])
            logits = self.action_net(feats)
            vfeats = self.ram_mlp(obs["ram"]) if self.ram_mlp is not None else feats
            values = self.value_net(vfeats)
            dist = self.action_dist.proba_distribution(action_logits=logits)
            log_prob = dist.log_prob(actions)
            return values, log_prob, dist.entropy()

        def predict_values(self, obs):
            # SB3 bootstraps the last rollout value through this hook
            vfeats = self.ram_mlp(obs["ram"]) if self.ram_mlp is not None \
                else self.pixel_cnn(obs["pixels"])
            return self.value_net(vfeats)

    return DualActorCriticPolicy


def main():
    import stable_baselines3 as sb3
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import SubprocVecEnv

    ap = argparse.ArgumentParser()
    ap.add_argument("--value-input", choices=("ram", "pixels"), default="ram",
                    help="what the CRITIC sees; the policy always sees pixels")
    ap.add_argument("--total-timesteps", type=int, default=200_000)
    ap.add_argument("--n-envs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--rollout", type=int, default=128,
                    help="steps per env per rollout (n_steps)")
    ap.add_argument("--out", default="", help="model save path (.zip)")
    args = ap.parse_args()

    set_random_seed(args.seed)
    env_fns = [lambda rank=i: Monitor(make_env(rank)) for i in range(args.n_envs)]
    venv = SubprocVecEnv(env_fns, start_method="spawn")

    policy_cls = build_policy_class(args.value_input)
    model = PPO(
        policy_cls,
        venv,
        n_steps=args.rollout,
        batch_size=args.n_envs * args.rollout // 4,
        n_epochs=4,
        learning_rate=2.5e-4,
        ent_coef=0.01,
        seed=args.seed,
        verbose=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    t0 = time.time()
    model.learn(total_timesteps=args.total_timesteps)
    dt = time.time() - t0
    sps = args.total_timesteps / dt if dt else 0.0
    print(f"\n[value-input={args.value_input}] {args.total_timesteps} steps in "
          f"{dt/60:.1f} min ({sps:.0f} steps/s)")

    if args.out:
        model.save(args.out)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
