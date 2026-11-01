"""Shared pieces of the v2 re-benchmark (PROJECT_STATUS 4az).

The historical n30 table (evals/benchmark_n30_merged.json) mixed two games:
the visual rows used the legacy 6-action table (no dash, B = a second jump)
under the flow reward, the RAM rows used 8 actions under the unified RAM
reward. This module fixes ONE ruler for every arm:

  * actions  : the v2 table of MarioRamEnv (8 actions, X dash, A jump)
  * reward   : the simple potential reward of 4ao-4at,
               r_t = max(0, x_t - max_x_so_far)/16 + 10*cleared - 1*died,
               via train_visual_simple.SimpleRewardPixelEnv (single source)
  * stall    : truncate when x moves < 2.5 px over 30 steps (train AND eval)
  * cap      : MarioRamEnv's 1000-step episode cap
  * pixels   : DualObsEnv's 84x84x1 grayscale top screen (the June pipeline)

Arms differ only in observation and architecture. Every arm is registered
in ARMS; the trainer and the evaluator both read it.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (HERE, os.path.join(ROOT, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN

import paths
from train_visual_simple import SimpleRewardPixelEnv

STALL_LIMIT = 30
MODEL_DIR = os.path.join("models", "rebench")
EVAL_DIR = os.path.join("evals", "rebench")
LOG_DIR = os.path.join("out", "rebench")

# obs: pixels | ram17 | ram23 | graph
# algo: rppo (sb3_contrib RecurrentPPO) | ppo (SB3 PPO)
# historical: the row of the old table this arm re-measures (None = new)
ARMS = {
    # -- the June visual family: every one of these was RecurrentPPO CnnLstm
    #    (enable_critic_lstm, lstm 256) -- read from the checkpoints' policy_class
    "rppo_pure":        dict(obs="pixels", algo="rppo", historical="PPO Pure / Recurrent PPO"),
    "rppo_ae":          dict(obs="pixels", algo="rppo", extractor="ae_frozen",
                             historical="PPO + Autoencoder"),
    "rppo_curl":        dict(obs="pixels", algo="rppo", curl=True, historical="PPO CURL"),
    "rppo_hybrid":      dict(obs="pixels", algo="rppo", extractor="ae_unfrozen", curl=True,
                             historical="PPO Hybrid AE+CURL"),
    "rppo_spr":         dict(obs="pixels", algo="rppo", spr=True, historical="PPO SPR"),
    "rppo_drq":         dict(obs="pixels", algo="rppo", extractor="drq", historical="PPO DrQ-v2"),
    "rppo_icm":         dict(obs="pixels", algo="rppo", icm=True, historical="PPO + ICM"),
    "rppo_icm_ae":      dict(obs="pixels", algo="rppo", extractor="ae_frozen", icm=True,
                             historical="PPO + ICM + Autoencoder"),
    "rppo_impala_curl": dict(obs="pixels", algo="rppo", extractor="impala", curl=True,
                             historical="IMPALA CURL Recurrent (1M)"),
    "rtrf_impala_spr":  dict(obs="pixels", algo="rppo", policy="transformer", extractor="impala",
                             spr=True, historical="IMPALA Transformer SPR (1M)"),
    # -- feedforward visual PPO
    "ppo_impala":       dict(obs="pixels", algo="ppo", extractor="impala",
                             historical="ImpalaCNN PPO (1M)"),
    "ppo_naturecnn":    dict(obs="pixels", algo="ppo", historical=None),
    # -- vision-free arms
    "ppo_ram17":        dict(obs="ram17", algo="ppo", historical="PPO RAM-only"),
    "ppo_ram23":        dict(obs="ram23", algo="ppo", historical="PPO RAM + Geometry"),
    "ppo_gnn":          dict(obs="graph", algo="ppo", extractor="gnn",
                             historical="PPO GNN Relational"),
}


# ---------------------------------------------------------------- envs
class PixelsOnly(gym.ObservationWrapper):
    """DualObsEnv -> the (84,84,1) pixel view alone (zero RAM in the obs)."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space["pixels"]

    def observation(self, obs):
        return obs["pixels"]


def _base_env(obs_kind, rom, state):
    if obs_kind == "pixels":
        from dual_obs_env import DualObsEnv
        return PixelsOnly(DualObsEnv(rom_path=rom, state_path=state, pixel_size=84))
    if obs_kind == "ram17":
        from ram_env import MarioRamEnv
        return MarioRamEnv(rom_path=rom, state_path=state, geo=False)
    if obs_kind == "ram23":
        from ram_env import MarioRamEnv
        return MarioRamEnv(rom_path=rom, state_path=state, geo=True)
    if obs_kind == "graph":
        from graph_env import MarioGraphEnv
        return MarioGraphEnv(rom_path=rom, state_path=state)
    raise ValueError(obs_kind)


def make_env(obs_kind, rom=paths.ROM_REL, state=paths.STATE_REL):
    """One potential-reward env in THIS process (one DeSmuME per process)."""
    return SimpleRewardPixelEnv(_base_env(obs_kind, rom, state), stall_limit=STALL_LIMIT)


class _EnvFactory:
    """Picklable SubprocVecEnv thunk: private ROM/state copies per worker
    (the June trainers' Windows file-sharing workaround) + staggered init."""

    def __init__(self, obs_kind, rank):
        self.obs_kind, self.rank = obs_kind, rank

    def __call__(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        torch.set_num_threads(1)
        time.sleep(self.rank * 1.5)
        rom, state = paths.resolve(paths.ROM_REL), paths.resolve(paths.STATE_REL)
        tmp = tempfile.gettempdir()
        tag = f"rebench_{self.obs_kind}_{self.rank}_{os.getpid()}"
        t_rom = os.path.join(tmp, tag + os.path.splitext(rom)[1])
        t_st = os.path.join(tmp, tag + os.path.splitext(state)[1])
        shutil.copy2(rom, t_rom)
        shutil.copy2(state, t_st)
        env = make_env(self.obs_kind, t_rom, t_st)
        inner_close = env.close

        def _close():
            try:
                inner_close()
            finally:
                for f in (t_rom, t_st):
                    try:
                        os.remove(f)
                    except OSError:
                        pass
        env.close = _close
        return env


def make_vec_env(obs_kind, n_envs):
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    fns = [_EnvFactory(obs_kind, i) for i in range(n_envs)]
    if n_envs == 1:
        return DummyVecEnv(fns)
    return SubprocVecEnv(fns, start_method="spawn")


# ---------------------------------------------------------------- extractors
class AEExtractor(BaseFeaturesExtractor):
    """The June autoencoder encoder (models/autoencoder.pth, NatureCNN-shaped,
    512 latents), frozen or fine-tuned. Module-level so checkpoints reload."""

    def __init__(self, observation_space, features_dim=512, freeze=True,
                 model_path="models/autoencoder.pth"):
        super().__init__(observation_space, features_dim)
        c = observation_space.shape[0]
        self.encoder = nn.Sequential(
            nn.Conv2d(c, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(3136, features_dim), nn.ReLU())
        path = paths.resolve(model_path) if not os.path.isabs(model_path) else model_path
        if os.path.exists(path):
            sd = torch.load(path, map_location="cpu")
            self.encoder.load_state_dict({k[len("encoder."):]: v for k, v in sd.items()
                                          if k.startswith("encoder.")})
        else:
            raise FileNotFoundError(f"autoencoder weights missing: {path}")
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, obs):
        return self.encoder(obs)


class DrQNatureCNN(BaseFeaturesExtractor):
    """NatureCNN with DrQ random shifts (pad 4, replicate) in train mode only;
    identical to the plain CNN at evaluation (module.eval())."""

    def __init__(self, observation_space, features_dim=512, pad=4):
        super().__init__(observation_space, features_dim)
        self.cnn = NatureCNN(observation_space, features_dim=features_dim)
        self.pad = pad

    def forward(self, obs):
        if self.training:
            from curl import random_crop
            obs = random_crop(obs, out_size=obs.shape[-1], padding=self.pad)
        return self.cnn(obs)


def policy_spec(arm):
    """(policy class or name, policy_kwargs) for an arm."""
    cfg = ARMS[arm]
    kw = {}
    ext = cfg.get("extractor")
    if ext == "ae_frozen":
        kw.update(features_extractor_class=AEExtractor,
                  features_extractor_kwargs=dict(features_dim=512, freeze=True))
    elif ext == "ae_unfrozen":
        kw.update(features_extractor_class=AEExtractor,
                  features_extractor_kwargs=dict(features_dim=512, freeze=False))
    elif ext == "drq":
        kw.update(features_extractor_class=DrQNatureCNN,
                  features_extractor_kwargs=dict(features_dim=512))
    elif ext == "impala":
        from impala_cnn import ImpalaCNNFeaturesExtractor
        kw.update(features_extractor_class=ImpalaCNNFeaturesExtractor,
                  features_extractor_kwargs=dict(features_dim=512))
    elif ext == "gnn":
        from gnn_extractor import GNNExtractor
        kw.update(features_extractor_class=GNNExtractor,
                  features_extractor_kwargs=dict(features_dim=256))
    if cfg["algo"] == "rppo":
        kw.update(enable_critic_lstm=True, lstm_hidden_size=256)
        if cfg.get("policy") == "transformer":
            from transformer_policy import RecurrentTransformerActorCriticPolicy
            kw.update(context_len=16, nhead=8)
            return RecurrentTransformerActorCriticPolicy, kw
        return "CnnLstmPolicy", kw
    if cfg["obs"] == "pixels":
        return "CnnPolicy", kw
    return "MlpPolicy", kw


def algo_class(arm):
    if ARMS[arm]["algo"] == "rppo":
        from sb3_contrib import RecurrentPPO
        return RecurrentPPO
    from stable_baselines3 import PPO
    return PPO


# ---------------------------------------------------------------- logging
class ProgressLogger(BaseCallback):
    """Per-episode training record (max x, return, clear) + an atomic
    checkpoint at every new max-x record (the 4ar convention)."""

    def __init__(self, arm, out_log, best_path, flush_every=10):
        super().__init__(0)
        self.arm, self.out_log, self.best_path = arm, out_log, best_path
        self.flush_every = flush_every
        self.best_max_x, self.history, self.n_ep = 0.0, [], 0
        self.first_clear_step = None
        self.t0 = time.time()

    def _on_step(self):
        for info, done in zip(self.locals.get("infos", []), self.locals.get("dones", [])):
            if not done:
                continue
            self.n_ep += 1
            mx = float(info.get("max_x_so_far", 0.0))
            cleared = bool(info.get("level_cleared", False))
            if cleared and self.first_clear_step is None:
                self.first_clear_step = int(self.num_timesteps)
            new_best = mx > self.best_max_x
            if new_best:
                self.best_max_x = mx
                self.best_step = int(self.num_timesteps)
                self._save_best()
            self.history.append({
                "episode": self.n_ep, "total_steps": int(self.num_timesteps),
                "ep_steps": int(info.get("ep_steps", 0)), "max_x_px": mx,
                "ep_reward": float(info.get("ep_reward", 0.0)), "cleared": cleared,
                "elapsed_min": (time.time() - self.t0) / 60.0})
            print(f"[{self.arm}] ep {self.n_ep:4d} step {self.num_timesteps:6d} "
                  f"max_x {mx:7.1f} best {self.best_max_x:7.1f} R {info.get('ep_reward', 0.0):6.2f}"
                  f"{' CLEAR' if cleared else ''}{' *' if new_best else ''}", flush=True)
            if self.n_ep % self.flush_every == 0:
                self.dump()
        return True

    def _save_best(self):
        """Write-then-rename with retries: on Windows a just-written zip can be
        briefly locked (scanner), and a failed save must not kill the run."""
        tmp = self.best_path + "_tmp"
        for attempt in range(5):
            try:
                self.model.save(tmp)
                os.replace(tmp + ".zip", self.best_path + ".zip")
                return
            except OSError as e:
                print(f"[{self.arm}] best-checkpoint save retry {attempt + 1}: {e}", flush=True)
                time.sleep(1.0)

    def dump(self, extra=None):
        d = {"arm": self.arm, "total_steps": int(self.num_timesteps), "episodes": self.n_ep,
             "best_max_x_px": self.best_max_x, "best_step": getattr(self, "best_step", None),
             "first_clear_step": self.first_clear_step,
             "elapsed_min": (time.time() - self.t0) / 60.0}
        if extra:
            d.update(extra)
        d["history"] = self.history
        os.makedirs(os.path.dirname(self.out_log) or ".", exist_ok=True)
        tmp = self.out_log + ".tmp"
        with open(tmp, "w") as f:
            json.dump(d, f, indent=1)
        os.replace(tmp, self.out_log)
