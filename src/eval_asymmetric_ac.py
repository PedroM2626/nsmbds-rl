"""Evaluate the trained asymmetric-AC pair: mean px per episode, the same
ruler as the transfer tables. det + stoch, 3 eps each. ONE DeSmuME per
process (the eternal constraint), constructed BEFORE any torch load."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from dual_obs_env import DualObsEnv
from train_asymmetric_ac import build_policy_class
from stable_baselines3 import PPO

ROOT = os.path.dirname(HERE)
ROM = os.path.join(ROOT, "data", "0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds")
STATE = os.path.join(ROOT, "data", "0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).ds1")

# the ONE emulator of this process, before torch
env = DualObsEnv(rom_path=ROM, state_path=STATE, max_steps=1000)

for tag, value_input, path in (
        ("pixel-critic (baseline)", "pixels", "models/ac_pixels_200k.zip"),
        ("ram-critic (asymmetric)", "ram", "models/ac_ram_200k.zip")):
    model = PPO.load(os.path.join(ROOT, path),
                     policy=build_policy_class(value_input), device="cpu")
    for det in (True, False):
        pxs, lens = [], []
        for ep in range(3):
            obs, info = env.reset()
            best = 0.0
            steps = 0
            done = trunc = False
            while not (done or trunc):
                action, _ = model.predict(obs, deterministic=det)
                obs, r, done, trunc, info = env.step(int(action))
                steps += 1
                best = max(best, info.get("mario_px", 0.0))
            pxs.append(best)
            lens.append(steps)
        print(f"{tag} {'det' if det else 'stoch'}: "
              f"mean px {np.mean(pxs):7.1f}  max {np.max(pxs):7.1f}  "
              f"steps {lens}", flush=True)
env.close()
