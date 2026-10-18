"""The dual-observation env: one emulator, two views per step.

obs = {"pixels": (84,84,1) uint8,   the DS top screen, grayscale -- the
                         SAME pipeline as MarioNdsEnv._get_obs (the June
                         visual arm), so pixel-PPO numbers stay comparable;
       "ram":     (23,) float32,     the unified 23-float scenario vector
                         MarioRamEnv builds (Mario kinematics, 3 enemies,
                         6 pit flags; geo=True)}

Reward: the unified RAM reward (4s) on both arms -- progress via
camera-RAM, death via RAM lives, clear via the flag. The A/B this env
exists for: the policy head reads ONLY "pixels" (a true CV policy), the
critic head reads "pixels" (baseline) or "ram" (asymmetric) -- same
code, same budget, the value input is the only difference.

One emulator per process (the two-init constraint); parallel envs go
through SubprocVecEnv like the June runs.
"""
from __future__ import annotations

import cv2
import numpy as np
from gymnasium import spaces

from ram_env import MarioRamEnv


class DualObsEnv(MarioRamEnv):
    """MarioRamEnv (reward + ram obs + course geometry) + the pixel view."""

    def __init__(self, *a, pixel_size=84, **k):
        super().__init__(*a, geo=True, **k)
        self.pixel_size = pixel_size
        dim = self.observation_space.shape[0]           # 23 with geo
        self.observation_space = spaces.Dict({
            "pixels": spaces.Box(low=0, high=255, shape=(pixel_size, pixel_size, 1),
                                 dtype=np.uint8),
            "ram": spaces.Box(low=-10.0, high=10.0, shape=(dim,),
                             dtype=np.float32),
        })

    # -- pixel pipeline (mirrors MarioNdsEnv._get_obs) ---------------------
    def _pixels(self):
        if not self.has_emulator:
            return np.zeros((self.pixel_size, self.pixel_size, 1), dtype=np.uint8)
        frame = np.array(self.emu.display_buffer_as_rgbx(), dtype=np.uint8)
        frame = frame.reshape((384, 256, 4))[:192, :, :3]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (self.pixel_size, self.pixel_size),
                             interpolation=cv2.INTER_AREA)
        return resized[:, :, None]

    def _wrap(self, obs_ram):
        return {"pixels": self._pixels(), "ram": obs_ram.astype(np.float32)}

    # -- gym api ----------------------------------------------------------
    def reset(self, seed=None, options=None):
        obs_ram, info = super().reset(seed=seed, options=options)
        return self._wrap(obs_ram), info

    def step(self, action):
        obs_ram, r, done, trunc, info = super().step(action)
        return self._wrap(obs_ram), r, done, trunc, info
