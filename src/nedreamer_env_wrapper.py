import gymnasium as gym
import numpy as np
import os

class MarioNDSNEDreamerWrapper(gym.Wrapper):
    def __init__(self, env, episode_log=None):
        super().__init__(env)
        self.env = env
        # Optional side channel (4az): one JSON line per finished episode with
        # max_x / cleared / died, which NE-Dreamer's own score does not expose.
        self._episode_log = episode_log
        self._max_x, self._steps = 0.0, 0

        img_shape = self.env.observation_space.shape
        self.observation_space = gym.spaces.Dict({
            "image": gym.spaces.Box(0, 255, img_shape, np.uint8),
            "is_first": gym.spaces.Box(0, 1, (), bool),
            "is_last": gym.spaces.Box(0, 1, (), bool),
            "is_terminal": gym.spaces.Box(0, 1, (), bool),
        })

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._max_x, self._steps = float(info.get("mario_px", 0.0) or 0.0), 0
        obs_dict = {
            "image": obs,
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
        }
        return obs_dict

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        self._steps += 1
        self._max_x = max(self._max_x, float(info.get("mario_px", 0.0)))
        if done and self._episode_log:
            import json
            import time
            with open(self._episode_log, "a") as f:
                f.write(json.dumps({"t": time.time(), "steps": self._steps, "max_x_px": self._max_x,
                                    "cleared": bool(info.get("level_cleared", False)),
                                    "died": bool(info.get("died", False))}) + "\n")

        obs_dict = {
            "image": obs,
            "is_first": False,
            "is_last": done,
            "is_terminal": terminated,
        }
        return obs_dict, reward, done, info

def make_nedreamer_env(**kwargs):
    from env import MarioNdsEnv

    # Resolve relative paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.getenv("NEDREAMER_MARIO_V2"):
        # Opt-in 4az re-benchmark env: v2 actions, simple potential reward,
        # stall truncation -- the same ruler as tools/rebench_common.py.
        import sys
        sys.path.insert(0, os.path.join(base_dir, "tools"))
        import rebench_common
        return MarioNDSNEDreamerWrapper(rebench_common.make_env("pixels"),
                                        episode_log=os.getenv("NEDREAMER_EPLOG"))
    rom_path = os.path.join(base_dir, "data/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds")
    state_path = os.path.join(base_dir, "data/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).ds1")
    
    render_mode = os.getenv("MARIO_RENDER_MODE", None)
    # legacy: the vendored NE-Dreamer config assumes the June 6-action env
    raw_env = MarioNdsEnv(rom_path=rom_path, state_path=state_path, render_mode=render_mode,
                          actions="legacy")
    
    return MarioNDSNEDreamerWrapper(raw_env)
