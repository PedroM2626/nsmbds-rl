"""100% RAM Environment (zero pixels, no CNN): state = vector of 17 floats.

Observation (Box 17, normalized for MLP):
  [x, y, vx, vy, on_ground, lives, time, cam_x,
   edx0, edy0, etype0, edx1, edy1, etype1, edx2, edy2, etype2]
  x,y relative to episode reset (px, /512); vx,vy in px/frame (/4);
  enemies: 3 closest in |dx|, dx/dy in px (/256), type/256.
Reward: identical formula to visual env (progress*2 - 0.05, death -15).
Done: death (lives drop), timer reaches zero (-50, equivalent to timeout), clear (+100),
or 1000 steps cap. The clear flag and flagpole fallback are checked from RAM/course data.
"""
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    from desmume.emulator import DeSmuME
    from desmume.controls import Keys, keymask
except ImportError:
    DeSmuME = None

from ram_state import RamState, _s32
import paths

ROM = paths.ROM_REL
STATE = paths.STATE_REL
TIME_ADDR = 0x020DC968
YVEL_ADDR = 0x021C1908
N_ENEMIES = 3
OBS_DIM = 8 + 3 * N_ENEMIES
GEO_DIM = 6  # Pit flags for the 6 columns ahead
MAX_STEPS = 1000

# Action -> held DS buttons. Discovered empirically (see docs): the face button
# A is JUMP and X (or Y) is the RUN/DASH button (doubles speed 1.5->3.0 px/frame),
# enabling running jumps that clear the 64-80 px pits. This agent emits A for jump
# and never emits B; a 1-1 standing-jump probe found matching A/B trajectories in
# one state/protocol (see out/dataset/jump_button_equivalence.json). Index layout
# is also mirrored in pinn_world_model.N_ACTIONS.
#   0 noop | 1 right | 2 right+jump | 3 right+dash | 4 right+dash+jump
#   5 left  | 6 dash+jump (broad running leap) | 7 jump in place


class MarioRamEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, rom_path=ROM, state_path=STATE, max_steps=MAX_STEPS, geo=False,
                 course_name="course/A01_1.bin"):
        super().__init__()
        self.rom_path = paths.resolve(rom_path)
        self.state_path = paths.resolve(state_path)
        self.max_steps = max_steps
        self.geo = geo
        self.action_space = spaces.Discrete(8)
        self.course = None
        # Load the course BEFORE fixing the observation space: if the parser
        # refuses a course, dropping the geometry features must also drop the
        # six pit dims, or the space says 23 floats while _observe returns 17
        # (a silent SB3 check failure later, instead of a loud one here).
        if self.geo:
            try:
                from course import Course
                self.course = Course(name=course_name, rom_path=self.rom_path)
            except Exception as e:
                print(f"Geometry disabled (failed to load course: {e})")
                self.geo = False
        dim = OBS_DIM + (GEO_DIM if self.geo else 0)
        self.observation_space = spaces.Box(low=-10.0, high=10.0,
                                            shape=(dim,), dtype=np.float32)
        try:
            # Check the files BEFORE allocating the native emulator: a
            # DeSmuME() whose open() then fails still leaves a native
            # instance whose __del__ heap-corrupts at GC time (0xc0000374,
            # measured in the test suite). A missing file must be a clean
            # Python error, not a crash on exit.
            if not os.path.exists(self.rom_path):
                raise FileNotFoundError(f"ROM not found: {self.rom_path}")
            if not os.path.exists(self.state_path):
                raise FileNotFoundError(f"savestate not found: {self.state_path}")
            self.emu = DeSmuME()
            self.emu.open(self.rom_path)
            self.emu.savestate.load_file(self.state_path)
            self.has_emulator = True
        except Exception as e:
            print(f"Failed to initialize emulator: {e}")
            self.has_emulator = False
            self.emu = None
        self.ram = RamState(self.emu) if self.has_emulator else None
        self.frameskip = 8
        self._x0 = self._y0 = None
        self._last_time = 400.0

    # -- helpers ------------------------------------------------------
    def _s32read(self, addr):
        return _s32(self.emu.memory.read(addr, addr, 4, False))

    def _dummy_state(self):
        """A poll-shaped state for the no-emulator stub (unit tests, CI).
        There is no game behind it: progress 0, timer full, nothing dead."""
        return {"lives": 5, "cam": 0, "cam_dx": 0, "acc_cam": 0.0,
                "progress": 0.0, "odo": 0, "y": 0, "vel": 0, "died": False,
                "deaths": 0, "mario_A": None, "mario_B": None, "cleared": False}

    def _observe(self):
        if not self.has_emulator:
            # The documented offline fallback, made real: reset()/step() must
            # work without DeSmuME instead of crashing on self.ram.poll()
            # (AttributeError on None, measured 2026-10-23).
            self._last_enemies = []
            st = self._dummy_state()
            return np.zeros(self.observation_space.shape, dtype=np.float32), st
        st = self.ram.poll()
        mb, ma = st["mario_B"], st["mario_A"]
        if self._x0 is None:
            # The origin must be Mario's own position at reset, but on the
            # first poll after a fresh load the player object is not yet in
            # the entity list and poll returns None -- falling back to 0
            # here made every later obs[0] ABSOLUTE instead of relative
            # (measured: second-poll obs[0]*512 == mario_B exactly; see
            # PROJECT_STATUS 4o). Discover the object before fixing the
            # origin. NOTE: _mario_pos() returns (A, B) = (y, x).
            if mb is None or ma is None:
                if self.ram.mario_base is None:
                    self.ram.discover_mario()
                ma, mb = self.ram._mario_pos()
            self._x0, self._y0 = (mb or 0), (ma or 0)
        x = ((mb or self._x0) - self._x0) / 4096.0 / 512.0
        y = ((ma or self._y0) - self._y0) / 4096.0 / 512.0
        vx = st["vel"] / 4096.0 / 4.0
        vy = self._s32read(YVEL_ADDR) / 4096.0 / 4.0
        on_ground = 1.0 if vy == 0.0 else 0.0
        lives = st["lives"] / 10.0
        time_raw = self._s32read(TIME_ADDR) / 4096.0
        self._last_time = time_raw
        time_left = time_raw / 400.0
        cam = st["acc_cam"] / 4096.0 / 512.0
        obs = [x, y, vx, vy, on_ground, lives, time_left, cam]
        ens = sorted(self.ram.enemies(), key=lambda e: abs(e["dx"] or 1e9))[:N_ENEMIES]
        self._last_enemies = ens
        for e in ens:
            obs += [e["dx"] / 256.0, e["dy"] / 256.0, e["type"] / 256.0]
        while len(obs) < OBS_DIM:
            obs += [0.0, 0.0, 0.0]
        obs = obs[:OBS_DIM]
        if self.geo and self.course is not None:
            mb = st["mario_B"]
            if mb is None:
                if self.ram.mario_base is None:
                    self.ram.discover_mario()
                mb, _ = self.ram._mario_pos()
            mario_px = (mb / 4096.0) if mb is not None else 0.0
            obs += self.course.pit_ahead(mario_px, GEO_DIM)
        return np.array(obs, dtype=np.float32), st

    # -- gym api ------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.has_emulator:
            self.emu.savestate.load_file(self.state_path)
            self.ram.reset()
            for k in [Keys.KEY_RIGHT, Keys.KEY_LEFT, Keys.KEY_A, Keys.KEY_B, Keys.KEY_X, Keys.KEY_Y, Keys.KEY_UP, Keys.KEY_DOWN]:
                self.emu.input.keypad_rm_key(keymask(k))
        self._x0 = self._y0 = None
        self.episode_steps = 0
        self._prev_keys = set()
        self._last_on_ground = True
        self._just_landed = False
        obs, _ = self._observe()
        # Origin fixed after first poll
        return obs, {}

    def _action_keys(self, action):
        R, L, A, X = Keys.KEY_RIGHT, Keys.KEY_LEFT, Keys.KEY_A, Keys.KEY_X
        table = {
            0: [],
            1: [R],
            2: [R, A],            # walk + jump
            3: [R, X],            # dash (run)
            4: [R, X, A],         # dash + jump  -> running leap
            5: [L],
            6: [X, A],            # jump in place with dash spin
            7: [A],               # jump in place
        }
        return table.get(int(action), [])

    def _emit_cycle(self):
        """Called after every emulated frame so a recorder can capture true 60 Hz
        footage instead of one image per action. Set self.on_cycle = callable."""
        cb = getattr(self, "on_cycle", None)
        if cb is not None:
            cb()

    def step(self, action):
        if not self.has_emulator:
            self.episode_steps += 1
            obs, st = self._observe()
            trunc = self.episode_steps >= self.max_steps
            info = {"x": 0.0, "mario_px": 0.0, "level_cleared": False,
                    "enemies": [], "died": False, "timeout": False}
            return obs, -0.05, False, trunc, info
        keys = self._action_keys(action)
        prev_keys = getattr(self, "_prev_keys", set())

        # If repeating a jump action while grounded or upon touchdown, release KEY_A for 1 cycle to generate a hardware rising edge
        is_jump = (Keys.KEY_A in keys)
        was_jump = (Keys.KEY_A in prev_keys)
        grounded = getattr(self, "_last_on_ground", False) or getattr(self, "_just_landed", False)
        if is_jump and was_jump and grounded:
            # 1 cycle with directional/dash keys but without KEY_A
            for k in [k for k in keys if k != Keys.KEY_A]:
                self.emu.input.keypad_add_key(keymask(k))
            self.emu.cycle()
            self._emit_cycle()
            for k in [k for k in keys if k != Keys.KEY_A]:
                self.emu.input.keypad_rm_key(keymask(k))
            skip = max(1, self.frameskip - 1)
        else:
            skip = self.frameskip

        for key in keys:
            self.emu.input.keypad_add_key(keymask(key))
        for _ in range(skip):
            self.emu.cycle()
            self._emit_cycle()
        for key in keys:
            self.emu.input.keypad_rm_key(keymask(key))

        self._prev_keys = set(keys)
        obs, st = self._observe()
        on_ground = (obs[4] > 0.5)
        self._just_landed = (not getattr(self, "_last_on_ground", False) and on_ground)
        self._last_on_ground = on_ground
        reward = st["progress"] / 4096.0 * 2.0 - 0.05
        done, trunc = False, False
        died = timeout = False

        # Check level completion (RAM flag or flagpole contact)
        mario_px = (st["mario_B"] / 4096.0) if st["mario_B"] is not None else 0.0
        flag_x = getattr(self.course, "flag_x_px", 4032.0) if self.course else 4032.0
        cleared = st.get("cleared", False) or (mario_px >= flag_x - 16.0)

        if st["died"]:
            done = True
            died = True
            reward -= 15.0
        elif cleared:
            done = True
            reward += 100.0  # Terminal reward for clearing the stage
        elif getattr(self, "_last_time", 1.0) <= 0:
            done = True
            timeout = True
            reward -= 50.0
        self.episode_steps += 1
        if self.episode_steps >= self.max_steps:
            trunc = True
        # died/timeout let the eval harness label terminations from the env's
        # own decision instead of inferring the cause from the remaining
        # stage clock (the old obs[6] > 0.02 heuristic mislabeled any death
        # with under ~8 s left as a timeout).
        info = {"x": float(obs[0]), "mario_px": mario_px, "level_cleared": cleared,
                "enemies": getattr(self, "_last_enemies", []),
                "died": died, "timeout": timeout}
        return obs, reward, done, trunc, info

    def close(self):
        if self.has_emulator:
            try:
                self.emu.destroy()
            except Exception:
                pass
