"""Opt-in evaluation environment for courses connected by pipes.

The legacy environment's horizontal finish heuristic is intentionally frozen.
It is insufficient for C01_1 (spawn right of its flag) and G01_1 (the flag is
on the lower route). Completion needs either the RAM clear flag after pole
contact, or measured capture/vertical descent on the exact ROM pole. The
legacy clear-byte address does not activate on the measured C01_1 exit.
"""
from __future__ import annotations

from gymnasium import spaces

from course import Course
from course_id import course_area_index, course_for_area_index
from ram_env import MarioRamEnv, Keys


def at_terminal_flag(course, x, y):
    """ROM flag proximity in pixels; RAM's vertical coordinate is negative."""
    flags = [s for s in course.sprites if s["type"] == 32]
    if not flags:
        return False
    flag = max(flags, key=lambda s: s["x_px"])
    return (abs(x - flag["x_px"]) <= 64.0
            and abs(y + flag["y_px"]) <= 128.0)


class MultiAreaMarioEnv(MarioRamEnv):
    """Pipe actions: 8=DOWN, 9=UP, 10=UP+A, 11=UP+R+A, 12=UP+R+X+A."""

    def __init__(self, *args, **kwargs):
        self.water_overrides = kwargs.pop("water_overrides", {})
        super().__init__(*args, **kwargs)
        self.initial_course = kwargs.get("course_name", "course/A01_1.bin")
        self.active_course = self.initial_course
        self.action_space = spaces.Discrete(13)
        self._courses = {self.initial_course: self.course}
        self._apply_water_override()

    def _apply_water_override(self):
        if self.active_course in self.water_overrides and self.course is not None:
            self.course.is_water_course = self.water_overrides[self.active_course]

    def _action_keys(self, action):
        if int(action) == getattr(self, "_pipe_action", None):
            return super()._action_keys(action) + [Keys.KEY_UP]
        if int(action) == 8:
            return [Keys.KEY_DOWN]
        if int(action) == 9:
            return [Keys.KEY_UP]
        return super()._action_keys(action)

    def reset(self, **kwargs):
        self._terminal_contact = False
        self._pole_start_y = None
        self._pole_samples = 0
        self._pole_captured = False
        self.active_course = self.initial_course
        self.course = self._courses[self.initial_course]
        obs, info = super().reset(**kwargs)
        if not self.has_emulator:
            raise RuntimeError("Multi-area evaluation requires a live emulator")
        actual = self.ram.course_index()
        expected = course_area_index(self.initial_course)
        if actual != expected:
            raise ValueError(f"State area {actual} disagrees with {self.initial_course} ({expected})")
        info["course"] = self.active_course
        return obs, info

    def step(self, action):
        # Reuse action 7's measured water rising-edge handling while holding
        # UP, so swimming into a ceiling pipe needs no RAM writes.
        self._pipe_action = {10: 7, 11: 2, 12: 4}.get(int(action))
        try:
            effective = self._pipe_action if self._pipe_action is not None else action
            obs, reward, _, trunc, info = super().step(effective)
        finally:
            self._pipe_action = None
        cid = course_for_area_index(self.ram.course_index())
        if cid is None:
            raise RuntimeError("Unknown course index during area transition")
        name = f"course/{cid}.bin"
        changed = name != self.active_course
        if changed:
            self._terminal_contact = False
            self._pole_start_y = None
            self._pole_samples = 0
            self._pole_captured = False
            if name not in self._courses:
                self._courses[name] = Course(name, self.rom_path)
            self.course = self._courses[name]
            self.active_course = name
            self._apply_water_override()
            self.ram.mario_base = None
            obs, _ = self._observe()
        st = self.ram.poll()
        x = st["mario_B"] / 4096.0 if st["mario_B"] is not None else None
        y = st["mario_A"] / 4096.0 if st["mario_A"] is not None else None
        contact = (x is not None and y is not None
                   and at_terminal_flag(self.course, x, y))
        # The game raises the clear flag after the pole/automatic walk, by
        # which time Mario may be outside the pole's proximity window.
        self._terminal_contact = getattr(self, "_terminal_contact", False) or contact
        flags = [s for s in self.course.sprites if s["type"] == 32]
        pole = max(flags, key=lambda s: s["x_px"]) if flags else None
        on_pole = bool(contact and pole and abs(x - pole["x_px"]) <= 0.25)
        if on_pole:
            if getattr(self, "_pole_start_y", None) is None:
                self._pole_start_y = y
                self._pole_samples = 0
            self._pole_samples += 1
            if self._pole_samples >= 3 and self._pole_start_y - y >= 8.0:
                self._pole_captured = True
        else:
            self._pole_start_y = None
            self._pole_samples = 0
        captured = getattr(self, "_pole_captured", False)
        raw_clear = bool(st["cleared"] and self._terminal_contact)
        cleared = bool((raw_clear or captured) and not info["died"])
        # Remove any bonus granted by the legacy horizontal fallback.
        if info["level_cleared"] and not info["died"]:
            reward -= 100.0
        if cleared:
            reward += 100.0
        info.update(course=name, area_changed=changed, mario_px=x, mario_y_px=y,
                    raw_clear_flag=bool(st["cleared"]), flag_contact=bool(contact),
                    terminal_contact_seen=bool(self._terminal_contact),
                    pole_captured=bool(captured),
                    completion_evidence=("ram_clear_flag" if raw_clear else
                                         "pole_slide" if captured else None),
                    level_cleared=cleared)
        done = bool(info["died"] or info["timeout"] or cleared)
        return obs, reward, done, trunc, info
