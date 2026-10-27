"""Pin false-positive finishes found while resuming the special courses."""
from types import SimpleNamespace

from multi_area_env import at_terminal_flag, MultiAreaMarioEnv
from ram_env import MarioRamEnv


def course(x, y):
    return SimpleNamespace(sprites=[dict(type=32, x_px=x, y_px=y)])


def test_31_entrance_to_right_of_flag_is_not_a_finish():
    assert not at_terminal_flag(course(272, 192), 808, -192)


def test_71_upper_route_above_flag_is_not_a_finish():
    assert not at_terminal_flag(course(2848, 1216), 2848, -560)
    assert at_terminal_flag(course(2848, 1216), 2836, -1248)


def test_unflagged_underwater_area_has_no_horizontal_fallback():
    assert not at_terminal_flag(SimpleNamespace(sprites=[]), 4032, -480)


def test_rightmost_flag_is_used():
    c = course(4000, 480)
    c.sprites.append(dict(type=32, x_px=1000, y_px=480))
    assert not at_terminal_flag(c, 1000, -480)
    assert at_terminal_flag(c, 3990, -480)


def test_multi_area_requires_live_emulator():
    import pytest
    env = MultiAreaMarioEnv(rom_path="missing.nds", state_path="missing.ds1")
    with pytest.raises(RuntimeError, match="live emulator"):
        env.reset()


def test_legacy_horizontal_finish_is_removed(monkeypatch):
    from course_id import course_area_index
    env = object.__new__(MultiAreaMarioEnv)
    env.active_course = "course/C01_1.bin"
    env.course = course(272, 192)
    env.ram = SimpleNamespace(
        course_index=lambda: course_area_index("C01_1"),
        poll=lambda: dict(mario_B=808 * 4096, mario_A=-192 * 4096, cleared=False))
    monkeypatch.setattr(MarioRamEnv, "step", lambda *_: (
        None, 99.95, True, False,
        dict(level_cleared=True, died=False, timeout=False, mario_px=808)))
    _, reward, done, _, info = env.step(3)
    assert abs(reward + 0.05) < 1e-10
    assert not done
    assert not info["level_cleared"]


def test_clear_requires_both_ram_flag_and_terminal_contact(monkeypatch):
    from course_id import course_area_index
    env = object.__new__(MultiAreaMarioEnv)
    env.active_course = "course/G01_1.bin"
    env.course = course(2848, 1216)
    state = dict(mario_B=2848 * 4096, mario_A=-560 * 4096, cleared=True)
    env.ram = SimpleNamespace(course_index=lambda: course_area_index("G01_1"),
                              poll=lambda: state)
    monkeypatch.setattr(MarioRamEnv, "step", lambda *_: (
        None, 99.95, True, False,
        dict(level_cleared=True, died=False, timeout=False, mario_px=2848)))
    assert env.step(0)[2] is False
    state["mario_A"] = -1248 * 4096
    assert env.step(0)[2] is True


def test_pole_contact_is_remembered_during_automatic_walk(monkeypatch):
    from course_id import course_area_index
    env = object.__new__(MultiAreaMarioEnv)
    env.active_course = "course/A01_1.bin"
    env.course = course(4032, 480)
    state = dict(mario_B=4032 * 4096, mario_A=-480 * 4096, cleared=False)
    env.ram = SimpleNamespace(course_index=lambda: course_area_index("A01_1"),
                              poll=lambda: state)
    monkeypatch.setattr(MarioRamEnv, "step", lambda *_: (
        None, 99.95, True, False,
        dict(level_cleared=True, died=False, timeout=False, mario_px=4032)))
    assert env.step(0)[2] is False
    state.update(mario_B=4248 * 4096, cleared=True)
    assert env.step(0)[2] is True


def test_exact_pole_slide_confirms_capture_without_legacy_clear_byte(monkeypatch):
    from course_id import course_area_index
    env = object.__new__(MultiAreaMarioEnv)
    env.active_course = "course/C01_1.bin"
    env.course = course(272, 192)
    state = dict(mario_B=272 * 4096, mario_A=-160 * 4096, cleared=False)
    env.ram = SimpleNamespace(course_index=lambda: course_area_index("C01_1"),
                              poll=lambda: state)
    monkeypatch.setattr(MarioRamEnv, "step", lambda *_: (
        None, -0.05, False, False,
        dict(level_cleared=False, died=False, timeout=False, mario_px=272)))
    assert env.step(0)[2] is False
    state["mario_A"] = -168 * 4096
    assert env.step(0)[2] is False
    state["mario_A"] = -176 * 4096
    _, _, done, _, info = env.step(0)
    assert done and info["completion_evidence"] == "pole_slide"


def test_falling_near_pole_does_not_count_as_capture(monkeypatch):
    from course_id import course_area_index
    env = object.__new__(MultiAreaMarioEnv)
    env.active_course = "course/C01_1.bin"
    env.course = course(272, 192)
    state = dict(mario_B=280 * 4096, mario_A=-160 * 4096, cleared=False)
    env.ram = SimpleNamespace(course_index=lambda: course_area_index("C01_1"),
                              poll=lambda: state)
    monkeypatch.setattr(MarioRamEnv, "step", lambda *_: (
        None, -0.05, False, False,
        dict(level_cleared=False, died=False, timeout=False, mario_px=280)))
    for y in (-160, -168, -176, -184):
        state["mario_A"] = y * 4096
        assert env.step(0)[2] is False
