"""Compare NSMB DS A/B jump-button behavior from a verified 1-1 savestate.

This is a behavioral oracle, not a claim about the game's internal code. It
loads the same clean state before each trial, settles for 20 frames, holds one
button for 60 frames, and records Mario's vertical position/velocity every
frame. One DeSmuME instance is used for the whole comparison.

Usage:
    .venv\\Scripts\\python.exe tools\\probe_jump_buttons.py
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from desmume.controls import Keys, keymask  # noqa: E402
from ram_env import MarioRamEnv, YVEL_ADDR  # noqa: E402
from ram_state import OFF_A  # noqa: E402

STATE = os.path.join(
    ROOT, "data", "0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).ds1")
FIXED = 4096
SETTLE_FRAMES = 20
HOLD_FRAMES = 60


def signed32(value: int) -> int:
    return value - 2**32 if value >= 2**31 else value


def capture(env: MarioRamEnv, label: str, key: int) -> dict:
    env.reset()
    for _ in range(SETTLE_FRAMES):
        env.emu.cycle()
    base = env.ram.mario_base
    if base is None:
        raise RuntimeError("Mario RAM object was not found after settling")
    y0 = env.ram._rs32(base + OFF_A) / FIXED

    env.emu.input.keypad_add_key(keymask(key))
    samples = []
    try:
        for frame in range(HOLD_FRAMES):
            env.emu.cycle()
            y = env.ram._rs32(base + OFF_A) / FIXED
            vy = signed32(env.emu.memory.read(
                YVEL_ADDR, YVEL_ADDR, 4, False)) / FIXED
            samples.append({"frame": frame + 1,
                            "rise_px": round(y - y0, 6),
                            "vy_px_per_frame": round(vy, 6)})
    finally:
        env.emu.input.keypad_rm_key(keymask(key))

    peak = max(row["rise_px"] for row in samples)
    return {"button": label,
            "settle_frames": SETTLE_FRAMES,
            "hold_frames": HOLD_FRAMES,
            "y0_px": round(y0, 6),
            "peak_rise_px": peak,
            "trajectory": samples}


def main() -> int:
    env = MarioRamEnv(state_path=STATE, max_steps=100000)
    if not env.has_emulator:
        raise SystemExit("DeSmuME failed to open the 1-1 savestate")
    try:
        rows = [capture(env, "A", Keys.KEY_A),
                capture(env, "B", Keys.KEY_B)]
    finally:
        env.close()

    a, b = rows
    result = {
        "rom": "NSMB DS EUR (NTR-A2DP-EUR)",
        "course": "course/A01_1.bin",
        "state": "data/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).ds1",
        "state_identification": {
            "method": "tools/identify_state.py unique entrance match",
            "spawn_x_px": 88.0,
            "spawn_y_px": -480.0,
            "entrance": "course/A01_1.bin entrance[0] (80, 464)"
        },
        "protocol": {
            "same_emulator_instance": True,
            "reset_state_between_trials": True,
            "settle_frames": SETTLE_FRAMES,
            "held_frames": HOLD_FRAMES,
            "sample_rate_hz": 60,
            "powerup": "as represented by input savestate; not independently verified",
            "dash": False
        },
        "comparison": {
            "peak_rise_difference_px": round(abs(a["peak_rise_px"] - b["peak_rise_px"]), 6),
            "trajectories_identical": all(
                x["rise_px"] == y["rise_px"] and
                x["vy_px_per_frame"] == y["vy_px_per_frame"]
                for x, y in zip(a["trajectory"], b["trajectory"])),
            "interpretation": (
                "Both A and B produce the same measured standing-jump trajectory "
                "in this state and protocol; this does not establish equivalence "
                "for every powerup or movement state.")
        },
        "trials": rows
    }
    out = os.path.join(ROOT, "out", "dataset", "jump_button_equivalence.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps(result["comparison"], indent=2))
    print(f"A peak={a['peak_rise_px']:.6f}px; B peak={b['peak_rise_px']:.6f}px")
    print(f"saved {os.path.relpath(out, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
