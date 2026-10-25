"""Record a SciML dashboard video of World 8-1 (H01_1) clearance."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from ram_env import MarioRamEnv, OBS_DIM, GEO_DIM
from course import Course, GROUND_OBJS
from world_8_1_controller import World81Agent, setup_world_8_1_planner_and_profile
import record_dashboard as rd

ROM = os.path.join(ROOT, "data", "0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds")
STATE = os.path.join(ROOT, "data", "0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).ds3")
COURSE = "course/H01_1.bin"
GOAL = 3920.0
SPAWN_PX = 56.0


def main():
    parser = argparse.ArgumentParser(description="Record World 8-1 SciML Dashboard Video & GIF")
    parser.add_argument("--out", default="media/nsmb_w81_dashboard_clear.mp4", help="Output MP4 video path")
    parser.add_argument("--gif", default="media/nsmb_w81_dashboard_preview.gif", help="Output GIF preview path")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--gif-len", type=float, default=10.0, help="GIF duration in seconds")
    args = parser.parse_args()

    planner, profile = setup_world_8_1_planner_and_profile(ROM, COURSE)
    env = MarioRamEnv(rom_path=ROM, state_path=STATE, max_steps=1200, geo=True, course_name=COURSE)

    # Geometry dict for draw_agent_view
    geo = {
        "min_col": planner.min_col,
        "max_col": planner.max_col,
        "reach": 7,
        "feasible": True,
        "surface": planner.surface,
        "obstacles": planner.obstacle_h,
    }
    physics = rd.physics_rows(env, geo)

    course = planner.course
    ABSORBED_BLOCKS = (0x2C, 0x30)
    blocks = [r for r in course.rects
              if (r["obj"] not in GROUND_OBJS or r["obj"] in ABSORBED_BLOCKS)
              and 20 <= r["ty"] <= 27 and r["w"] <= 3 and r["h"] <= 3]

    model = None
    device = "cpu"
    try:
        from pinn_world_model import MarioPINNWorldModel
        ck = torch.load(os.path.join(ROOT, "models", "pinn_nsmb.pt"),
                        map_location="cuda" if torch.cuda.is_available() else "cpu",
                        weights_only=True)
        model = MarioPINNWorldModel(hid=ck["hid"])
        model.load_state_dict(ck["state_dict"])
        model.eval()
        device = next(model.parameters()).device
    except Exception as e:
        print(f"[dash-8-1] PINN not loaded ({e}): reward panel stays empty")

    tmp_avi = os.path.join(ROOT, "out", "dashboard_w81_tmp.avi")
    os.makedirs(os.path.dirname(tmp_avi), exist_ok=True)
    writer = cv2.VideoWriter(tmp_avi, cv2.VideoWriter_fourcc(*"MJPG"), args.fps, (rd.W, rd.H))

    made = 0
    obs, _ = env.reset()
    agent = World81Agent(rom_path=ROM, course_name=COURSE, ram=env.ram)
    agent.reset()
    recent, hist = [], []
    step = 0
    cycle_frames = []
    env.on_cycle = lambda: cycle_frames.append(rd.grab_frame(env.emu))
    game = rd.grab_frame(env.emu)

    st = {
        "level": "World 8-1 (course H01_1)", "ep": 1, "planner": "python",
        "controller": "world81", "geo": geo, "goal_px": GOAL,
        "physics": physics, "cleared": False, "step": 0, "abs_px": SPAWN_PX,
        "blocks": blocks, "lives": 0.0, "timer": 0.0, "vx": 0.0, "vy": 0.0,
        "ground": 1.0, "obs": obs, "action_idx": 0, "action_name": rd.ACTION_NAMES[0],
        "held": [], "rule": "kinematic drive", "recent": [], "hist": []
    }

    print("[dash-8-1] Recording trajectory...")
    while step < 1200:
        cycle_frames.clear()
        a = int(agent.act(obs))
        pred_r = float("nan")
        if model is not None:
            with torch.no_grad():
                s = torch.as_tensor(np.asarray(obs).reshape(1, -1), device=device).float()
                _, r_hat, _, _ = model.step_batch(s, torch.tensor([a], device=device))
            pred_r = float(r_hat.item())

        obs, r, done, trunc, info = env.step(a)
        recent.append(a)
        if len(recent) > rd.WINDOW:
            recent.pop(0)
        hist.append((pred_r, float(r)))
        step += 1
        game = cycle_frames[-1] if cycle_frames else rd.grab_frame(env.emu)
        abs_px = SPAWN_PX + float(obs[0]) * 512.0
        finished = abs_px >= (GOAL - 16.0) or bool(info.get("level_cleared"))

        st.update({
            "step": step, "abs_px": abs_px,
            "lives": float(obs[5]) * 10.0, "timer": float(obs[6]) * 400.0,
            "vx": float(obs[2]) * 4.0, "vy": float(obs[3]) * 4.0,
            "ground": float(obs[4]), "obs": obs,
            "action_idx": a, "action_name": rd.ACTION_NAMES[a],
            "held": rd.ACTION_BUTTONS[a],
            "rule": "staircase leap" if 3200 <= abs_px <= 3480 else ("blaster leap" if 2660 <= abs_px <= 2950 else "dash"),
            "recent": list(recent), "hist": list(hist),
            "cleared": finished
        })

        seq = cycle_frames or [game]
        for cf in seq:
            frame = rd.compose(cf, st)
            writer.write(frame)
            made += 1

        if finished or done or trunc:
            break

    writer.release()
    env.close()
    print(f"[dash-8-1] Finished at step {step}, abs_px={st['abs_px']:.1f}, cleared={st['cleared']}")

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("[dash-8-1] ffmpeg missing; left MJPG avi", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", tmp_avi, "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "23", "-movflags", "+faststart", args.out], check=True)
    print(f"[dash-8-1] Encoded MP4: {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)")

    if args.gif:
        clip_seconds = made / args.fps
        gstart = max(0.0, clip_seconds - args.gif_len)
        filt = "fps=15,scale=640:-1:flags=lanczos"
        pal = os.path.join(ROOT, "out", "gif_w81_palette.png")
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-ss", str(gstart),
                        "-t", str(args.gif_len), "-i", args.out,
                        "-vf", filt + ",palettegen=stats_mode=diff", pal], check=True)
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-ss", str(gstart),
                        "-t", str(args.gif_len), "-i", args.out, "-i", pal,
                        "-lavfi", f"{filt}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3",
                        args.gif], check=True)
        print(f"[dash-8-1] Wrote GIF: {args.gif} ({os.path.getsize(args.gif)/1e6:.1f} MB)")

    with open(os.path.join(ROOT, "out", "dashboard_w81_run.json"), "w", encoding="utf-8") as f:
        json.dump({
            "steps": step, "abs_px": round(st["abs_px"]), "goal_px": GOAL,
            "cleared": st["cleared"], "video": os.path.basename(args.out),
            "seconds": round(made / args.fps), "width": rd.W, "height": rd.H
        }, f, indent=2)

    return 0


if __name__ == "__main__":
    main()
