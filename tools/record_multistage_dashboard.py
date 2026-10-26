"""Record official SciML telemetry dashboard videos for cleared worlds (4-1, 5-1, 6-1).

Usage:
    python tools/record_multistage_dashboard.py --world 5-1
    python tools/record_multistage_dashboard.py --world 4-1
    python tools/record_multistage_dashboard.py --world 6-1
    python tools/record_multistage_dashboard.py --all
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import eval_pinn_realtime as ev
from ram_env import MarioRamEnv, OBS_DIM, GEO_DIM, TIME_ADDR, YVEL_ADDR
from course import Course, GROUND_OBJS
from tilemap_planner import TilemapAStar, TILE
import record_dashboard as rd

ROM = os.path.join(ROOT, "data", "0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds")

CONFIGS = {
    "4-1": {
        "course": "course/D01_1.bin",
        "state": os.path.join(ROOT, "data", "state_4-1.ds1"),
        "goal_px": 4176.0,
        "spawn_px": 72.0,
        "controller": "world41",
        "title": "World 4-1 (course D01_1 - Swamp & Dorrie)",
        "out_mp4": os.path.join(ROOT, "media", "nsmb_w41_dashboard_clear.mp4"),
        "out_gif": os.path.join(ROOT, "media", "nsmb_w41_dashboard_preview.gif"),
        "meta_out": os.path.join(ROOT, "out", "dashboard_w41_run.json"),
        "max_steps": 1200,
    },
    "5-1": {
        "course": "course/E01_1.bin",
        "state": os.path.join(ROOT, "data", "0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).ds2"),
        "goal_px": 4576.0,
        "spawn_px": 56.0,
        "controller": "world51",
        "title": "World 5-1 (course E01_1 - Snow & High Canopy)",
        "out_mp4": os.path.join(ROOT, "media", "nsmb_w51_dashboard_clear.mp4"),
        "out_gif": os.path.join(ROOT, "media", "nsmb_w51_dashboard_preview.gif"),
        "meta_out": os.path.join(ROOT, "out", "dashboard_w51_run.json"),
        "max_steps": 1200,
    },
    "6-1": {
        "course": "course/F01_1.bin",
        "state": os.path.join(ROOT, "data", "state_6-1.ds1"),
        "goal_px": 4080.0,
        "spawn_px": 120.0,
        "controller": "world61",
        "title": "World 6-1 (course F01_1 - Marine Trench & Mountain Summit)",
        "out_mp4": os.path.join(ROOT, "media", "nsmb_w61_dashboard_clear.mp4"),
        "out_gif": os.path.join(ROOT, "media", "nsmb_w61_dashboard_preview.gif"),
        "meta_out": os.path.join(ROOT, "out", "dashboard_w61_run.json"),
        "max_steps": 1200,
    },
}


def make_agent(ctrl_name: str, course_name: str, planner: TilemapAStar, env: MarioRamEnv):
    if ctrl_name == "world41":
        from world_4_1_controller import World41Agent
        agent = World41Agent(rom_path=ROM, course_name=course_name, ram=env.ram)
    elif ctrl_name == "world51":
        from world_5_1_controller import World51Agent
        agent = World51Agent(rom_path=ROM, course_name=course_name, ram=env.ram)
    elif ctrl_name == "world61":
        from world_6_1_controller import World61Agent
        agent = World61Agent(planner=planner, ram=env.ram)
    else:
        raise ValueError(f"Unknown controller {ctrl_name}")
    agent.reset()
    return agent


def record_world(world_id: str, fps: int = 60, gif_len: float = 10.0):
    if world_id not in CONFIGS:
        raise ValueError(f"Unknown world {world_id}. Choose from: {list(CONFIGS.keys())}")
    cfg = CONFIGS[world_id]
    print(f"\n=======================================================")
    print(f"  RECORDING DASHBOARD VIDEO: {cfg['title']}")
    print(f"  Controller: {cfg['controller']} | Goal: {cfg['goal_px']} px")
    print(f"  Target: {cfg['out_mp4']}")
    print(f"=======================================================")

    planner = TilemapAStar(rom_path=ROM, name=cfg["course"])
    env = MarioRamEnv(rom_path=ROM, state_path=cfg["state"],
                      max_steps=cfg["max_steps"], geo=True,
                      course_name=cfg["course"])

    geo = {
        "min_col": planner.min_col,
        "max_col": planner.max_col,
        "reach": 7,
        "feasible": bool(planner.feasible),
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
        print(f"[dash] PINN not loaded ({e}): reward panel stays empty")

    tmp_avi = os.path.join(ROOT, "out", f"dashboard_{world_id}_tmp.avi")
    os.makedirs(os.path.dirname(tmp_avi), exist_ok=True)
    writer = cv2.VideoWriter(tmp_avi, cv2.VideoWriter_fourcc(*"MJPG"), fps, (rd.W, rd.H))
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter failed to open {tmp_avi}")

    cycle_frames = []
    # Hook native emu.cycle to capture every single 60 Hz frame, even inside controller sequences
    orig_emu_cycle = env.emu.cycle

    def cycle_hook():
        orig_emu_cycle()
        cycle_frames.append(rd.grab_frame(env.emu))

    env.emu.cycle = cycle_hook

    agent = make_agent(cfg["controller"], cfg["course"], planner, env)
    obs, _ = env.reset()

    spawn_px = cfg["spawn_px"]
    goal_px = cfg["goal_px"]

    st = {
        "level": cfg["title"], "ep": 1, "planner": "python",
        "controller": cfg["controller"], "geo": geo, "goal_px": int(goal_px),
        "physics": physics, "cleared": False, "step": 0, "abs_px": spawn_px,
        "blocks": blocks, "lives": 5.0, "timer": 400.0, "vx": 0.0, "vy": 0.0,
        "ground": 1.0, "obs": obs, "action_idx": 0, "action_name": rd.ACTION_NAMES[0],
        "held": [], "rule": "kinematic trajectory", "recent": [], "hist": []
    }

    recent, hist = [], []
    step = 0
    made = 0
    done = trunc = False
    finished = False
    last_game_frame = rd.grab_frame(env.emu)

    print(f"[dash-{world_id}] Running trajectory...")
    t_start = time.time()

    while not (done or trunc) and step < cfg["max_steps"]:
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

        # Determine real coordinates
        st_poll = env.ram.poll()
        if st_poll.get("mario_B") is not None:
            current_abs_px = float(st_poll["mario_B"]) / 4096.0
        else:
            current_abs_px = spawn_px + float(obs[0]) * 512.0

        finished = current_abs_px >= (goal_px - 16.0) or bool(info.get("level_cleared"))

        st.update({
            "step": step,
            "abs_px": current_abs_px,
            "lives": float(obs[5]) * 10.0,
            "timer": float(obs[6]) * 400.0,
            "vx": float(obs[2]) * 4.0,
            "vy": float(obs[3]) * 4.0,
            "ground": float(obs[4]),
            "obs": obs,
            "action_idx": a,
            "action_name": rd.ACTION_NAMES[a],
            "held": rd.ACTION_BUTTONS[a],
            "rule": "flagpole contact" if finished else ("kinematic sequence" if len(cycle_frames) > 16 else "planner drive"),
            "recent": list(recent),
            "hist": list(hist),
            "cleared": finished,
        })

        # Write all frames produced in this step (at 60 Hz)
        seq = cycle_frames if cycle_frames else [last_game_frame]
        for cf in seq:
            frame = rd.compose(cf, st)
            writer.write(frame)
            made += 1
        if seq:
            last_game_frame = seq[-1]

        if step % 25 == 0 or finished or done:
            print(f"  step {step:3d} | x={current_abs_px:6.1f} / {goal_px:.0f} px ({current_abs_px/goal_px*100:4.1f}%) | "
                  f"frames={made} ({made/fps:.1f}s) | cleared={finished}")

        if finished:
            # Hold final victory frame for 2.0 seconds (120 frames at 60 Hz)
            victory_frame = rd.compose(last_game_frame, st)
            for _ in range(fps * 2):
                writer.write(victory_frame)
                made += 1
            break

    writer.release()
    env.close()
    elapsed = time.time() - t_start
    print(f"[dash-{world_id}] Simulation complete in {elapsed:.1f}s: {step} steps, {made} frames ({made/fps:.1f}s), "
          f"final_x={st['abs_px']:.1f}px, cleared={st['cleared']}")

    # Encode with FFmpeg
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("[dash] ffmpeg missing; left MJPG avi", file=sys.stderr)
        return False

    os.makedirs(os.path.dirname(cfg["out_mp4"]), exist_ok=True)
    subprocess.run([
        ffmpeg, "-y", "-loglevel", "error", "-i", tmp_avi,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
        "-movflags", "+faststart", cfg["out_mp4"]
    ], check=True)
    mp4_mb = os.path.getsize(cfg["out_mp4"]) / 1e6
    print(f"[dash-{world_id}] Generated MP4: {cfg['out_mp4']} ({mp4_mb:.1f} MB)")

    # Generate preview GIF of the closing seconds
    clip_seconds = made / fps
    gstart = max(0.0, clip_seconds - gif_len)
    filt = "fps=15,scale=640:-1:flags=lanczos"
    pal = os.path.join(ROOT, "out", f"gif_{world_id}_palette.png")
    subprocess.run([
        ffmpeg, "-y", "-loglevel", "error", "-ss", str(gstart),
        "-t", str(gif_len), "-i", cfg["out_mp4"],
        "-vf", filt + ",palettegen=stats_mode=diff", pal
    ], check=True)
    subprocess.run([
        ffmpeg, "-y", "-loglevel", "error", "-ss", str(gstart),
        "-t", str(gif_len), "-i", cfg["out_mp4"], "-i", pal,
        "-lavfi", f"{filt}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3",
        cfg["out_gif"]
    ], check=True)
    gif_mb = os.path.getsize(cfg["out_gif"]) / 1e6
    print(f"[dash-{world_id}] Generated GIF: {cfg['out_gif']} ({gif_mb:.1f} MB)")

    # Save level-specific metadata run JSON
    with open(cfg["meta_out"], "w", encoding="utf-8") as f:
        json.dump({
            "world": world_id,
            "steps": step,
            "abs_px": round(st["abs_px"]),
            "goal_px": int(goal_px),
            "cleared": bool(st["cleared"]),
            "controller": cfg["controller"],
            "video": os.path.basename(cfg["out_mp4"]),
            "gif": os.path.basename(cfg["out_gif"]),
            "seconds": round(made / fps),
            "frames": made,
            "fps": fps,
            "width": rd.W,
            "height": rd.H
        }, f, indent=2)

    # Clean up temporary avi and palette
    if os.path.exists(tmp_avi):
        os.remove(tmp_avi)
    if os.path.exists(pal):
        os.remove(pal)

    return True


def main():
    ap = argparse.ArgumentParser(description="Multi-world NSMB-DS Telemetry Dashboard Video Recorder")
    ap.add_argument("--world", choices=["4-1", "5-1", "6-1"], default=None, help="Specific world to record")
    ap.add_argument("--all", action="store_true", help="Record all cleared worlds (4-1, 5-1, 6-1)")
    ap.add_argument("--fps", type=int, default=60, help="Framerate (60 fps native DS)")
    ap.add_argument("--gif-len", type=float, default=10.0, help="Duration of preview GIF in seconds")
    args = ap.parse_args()

    targets = ["4-1", "5-1", "6-1"] if args.all else ([args.world] if args.world else ["5-1"])
    for w in targets:
        record_world(w, fps=args.fps, gif_len=args.gif_len)


if __name__ == "__main__":
    main()
