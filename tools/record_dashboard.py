"""Record a dashboard-style video of the NSMB-DS agent: gameplay, what the
agent observes, its decision mix, the PINN's reward prediction and stage progress.

The point of this tool is evidence: the frame carries the level, the distance,
the goal and the controller identity, so a viewer can tell what a run proved
without opening the metrics json.

Usage:
    python tools/record_dashboard.py --planner rust --out media/nsmb_dashboard_clear.mp4
"""
from __future__ import annotations

import argparse
import collections
import os
import shutil
import struct
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import eval_pinn_realtime as ev  # noqa: E402
from ram_env import MarioRamEnv, OBS_DIM, GEO_DIM  # noqa: E402
from tilemap_planner import TilemapAStar, TILE, JUMP_REACH_PX  # noqa: E402

ROM = os.path.join(ROOT, "data", "0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds")
STATE = os.path.join(ROOT, "data", "0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).ds1")
GEOM = os.path.join(ROOT, "out", "nsmb_geometry.bin")

W, H = 1280, 720
BG = (24, 20, 14)
PANEL = (34, 30, 22)
FG = (216, 210, 200)
DIM = (140, 134, 124)
ACCENT = (60, 190, 245)
GOOD = (90, 200, 120)
BAD = (80, 90, 235)
TILE_FLOOR = (120, 96, 58)
TILE_BLOCK = (60, 190, 235)
TILE_PIT = (18, 16, 12)

ACTION_NAMES = ["noop", "walk", "walk+jump", "dash", "dash+jump (leap)",
                "back", "dash+jump (in place)", "jump in place"]
ACTION_BUTTONS = [[], ["right"], ["right", "A"], ["right", "X"],
                  ["right", "X", "A"], ["left"], ["X", "A"], ["A"]]
WINDOW = 60  # decisions shown as a percentage in the "what it thinks" panel


# ---------------------------------------------------------------- geometry
def load_geometry(path):
    """Read out/nsmb_geometry.bin (the same file the Rust planner loads)."""
    b = open(path, "rb").read()
    assert b[:8] == b"NSMBGEO1", "bad geometry magic"
    o = 12
    min_col, max_col, reach, feasible = struct.unpack_from("<iiii", b, o)
    o += 16
    surface = {}
    n = struct.unpack_from("<I", b, o)[0]
    o += 4
    for _ in range(n):
        c, v = struct.unpack_from("<ii", b, o)
        surface[c] = v
        o += 8
    obstacles = {}
    n = struct.unpack_from("<I", b, o)[0]
    o += 4
    for _ in range(n):
        c, v = struct.unpack_from("<ii", b, o)
        obstacles[c] = v
        o += 8
    return {"min_col": min_col, "max_col": max_col, "reach": reach,
            "feasible": bool(feasible), "surface": surface, "obstacles": obstacles}


# --------------------------------------------------------------- rule trace
RULES = {}
_orig_surface_threat = ev.BaseReflexController._surface_threat


def _traced_surface_threat(self, obs):
    r = _orig_surface_threat(self, obs)
    RULES["surface_threat"] = r
    return r


ev.BaseReflexController._surface_threat = _traced_surface_threat


# ------------------------------------------------------------------ drawing
def panel(img, x, y, w, h, title):
    cv2.rectangle(img, (x, y), (x + w, y + h), PANEL, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (60, 56, 48), 1)
    cv2.putText(img, title.upper(), (x + 10, y + 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.46, ACCENT, 1, cv2.LINE_AA)


def text(img, s, x, y, scale=0.45, color=FG, thick=1):
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_agent_view(img, x, y, w, h, geo, abs_px, obs):
    """The 23 numbers the policy receives, drawn spatially: surface profile from
    the tile geometry, the 6 pit-lookahead flags, the 3 nearest enemies, Mario."""
    mario_col = int(abs_px / TILE)
    cols = 30
    c0 = mario_col - 6
    rows = 10
    r0 = 21                      # first tile row drawn (row 30 = the floor)
    cw, ch = w // cols, h // rows
    cv2.rectangle(img, (x, y), (x + cols * cw, y + rows * ch), (30, 27, 22), -1)
    for i in range(cols):
        c = c0 + i
        top = geo["surface"].get(c)
        for r in range(rows):
            px, py = x + i * cw, y + r * ch
            if top is None:      # pit: no ground at all in this column
                continue
            if r0 + r >= top:
                col = TILE_FLOOR if (r0 + r) - top < 2 else TILE_BLOCK
                cv2.rectangle(img, (px, py), (px + cw - 1, py + ch - 1), col, -1)
    # pit-lookahead flags the agent reads directly (obs[17:23]): 1 = solid, 0 = pit
    for k in range(GEO_DIM):
        flag = obs[OBS_DIM + k] if OBS_DIM + k < len(obs) else 0.0
        px = x + (6 + k) * cw
        cv2.rectangle(img, (px, y + rows * ch + 4), (px + cw - 1, y + rows * ch + 9),
                      BAD if flag < 0.5 else GOOD, -1)
    # enemies from RAM: dx,dy in px relative to Mario (positive dy is UP)
    mario_top = geo["surface"].get(mario_col)
    mario_r = rows - 1 if mario_top is None else (mario_top - r0) - 1
    for k in range(3):
        dx, dy, et = obs[8 + 3 * k] * 256.0, obs[9 + 3 * k] * 256.0, obs[10 + 3 * k] * 256.0
        if abs(dx) < 1e-6 and abs(et) < 1e-6:
            continue
        i = 6 + int(round(dx / TILE))
        j = mario_r - int(round(dy / TILE))
        if 0 <= i < cols and 0 <= j < rows:
            cv2.rectangle(img, (x + i * cw, y + j * ch), (x + (i + 1) * cw - 1, y + (j + 1) * ch - 1),
                          BAD, -1)
    # Mario stands on the surface column he occupies
    j = max(0, min(rows - 1, mario_r))
    cv2.rectangle(img, (x + 6 * cw - 2, y + j * ch - ch // 2), (x + 7 * cw + 1, y + (j + 1) * ch - 1),
                  FG, 2, cv2.LINE_AA)
    text(img, f"cols {c0}..{c0 + cols - 1}  (Mario at col {mario_col})", x, y - 6, 0.34, DIM)
    text(img, "pit flags", x, y + rows * ch + 9, 0.30, DIM)


def draw_thinks(img, x, y, w, h, recent, current):
    counts = collections.Counter(recent)
    n = max(1, len(recent))
    bh = (h - 8) / len(ACTION_NAMES)
    for i, name in enumerate(ACTION_NAMES):
        pct = 100.0 * counts.get(i, 0) / n
        yy = int(y + 6 + i * bh)
        text(img, name, x, yy + 12, 0.40, FG if i == current else DIM)
        bx = x + 150
        bw = w - 150 - 52
        cv2.rectangle(img, (bx, yy + 4), (bx + bw, yy + 13), (52, 48, 42), -1)
        fill = int(bw * min(1.0, pct / 100.0))
        if fill:
            cv2.rectangle(img, (bx, yy + 4), (bx + fill, yy + 13),
                          GOOD if i == current else ACCENT, -1)
        text(img, f"{pct:.0f}%", bx + bw + 6, yy + 13, 0.38, DIM)


def draw_reward(img, x, y, w, h, hist):
    if len(hist) < 2:
        return
    win = hist[-200:]
    pred = np.array([p for p, _ in win], dtype=float)
    act = np.array([a for _, a in win], dtype=float)
    finite = np.concatenate([pred[np.isfinite(pred)], act[np.isfinite(act)]])
    if finite.size == 0:
        text(img, "no reward prediction available", x, y + h, 0.40, DIM)
        return
    lo, hi = float(finite.min()), float(finite.max())
    rng = (hi - lo) or 1.0

    def draw_series(arr, color, thick):
        xs = np.linspace(x, x + w, len(arr))
        ys = y + h - (np.nan_to_num(arr, nan=lo) - lo) / rng * h
        pts = np.stack([xs, ys], 1).astype(np.int32)
        i = 0
        while i < len(arr):
            if not np.isfinite(arr[i]):
                i += 1
                continue
            j = i
            while j < len(arr) and np.isfinite(arr[j]):
                j += 1
            if j - i > 1:
                cv2.polylines(img, [pts[i:j]], False, color, thick, cv2.LINE_AA)
            i = j

    draw_series(act, (70, 70, 70), 1)
    draw_series(pred, GOOD, 2)
    last_p = pred[np.isfinite(pred)]
    last_a = act[np.isfinite(act)]
    text(img, f"PINN predicted r  |  grey: actual  |  "
              f"{last_p[-1]:+.3f} vs {last_a[-1]:+.3f}", x, y + h + 14, 0.36, DIM)


def draw_buttons(img, held):
    """DS controls: crosspad left, A/B/X/Y diamond right, shoulder buttons below."""
    def btn(label, short, cx, cy, r=15):
        on = label in held
        cv2.circle(img, (cx, cy), r, (60, 190, 245) if on else (52, 48, 42), -1)
        text(img, short, cx - 6 if len(short) < 2 else cx - 9, cy + 5, 0.38,
             BG if on else DIM)

    for lab, s, dx, dy in [("up", "up", 0, -30), ("down", "dn", 0, 30),
                           ("left", "lf", -30, 0), ("right", "rt", 30, 0)]:
        btn(lab, s, 1076 + dx, 158 + dy)
    for lab, dx, dy in [("Y", -26, 0), ("A", 26, 0), ("X", 0, -26), ("B", 0, 26)]:
        btn(lab, lab, 1176 + dx, 158 + dy)
    btn("L", "L", 1076, 232, 14)
    btn("R", "R", 1176, 232, 14)
    text(img, "crosspad                A/B/X/Y", 1034, 258, 0.30, DIM)


def grab_frame(emu):
    buf = np.array(emu.display_buffer_as_rgbx(), dtype=np.uint8).reshape((384, 256, 4))
    return cv2.cvtColor(buf[:192, :, :3], cv2.COLOR_RGB2BGR)


def compose(game, st):
    img = np.full((H, W, 3), BG, dtype=np.uint8)
    text(img, "Reinforcement learning on New Super Mario Bros. DS", 24, 30, 0.72, FG, 2)
    text(img, f"{st['level']}   ·   episode {st['ep']}   ·   planner: {st['planner']}",
         24, 48, 0.44, DIM)
    text(img, f"{st['controller']} controller", W - 250, 34, 0.48, ACCENT)
    text(img, f"step {st['step']}", W - 250, 52, 0.42, DIM)

    # gameplay, 2x nearest (the DS top screen is 256x192)
    gs = cv2.resize(game, (512, 384), interpolation=cv2.INTER_NEAREST)
    gx, gy = 24, 66
    panel(img, gx - 8, gy - 8, 528, 400, "")
    img[gy:gy + 384, gx:gx + 512] = gs
    text(img, f"{st['abs_px']:.0f} px / {st['goal_px']} px to the flagpole", gx, 490, 0.52, FG)
    text(img, f"lives {st['lives']:.0f}   timer {st['timer']:.0f}   "
              f"vx {st['vx']:+.2f} px/f   vy {st['vy']:+.2f}   on ground {int(st['ground'])}",
         gx, 512, 0.44, DIM)
    text(img, f"action: {st['action_name']}  [{', '.join(st['held']) or 'none'}]   "
              f"rule fired: {st['rule']}", gx, 534, 0.44, ACCENT)

    x0 = 572
    panel(img, x0, 66, 440, 226, "what the agent sees (RAM + tile geometry)")
    draw_agent_view(img, x0 + 12, 104, 416, 130, st["geo"], st["abs_px"], st["obs"])
    lx = x0 + 12
    for i, (lab, col) in enumerate([("floor/step", TILE_FLOOR), ("block", TILE_BLOCK),
                                    ("pit (no ground)", (30, 27, 22)), ("enemy", BAD)]):
        cv2.rectangle(img, (lx + i * 106, 268), (lx + i * 106 + 10, 278), col, -1)
        text(img, lab, lx + i * 106 + 15, 277, 0.32, DIM)

    panel(img, x0, 302, 440, 174, "what it thinks (share of last 60 decisions)")
    draw_thinks(img, x0 + 12, 314, 416, 152, st["recent"], st["action_idx"])

    panel(img, x0, 486, 440, 110, "reward prediction (PINN world model)")
    draw_reward(img, x0 + 12, 512, 416, 56, st["hist"])

    panel(img, 1024, 66, 232, 210, "buttons")
    draw_buttons(img, st["held"])

    panel(img, 1024, 288, 232, 308, "SciML / physics")
    for i, (k, v) in enumerate(st["physics"]):
        text(img, k, 1034, 318 + i * 30, 0.34, DIM)
        text(img, v, 1034, 336 + i * 30, 0.38, FG)
    text(img, f"held now: {', '.join(st['held']) or 'none'}", 1034, 560, 0.34, DIM)

    # progress bar
    y = 640
    text(img, f"{st['level']} progress", 24, y - 8, 0.44, DIM)
    frac = max(0.0, min(1.0, st["abs_px"] / float(st["goal_px"])))
    cv2.rectangle(img, (24, y), (W - 24, y + 22), (52, 48, 42), -1)
    cv2.rectangle(img, (24, y), (int(24 + (W - 48) * frac), y + 22), ACCENT, -1)
    text(img, f"{frac * 100:.1f}%", W - 92, y + 17, 0.5, BG if frac > 0.06 else FG, 2)
    if st["cleared"]:
        text(img, "STAGE CLEARED - flagpole contact", 24, y + 50, 0.55, GOOD, 2)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--planner", choices=["python", "rust"], default="rust")
    ap.add_argument("--controller", default="reactive")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--record-episode", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--hold", type=int, default=4, help="video frames per agent step")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out", default=os.path.join(ROOT, "media", "nsmb_dashboard_clear.mp4"))
    ap.add_argument("--png-frame", default=os.path.join(ROOT, "out", "dashboard_frame.png"))
    args = ap.parse_args()

    planner_obj = None
    if args.planner == "rust":
        from rust_planner import RustPlanner
        planner_obj = RustPlanner()
    else:
        planner_obj = TilemapAStar(rom_path=ROM)

    env = MarioRamEnv(rom_path=ROM, state_path=STATE, max_steps=args.max_steps, geo=True)
    goal = ev.goal_px()
    geo = load_geometry(GEOM)
    physics = physics_rows(env, geo)

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
        model = None

    tmp_avi = os.path.join(ROOT, "out", "dashboard_tmp.avi")
    os.makedirs(os.path.dirname(tmp_avi), exist_ok=True)
    writer = cv2.VideoWriter(tmp_avi, cv2.VideoWriter_fourcc(*"MJPG"), args.fps, (W, H))
    if not writer.isOpened():
        print("[dash] VideoWriter failed to open", tmp_avi, file=sys.stderr)
        return 2

    made = 0
    snapdir = os.path.join(ROOT, "out", "dashboard_frames")
    os.makedirs(snapdir, exist_ok=True)
    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        agent = ev.ReactiveAgent(planner=planner_obj)
        agent.reset()
        recent, hist = [], []
        done = trunc = False
        step = 0
        info = {}
        game = grab_frame(env.emu)
        st = {"level": "World 1-1 (course A01_1)", "ep": ep, "planner": args.planner,
              "controller": args.controller, "geo": geo, "goal_px": goal,
              "physics": physics, "cleared": False, "step": 0, "abs_px": 0.0,
              "lives": 0.0, "timer": 0.0, "vx": 0.0, "vy": 0.0, "ground": 1.0,
              "obs": obs, "action_idx": 0, "action_name": ACTION_NAMES[0],
              "held": [], "rule": "-", "recent": [], "hist": []}
        while not (done or trunc) and step < args.max_steps:
            RULES["surface_threat"] = None
            a = int(agent.act(obs))
            pred_r = float("nan")
            if model is not None:
                with torch.no_grad():
                    s = torch.as_tensor(np.asarray(obs).reshape(1, -1), device=device).float()
                    _, r_hat, _, _ = model.step_batch(s, torch.tensor([a], device=device))
                pred_r = float(r_hat.item())
            obs, r, done, trunc, info = env.step(a)
            recent.append(a)
            if len(recent) > WINDOW:
                recent.pop(0)
            hist.append((pred_r, float(r)))
            step += 1
            game = grab_frame(env.emu)
            st.update({"step": step, "abs_px": float(info.get("mario_px", 0.0)),
                       "lives": float(obs[5]) * 10.0, "timer": float(obs[6]) * 400.0,
                       "vx": float(obs[2]) * 4.0, "vy": float(obs[3]) * 4.0,
                       "ground": float(obs[4]), "obs": obs,
                       "action_idx": a, "action_name": ACTION_NAMES[a],
                       "held": ACTION_BUTTONS[a],
                       "rule": RULES["surface_threat"] or "drive right",
                       "recent": list(recent), "hist": list(hist),
                       "cleared": bool(info.get("level_cleared"))})
            frame = compose(game, st)
            for _ in range(args.hold):
                writer.write(frame)
                made += 1
            if step % 60 == 0 or done or trunc:
                cv2.imwrite(os.path.join(snapdir, f"ep{ep}_step{step:04d}.png"), frame)
        cv2.imwrite(args.png_frame, compose(game, st))
        print(f"[dash] ep {ep}: {step} steps, {st['abs_px']:.0f}/{goal}px, "
              f"cleared={st['cleared']}")
        # the numbers the README quotes are the numbers this run displayed
        import json
        with open(os.path.join(ROOT, "out", "dashboard_run.json"), "w") as f:
            json.dump({"steps": step, "abs_px": round(st["abs_px"]), "goal_px": goal,
                       "cleared": st["cleared"], "planner": args.planner,
                       "controller": args.controller, "video": os.path.basename(args.out),
                       "seconds": round(made / args.fps), "width": W, "height": H},
                      f, indent=2)
    writer.release()
    if planner_obj is not None and hasattr(planner_obj, "close"):
        planner_obj.close()
    print(f"[dash] wrote {tmp_avi} ({made} frames, {made / args.fps:.1f}s)")

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("[dash] ffmpeg missing; leaving the MJPG avi", file=sys.stderr)
        return 1
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", tmp_avi, "-c:v", "libx264",
           "-pix_fmt", "yuv420p", "-crf", "23", "-movflags", "+faststart", args.out]
    subprocess.run(cmd, check=True)
    mb = os.path.getsize(args.out) / 1e6
    print(f"[dash] encoded {args.out} ({mb:.1f} MB)")
    return 0


def physics_rows(env, geo):
    """Panel content pulled from the code paths that actually use these numbers,
    so the video cannot drift away from the kinematics in the repo."""
    import tilemap_planner as tp
    surf = geo["surface"]
    pits = gaps = 0
    run = 0
    for c in range(geo["min_col"], geo["max_col"] + 1):
        if c not in surf:
            run += 1
        elif run:
            pits += 1
            gaps += run
            run = 0
    if run:
        pits += 1
        gaps += run
    return [("run-jump reach", f"{tp.JUMP_REACH_PX:.2f} px / 52 frames"),
            ("walk-jump reach", f"{tp.dynamic_jump_reach_px(1.47):.2f} px"),
            ("standing jump", "0 px forward (the 1-1 trap)"),
            ("frames per action", f"{env.frameskip} @ 60 Hz"),
            ("tile / units", f"{TILE} px, 0x1000 = 1 px"),
            ("level cols / pits", f"{geo['max_col']} cols, {pits} pits ({gaps} empty)"),
            ("goal (flagpole)", f"{ev.goal_px()} px")]


if __name__ == "__main__":
    sys.exit(main())
