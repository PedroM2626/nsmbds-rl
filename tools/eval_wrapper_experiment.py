"""Experiment: Evaluate PPO checkpoints with RisingEdgeActionWrapper under deterministic argmax."""
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = r"d:\mario-ds"
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "src"))


def evaluate_single(model_path, mode, cooldown=1, max_steps=1000, recurrent=False):
    import numpy as np
    import rebench_common as rc
    from action_wrapper import RisingEdgeActionWrapper

    raw_env = rc.make_env("pixels")
    if mode == "baseline":
        env = raw_env
    else:
        env = RisingEdgeActionWrapper(raw_env, mode=mode, cooldown_steps=cooldown)

    if recurrent:
        from sb3_contrib import RecurrentPPO
        model = RecurrentPPO.load(model_path, device="cpu")
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)
    else:
        from stable_baselines3 import PPO
        model = PPO.load(model_path, device="cpu")

    obs, info = env.reset()

    max_x = final_x = float(info.get("mario_px", 0.0) or 0.0)
    ret, steps = 0.0, 0
    t0 = time.time()

    while steps < max_steps:
        if recurrent:
            act, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
            episode_starts = np.zeros((1,), dtype=bool)
        else:
            act, _ = model.predict(obs, deterministic=True)

        obs, r, done, trunc, info = env.step(int(act))
        steps += 1
        ret += float(r)
        x = float(info.get("mario_px", 0.0))
        final_x = x
        max_x = max(max_x, x)
        if done or trunc:
            break

    elapsed = time.time() - t0
    term = "clear" if info.get("level_cleared") else ("death" if info.get("died") else ("timeout" if info.get("timeout") else "stall"))

    env.close()
    return {
        "mode": mode,
        "cooldown": cooldown,
        "max_x_px": max_x,
        "final_x_px": final_x,
        "steps": steps,
        "return": ret,
        "cleared": info.get("level_cleared", False),
        "termination": term,
        "elapsed_s": round(elapsed, 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/rebench/ppo_naturecnn_s2_best.zip")
    parser.add_argument("--mode", default=None)
    parser.add_argument("--cooldown", type=int, default=1)
    parser.add_argument("--recurrent", action="store_true")
    args = parser.parse_args()

    # If --mode is passed, run worker mode (one emulator per process)
    if args.mode is not None:
        res = evaluate_single(args.model, args.mode, args.cooldown, recurrent=args.recurrent)
        print(f"RESULT_JSON:{json.dumps(res)}")
        return

    # Master orchestrator mode: spawn subprocess for each config to obey "one DeSmuME per process"
    print(f"=== Testing model: {args.model} (recurrent={args.recurrent}) ===")

    configs = [
        ("baseline", 0),
        ("macro_edge", 0),
        ("stride_cooldown", 1),
        ("stride_cooldown", 2),
        ("stride_cooldown", 3),
        ("stride_cooldown", 4),
        ("stride_cooldown", 5),
    ]

    py = sys.executable
    script = os.path.abspath(__file__)

    for mode, cd in configs:
        cmd = [py, script, "--model", args.model, "--mode", mode, "--cooldown", str(cd)]
        if args.recurrent:
            cmd.append("--recurrent")
        out = subprocess.check_output(cmd, text=True)
        for line in out.splitlines():
            if line.startswith("RESULT_JSON:"):
                res = json.loads(line[len("RESULT_JSON:"):])
                print(f"[{mode:15s} cd={cd}] max_x={res['max_x_px']:7.1f} px | steps={res['steps']:3d} | term={res['termination']:7s} | ret={res['return']:6.2f} | clear={res['cleared']}")


if __name__ == "__main__":
    main()
