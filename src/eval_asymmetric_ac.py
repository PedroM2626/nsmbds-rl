"""Evaluate the trained asymmetric-AC pair: mean px per episode, the same
ruler as the transfer tables. det + stoch, N eps each. ONE DeSmuME per
process (the eternal constraint), constructed BEFORE any torch load.

Writes a provenance-annotated evaluation artifact to evals/asymmetric_ac_eval.json.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import paths
from course_id import course_area_index, course_for_area_index
from dual_obs_env import DualObsEnv
from eval_provenance import collect as collect_provenance, sha1_file

ROOT = os.path.dirname(HERE)
ROM_DEFAULT = paths.resolve(paths.ROM_REL)
STATE_DEFAULT = paths.resolve(paths.STATE_REL)

MODELS = (
    ("pixel_critic", "pixel-critic (baseline)", "pixels", "models/ac_pixels_200k.zip"),
    ("ram_critic", "ram-critic (asymmetric)", "ram", "models/ac_ram_200k.zip"),
)


def evaluate_asymmetric_ac(
    rom_path: str = ROM_DEFAULT,
    state_path: str = STATE_DEFAULT,
    course_path: str = "course/A01_1.bin",
    episodes_det: int = 3,
    episodes_stoch: int = 3,
    max_steps: int = 1000,
    seed: int = 0,
    out_path: str | None = os.path.join(ROOT, "evals", "asymmetric_ac_eval.json"),
) -> dict:
    t0 = time.time()
    # HARD RULE (AGENTS.md): DeSmuME must be initialized BEFORE any torch/PPO load
    env = DualObsEnv(rom_path=rom_path, state_path=state_path, max_steps=max_steps)
    verified_course_index = None
    try:
        obs, _ = env.reset()
        expected_area = course_area_index(course_path) if course_path else None
        state_area = env.ram.course_index() if env.ram is not None else None
        if expected_area is not None and state_area is not None and state_area != expected_area:
            raise RuntimeError(
                f"course/state mismatch: --course {course_path} is area index "
                f"{expected_area}, but savestate is running area index "
                f"{state_area} ({course_for_area_index(state_area)})."
            )
        verified_course_index = state_area

        # Import torch & SB3 only AFTER the emulator is live
        import torch
        from stable_baselines3 import PPO
        from train_asymmetric_ac import build_policy_class

        results = {}
        for idx, (key, tag, value_input, rel_path) in enumerate(MODELS):
            abs_model = os.path.join(ROOT, rel_path)
            model = PPO.load(
                abs_model,
                policy=build_policy_class(value_input),
                device="cpu",
            )
            model_res = {
                "tag": tag,
                "value_input": value_input,
                "checkpoint": rel_path,
                "checkpoint_sha1": sha1_file(abs_model),
            }

            for mode_idx, (mode, det, n_eps) in enumerate((
                ("det", True, episodes_det),
                ("stoch", False, episodes_stoch),
            )):
                sub_seed = int(seed + 100 * idx + 10 * mode_idx)
                random.seed(sub_seed)
                np.random.seed(sub_seed % (2 ** 32))
                torch.manual_seed(sub_seed)

                pxs, lens, rews, clears = [], [], [], []
                for ep in range(n_eps):
                    obs, info = env.reset()
                    best_px = 0.0
                    total_r = 0.0
                    steps = 0
                    done = trunc = False
                    while not (done or trunc):
                        action, _ = model.predict(obs, deterministic=det)
                        obs, r, done, trunc, info = env.step(
                            int(action) if np.ndim(action) == 0 else int(action[0])
                        )
                        steps += 1
                        total_r += float(r)
                        best_px = max(best_px, float(info.get("mario_px", 0.0)))
                    pxs.append(best_px)
                    lens.append(int(steps))
                    rews.append(float(total_r))
                    clears.append(bool(best_px >= 4016.0))

                model_res[mode] = {
                    "episodes": int(n_eps),
                    "seed": sub_seed,
                    "max_x_px": [round(float(x), 2) for x in pxs],
                    "mean_x_px": round(float(np.mean(pxs)), 2),
                    "std_x_px": round(float(np.std(pxs)), 2),
                    "best_x_px": round(float(np.max(pxs)), 2),
                    "steps": lens,
                    "mean_steps": round(float(np.mean(lens)), 2),
                    "rewards": [round(float(r), 4) for r in rews],
                    "mean_reward": round(float(np.mean(rews)), 4),
                    "clear_rate": round(float(np.mean(clears)), 3),
                }
                print(
                    f"{tag} {mode}: mean px {np.mean(pxs):7.1f}  "
                    f"max {np.max(pxs):7.1f}  steps {lens}",
                    flush=True,
                )
            results[key] = model_res
            del model
    finally:
        env.close()

    payload = {
        "protocol": {
            "course": paths.display(course_path),
            "episodes_det": int(episodes_det),
            "episodes_stoch": int(episodes_stoch),
            "max_steps": int(max_steps),
            "seed": int(seed),
            "action_space": "v2_8_actions",
            "actor_input": "pixels_1x84x84",
        },
        "elapsed_s": round(time.time() - t0, 2),
        "results": results,
        "provenance": collect_provenance(
            rom=rom_path,
            state=state_path,
            extra={
                "course": paths.display(course_path),
                "verified_course_index": verified_course_index,
                "episodes_det": int(episodes_det),
                "episodes_stoch": int(episodes_stoch),
                "max_steps": int(max_steps),
                "seed": int(seed),
            },
        ),
    }
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved -> {out_path}", flush=True)
    return payload


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rom", default=ROM_DEFAULT)
    ap.add_argument("--state", default=STATE_DEFAULT)
    ap.add_argument("--course", default="course/A01_1.bin")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--episodes-stoch", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out",
        default=os.path.join(ROOT, "evals", "asymmetric_ac_eval.json"),
    )
    args = ap.parse_args()
    try:
        evaluate_asymmetric_ac(
            rom_path=args.rom,
            state_path=args.state,
            course_path=args.course,
            episodes_det=args.episodes,
            episodes_stoch=args.episodes_stoch,
            max_steps=args.max_steps,
            seed=args.seed,
            out_path=args.out,
        )
    except RuntimeError as e:
        print(f"REFUSING TO EVALUATE: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
