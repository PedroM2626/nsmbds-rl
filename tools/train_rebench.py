"""Train one arm of the v2 re-benchmark (PROJECT_STATUS 4az).

Every arm: v2 actions (8), simple potential reward, stall 30, SubprocVecEnv
of --num-envs emulators, and ONE shared PPO hyperparameter set (below) --
the architecture is the only thing an arm changes. Outputs:

  models/rebench/<arm>_final.zip   last weights
  models/rebench/<arm>_best.zip    weights at the training max-x record
  evals/rebench/<arm>_training_log.json   per-episode history + provenance

  .venv\\Scripts\\python.exe tools\\train_rebench.py --arm rppo_curl --timesteps 100000
"""
from __future__ import annotations

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import rebench_common as rc  # noqa: E402  (sets sys.path for src/)

# One hyperparameter set for every arm (the 4as visual PPO values).
HPARAMS = dict(learning_rate=2.5e-4, n_steps=256, batch_size=64, n_epochs=4,
               gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", required=True, choices=sorted(rc.ARMS))
    ap.add_argument("--timesteps", type=int, default=100_000)
    ap.add_argument("--num-envs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None, help="default: cuda for pixel arms, cpu otherwise")
    args = ap.parse_args()

    cfg = rc.ARMS[args.arm]
    import torch
    device = args.device or ("cuda" if cfg["obs"] == "pixels" and torch.cuda.is_available() else "cpu")
    torch.set_num_threads(2)

    os.makedirs(rc.MODEL_DIR, exist_ok=True)
    tag = args.arm + (f"_s{args.seed}" if args.seed else "")   # seed 0 keeps the 4az names
    final_path = os.path.join(rc.MODEL_DIR, f"{tag}_final")
    best_path = os.path.join(rc.MODEL_DIR, f"{tag}_best")
    log_path = os.path.join(rc.EVAL_DIR, f"{tag}_training_log.json")

    # Emulators first (AGENTS.md: DeSmuME before any model construction).
    env = rc.make_vec_env(cfg["obs"], args.num_envs)
    if cfg.get("icm"):
        from icm import ICMVecEnvWrapper
        env = ICMVecEnvWrapper(env, update_freq=8)

    policy, policy_kwargs = rc.policy_spec(args.arm)
    Algo = rc.algo_class(args.arm)
    model = Algo(policy, env, policy_kwargs=policy_kwargs, seed=args.seed,
                 device=device, verbose=0, **HPARAMS)
    n_params = sum(p.numel() for p in model.policy.parameters())

    logger = rc.ProgressLogger(args.arm, log_path, best_path)
    callbacks = [logger]
    if cfg.get("curl"):
        from curl import CURLCallback
        callbacks.append(CURLCallback(curl_lr=1e-4, batch_size=64, epochs=5))
    if cfg.get("spr"):
        from spr import SPRCallback
        callbacks.append(SPRCallback(spr_lr=1e-4, batch_size=64, epochs=5, k_steps=3))

    print(f"[{args.arm}] {Algo.__name__} policy={policy if isinstance(policy, str) else policy.__name__} "
          f"params={n_params} device={device} envs={args.num_envs} steps={args.timesteps}", flush=True)
    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, callback=callbacks)
    wall = time.time() - t0
    model.save(final_path)
    env.close()

    from eval_provenance import collect
    logger.dump(extra={
        "algo": Algo.__name__, "config": cfg, "hparams": HPARAMS, "num_envs": args.num_envs,
        "seed": args.seed, "device": device, "policy_params": n_params,
        "stall_limit": rc.STALL_LIMIT, "wall_min": wall / 60.0,
        "final_model": final_path + ".zip", "best_model": best_path + ".zip",
        "provenance": collect(rom=rc.paths.ROM_REL, state=rc.paths.STATE_REL,
                              extra={"seed": args.seed, "arm": args.arm,
                                     "timesteps": args.timesteps})})
    print(f"[{args.arm}] done in {wall/60:.1f} min, best max_x {logger.best_max_x:.1f} px, "
          f"first clear at step {logger.first_clear_step}", flush=True)


if __name__ == "__main__":
    main()
