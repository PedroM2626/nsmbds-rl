"""Evaluate a checkpoint under the v2 re-benchmark protocol (PROJECT_STATUS 4az).

Protocol (identical for every arm):
  * env: the arm's observation over the v2 action table, the simple potential
    reward and the 30-step stall truncation used in training, 1000-step cap
  * det:   --det-eps argmax episodes (the emulator and argmax are deterministic,
           so they are expected to be identical -- the JSON records whether
           they were)
  * stoch: 3 blocks x 10 sampled episodes, torch/numpy seeded 100/200/300
           (the n30 layout of the historical benchmark)
  * per episode: max_x_px, final_x_px, steps, potential return, cleared,
    termination (clear | death | stall | timeout | step_cap)

  .venv\\Scripts\\python.exe tools\\eval_rebench.py --arm rppo_curl --which best
  .venv\\Scripts\\python.exe tools\\eval_rebench.py --model models/visual_simple_best.zip \\
        --obs pixels --algo ppo --name ref_visual_simple
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import rebench_common as rc  # noqa: E402

STOCH_SEEDS = (100, 200, 300)
T_975_DF29 = 2.045  # two-sided 95% Student-t critical value, n = 30


def run_episode(env, act, reset_policy):
    obs, info = env.reset()
    reset_policy()
    max_x = final_x = float(info.get("mario_px", 0.0) or 0.0)
    ret, steps = 0.0, 0
    while True:
        obs, r, done, trunc, info = env.step(act(obs))
        steps += 1
        ret += float(r)
        x = float(info.get("mario_px", 0.0))
        final_x = x
        max_x = max(max_x, x)
        if done or trunc:
            break
    if info.get("level_cleared", False):
        term = "clear"
    elif info.get("died", False):
        term = "death"
    elif info.get("timeout", False):
        term = "timeout"
    elif steps >= env.unwrapped.max_steps:
        term = "step_cap"
    else:
        term = "stall"
    return {"max_x_px": max_x, "final_x_px": final_x, "steps": steps,
            "return": ret, "cleared": term == "clear", "termination": term}


def sb3_actor(model, recurrent, deterministic):
    state = {"lstm": None, "start": np.ones((1,), dtype=bool)}

    def reset():
        state["lstm"], state["start"] = None, np.ones((1,), dtype=bool)

    def act(obs):
        if recurrent:
            a, state["lstm"] = model.predict(obs, state=state["lstm"],
                                             episode_start=state["start"],
                                             deterministic=deterministic)
            state["start"] = np.zeros((1,), dtype=bool)
        else:
            a, _ = model.predict(obs, deterministic=deterministic)
        return int(a)
    return act, reset


def wm_actor(variant):
    """World Models controller (argmax only: it has no sampling distribution)."""
    import torch
    import train_rebench_worldmodels as wmr
    vec = np.load(os.path.join(rc.MODEL_DIR, f"wm_{variant}_best.npz"))["vec"]
    vision = wmr.build_vision(variant)
    memory = wmr.build_memory()
    memory.load_state_dict(torch.load(os.path.join(rc.MODEL_DIR, f"wm_{variant}_memory.pth")))
    memory.eval()
    L, H, A = wmr.LATENT, wmr.HIDDEN, wmr.N_ACT
    W, b = vec[: (L + H) * A].reshape(A, L + H), vec[(L + H) * A:]
    st = {}

    def reset():
        st["h"] = st["c"] = torch.zeros(1, 1, H)

    @torch.no_grad()
    def act(obs):
        x = torch.from_numpy(obs.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        z = vision(x)[0].numpy()
        a = int(np.argmax(W @ np.concatenate([z, st["h"][0, 0].numpy()]) + b))
        inp = torch.cat([torch.from_numpy(z)[None, None],
                         torch.nn.functional.one_hot(torch.tensor([[a]]), A).float()], -1)
        _, (st["h"], st["c"]) = memory.lstm(inp, (st["h"], st["c"]))
        return a
    return act, reset


REFS = {  # earlier potential-reward policies (4ao-4at), re-measured under this protocol
    "ref_genetic":        ("ram23", "models/genetic_policy.npz"),
    "ref_iqn":            ("ram23", "models/iqn_policy.pt"),
    "ref_rainbow":        ("ram23", "models/rainbow_policy.pt"),
    "ref_visual_genetic": ("pixels", "models/visual_genetic_best.pt"),
}


def ref_actor(name, path):
    """Argmax actors of the non-SB3 reference policies (their own eval code's math)."""
    import torch
    if name == "ref_genetic":
        d = np.load(path, allow_pickle=True)
        w1, b1, w2, b2 = d["w1"], d["b1"], d["w2"], d["b2"]
        return (lambda obs: int(np.argmax(w2 @ np.tanh(w1 @ obs + b1) + b2))), (lambda: None)
    if name in ("ref_iqn", "ref_rainbow"):
        if name == "ref_iqn":
            from iqn import IQNNetwork as Net
        else:
            from rainbow_dqn import RainbowNetwork as Net
        net = Net(obs_dim=23, n_actions=8, hidden_dim=128)
        net.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model_state"])
        net.eval()

        @torch.no_grad()
        def act(obs):
            q = net.get_q_values(torch.as_tensor(obs, dtype=torch.float32)[None])
            return int(q.argmax(dim=-1).item())
        return act, (lambda: None)
    from train_visual_neuroevolution import VisualIndividual
    ind = VisualIndividual(weights=torch.load(path, map_location="cpu", weights_only=False)["theta"])
    return (lambda obs: ind.act_batch(obs[None])[0]), (lambda: None)


def summarize(eps):
    xs = np.array([e["max_x_px"] for e in eps], dtype=float)
    n = len(xs)
    sd = float(xs.std(ddof=1)) if n > 1 else 0.0
    half = T_975_DF29 * sd / math.sqrt(n) if n == 30 else None
    terms = {}
    for e in eps:
        terms[e["termination"]] = terms.get(e["termination"], 0) + 1
    return {"n": n, "mean_max_x_px": float(xs.mean()), "sd_max_x_px": sd,
            "ci95_halfwidth_px": half, "median_max_x_px": float(np.median(xs)),
            "max_max_x_px": float(xs.max()), "min_max_x_px": float(xs.min()),
            "clears": int(sum(e["cleared"] for e in eps)),
            "clear_rate": float(np.mean([e["cleared"] for e in eps])),
            "mean_return": float(np.mean([e["return"] for e in eps])),
            "mean_steps": float(np.mean([e["steps"] for e in eps])),
            "terminations": terms}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", choices=sorted(rc.ARMS) + ["wm_ga", "wm_cma"] + sorted(REFS))
    ap.add_argument("--which", default="best", choices=["best", "final"])
    ap.add_argument("--model", help="external SB3 checkpoint (with --obs/--algo/--name)")
    ap.add_argument("--obs", choices=["pixels", "ram17", "ram23", "graph"])
    ap.add_argument("--algo", choices=["ppo", "rppo"])
    ap.add_argument("--name")
    ap.add_argument("--det-eps", type=int, default=3)
    ap.add_argument("--stoch-per-seed", type=int, default=10)
    ap.add_argument("--out")
    ap.add_argument("--state", default=rc.paths.STATE_REL)
    args = ap.parse_args()

    if args.arm in ("wm_ga", "wm_cma"):
        obs_kind, algo, name = "pixels", "wm", args.arm
        model_path = os.path.join(rc.MODEL_DIR, f"{args.arm}_best.npz")
    elif args.arm in REFS:
        (obs_kind, model_path), algo, name = REFS[args.arm], "ref", args.arm
    elif args.arm:
        obs_kind, algo = rc.ARMS[args.arm]["obs"], rc.ARMS[args.arm]["algo"]
        model_path = os.path.join(rc.MODEL_DIR, f"{args.arm}_{args.which}.zip")
        name = f"{args.arm}_{args.which}"
    else:
        if not (args.model and args.obs and args.algo and args.name):
            ap.error("--model needs --obs, --algo and --name")
        obs_kind, algo, model_path, name = args.obs, args.algo, args.model, args.name
    out = args.out or os.path.join(rc.EVAL_DIR, f"eval_{name}.json")

    # Env first, then the model (AGENTS.md: DeSmuME before torch.load).
    env = rc.make_env(obs_kind, state=args.state)
    import torch
    torch.set_num_threads(2)
    t0 = time.time()
    if algo == "wm":
        det_actor, stoch_actor = wm_actor(args.arm[3:]), None
    elif algo == "ref":
        torch.manual_seed(0)   # IQN samples its quantile fractions
        det_actor, stoch_actor = ref_actor(name, model_path), None
    else:
        if algo == "rppo":
            from sb3_contrib import RecurrentPPO as Algo
        else:
            from stable_baselines3 import PPO as Algo
        import train_visual_impala  # noqa: F401  (class the 4at checkpoint pickles by reference)
        model = Algo.load(model_path, device="cpu")
        recurrent = algo == "rppo"
        det_actor = sb3_actor(model, recurrent, deterministic=True)
        stoch_actor = sb3_actor(model, recurrent, deterministic=False)

    det = []
    act, reset = det_actor
    for i in range(args.det_eps):
        det.append(run_episode(env, act, reset))
        print(f"[{name}] det {i+1}: {det[-1]}", flush=True)
    stoch = []
    for seed in (STOCH_SEEDS if stoch_actor else ()):
        act, reset = stoch_actor
        torch.manual_seed(seed)
        np.random.seed(seed)
        for i in range(args.stoch_per_seed):
            e = run_episode(env, act, reset)
            e["seed_block"] = seed
            stoch.append(e)
            print(f"[{name}] stoch s{seed} {i+1}: max_x {e['max_x_px']:.1f} {e['termination']}",
                  flush=True)
    env.close()

    keys = ("max_x_px", "final_x_px", "steps", "termination")
    det_identical = all(tuple(d[k] for k in keys) == tuple(det[0][k] for k in keys) for d in det)
    from eval_provenance import collect
    result = {
        "name": name, "arm": args.arm, "checkpoint": model_path.replace("\\", "/"),
        "obs": obs_kind, "algo": algo,
        "protocol": {"actions": "v2 (8)", "reward": "potential max-x /16, +10 clear, -1 death",
                     "stall_limit": rc.STALL_LIMIT, "step_cap": 1000,
                     "det_eps": args.det_eps, "stoch_seeds": list(STOCH_SEEDS),
                     "stoch_per_seed": args.stoch_per_seed,
                     "ci": "Student-t 95%, sd ddof=1"},
        "det": {"episodes": det, "identical": det_identical, **summarize(det)},
        "stoch": {"episodes": stoch, **summarize(stoch)} if stoch else None,
        "wall_s": time.time() - t0,
        "provenance": collect(rom=rc.paths.ROM_REL, state=args.state,
                              extra={"seed": list(STOCH_SEEDS), "checkpoint": model_path}),
    }
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=1)
    s = result["stoch"]
    tail = (f"stoch {s['mean_max_x_px']:.1f} +- {s['sd_max_x_px']:.1f} max {s['max_max_x_px']:.1f} "
            f"clears {s['clears']}/{s['n']}") if s else "stoch n/a (argmax controller)"
    print(f"[{name}] det {det[0]['max_x_px']:.1f} px ({det[0]['termination']}, "
          f"identical={det_identical}) | {tail} -> {out}", flush=True)


if __name__ == "__main__":
    main()
