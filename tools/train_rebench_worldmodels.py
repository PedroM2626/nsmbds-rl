"""World Models (Ha & Schmidhuber, 2018) arms of the v2 re-benchmark (4az).

Same V-M-C design as src/train_worldmodels_ga.py / _cma.py, re-run under
the re-benchmark ruler (v2 8 actions, simple potential reward, stall 30):

  V  frozen 512-d encoder: --variant ga  -> models/autoencoder.pth (as in 4-era GA)
                           --variant cma -> the hybrid AE+CURL encoder of
                              models/ppo_hybrid_ae_curl_100k.zip (as in the CMA run)
  M  LSTM(512+8 -> 256) trained on --mem-frames random-policy frames to predict z_{t+1}
  C  linear argmax(W[z; h] + b), 6,152 params, evolved by
       ga : elitist GA (pop 24, top 25 %, sigma 0.05, champion clone)
       cma: diagonal sep-CMA-ES (pop 24, sigma0 0.08)
     fitness = the episode's potential return (= (max_x - x0)/16 + 10 clear - 1 death)

The env-step budget (--timesteps) counts the memory collection AND every
evolution episode. Individuals are evaluated in --workers emulator processes.

  .venv\\Scripts\\python.exe tools\\train_rebench_worldmodels.py --variant ga --timesteps 100000
"""
from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
import os
import sys
import time
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import rebench_common as rc  # noqa: E402

LATENT, HIDDEN, N_ACT = 512, 256, 8
DIM = (LATENT + HIDDEN) * N_ACT + N_ACT


# ---------------------------------------------------------------- V and M
def build_vision(variant):
    import torch
    import train_worldmodels_ga as wm
    vision = wm.VisionEncoder()
    if variant == "ga":
        sd = torch.load(rc.paths.resolve("models/autoencoder.pth"), map_location="cpu")
        vision.encoder.load_state_dict({k[len("encoder."):]: v for k, v in sd.items()
                                        if k.startswith("encoder.")})
    else:
        with zipfile.ZipFile(rc.paths.resolve("models/ppo_hybrid_ae_curl_100k.zip")) as z:
            sd = torch.load(io.BytesIO(z.read("policy.pth")), map_location="cpu")
        pre = "features_extractor.encoder."
        enc = {k.replace(pre, "encoder."): v for k, v in sd.items() if k.startswith(pre)}
        assert len(enc) == 8, len(enc)
        vision.load_state_dict(enc)
    for p in vision.parameters():
        p.requires_grad = False
    return vision.eval()


def build_memory():
    import torch
    import torch.nn as nn

    class Memory(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(LATENT + N_ACT, HIDDEN, batch_first=True)
            self.pred_z = nn.Linear(HIDDEN, LATENT)

        def forward(self, z_seq, a_seq, hidden=None):
            a = torch.nn.functional.one_hot(a_seq, num_classes=N_ACT).float()
            out, hidden = self.lstm(torch.cat([z_seq, a], dim=-1), hidden)
            return self.pred_z(out), hidden
    return Memory()


def collect_random(env, n, seed):
    """Sequential random-policy frames, biased to the rightward actions 1-4."""
    rng = np.random.default_rng(seed)
    frames, acts, dones = [], [], []
    obs, _ = env.reset()
    for _ in range(n):
        a = int(rng.integers(1, 5)) if rng.random() < 0.6 else int(rng.integers(0, N_ACT))
        frames.append(obs.copy())
        acts.append(a)
        obs, _, d, tr, _ = env.step(a)
        dones.append(bool(d or tr))
        if d or tr:
            obs, _ = env.reset()
    return np.array(frames), np.array(acts, dtype=np.int64), np.array(dones, dtype=bool)


# ---------------------------------------------------------------- C rollout
def rollout(vec, env, vision, memory, max_steps=1000):
    """One argmax episode. Returns (fitness, steps, max_x, cleared)."""
    import torch
    W = vec[: (LATENT + HIDDEN) * N_ACT].reshape(N_ACT, LATENT + HIDDEN)
    b = vec[(LATENT + HIDDEN) * N_ACT:]
    obs, info = env.reset()
    hx = cx = torch.zeros(1, 1, HIDDEN)
    fit, steps, max_x, cleared = 0.0, 0, float(info.get("mario_px", 0.0) or 0.0), False
    with torch.no_grad():
        while steps < max_steps:
            x = torch.from_numpy(obs.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
            z = vision(x)[0].numpy()
            a = int(np.argmax(W @ np.concatenate([z, hx[0, 0].numpy()]) + b))
            obs, r, d, tr, info = env.step(a)
            inp = torch.cat([torch.from_numpy(z)[None, None],
                             torch.nn.functional.one_hot(torch.tensor([[a]]), N_ACT).float()], -1)
            _, (hx, cx) = memory.lstm(inp, (hx, cx))
            fit += float(r)
            steps += 1
            max_x = max(max_x, float(info.get("mario_px", 0.0)))
            if d or tr:
                cleared = bool(info.get("level_cleared", False))
                break
    return fit, steps, max_x, cleared


_W = {}


def _worker_init(rank_q, variant, mem_state):
    import torch
    torch.set_num_threads(1)
    rank = rank_q.get()
    _W["env"] = rc._EnvFactory("pixels", rank)()       # emulator first
    _W["vision"] = build_vision(variant)
    mem = build_memory()
    mem.load_state_dict(mem_state)
    _W["memory"] = mem.eval()


def _worker_eval(vec):
    return rollout(vec, _W["env"], _W["vision"], _W["memory"])


# ---------------------------------------------------------------- optimizers
def ga_step_factory(rng, pop_size=24, elite_frac=0.25, sigma=0.05):
    n_elite = max(1, int(pop_size * elite_frac))
    state = {"pop": rng.normal(0, 0.1, size=(pop_size, DIM))}

    def ask():
        return state["pop"]

    def tell(X, fits):
        order = np.argsort(-fits)
        elites = X[order[:n_elite]]
        kids = [elites[0].copy()]
        while len(kids) < pop_size:
            kids.append(elites[rng.integers(0, n_elite)] + rng.normal(0, sigma, size=DIM))
        state["pop"] = np.array(kids)
        return {}
    return ask, tell


def cma_step_factory(rng, pop_size=24, sigma0=0.08):
    """Diagonal sep-CMA-ES (Ros & Hansen 2008), the constants of src/train_worldmodels_cma.py."""
    lam, mu, n = pop_size, pop_size // 2, DIM
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    w /= w.sum()
    mueff = 1.0 / np.sum(w ** 2)
    cs = (mueff + 2) / (n + mueff + 5)
    cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
    c1 = 2 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
    chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
    st = {"m": rng.normal(0, 0.05, size=n), "sigma": sigma0, "C": np.ones(n),
          "ps": np.zeros(n), "pc": np.zeros(n), "gen": 0}

    def ask():
        Z = rng.normal(0, 1, size=(lam, n))
        return st["m"] + st["sigma"] * Z * np.sqrt(st["C"])

    def tell(X, fits):
        st["gen"] += 1
        order = np.argsort(-fits)
        m_old = st["m"].copy()
        st["m"] = (w[:, None] * X[order[:mu]]).sum(axis=0)
        Y = (X[order[:mu]] - m_old) / st["sigma"]
        y_w = (st["m"] - m_old) / st["sigma"]
        st["ps"] = (1 - cs) * st["ps"] + np.sqrt(cs * (2 - cs) * mueff) * y_w / np.sqrt(st["C"])
        nps = float(np.linalg.norm(st["ps"]))
        hsig = 1.0 if nps ** 2 / (1 - (1 - cs) ** (2 * st["gen"])) / n < 2 + 4 / (n + 1) else 0.0
        st["pc"] = (1 - cc) * st["pc"] + hsig * np.sqrt(cc * (2 - cc) * mueff) * y_w
        st["C"] = np.maximum((1 - c1 - cmu) * st["C"]
                             + c1 * (st["pc"] ** 2 + (1 - hsig) * cc * (2 - cc) * st["C"])
                             + cmu * (w[:, None] * Y * Y).sum(axis=0), 1e-12)
        st["sigma"] = float(np.clip(st["sigma"] * np.exp((cs / damps) * (nps / chiN - 1)), 1e-4, 1.0))
        return {"sigma": st["sigma"]}
    return ask, tell


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variant", required=True, choices=["ga", "cma"])
    ap.add_argument("--timesteps", type=int, default=100_000)
    ap.add_argument("--mem-frames", type=int, default=None, help="default 10000 (ga) / 5000 (cma)")
    ap.add_argument("--mem-epochs", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    mem_frames = args.mem_frames or (10_000 if args.variant == "ga" else 5_000)
    arm = f"wm_{args.variant}"

    t0 = time.time()
    env = rc._EnvFactory("pixels", 0)()                  # emulator before any torch model
    import torch
    torch.manual_seed(args.seed)
    import train_worldmodels_ga as wm
    vision = build_vision(args.variant)
    frames, acts, dones = collect_random(env, mem_frames, args.seed)
    env.close()
    memory = build_memory()
    mem_loss, _ = wm.train_memory(memory, frames, acts, dones, vision, torch.device("cpu"),
                                  epochs=args.mem_epochs)
    used = mem_frames
    mem_state = {k: v.cpu() for k, v in memory.state_dict().items()}
    os.makedirs(rc.MODEL_DIR, exist_ok=True)
    torch.save(mem_state, os.path.join(rc.MODEL_DIR, f"{arm}_memory.pth"))
    print(f"[{arm}] memory trained on {mem_frames} frames, loss {mem_loss:.6f}", flush=True)

    rng = np.random.default_rng(args.seed)
    ask, tell = (ga_step_factory if args.variant == "ga" else cma_step_factory)(rng)
    ctx = mp.get_context("spawn")
    rank_q = ctx.Queue()
    for r in range(args.workers):
        rank_q.put(r)
    history, best = [], {"fit": -1e18, "vec": None, "max_x": 0.0}
    first_clear = None
    with ctx.Pool(args.workers, initializer=_worker_init,
                  initargs=(rank_q, args.variant, mem_state)) as pool:
        gen = 0
        while used < args.timesteps:
            gen += 1
            X = ask()
            res = pool.map(_worker_eval, list(X))
            fits = np.array([r[0] for r in res])
            used += int(sum(r[1] for r in res))
            for x, (f, s, mx, cl) in zip(X, res):
                if f > best["fit"]:
                    best.update(fit=float(f), vec=x.copy(), max_x=float(mx))
                if cl and first_clear is None:
                    first_clear = used
            extra = tell(X, fits)
            rec = {"gen": gen, "env_steps": used, "gen_best_fit": float(fits.max()),
                   "gen_mean_fit": float(fits.mean()),
                   "gen_best_max_x_px": float(max(r[2] for r in res)),
                   "gen_clears": int(sum(r[3] for r in res)),
                   "best_fit": best["fit"], "best_max_x_px": best["max_x"],
                   "elapsed_min": (time.time() - t0) / 60.0, **extra}
            history.append(rec)
            print(f"[{arm}] gen {gen} steps {used}/{args.timesteps} gen_max_x "
                  f"{rec['gen_best_max_x_px']:.1f} best_max_x {best['max_x']:.1f} "
                  f"fit {fits.max():.2f}/{fits.mean():.2f}", flush=True)
            np.savez(os.path.join(rc.MODEL_DIR, f"{arm}_best.npz"), vec=best["vec"],
                     variant=args.variant, fitness=best["fit"])

    from eval_provenance import collect
    log = {"arm": arm, "variant": args.variant, "total_steps": used, "mem_frames": mem_frames,
           "mem_loss": mem_loss, "params": DIM, "best_fitness": best["fit"],
           "best_max_x_px": best["max_x"], "first_clear_step": first_clear,
           "wall_min": (time.time() - t0) / 60.0, "stall_limit": rc.STALL_LIMIT,
           "workers": args.workers, "seed": args.seed, "history": history,
           "provenance": collect(rom=rc.paths.ROM_REL, state=rc.paths.STATE_REL,
                                 extra={"seed": args.seed, "arm": arm})}
    os.makedirs(rc.EVAL_DIR, exist_ok=True)
    with open(os.path.join(rc.EVAL_DIR, f"{arm}_training_log.json"), "w") as f:
        json.dump(log, f, indent=1)
    print(f"[{arm}] done: {used} steps, best max_x {best['max_x']:.1f} px, "
          f"{(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
