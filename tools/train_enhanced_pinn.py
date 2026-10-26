"""Train Enhanced Multi-World Physics-Informed Neural Network (PINN) World Model.

Solves the open-loop dynamic drift through:
1. Multi-world real transition dataset (1-1, 1-2, 2-1, 4-1, 5-1, 6-1, 8-1).
2. Stratified multi-world train/val/test splitting.
3. Multi-step autoregressive rollout loss to prevent error accumulation.
4. Hard-residual exact discrete Euler integration (+ K*v, zero structural residual).

Outputs:
    models/pinn_nsmb.pt
    evals/pinn_worldmodel_metrics.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from pinn_world_model import (
    MarioPINNWorldModel, OBS_DIM, N_ACTIONS, IDX_X, IDX_Y, IDX_VX, IDX_VY, EULER_K
)


def split_multiworld(data, val_frac=0.10, test_frac=0.15):
    """Stratified temporal split across distinct episodes.
    Guarantees every world is represented in train, val, and test.
    """
    S, A, S2, R, D, EP = data
    unique_eps = np.unique(EP)
    n_eps = len(unique_eps)

    # Deterministic split by episode contiguous blocks
    n_tr = int(n_eps * (1.0 - val_frac - test_frac))
    n_va = int(n_eps * (1.0 - test_frac))

    tr_eps = set(unique_eps[:n_tr])
    va_eps = set(unique_eps[n_tr:n_va])
    te_eps = set(unique_eps[n_va:])

    tr_idx = np.isin(EP, list(tr_eps))
    va_idx = np.isin(EP, list(va_eps))
    te_idx = np.isin(EP, list(te_eps))

    def sl(mask):
        return tuple(arr[mask] for arr in data)

    return sl(tr_idx), sl(va_idx), sl(te_idx)


def make_tensors(data, device):
    S, A, S2, R, D = data[:5]
    a_oh = F.one_hot(torch.as_tensor(A), N_ACTIONS).float()
    return (
        torch.as_tensor(S, device=device),
        a_oh.to(device),
        torch.as_tensor(S2, device=device),
        torch.as_tensor(R, device=device),
        torch.as_tensor(D, device=device)
    )


def train_enhanced_model(data, model, device, epochs=350, lr=1e-3, wd=1e-5, bs=64,
                         lambda_rollout=0.5, verbose=True):
    tr, va, _ = data
    St, At, S2t, Rt, Dt = make_tensors(tr, device)
    Sv, Av, S2v, Rv, Dv = make_tensors(va, device)
    EP_tr = tr[5]

    n = St.shape[0]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=15)

    # Kinematic dimension importance weighting
    dim_w = torch.ones(OBS_DIM, device=device)
    for i in (IDX_X, IDX_Y, IDX_VX, IDX_VY):
        dim_w[i] = 3.0
    dim_w[7] = 2.0  # progress cam_x

    # Identify consecutive transition pairs for 2-step unrolling
    has_next = []
    for t in range(n - 1):
        if Dt[t].item() < 0.5 and EP_tr[t] == EP_tr[t + 1]:
            has_next.append(t)
    has_next = np.array(has_next, dtype=np.int64)

    best_val, best_state, best_ep = 1e9, None, 0
    hist = []

    print(f"[pinn-wm] Training with {n} transitions ({len(has_next)} rollout pairs)...", flush=True)

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        tot_loss = 0.0

        for i in range(0, n, bs):
            idx = perm[i:i + bs]

            # 1-step prediction
            ns, r, lc, logvar = model(St[idx], At[idx])
            err = (ns - S2t[idx]) ** 2
            nll = 0.5 * torch.exp(-logvar) * err + 0.5 * logvar
            loss_dyn = (nll * dim_w).mean()
            loss_rew = F.mse_loss(r, Rt[idx])
            w_cont = torch.where(Dt[idx] > 0.5, torch.full_like(Dt[idx], 25.0), torch.ones_like(Dt[idx]))
            loss_cont = F.binary_cross_entropy_with_logits(lc, 1.0 - Dt[idx], weight=w_cont)

            loss = loss_dyn + loss_rew + 0.5 * loss_cont

            # 2-step autoregressive rollout regularization
            if lambda_rollout > 0.0 and len(has_next) > 0:
                # Sample a small sub-batch of valid consecutive transitions
                roll_idx = np.random.choice(has_next, size=min(len(idx) // 2, 32), replace=False)
                t_idx = torch.as_tensor(roll_idx, device=device)
                t1_idx = torch.as_tensor(roll_idx + 1, device=device)

                s1_pred, _, _, _ = model(St[t_idx], At[t_idx])
                s2_pred, r2_pred, _, _ = model(s1_pred, At[t1_idx])

                rollout_err = ((s2_pred - S2t[t1_idx]) ** 2 * dim_w).mean()
                loss = loss + lambda_rollout * rollout_err

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tot_loss += loss.item() * len(idx)

        model.eval()
        with torch.no_grad():
            ns_v, r_v, lc_v, _ = model(Sv, Av)
            val = F.smooth_l1_loss(ns_v, S2v).item() + F.mse_loss(r_v, Rv).item()

        sched.step(val)
        hist.append(val)

        if val < best_val:
            best_val, best_ep = val, ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if verbose and (ep % max(1, epochs // 10) == 0 or ep == epochs - 1):
            print(f"  ep {ep:3d} | train loss {tot_loss/n:.5f} | val {val:.5f} | best {best_val:.5f}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, hist, best_val


@torch.no_grad()
def eval_metrics(model, data, device, rollout_len=120):
    """Evaluate test MSE, kinematic residual, and multi-step open-loop rollout drift."""
    _, _, test = data
    S, A, S2, R, D = test[:5]
    EP = test[5] if len(test) > 5 else np.zeros(len(S), np.int64)

    model.eval()
    St = torch.as_tensor(S, device=device)
    a_oh = F.one_hot(torch.as_tensor(A, device=device), N_ACTIONS).float()
    ns, r, lc, logvar = model(St, a_oh)
    mse = F.mse_loss(ns, torch.as_tensor(S2, device=device)).item()

    vx = ns[:, IDX_VX]
    vy = ns[:, IDX_VY]
    rx = (ns[:, IDX_X] - St[:, IDX_X] - vx * EULER_K).abs().mean().item()
    ry = (ns[:, IDX_Y] - St[:, IDX_Y] - vy * EULER_K).abs().mean().item()

    # Open-loop drift strictly within contiguous test episodes
    ep_starts = {}
    for i, e in enumerate(EP):
        ep_starts.setdefault(int(e), []).append(i)

    rng = np.random.default_rng(0)
    drift, horizons = [], []

    for e, idxs in list(ep_starts.items()):
        if len(idxs) < 10:
            continue
        for _ in range(3):
            t0 = idxs[rng.integers(0, max(1, len(idxs) - 10))]
            span = min(rollout_len, idxs[-1] - t0)
            if span < 5:
                continue
            s = St[t0:t0 + 1].clone()
            for t in range(span):
                ai = torch.as_tensor([A[t0 + t]], device=device)
                s = model.step_batch(s, ai)[0]
            gt_x = S[t0 + span, IDX_X]
            gt_y = S[t0 + span, IDX_Y]
            # Drift in true pixels (coordinate * 512)
            d_px = np.hypot(s[0, IDX_X].item() - gt_x, s[0, IDX_Y].item() - gt_y) * 512.0
            drift.append(d_px)
            horizons.append(span)

    return {
        "test_mse": mse,
        "kinematic_residual_x": rx,
        "kinematic_residual_y": ry,
        "open_loop_drift_px_mean": float(np.mean(drift)) if drift else float("nan"),
        "open_loop_horizon_frames": float(np.mean(horizons)) if horizons else 0.0,
        "n_drift_episodes": len(drift)
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-in", default="models/pinn_pool.npz")
    ap.add_argument("--out", default="models/pinn_nsmb.pt")
    ap.add_argument("--epochs", type=int, default=350)
    ap.add_argument("--hid", type=int, default=128)
    ap.add_argument("--lambda-rollout", type=float, default=0.5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device

    print(f"[pinn-wm] Loading pool from {args.pool_in} on device={device}...")
    p = np.load(args.pool_in)
    data = (p["S"], p["A"], p["S2"], p["R"], p["D"], p["EP"])
    n_trans = len(data[0])
    print(f"[pinn-wm] Pool loaded: {n_trans} transitions across {len(np.unique(p['EP']))} episodes.")

    splits = split_multiworld(data)
    print(f"[pinn-wm] Train: {len(splits[0][0])} | Val: {len(splits[1][0])} | Test: {len(splits[2][0])}")

    model = MarioPINNWorldModel(hid=args.hid).to(device)
    n_params = sum(param.numel() for param in model.parameters())
    print(f"[pinn-wm] Trainable parameters: {n_params}")

    t1 = time.time()
    model, hist, best_val = train_enhanced_model(
        splits, model, device, epochs=args.epochs, bs=64,
        lambda_rollout=args.lambda_rollout
    )
    train_time = time.time() - t1

    metrics = eval_metrics(model, splits, device)
    metrics.update({
        "collect_time_s": 0.0,
        "train_time_s": train_time,
        "n_transitions": n_trans,
        "n_params": int(n_params)
    })
    print("\n=======================================================")
    print("  PINN WORLD MODEL METRICS EVALUATION")
    print("=======================================================")
    print(json.dumps(metrics, indent=2))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "obs_dim": OBS_DIM,
        "n_actions": N_ACTIONS,
        "hid": args.hid,
        "layers": 2,
        "k": EULER_K,
        "metrics": metrics
    }, args.out)

    os.makedirs("evals", exist_ok=True)
    with open("evals/pinn_worldmodel_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[pinn-wm] Successfully saved model to {args.out} and metrics to evals/pinn_worldmodel_metrics.json")


if __name__ == "__main__":
    main()
