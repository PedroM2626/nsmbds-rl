# models/ — artifact index

`models/checkpoints/` and every `*_<n>_steps.zip` are intermediate training
snapshots: local scratch, gitignored, never load-bearing. The canonical
artifacts are the ones below (the files this README lists are the tracked set).

| Artifact | Role |
|---|---|
| `pinn_nsmb.pt` | Hard-residual PINN world model (43,440 params; MSE 0.0555) |
| `pinn_policy.zip` | Dyna-PPO policy trained in PINN imagination (200k steps) |
| `pinn_pool.npz` | 3,500 real RAM transitions (46 episodes; x max 2,601 px) |
| `ac_ram_200k.zip` | Asymmetric AC, **RAM-critic** arm — det 1,250.5 px, stoch max 2,610.9 (n=3, PROJECT_STATUS 4ac) |
| `ac_pixels_200k.zip` | Asymmetric AC, pixel-critic baseline — det 88.0 px (the 14x contrast arm) |
| `ram_ppo_100k.zip` / `ram_geo_100k.zip` / `gnn_100k.zip` | RAM-arm 100k baselines (17/23-dim vector, entity graph) |
| `ppo_*_100k.zip` (8 files) | June visual baselines — **legacy 6-action, no dash** (PROJECT_STATUS 4af); flow-reward era |
| `recurrent_ppo_mario.zip`, `mario_impala.zip` | Long-run visual baselines (same legacy caveat) |
| `impala_*_1m.zip`, `impala_transformer_spr_1m.zip` | 1M-step IMPALA runs (same legacy caveat) |
| `autoencoder.pth` | Conv-AE used for frozen-latent PPO and the world-models V tier |
| `worldmodels_ga/cma_100k*` | Ha&Schmidhuber-style world models (GA / sep-CMA-ES controllers) |

**Not reproducible from a clone:** `mlflow.db` and `mlruns/` are gitignored, so
the 34-run training ledger (PROJECT_STATUS 4n) lives only on the dev machine;
the cards in `docs/MODEL_CARDS.md` are the portable record.

## What a clone contains (post-4ag, 2026-10-23)

The June-era visual binaries (the `ppo_*`, `impala_*`, `mario_impala`,
`recurrent_ppo`, `autoencoder.pth`, `worldmodels_*_memory.pth`, `test_*.zip`
set, ~586 MB) were **removed from git history and HEAD** by the 4ag rewrite:
they are legacy-action/no-dash, flow-reward-era artifacts whose measured
results are already recorded in `evals/` and the docs. Like the ROM under
`data/`, they live only on the dev machine. The media files and the AC pair
are served through **git-lfs** (see `.gitattributes`); a fresh clone needs
`git lfs pull` for those.

Tracked after the slim: the PINN trio (`pinn_nsmb.pt`, `pinn_policy.zip`,
`pinn_pool.npz`), the RAM baselines (`ram_ppo_100k`, `ram_geo_100k`,
`ram_ppo_smoke`), `gnn_100k.zip`, the world-model `*.npz` controllers, and
`ac_pixels_200k.zip` / `ac_ram_200k.zip` via LFS.

**Measured, before -> after (4ag):** git pack **2.04 GiB -> 4.72 MiB**
(`git count-objects -vH`; 109 commits preserved, working tree byte-identical,
all four gates green on the rewritten HEAD). The pre-rewrite history is archived
at `D:/mario-ds-backup-2026-10-23/mario-ds-full-history.bundle` (0.79 GB,
verified) alongside the local-only model binaries and `media/`.