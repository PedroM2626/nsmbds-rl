# Nintendo DS Reinforcement Learning Pipeline

This repository implements an academic and experimental Reinforcement Learning (RL) research framework designed for simulated Nintendo DS environments (specifically *New Super Mario Bros.*). The pipeline explores both computer-vision-based spatial tracking (bypassing volatile memory pointers via dense optical flow and differential frame subtraction) and direct emulator memory reverse engineering (extracting RAM telemetry, entity graphs, and ROM tile geometries).

The codebase is engineered with MLOps best practices and supports diverse learning paradigms: model-free policy gradients (PPO, RecurrentPPO, ImpalaCNN), self-supervised auxiliary visual representations (Autoencoders, CURL, SPR, DrQ-v2), intrinsic curiosity (ICM), causal sequence modeling (Causal Transformers), evolutionary and model-based World Models (DreamerV3, NE-Dreamer, GA, sep-CMA-ES), and relational Graph Neural Networks (MeanMPNN).

### World 1-1, start to flagpole — the whole run in one clip

![World 1-1 clear dashboard, closing 10 s](media/nsmb_dashboard_preview.gif)

The animation above is the closing 10 s of the run — its last frame is the
stage-clear one. The full clip (27 s, 1280×720) is
`media/nsmb_dashboard_clear.mp4` — GitHub does not stream video out of a
repository tree, so open or download it from
[`media/nsmb_dashboard_clear.mp4`](https://github.com/PedroM2626/nsmbds-rl/blob/main/media/nsmb_dashboard_clear.mp4).

**Every recording in this repository, and what it proves:**

| File | What it shows | Status |
|---|---|---|
| [`media/nsmb_dashboard_clear.mp4`](media/nsmb_dashboard_clear.mp4) | the full 33 s World 1-1 clear with all panels (206 decisions, 4,020 of 4,032 px) | current |
| [`media/nsmb_dashboard_preview.gif`](media/nsmb_dashboard_preview.gif) | the closing 10 s of the same run, inline above | current |
| [`media/nsmb_w11_genetic_clear.mp4`](media/nsmb_w11_genetic_clear.mp4) | the 100% World 1-1 clear evolved via pure Neuroevolution (206 decisions, 4,020 px, 1,032 params) | current |
| [`media/nsmb_w11_genetic_preview.gif`](media/nsmb_w11_genetic_preview.gif) | the closing 10 s of the neuroevolutionary clear | current |
| [`media/nsmb_pipe_top_timesup.png`](media/nsmb_pipe_top_timesup.png) | the frame + velocity trace of the pipe trap: a standstill jump that never advances, timer expired | current |
| [`media/nsmb_w41_dashboard_clear.mp4`](media/nsmb_w41_dashboard_clear.mp4) | the full World 4-1 clear with Dorrie swamp navigation and all panels (106 decisions, 4,344 px) | current |
| [`media/nsmb_w41_dashboard_preview.gif`](media/nsmb_w41_dashboard_preview.gif) | the closing 10 s of the World 4-1 run | current |
| [`media/nsmb_w51_dashboard_clear.mp4`](media/nsmb_w51_dashboard_clear.mp4) | the full World 5-1 clear on ice/snow with all panels (300 decisions, 4,536 px) | current |
| [`media/nsmb_w51_dashboard_preview.gif`](media/nsmb_w51_dashboard_preview.gif) | the closing 10 s of the World 5-1 run | current |
| [`media/nsmb_w61_dashboard_clear.mp4`](media/nsmb_w61_dashboard_clear.mp4) | the full World 6-1 clear with marine trench exit and all panels (133 decisions, 4,119 px) | current |
| [`media/nsmb_w61_dashboard_preview.gif`](media/nsmb_w61_dashboard_preview.gif) | the closing 10 s of the World 6-1 run | current |
| [`media/nsmb_w81_dashboard_clear.mp4`](media/nsmb_w81_dashboard_clear.mp4) | the full World 8-1 clear with all panels (205 decisions, 3,910 of 3,920 px) | current |
| [`media/nsmb_w81_dashboard_preview.gif`](media/nsmb_w81_dashboard_preview.gif) | the closing 10 s of the World 8-1 run | current |
| [`media/pinn_reflex_astar_run.avi`](media/pinn_reflex_astar_run.avi) | reflex + A\* planner run, raw gameplay | historical, **red/blue swapped** |
| [`media/pinn_reflex_pipe.avi`](media/pinn_reflex_pipe.avi) | the pipe trap in motion: timer reaches 0001 with Mario still at the pipe base | historical, **red/blue swapped** |
| [`media/pinn_mpc_run.avi`](media/pinn_mpc_run.avi) | CEM-MPC run over the learned model | historical, **red/blue swapped** |

The three `.avi` files were recorded before the display-buffer fix
(`src/eval_pinn_realtime.py` used to reverse channels the other way), so their
red and blue are exchanged: measured sky in them is R≈247/B≈43 where the
corrected clip reads B≈247/R≈70, and their coins render blue. The runs they
recorded are real and the trajectories are unaffected — the pixels are the only
thing inverted — but read them as history, not as the current look.

It records one episode of the **learned neural policy** (Neuroevolution champion MLP, 1,032 parameters, `models/genetic_policy.npz`) on the live DeSmuME core. Every panel is read
from the same step loop that drives the emulator, so the frame is the evidence:

| on screen | where it comes from |
| --- | --- |
| gameplay, HUD, flagpole | DeSmuME top screen, `display_buffer_as_rgbx()` |
| which stage, px travelled, % of the level | `info["mario_px"]` over the 4032 px flagpole (`eval_pinn_realtime.goal_px()`) |
| the tile grid + pit flags + enemies + floating blocks | the 23-float observation of `src/ram_env.py` + `out/nsmb_geometry.bin` + the parser's bgdat block rects; enemies are anchored to Mario's *live* height and split by the measured hostility set -- hostile types red, coins/items/entities violet, off-window ones tick the panel edge; the ?/brick rows the game renders above the surface draw as one parser-identified class |
| action share (%) over the last 60 decisions | the controller's own action history |
| buttons held this step | `MarioRamEnv._action_keys` |
| reward prediction (green) vs actual (grey) | `models/pinn_nsmb.pt` `step_batch()` reward head |
| SciML panel (reach, units, pits) | `src/tilemap_planner.py` + `out/nsmb_physics.json` |

Measured on this run: **206 steps, 4020 px of 4032 px, stage-clear flag raised** —
the same numbers `evals/pinn_realtime_eval.json` reports, because the recorder
now stops on the criterion the eval uses. The grey spike at the end is the `+100`
terminal bonus the PINN does not predict.

```bash
python tools/export_geometry.py                      # ROM -> out/nsmb_geometry.bin
bash rust/build.sh && python tools/parity_planner.py # Rust == tilemap_planner.py
python tools/record_dashboard.py --controller genetic      # -> mp4 + the inline .gif above
```

**Not only 1-1.** Clearing a level hands the game back to its World 1 map, and
`tools/reach_map.py` turns any unlocked node into a savestate, so the same
pipeline runs on a second course without any hand-made state.
[`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md) §4i and §4j report that route,
the corrected course-file-name mapping, and the first transfer numbers.

## Contents
### World 8-1, start to flagpole — the final gauntlet cleared

![World 8-1 clear dashboard, closing 10 s](media/nsmb_w81_dashboard_preview.gif)

The animation above is the closing 10 s of the World 8-1 run (205 steps, 3,910 px of 3,920 px) touching the flagpole, executed at native 60 Hz by the **Learned Neural Policy** (`models/neural_policy_w81.pt`, MLP 64x64). The full clip with all SciML panels and synchronized spatial grid is [`media/nsmb_w81_dashboard_clear.mp4`](media/nsmb_w81_dashboard_clear.mp4).

- [Prerequisites and Installation](#-prerequisites-and-installation)
- [Visual Reward Methodology](#-visual-reward-methodology)
- [Training Architectures & Paradigms](#-training-architectures--paradigms) (1–17, incl. [§14 PINN](#14-physics-informed-pinn-world-model--fast-training--real-time-play), [§15 Neuroevolution](#15-compact-neuroevolution-genetic-algorithm), [§16 IQN](#16-implicit-quantile-networks-iqn), [§17 Rainbow DQN](#17-rainbow-dqn--pure-potential-maximum-reward))
- [Experiment Management (MLOps)](#-experiment-management-mlops)
- [Acknowledgments](#-acknowledgments)

---

## 🛠 Prerequisites and Installation

A dedicated virtual environment (conda or venv) is recommended for dependency isolation:

```bash
pip install gymnasium stable-baselines3[extra] sb3-contrib opencv-python py-desmume mlflow torch torchvision tensorboard sheeprl ndspy tqdm numpy
```

> **Notice:** The `py-desmume` package provides Python bindings for the DeSmuME Nintendo DS emulation core. Without it, the environment falls back to static dummy stubs for non-emulated unit testing.

### Directory Structure

```
mario-ds/
├── data/         # NSMB (EUR) ROM, savestate (.ds1), collected dataset (.npz), test trajectories
├── src/          # all environments, models, training + evaluation scripts (flat, import each other)
│   ├── env.py / course.py / ram_env.py / ram_state.py   # DeSmuME env, ROM tile parser, RAM env/state
│   ├── train*.py / evaluate*.py                          # per-paradigm trainers + benchmark eval
│   └── pinn_world_model.py / tilemap_planner.py          # physics-informed world model + A*/surface planner
├── nedreamer/    # NE-Dreamer (vendored in-tree, not a submodule) — see Acknowledgments
├── models/       # saved policies/checkpoints (.zip SB3, .pt torch, .npz world models)
├── evals/        # benchmark result JSONs + PINN real-time eval output
├── media/        # recorded gameplay clips (.avi) + evidence frames (.png)
├── docs/         # deep-dive methodology (e.g. WORLD_MODEL_PINN.md)
└── tensorboard_logs/ mlruns/ mlflow.db   # experiment tracking
```

Place the target ROM binary and initial episode savestate in `data/`:
- `data/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds` (or symlink `data/mario.nds`)
- `data/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).ds1` (or `data/state.dst`)

> **NE-Dreamer** lives in-tree at `nedreamer/` (a first-class package, not a git submodule). Run its
> Mario config via `python src/train_nedreamer_mlflow.py` or from `nedreamer/train.py` with
> `--config configs/env/marionds.yaml`. Credit to the upstream project — see [Acknowledgments](#-acknowledgments).

---

## 🔬 Visual Reward Methodology

> **Status (2026-10-13):** the default reward on the visual arm is now the
> **unified RAM reward** (camera-RAM progress, RAM-lives death, RAM clear
> flag) — the same one the RAM arm trains on, so the two arms differ only in
> observations and their numbers are comparable. The visual tracker below is
> retained as the legacy opt-in (`reward_mode="flow"`) and as redundant
> terminal detection; the June-era runs and the n30 benchmark rows were
> measured under it. See PROJECT_STATUS 4s for the full reward ledger.

To avoid rigid coupling with volatile emulator RAM addresses, the baseline visual reward function combines two computer vision heuristics operating on top-screen grayscale observation tensors ($84 \times 84$):

1. **Dense Optical Flow (Farneback Method):** Computes pixel velocity vectors across the background. When the virtual camera tracks the agent to the right, the background shifts leftward, yielding an advance gradient along the X-axis. Directional median filtering isolates background translation from foreground agent motion.
2. **Differential Frame Subtraction (AbsDiff):** Solves the mechanical camera "dead zone" problem, where the agent moves across the screen while the camera remains stationary. Thresholding on frame differences (`cv2.absdiff`) isolates the active entity, extracts its geometric center, and fuses this local displacement into the global camera odometer.

This hybrid visual tracker constructs a synthetic global X coordinate. It prevents *Reward Hacking* failure modes where agents deliberately triggered premature episode termination to exploit camera dead-zone estimation gaps.

---

## 🚀 Training Architectures & Paradigms

All training routines can be invoked via CLI flags.

> **⚠️ Action-mapping correction (2026-10-23) — the visual arm's "dash" was never a dash.**
> The June visual env (src/env.py) labeled KEY_B as "dash". Measured on the 1-1
> savestate (one emulator, state reload per trial): **B is a jump button** — it produces
> the *identical* standing jump as A (peak **67.40625 px**, matching arcs) and adds **no
> speed** (RIGHT+B advances **246.996 px** in 180 frames, exactly the walk), while
> **X/Y are the real dash** (RIGHT+X: **269.957 px**, +9.3% over 180 frames, +23.9% over
> the first 60). So the legacy visual arm had *no dash action and two identical jump
> actions* — its policies could never reach running speed. `MarioNdsEnv` now defaults to
> `actions="v2"` (8 actions mirroring `MarioRamEnv`: X dash, A jump); `actions="legacy"`
> reproduces the old 6-action table byte-for-byte so the committed checkpoints keep
> loading. Consequences, stated honestly: the **n30 benchmark's visual rows are
> legacy/no-dash and were measured under the flow reward** (PROJECT_STATUS 4s) — they
> are **not comparable** with the RAM rows (8 actions, unified reward) or with any `v2`
> run. Details: PROJECT_STATUS §4af, docs/BENCHMARK_PROTOCOL.md.

### 1. Base Reinforcement Learning (PPO)
Standard Actor-Critic optimization where convolutional layers (`NatureCNN`) and policy MLP heads are trained simultaneously from scratch using only extrinsic environmental rewards.

```bash
python src/train.py --rom "data/mario.nds" --state "data/state.dst" --timesteps 1000000 --num-envs 4
```
*(The `--num-envs 4` flag launches parallel emulator sub-processes via `SubprocVecEnv`, decorrelating rollouts and maximizing sampling throughput).*

---

### 2. Pretrained Representation Learning (Autoencoder)
To mitigate sample inefficiency when learning directly from high-dimensional visual tensors, the pipeline supports decoupled visual feature pretraining:

* **Unsupervised Pretraining:** A convolutional Autoencoder (CNN Encoder + CNN Decoder) is trained on transition frames gathered via a random exploration policy. The MSE reconstruction loss forces the 512-dimensional latent bottleneck to encode structural geometry, obstacles, and entity boundaries independently of task reward.
* **Feature Transfer:** During downstream PPO training, the pretrained Encoder weights are loaded and frozen (`requires_grad=False`). PPO trains exclusively on top of the latent representation, reducing policy convergence time.

**Execution Pipeline:**
```bash
# Collect random exploration transitions
python src/collect_data.py --num-frames 10000

# Train unsupervised Autoencoder
python src/train_autoencoder.py --dataset "data/mario_dataset.npz" --epochs 20

# Train PPO using frozen visual representations
python src/train.py --timesteps 1000000 --num-envs 4 --use-autoencoder
```

---

### 3. Intrinsic Motivation & Exploration (ICM)
In sparse-reward environments, absence of positive feedback causes policy gradients to vanish. The Intrinsic Curiosity Module (ICM) formulates exploration as internal predictive error minimization:

* **Module Dynamics:** The *Inverse Model* predicts action $A_t$ given feature transitions $\phi(S_t) \rightarrow \phi(S_{t+1})$, ensuring features capture only policy-controllable factors. The *Forward Model* predicts next latent state $\hat{\phi}(S_{t+1})$ from $\phi(S_t)$ and $A_t$.
* **Curiosity Reward:** The Forward Model mean squared error is added to the extrinsic reward. Agents are incentivized to visit states with high predictive novelty.

```bash
python src/train.py --timesteps 1000000 --num-envs 4 --use-icm
```

**Evaluated ICM Runs (100k steps, RecurrentPPO family):**
```bash
python src/train_ppo_recurrent.py --timesteps 100000 --num-envs 4 --use-icm --icm-update-freq 8 --no-tensorboard --run-id ppo_icm_100k
python src/train_ppo_recurrent.py --timesteps 100000 --num-envs 4 --use-icm --use-autoencoder --icm-update-freq 8 --no-tensorboard --run-id ppo_icm_ae_100k
```
*Engineering notes: `--icm-update-freq 8` batches the ICM backward step (intrinsic rewards remain computed at every step, yielding identical learning signal at ~6x faster throughput); `--no-tensorboard` avoids importing TensorFlow overhead; `MLflowCallback` samples `intrinsic_reward` every 50 steps to prevent SQLite lock contention. Measured runtimes: **ppo_icm_100k ≈ 48 min, ppo_icm_ae_100k ≈ 49 min** (4 parallel envs, CUDA).*

---

### 4. Deep Residual Vision (ImpalaCNN)
To improve spatial generalization and representations across challenging platforms, the pipeline integrates the **ImpalaCNN** residual architecture (`src/impala_cnn.py`), featuring stacked ResNet blocks, spectral scaling, and dense convolutional filters.

```bash
python src/train_impala.py --run-id "mario_impala" --timesteps 1000000 --num-envs 6 --n-steps 1376 --lr 0.0003 --ent-coef 0.01
```

**Evaluation:**
```bash
python src/evaluate.py --model "models/mario_impala.zip" --stochastic
```

---

### 5. Model-Based World Models (DreamerV3 via SheepRL)
For sample efficiency via latent imagination rollouts, the project integrates **DreamerV3** through the **SheepRL** framework (Lightning Fabric):

* **Mechanism:** DreamerV3 learns a Recurrent State Space Model (RSSM) consisting of representation, transition, decoder, and reward predictors. The Actor-Critic policy trains entirely in latent imagination rollouts, reducing emulator physical step requirements.
* **Hardware Adaptation:** Configured for CUDA 12.x execution with in-memory ring buffers (`buffer.memmap=false`, capacity 100,000 transitions).

```bash
python src/train_dreamer.py
```

---

### 6. Recurrent PPO, Causal Transformers, and Auxiliary Representation Learning (CURL / SPR)
To handle temporal partial observability without manual frame stacking, the codebase implements **Recurrent PPO** (LSTM policy) and **Causal Transformers**, integrated with online self-supervised representation learning:

#### 6.1. CURL (Contrastive Unsupervised Representations for Reinforcement Learning)
Maximizes agreement between augmented views of the same observation via InfoNCE contrastive loss with bilinear projection $W$:
* Applies spatial `random_crop` transformations across rollout minibatches.
* Target network updated via Exponential Moving Average (EMA).
* Hybrid mode (`--unfreeze-encoder`) fine-tunes pretrained Autoencoder representations online without catastrophic forgetting.

#### 6.2. SPR (Self-Predictive Representations)
Forces the encoder to predict multi-step future latent transitions conditioned on action sequences:
* Evaluates latent prediction consistency via transition model and projection heads without reconstructing pixel tensors.
* Encourages representation geometry to preserve game physics and temporal dynamics.

#### 6.3. Causal Transformer Policy
Replaces recurrent LSTMs with multi-head causal self-attention (`src/transformer_policy.py`), allowing policies to attend over previous trajectory tokens without vanishing gradient degradation.

---

### Historical Architecture Benchmark (100k vs 1M Steps, Det-10 Protocol)
Initial baseline evaluation across 10 deterministic episodes per model (`num_episodes=10`):

| Configuration (Model) | Mean Reward | Standard Deviation | Maximum Reward |
| :--- | :---: | :---: | :---: |
| **NE-Dreamer (100k steps)** | 378.78 | 179.61 | 847.52 |
| **World Models GA: Encoder + LSTM + Genetic Alg. (100k steps)** | 425.58 | 0.00 | 425.58 |
| **World Models sep-CMA-ES: Hybrid Encoder + LSTM + CMA-ES (104k steps)** | 242.69 | 0.00 | 242.69 |
| **PPO Pure (100k steps)** | 13.10 | 0.00 | 13.10 |
| **PPO SPR (100k steps)** | 254.34 | 0.00 | 254.34 |
| **PPO CURL (100k steps)** | 105.00 | 0.00 | 105.00 |
| **PPO DrQ-v2 (100k steps)** | 74.92 | 0.00 | 74.92 |
| **IMPALA Transformer SPR (1M steps)** | 74.74 | 0.00 | 74.74 |
| **IMPALA CURL Recurrent (1M steps)** | 324.79 | 0.00 | 324.79 |

> **Technical Notes (Historical Det-10 Protocol):**
> 1. **NE-Dreamer (100k):** Exhibited highest peak score (847.52) in 100k steps due to unsupervised world dynamics modeling, though showing variance across seeds (Std 179.61).
> 2. **IMPALA CURL (1M):** Highest stability among model-free architectures (324.79 mean with zero catastrophic drops).
> 3. **SPR vs CURL vs DrQ-v2 (100k):** SPR (254.34) outperformed CURL (105.00) and DrQ-v2 (74.92), as temporal transition prediction accelerated dynamics learning faster than static pixel augmentations.
> 4. **World Models GA (100k):** Reached 425.58 mean with zero standard deviation, reflecting consistent survival up to the 1000-step evaluation cap.

---

### 7. World Models via Genetic Algorithm (Encoder + LSTM Memory + GA)
Three-tier architecture inspired by Ha & Schmidhuber (2018):
* **V (Vision):** Frozen pretrained CNN Encoder (`models/autoencoder.pth`, 512 latents) as feature extractor.
* **M (Memory):** Supervised LSTM (512+6 $\rightarrow$ 256) trained on 10k random policy frames to predict $z_{t+1}$ from $(z_t, a_t)$.
* **C (Controller):** Compact linear policy $a = \text{argmax}(W[z; h] + b)$ with 4,614 parameters evolved via elitist GA (population 24, Gaussian mutation $\sigma=0.05$).

```bash
python src/train_worldmodels_ga.py --timesteps 100000 --mem-frames 10000 --pop-size 24 --run-id worldmodels_ga_100k
```
*Measured training time (wall-clock, CUDA GPU + single env): 5510.9s (91.8 min) — data collection 498.4s (~8.3 min) + LSTM training 2.9s + GA evolution 5003.9s (~83.4 min). Artifacts: `models/worldmodels_ga_100k.npz` and `models/worldmodels_ga_100k_memory.pth`.*

---

### 8. World Models v2: Hybrid Encoder + sep-CMA-ES + Parallel Workers
Optimized evolution pipeline (`src/train_worldmodels_cma.py`):
* **V:** Pretrained hybrid AE+CURL encoder extracted from `models/ppo_hybrid_ae_curl_100k.zip`.
* **C:** Diagonal sep-CMA-ES (Ros & Hansen, 2008) in pure NumPy.
* **Throughput Optimization:** Parallel pool of 4 persistent environments with staggered initialization, 500-step evolution cap with 1000-step full validation on top-3 candidates, memory collection reduced to 5k frames.

```bash
python src/train_worldmodels_cma.py --timesteps 100000 --mem-frames 5000 --workers 4 --run-id worldmodels_cma_100k
```
*Measured training time: 2648.8s (44.1 min, 2.1x faster than v1) — encoder loading 0.3s + collection 241.2s (~4 min) + LSTM training 3.8s + CMA-ES 8 generations 2039.3s (~34 min). Environment steps: 104,000. Artifacts: `models/worldmodels_cma_100k.npz` and `models/worldmodels_cma_100k_memory.pth`.*

---

### 9. Unified Benchmark: 10 Deterministic + 30 Stochastic Episodes
All Stable-Baselines3 alternatives were systematically benchmarked using identical evaluation criteria (`src/evaluate_benchmark.py --eps 10 --eps-stoch 30`, full episodes, headless, stored in `evals/benchmark_n30_merged.json`):

| Model | Budget | Deterministic Mean | Stochastic n=30 (Mean ± Std / Max) |
| :--- | :---: | :---: | :--- |
| PPO Pure | 100k | 120.80 | 295.51 ± 80.93 / 437.83 |
| PPO + Autoencoder | 100k | 74.77 | 234.27 ± 140.10 / 713.77 |
| PPO CURL | 100k | 226.72 | 320.38 ± 139.19 / 704.05 |
| PPO Hybrid AE+CURL | 100k | 74.74 | 286.14 ± 141.60 / 590.90 |
| PPO SPR | 100k | 254.34 | 259.95 ± 112.75 / 495.76 |
| **PPO + ICM** | 100k | 222.49 | 231.37 ± 94.60 / 402.75 |
| **PPO + ICM + Autoencoder** | 100k | 74.77 | 230.09 ± 99.38 / 394.63 |
| **PPO DrQ-v2** | 100k | 74.92 | 214.82 ± 105.80 / 434.91 |
| **PPO RAM-only (Vision-Free)** | 100k | 61.40 | 392.39 ± 223.91 / 1001.80 |
| **PPO RAM + Geometry (Pits)** | 100k | 61.40 | 521.95 ± 425.36 / **1538.48** |
| **PPO GNN Relational (Entity Graph)** | 100k | 61.40 | 401.11 ± 312.21 / 1311.86 |
| Recurrent PPO | Long-run | 31.74 | 516.80 ± 131.04 / 771.03 |
| ImpalaCNN PPO | 1M | 74.74 | 290.16 ± 161.17 / 686.32 |

> **Scientific Observations:**
> 1. **Deterministic Standard Deviation is 0.00:** Both the environment and policy argmax execution are strictly deterministic.
> 2. **First Goomba Trap (Deterministic Death):** Models in the ~74.7 / ~61.4 deterministic range die deterministically at step 39 due to an incoming Goomba (RAM: dx 164px $\rightarrow$ contact $\rightarrow$ life lost), not a pit. Stochastic exploration is required to reveal true policy capabilities.
> 3. **Stochastic n=30 Sample Efficiency:** Under stochastic exploration, `ram_geo` (521.95), `recurrent` (516.80), and `gnn` (401.11) achieve leading returns, with `ram_geo` reaching a maximum reward of 1538.48.

---

### 10. RAM Memory Map (EUR NTR-A2DP-EUR) and `reward_mode="ram"`
Direct memory variable extraction via `emu.memory` (`src/ram_state.py`, discovery tool in `src/ram_search.py`). Notice that US memory addresses (TASVideos/DataCrystal) **do not apply** to the European ROM:

| Address (EUR) | Variable | Status |
| :--- | :--- | :---: |
| `0x0209DC00` u8 | Lives (decrements from 5 to 4 on death; cf. AR code `2209DC00`) | ✅ Verified |
| `0x02098240` / `0x020DCFA0` u32 | Camera absolute X (deadzone filtered) | ✅ Verified |
| `0x0209AE9C` u32 | Horizontal activity odometer | ✅ Verified |
| `0x021C1904`... s32 | Velocity mirrors (±6144 = ±1.5 px/frame in 20.12 fixed point) | ✅ Verified |
| `0x021C1890` +`0x44/0x60/0x68` | **Mario Object**: type `0x1C`, A=Y (jump arc, floor baseline -480px), B=X | ✅ Verified |
| Linked list at `obj+0x38` | **Circular linked list of active entities** (enemies link upon approach) | ✅ Verified |
| `RamState.enemies()` | `[{type, dx, dy}]` in px — Goomba verified: 164px $\rightarrow$ contact $\rightarrow$ life lost | ✅ Verified |
| `0x020DC968` u32 | Stage timer in 20.12 fixed point (398.3 HUD equivalent) | ✅ Verified |
| `0x021C1908` s32 | Vertical velocity Y-vel (0.0 on ground, parabolas during jump) | ✅ Verified |
| `0x020DCF27` u8 | Stage Clear Flag (transitions 0 -> 1 on flagpole descent / castle entry) | ✅ Verified |
| `0x020A703C` u32 | Complementary vertical position | 🧪 Experimental |

**Validation (RAM Telemetry vs Optical Flow):** Across identical 39-step rollouts, accumulated progress correlation is **0.91**, verifying the `RAM_PX_PER_UNIT = 1/4096` scaling factor. Death detection via RAM lives occurs **3 frames earlier** than pixel template matching (which triggers on the black game-over screen).

**Usage:** `MarioNdsEnv(..., reward_mode="ram")` uses RAM camera progress and life counters for reward/termination while retaining visual observation tensors. **The default is `"ram"` since the 4s unification** (both arms score the same reward; only the observation differs); `"flow"` is the legacy opt-in kept for reproducing the June-era runs.

---

### 11. Vision-Free Reinforcement Learning (NES/Gym-Retro Style)
`src/ram_env.py` (`MarioRamEnv`): Pure vector observation `Box(17)` directly from RAM (zero pixels, zero CNN processing):
`[x, y, vx, vy, on_ground, lives, time, cam_x + 3 * (dx, dy, type)]` (coordinates relative to episode reset, dynamic entities sorted by distance).

```bash
python src/train_ram.py --timesteps 100000 --num-envs 2 --run-id ram_ppo_100k
```
*Results: 102,400 actual steps completed in 2725s (45 min, 36.7 fps on 2 envs), yielding stochastic return **392.39 ± 223.91 / max 1001.80** — matching top visual methods with zero convolutional operations.*

---

### 12. Level Geometry Extraction via ROM Parsing (Pits, Walls, Spawns)
NSMB DS stage binary files are stored in `course/X##_#.bin` and `course/X##_#_bgdat.bin` inside the ROM (`src/course.py`, parsed via `ndspy`):
* `Blocks[5]`: 20-byte player entrance vectors $(x, y)$ in pixels.
* `Blocks[6]`: 12-byte sprite definitions $(type, x, y)$ in tiles.
* `_bgdat.bin`: Ground and wall tile objects $(obj, tileset, x, y, w, h)$.

Cross-validation for World 1-1 (`A01_1`): entrance at (80, 464)px; floor row at 30 ($-480$px); pit at tile 30; Goomba spawn at tile 24 ($384$px, matching RAM detection at $365$px).
`MarioRamEnv(geo=True)` appends 6 binary pit flags (`pit_ahead`, columns +1 to +6, expanding observation space to `Box(23)`):

```bash
python src/train_ram.py --timesteps 100000 --num-envs 2 --geo --run-id ram_geo_100k
```
*Evaluated performance: Stochastic return **521.95 ± 425.36 / max 1538.48**.*

---

### 13. Relational Graph Neural Network (Pure-Torch MeanMPNN)
`src/graph_env.py` models the environment as an entity graph: 14 nodes (Mario + $\le 5$ enemies + $\le 8$ static ROM geometry rectangles) with 8 node features each `[dx, dy, w, h, is_mario, is_enemy, is_static, type_norm]` + 8-dim global state vector + binary node mask (`Box(134)`).
`src/gnn_extractor.py` performs 2-layer masked mean message passing followed by global mean pooling:

```bash
python src/train_gnn.py --timesteps 100000 --num-envs 2 --run-id gnn_100k
```
*Evaluated performance: Stochastic return **401.11 ± 312.21 / max 1311.86**.*

---

### 14. Physics-Informed (PINN) World Model — Fast Training + Real-Time Play
Ported in methodology from [`PedroM2626/smw-pinn`](https://github.com/PedroM2626/smw-pinn) (Super Mario World) to **New Super Mario Bros. DS**, built on the RAM-telemetry stack. A **Hard-Residual PINN** world model embeds the engine's exact discrete integration `x_{t+1}=x_t+v_{x,t+1}/16` (verified on real data) into the computation graph, learning only the un-modelled force/contact residual. Result: trained from **~200 real transitions in ≈1 minute** to test MSE **0.056** with a **structurally zero kinematic residual**. Three-stage pipeline (with three upgrades: a discovered **dash action** (`X`), a **probabilistic `logvar` head** the planner can be pessimistic about, and a **tilemap A\* jump-reachability planner**):

```bash
python src/train_pinn_world_model.py --transitions 3500 --epochs 350 --sample-efficiency   # Stage 1: world model
python src/train_pinn_policy.py --timesteps 200000 --n-envs 8                              # Stage 2: Dyna-PPO in imagination
python src/eval_pinn_realtime.py --controller reactive --render                            # Stage 3: real-time on console (A*+reflex)
python src/tilemap_planner.py                                                             # A* feasibility proof (stage is completable)
```

**Real-time console results** (World 1-1; the flagpole is at 4,032 px and the last solid tile column ends at 4,256 px — the first is what `level_cleared` fires on): the dash unlock **doubles** reach, and with ROM **wall/pipe perception** the reflex + A\* planner now **clears the stage end to end**: `reactive` reaches **4,021 px of 4,032 px in 5/5 episodes** ([`evals/pinn_realtime_eval.json`](evals/pinn_realtime_eval.json), and the clip at the top of this file). The learned arms still stop short — CEM-MPC peaks at 1,510 px, PPO+Reflex at 1,139 px, pixel PPO at 120.8 px — and the imagination PPO (2.5 ms/frame, >400 FPS) also runs live. The **tilemap A\*** proves the stage is **geometrically feasible** (4 pits, max 3 tiles, all within a running jump — the true pit census measured against the game's own physics, five slope ramps the static classes miscounted as pits notwithstanding). The barrier that had to be broken sat at the **pipe + dense block-staircase** just after it: a standstill jump on a narrow surface goes straight up (`vx=0`) so Mario oscillates until the **stage timer** expires — direct in-game evidence (rendered frame + velocity trace) is in [`media/nsmb_pipe_top_timesup.png`](media/nsmb_pipe_top_timesup.png). The reflex layer clears it by taking off from the exact edge column with dash speed still on; a *learned* controller clearing it still needs a full solid-tile 2D platformer planner or large-scale direct RL. Full methodology + honest limits: **[`docs/WORLD_MODEL_PINN.md`](docs/WORLD_MODEL_PINN.md)**. Clips: `media/pinn_mpc_run.avi`, `media/nsmb_dashboard_clear.mp4`.

---

### 15. Compact Neuroevolution (Genetic Algorithm) — Zero-Gradient 100% Clear in 206 Steps
Direct policy search via an elitist Genetic Algorithm ($P=24$, top-4 elite, Gaussian mutation $\sigma=0.15$) operating on an ultra-compact MLP (23 inputs $\to$ 32 tanh $\to$ 8 actions, only **1,032 parameters** / 11.6 KB).
By replacing gradient descent and value-function bootstrapping with pure episodic progress fitness ($f = \max_t x_t$), the policy cleared World 1-1 at Generation 12 (29.9 min) and converged to an all-time speedrun record of **206 steps** (beating the 245-step baseline). Evaluated deterministically ($N=5$), it achieves **100% clear rate (5/5)** with zero catastrophic forgetting:

```bash
python tools/train_neuroevolution.py --generations 25 --pop-size 24 --elite 4
python tools/eval_neuroevolution.py --model models/best_genetic_policy.npz --episodes 5
```
*Evaluated performance: Deterministic **4,020.0 px (100% clear 5/5, 206 steps)** ([`evals/eval_genetic_w11.json`](evals/eval_genetic_w11.json), clip at [`media/nsmb_w11_genetic_clear.mp4`](media/nsmb_w11_genetic_clear.mp4)).*

---

### 16. Implicit Quantile Networks (IQN) — Quantile Distributional RL
Implements continuous quantile distributional RL (Dabney et al., 2018) via a 64-frequency cosine embedding $\phi(\tau)$ interacting with an observation encoder $\psi(s) \in \mathbb{R}^{128}$ (45,448 parameters). Trained on live DeSmuME under the simple potential maximum reward ($r_t = \max(0, x_t - x_{\text{max\_so\_far}})/16.0$):
First level clear achieved at **6.33 minutes** (Episode 82, step 8,388, $4,023.8\text{ px}$), reaching the top of the flagpole ($4,032.0\text{ px}$) during training:

```bash
python tools/train_iqn.py --total-steps 20000 --batch-size 64
python tools/eval_iqn.py --model models/iqn_policy.pt --episodes 5
```
*Evaluated performance: Peak training progress **4,032.0 px (Clear)**, greedy evaluation peak **2,913.0 px (72.2%)** ([`evals/eval_iqn_w11.json`](evals/eval_iqn_w11.json)).*

---

### 17. Rainbow DQN & Pure Potential Maximum Reward
Combines all six extensions of Rainbow (Hessel et al., AAAI 2018): Double Q-learning, Prioritized Experience Replay (PER, $\alpha=0.6, \beta_0=0.4 \to 1.0$), Dueling network, multi-step returns ($n=3, \gamma=0.99$), C51 categorical distribution (51 atoms over $[-2, 16]$), and factorized NoisyLinear exploration layers (204,054 parameters).
Under the pure potential maximum reward, Rainbow achieved **3,850.9 px** (95.8% of stage) in training (16.4 min) and maintains a rock-solid deterministic greedy performance of **2,996.8 px** (74.5%) across all 5 evaluation runs without on-policy policy collapse:

```bash
python tools/train_rainbow.py --total-steps 20000 --batch-size 64
python tools/eval_rainbow.py --model models/rainbow_policy.pt --episodes 5
```
*Evaluated performance: Peak training progress **3,850.9 px**, greedy evaluation **2,996.8 px** across 5/5 runs ([`evals/eval_rainbow_w11.json`](evals/eval_rainbow_w11.json)). Full comparative study across PPO, Rainbow, IQN, and GA in [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md) §4aq.*

---

## 💾 Experiment Management (MLOps)

### Run Versioning
Unique experiment runs are versioned via `--run-id`, configuring model checkpoints and unifying MLflow metric namespaces:
```bash
python src/train.py --run-id "experiment_icm_alpha" --use-icm
```

### Checkpoint Resumption
Graceful interruption handlers capture `SIGINT` (Ctrl+C) to save model state prior to termination. To resume training preserving global timesteps and learning rate decay schedules:
```bash
python src/train.py --timesteps 1000000 --resume "models/experiment_icm_alpha"
```

### Qualitative Evaluation
To execute deterministic or stochastic policy rollouts:
```bash
python src/evaluate.py --model "models/experiment_icm_alpha.zip" --stochastic
```

### Metric Observability
All metrics (extrinsic reward, intrinsic curiosity, rollout lengths, loss curves) are logged to local SQLite tracking databases:
```bash
mlflow ui
```
Navigate to `http://localhost:5000` to inspect comparative metric runs and parameter sweeps.

---

## 🙏 Acknowledgments

* **NE-Dreamer** — the `nedreamer/` package vendored in this repository is based on
  [**corl-team/nedreamer**](https://github.com/corl-team/nedreamer) (Bredis, Balagansky,
  Gavrilov, Rakhimov), which itself builds on
  [**NM512/r2dreamer**](https://github.com/NM512/r2dreamer).
  It is included here as a first-class part of the codebase (not a git submodule) for
  the NSMB-DS `configs/env/marionds.yaml` experiments; all credit for the original
  NE-Dreamer design and implementation goes to its authors (see `nedreamer/LICENSE`).
* **New Super Mario Bros. (DS)** is © Nintendo. This project uses ROM telemetry for
  research only.
