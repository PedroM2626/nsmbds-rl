# Formal Evaluation Protocol: *New Super Mario Bros. DS* (EUR) RL Benchmark

> **⚠️ Status of the n=30 table (2026-10-23).** Three caveats apply to the committed
> `benchmark_n30_merged.json` and to every number in §7 below:
> 1. **The visual rows are legacy-action and no-dash.** The June visual env used KEY_B
>    where the code claimed "dash"; measured, B is a *jump* button (67.40625 px arc,
>    identical to A) and adds no speed, so the 6-action visual arm had no dash at all
>    (PROJECT_STATUS §4af). `MarioNdsEnv` now defaults to `actions="v2"` (8 actions);
>    the benchmark loads its checkpoints with `actions="legacy"` to stay reproducible.
> 2. **The visual rows were measured under the flow reward** (before the 4s
>    unification), the RAM rows under the unified RAM reward — the two halves of the
>    table are not on the same reward scale.
> 3. **Stochastic n=30 is policy-sampling variance only.** The emulator is
>    deterministic and every episode starts from the same savestate; there is no
>    environmental seed or frame jitter (the §5.2 seed blocks were run as three separate
>    invocations, not in-code). The README already notes deterministic σ = 0.00.
> Treat cross-arm comparisons accordingly; a v2 re-benchmark under the unified reward
> is future work.

## 1. Executive Summary & Objective

This document defines the formal, reproducible evaluation protocol for benchmarking Reinforcement Learning (RL), Model-Based Planning (MBRL), and Physics-Informed Neural Network (PINN) architectures on the Nintendo DS title *New Super Mario Bros.* (EUR release NTR-A2DP-EUR).

The primary goal is to establish an academically rigorous standard for measuring horizontal level progression, survival time, collision resilience, and sample efficiency under both deterministic policy execution ($n=10$) and stochastic environmental exploration ($n=30$).

---

## 2. Emulation and Hardware Environment

### 2.1 Emulation Core and ROM Identification
All evaluations are conducted using `py-desmume` (v0.0.9, DeSmuME core 0.9.12), exposing C++ bindings directly into the DeSmuME Nintendo DS hardware emulator.
* **ROM Title:** *New Super Mario Bros.* (Europe) (En,Fr,De,Es,It)
* **Internal Serial:** `NTR-A2DP-EUR`
* **ROM SHA-1 (measured 2026-10-23):** `B0F7886720E842ED922569E1BB5C39C0FA090331`
  (the value this document previously carried — `809618B5...` — does not hash-match
  the ROM every committed eval ran on; use the measured one above, and see
  `src/eval_provenance.py`, which now records it in every new eval JSON)
* **Target Stage:** World 1-1 (`course/A01_1.bin`, Course ID `A01_1`)
* **Standard Level Entrance Savestate:** `data/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).ds1` (Mario initialized at absolute entrance tile $X = 88.0\text{ px}$, $Y = 448.0\text{ px}$).

### 2.2 Temporal Resolution & Timing
* **Base Frame Rate:** 60.0 frames per second (emulated ARM946E-S / ARM7TDMI hardware).
* **Frame Skipping:** $\Delta t_{\text{skip}} = 8$ frames per agent action step.
* **Control Frequency:**
  $$\nu_{\text{control}} = \frac{60.0}{8} = 7.5\text{ Hz} \quad (\Delta \tau = 133.\overline{3}\text{ ms})$$
* **Episode horizon:** `MarioRamEnv` defaults to 1,000 steps. The benchmark loop has a separate 5,000-step safety cap; other environments may terminate earlier. Do not infer every model used a 1,000-step cap.
* **Concurrency Invariant:** Due to DeSmuME C++ core static variables, instantiating multiple emulator instances within a single operating system process triggers memory access violations. Concurrent evaluations across seeds or models MUST execute as isolated child processes with staggered process spawning ($\ge 2.0\text{ s}$ sleep between worker initializations).

---

## 3. Observation Spaces and Sensor Modalities

The benchmark evaluates policies across three distinct sensor paradigms:

### 3.1 Pixel Modality (Vision Baselines)
* **Raw Hardware Display:** DeSmuME top-screen RGBX frame buffer ($256 \times 192 \times 4$, uint8).
* **Preprocessing Pipeline:**
  1. Alpha channel truncation $\to 256 \times 192 \times 3$ BGR.
  2. Bilinear spatial downsampling $\to 84 \times 84 \times 1$ grayscale.
  3. Frame Stacking: Most non-recurrent visual rows use $k=4$ observations; the recurrent row uses one grayscale frame and LSTM state. See `evaluate_benchmark.py` environment selection.

### 3.2 Egocentric RAM Telemetry Modality (Low-Dimensional State)
Extracted directly from DS Main RAM (`0x02000000`–`0x023FFFFF`) via pointer chasing:
* **State Vector $\mathbf{s}_t \in \mathbb{R}^{17}$ (`ram100k`; the "16" this document carried before 2026-10-23 miscounted the Mario block):**
  - Mario kinematics: normalized horizontal velocity $v_x / 4096.0$, vertical velocity $v_y / 4096.0$, grounded flag $\mathbf{1}_{\text{ground}}$, odometer tick delta.
  - Camera differential: $\Delta X_{\text{cam}}$.
  - Local entity tracking: relative offsets $(dx_i, dy_i)$ and type tags $e_i$ for the three nearest active actors in the sprite linked list ($0 \le i < 3$).
* **Geometry-Augmented Vector $\mathbf{s}_t \in \mathbb{R}^{23}$ (`ramgeo100k`; the "21" was the same miscount):**
  - Includes all 17 RAM dimensions plus a 6-column spatial lookahead of forward terrain surface continuity $\{g_{t+1}, \dots, g_{t+6}\}$ indicating upcoming pits and elevation jumps extracted from ROM tilemap slices.

### 3.3 Graph Entity Modality (`gnn100k`)
* **Node Features $\mathcal{V}$:** Node 0 represents Mario; nodes $1 \le j \le M$ represent active dynamic entities (Goombas, Koopas, Piranha Plants).
* **Adjacency $\mathcal{E}$:** Fully connected spatial edge weights $w_{ij} = \exp(-\|\mathbf{p}_i - \mathbf{p}_j\|_2 / \sigma)$ with relative coordinate displacement attributes $(\Delta x, \Delta y)$.

---

## 4. Action Space Parameterization

The agent operates over a discrete 8-element control repertoire mapped to Nintendo DS digital gamepad registers (`MarioRamEnv._action_keys`, src/ram_env.py):

| Action ID | Nintendo DS Key Mask | Semantic Maneuver | Purpose |
|:---:|:---|:---|:---|
| `0` | None | NOOP / Coast | Dynamic deceleration / latency buffer |
| `1` | `KEY_RIGHT` | Walk Right | Low-speed navigation |
| `2` | `KEY_RIGHT` + `KEY_A` | Walk + Jump | Short hurdle / 1-tile stair climb |
| `3` | `KEY_RIGHT` + `KEY_X` | Dash Right | Maximum horizontal velocity (~2.14 px/frame, measured) |
| `4` | `KEY_RIGHT` + `KEY_X` + `KEY_A` | Dash + Jump | Ballistic long leap (reach 109.73 px, apex 71.06 px @ f26, measured) |
| `5` | `KEY_LEFT` | Walk Left | Decoupled braking / runway recovery |
| `6` | `KEY_X` + `KEY_A` | Dash + Jump in Place | Vertical evasion with spin |
| `7` | `KEY_A` | Standing Vertical Jump | Vertical evasion without horizontal displacement |

> Corrected 2026-10-14: this table previously described the June-era
> mapping (KEY_B for jump, KEY_Y for dash, `v_max = 3.6 px/frame`,
> reach 172.8 px) which matches no current code path -- the RAM env
> has jumped with KEY_A and dashed with KEY_X since the RAM era.
>
> **Measured 2026-10-23 (the visual arm's own table):** the June
> `MarioNdsEnv` used KEY_B where it claimed "dash" -- B is a *jump*
> button (identical 67.40625 px standing arc as A; RIGHT+B advances
> exactly like RIGHT: 246.996 px / 180 frames) while X/Y carry the dash
> (269.957 px, +9.3%). The legacy visual action space was therefore
> 6 actions with **no dash** and two identical jumps. `MarioNdsEnv` now
> defaults to `actions="v2"` = the 8-action table above; every benchmark
> visual checkpoint is loaded with `actions="legacy"` for reproducibility
> (see the banner at the top of this document and PROJECT_STATUS §4af).

---

## 5. Evaluation Protocol and Trial Regimes

Each baseline model is subjected to two distinct test protocols. **Honest caveat (2026-10-23):** the emulator is deterministic and every episode boots the same savestate, so "seeds" vary only the policy's action sampling -- there is no environmental seed, no frame jitter, and deterministic runs are literally identical (σ = 0.00 in every committed det block). Treat the stochastic spread as policy-sampling variance, not environment variance:

### 5.1 Deterministic Evaluation Regime ($n = 10$)
* **Objective:** Measure policy peak asymptotic execution and verify if deterministic state exploitation leads to repetitive failure loops (e.g. wall collisions or periodic enemy death loops).
* **Sampling Rule:**
  $$a_t = \arg\max_{a \in \mathcal{A}} \pi_\theta(a \mid s_t)$$
  For actor-critic / PPO models, the mode of the categorical distribution is selected.
* **Sample Size:** $n = 10$ repeated episodes from the same entrance savestate; deterministic repetitions are identical under this protocol.

### 5.2 Stochastic Evaluation Regime ($n = 30$)
* **Objective:** Measure policy robustness under action sampling.
* **Sampling Rule:**
  $$a_t \sim \pi_\theta(a \mid s_t)$$
* **Sample Size & Seed Partition:** $n = 30$ episodes, run as **three separate
  invocations** of `evaluate_benchmark.py --seed {100,200,300}` (the blocks are
  not implemented in code; each invocation seeds torch/numpy once and runs 10
  stochastic episodes from the same savestate):

---

## 6. Mathematical Formulation of Benchmark Metrics

For each episode $i \in \{1, \dots, N\}$:

### 6.1 Cumulative Episodic Return ($R_i$)
$$R_i = \sum_{t=0}^{T_i} r_t$$
For RAM rows, step reward $r_t$ is the unified RAM reward (4s): camera-RAM
advancement $\times 2.0$, per-step time penalty ($-0.05$), death via
RAM lives ($-15.0$), stage-clock timeout ($-50.0$), and terminal goal
completion ($+100.0$). Historical visual rows instead used the legacy flow reward; their returns are not on the same scale.

> Corrected 2026-10-14: the previous text read "life loss penalty
> ($-50.0$), time penalty ($-0.01$)" -- both wrong against the code:
> death is $-15$ (and was $-15$ before the 4s unification too), the
> per-step penalty is $-0.05$, and $-50$ is the stage-clock timeout.

### 6.2 Absolute Spatial Level Progression ($X_{\text{prog}}$)
$$X_{\text{prog}, i} = \max_{0 \le t \le T_i} X_t - X_0$$
Normalized level completion fraction:
$$\rho_i = \min\left(1.0, \frac{X_{\text{prog}, i}}{X_{\text{goal}} - X_{\text{spawn}}}\right) \times 100\%$$
where $X_{\text{spawn}} = 88.0\text{ px}$ and $X_{\text{goal}} = 4,032.0\text{ px}$ (flagpole contact coordinate).

### 6.3 Completion Rate ($C$)
$$C = \frac{1}{N} \sum_{i=1}^N \mathbf{1}[\text{Flagpole Reached}_i]$$

### 6.4 Statistical Confidence Bounds
The historical table reports a parametric interval and the empirical standard deviation; it does not provide a non-parametric interval. Its dispersion and intervals describe policy sampling from a fixed savestate, not variability across game starts:
1. **Parametric 95% Confidence Interval ($t$-distribution):**
   $$\text{CI}_{95\%} = \hat{\mu} \pm t_{0.975, N-1} \cdot \frac{s}{\sqrt{N}}$$
   where $\hat{\mu} = \frac{1}{N} \sum R_i$. The saved `std` uses `numpy.std` with the default $N$ denominator, not the $N-1$ sample-standard-deviation convention; the table uses that reported standard deviation in its approximate intervals.
2. **Empirical Standard Deviation ($s$):** Reported directly alongside means.

---

## 7. Official Baseline Performance Summary ($n=30$ Stochastic vs. $n=10$ Deterministic)

The following metrics are compiled directly from empirical benchmark logs in [`evals/benchmark_n30_merged.json`](../evals/benchmark_n30_merged.json):

| Model Key | Observation Space | Architecture / Algorithm | Det Mean ($n=10$) | Stoch Mean ($n=30$) | Stoch Std ($s$) | Stoch 95% CI |
|:---|:---|:---|:---:|:---:|:---:|:---:|
| `drq` | $4 \times 84 \times 84$ Pixels | PPO + Random Shift Augmentation | 74.92 | 214.82 | 105.80 | $[175.31, 254.33]$ |
| `pure` | $4 \times 84 \times 84$ Pixels | Nature CNN PPO | 120.80 | 295.51 | 80.93 | $[265.26, 325.76]$ |
| `ae_frozen` | Latent $\mathbf{z} \in \mathbb{R}^{64}$ | Conv-Autoencoder (Frozen) + PPO | 74.77 | 234.27 | 140.10 | $[181.91, 286.63]$ |
| `hybrid` | Latent $\mathbf{z} \in \mathbb{R}^{64}$ | AE + CURL Contrastive Latent | 74.74 | 286.14 | 141.60 | $[233.20, 339.08]$ |
| `spr` | $4 \times 84 \times 84$ Pixels | Self-Predictive Representations | 254.34 | 259.95 | 112.75 | $[217.82, 302.08]$ |
| `curl` | $4 \times 84 \times 84$ Pixels | Contrastive Unsupervised RL | 226.72 | 320.38 | 139.19 | $[268.32, 372.44]$ |
| `icm` | $4 \times 84 \times 84$ Pixels | Intrinsic Curiosity Module PPO | 222.49 | 231.37 | 94.60 | $[196.02, 266.72]$ |
| `icm_ae` | Latent $\mathbf{z} \in \mathbb{R}^{64}$ | ICM over Autoencoder Latents | 74.77 | 230.09 | 99.38 | $[192.96, 267.22]$ |
| `impala` | $4 \times 84 \times 84$ Pixels | Deep Residual Network (IMPALA) | 74.74 | 290.16 | 161.17 | $[229.95, 350.37]$ |
| `recurrent`| $1 \times 84 \times 84$ Pixels + recurrent state | Recurrent PPO (LSTM Memory) | 31.74 | 516.80 | 131.04 | $[467.84, 565.76]$ |
| `ram100k` | $\mathbf{s} \in \mathbb{R}^{17}$ RAM | MLP-PPO ($[64, 64]$) | 61.40 | 392.39 | 223.91 | $[308.70, 476.08]$ |
| `ramgeo100k`| $\mathbf{s} \in \mathbb{R}^{23}$ Geo-RAM | MLP-PPO ($[64, 64]$) + Terrain Lookahead | 61.40 | 521.95 | 425.36 | $[362.96, 680.94]$ |
| `gnn100k` | Dynamic Graph | Interaction Network GNN | 61.40 | 401.11 | 312.21 | $[284.45, 517.77]$ |
