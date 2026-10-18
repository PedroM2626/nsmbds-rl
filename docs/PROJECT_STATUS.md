# Project Status & Gap Analysis

*Last updated: 2026-10-11. ROM: NSMB DS EUR (NTR-A2DP-EUR). All code lives under `src/`.*

---

## Executive Summary

The pipeline goes from raw DS emulator → RAM telemetry → physics-informed world model → real-time agent. All three PINN pipeline stages are executable and produce metrics. **The rule-based `reactive` controller clears World 1-1 end to end**: 5/5 episodes, 4,022 px of the 4,032 px flagpole, `clear_rate = 1.0` ([`evals/pinn_realtime_eval.json`](../evals/pinn_realtime_eval.json), reproduced 2026-10-11 on a second machine with an identical `total_reward`). The learned controllers are still far from that: CEM-MPC peaks at 1,510 px and PPO+Reflex at 1,139 px, both dying at pit/enemy approaches, and the pixel PPO baselines still show the early-death cluster at ~336 px caused by the first Goomba on flat ground. An earlier revision of this page reported "no controller has cleared 1-1 (max ≈ 1730 px)"; that predates commit `f70b262`.

---

## 1. What Is Done ✅

### 1.1 Environment & RAM Telemetry (`src/ram_env.py`, `src/ram_state.py`)

| Component | Status | Notes |
|-----------|--------|-------|
| DeSmuME Python binding (`py-desmume`) | ✅ | Fallback stubs for offline testing |
| Mario RAM addresses (EUR) | ✅ | Lives `0x0209DC00`, cam `0x02098240/0x020DCFA0`, odometer `0x0209AE9C`, Mario base `0x021C1890`, velocity addrs validated |
| Object type/A(Y)/B(X) at `+0x44/0x60/0x68` | ✅ | Validated with differential search + screenshots + AR codes |
| Enemy linked-list scanner (`SCAN_LO`–`SCAN_HI`) | ✅ | Finds up to 3 closest enemies by `|dx|`; links at ~20 steps proximity |
| 17-dim RAM observation vector | ✅ | `[x, y, vx, vy, on_ground, lives, time, cam_x, 3×(edx,edy,etype)]` |
| 8-action space with dash | ✅ | Reverse-engineered: X/Y = dash; running jump ≈ 120 px clears wide pits |
| Reward: progress×2 − 0.05/step, death −15, clear +100 | ✅ | Terminal +100 reward on flagpole contact |
| Level-finish detection | ✅ | Hardware RAM flag isolated at `CLEAR_FLAG_ADDR = 0x020DCF27` (Area/Camera controller) + flagpole coordinate (`4032.0 px`) |
| Dynamic enemy velocities | ✅ | Finite-difference `(vx, vy)` tracked in `RamState.enemies()` |
| Absolute screen-pixel X | ⚠️ Partial | Two consistent bases (`0x021C1828` vs object B); **delta is correct** (reward works), but absolute X is ±40 px ambiguous |

### 1.2 ROM Level Parser (`src/course.py`)

| Component | Status | Notes |
|-----------|--------|-------|
| `course/A##_#.bin` section table | ✅ | Header = u32 size + 14 (offset,length) pairs tiling the file |
| Sprite/object block (Block 6) | ✅ | 12 B/record: `type u16, x u16, y u16 (TILES), data 6B` ended by `0xFFFFFFFF` |
| Entrance block (Block 5) | ✅ | 20 B/record: x,y,cam in pixels |
| `_bgdat.bin` tile objects | ✅ | All walkable ground/slope/pipe/stair objects included; false pits eliminated |
| Calibration vs RAM (1-1) | ✅ | Spawn tile 3 → RAM 48 px, ground row 30 → RAM −480 px, Goomba tile 24 → RAM 365 px |
| All 382 course files parsed | ✅ | ndspy `NintendoDSRom`, 2449-file FS, manifest at `tools/fs_manifest.json`. Parse ≠ plannable: only 29 of 191 courses pass the coverage guard (`src/course.py:require_plannable`); see §4e |
| JSON export for all levels | ✅ | 191 levels exported (`out/dataset/levels.jsonl`, `entities.csv` via `tools/build_dataset.py`) |
| Enemy-type ID mapping | ✅ (85%) | 342 profile slots extracted from CC0 decomp into `out/dataset/object_ids.json`, loaded by `src/course.py` |
| Level-finish tile detection | ✅ | Flagpole detected at tile 252 (`4032.0 px`) across course sprites |

### 1.3 Tilemap A\* Planner (`src/tilemap_planner.py`)

| Component | Status | Notes |
|-----------|--------|-------|
| Ground/pit mask from `_bgdat.bin` | ✅ | `GROUND_OBJS` includes all solid surfaces; real 1-1 pits detected at width 2-3 tiles |
| Run-jump reachability | ✅ | Dynamic ballistic reach: $X_{reach} = v_x \cdot t_{airborne}$ ($t_{air} = 52\text{ frames}$ in code; documented as 48 in earlier sections — use code value) |
| A\* over tile columns | ✅ | Global layer; feeds takeoff column to CEM-MPC |
| Enemy-aware planning | ✅ | `enemy_threat_ahead()` checks takeoff/landing corridors; triggers `dash_leap` or `retreat_wait` |
| Multi-level support | ⚠️ Partial | Validated on 1-1; tested headless on 2-1–8-1 via `tools/enter_course.py`; only grass-tileset family plannable (92.1% on A01_1; others refused at coverage guard) |

### 1.4 PINN World Model (`src/pinn_world_model.py`, `src/train_pinn_world_model.py`)

| Component | Status | Notes |
|-----------|--------|-------|
| Hard-residual kinematics identity | ✅ | Verified: `x_{t+1} = x_t + vx_{t+1}/16`, residual mean ≈ 4×10⁻⁴ |
| MLP trunk (128×128, LayerNorm, GELU) | ✅ | Δobs head + reward head + continue head |
| Heteroscedastic `head_logvar` | ✅ | Gaussian-NLL loss; uncertainty used by pessimistic planner |
| Transition pool | ✅ | 3500 real transitions, `models/pinn_pool.npz` |
| Trained model | ✅ | `models/pinn_nsmb.pt`, 43 440 params, test MSE 0.0558 |
| Open-loop drift | ⚠️ Moderate | Mean drift 1738 px over 44-frame horizon across 18 episodes → model diverges in extended rollouts |
| Sample efficiency | ✅ | MSE at 500 transitions ≈ MSE at 2000 (plateau early) |

### 1.5 PINN Policy & Real-Time Eval (`src/train_pinn_policy.py`, `src/eval_pinn_realtime.py`)

| Component | Status | Notes |
|-----------|--------|-------|
| Dyna-PPO in imagination | ✅ | 200 k steps, 8 imagined envs, 222 steps/s, `models/pinn_policy.zip` |
| CEM-MPC controller | ✅ | Re-plans every real step; 80 ms latency |
| Reactive baseline | ✅ | Rule-based: run right + jump on pit/enemy |
| **Results (n=8 each)** | | |
| — PPO | ❌ | Mean 347 px, max 424 px; dies at first Goomba |
| — Reactive | ✅ | Mean 4,020.6 px, max 4,020.6 px; clears 5/5; `clear_rate = 1.0` (`evals/pinn_realtime_eval.json`) |
| — MPC | ❌ | Mean 1269 px, max 1730 px; best so far but never clears pit cluster at ~1730 px |
| Level clear | ✅ **Achieved** | Reactive controller clears World 1-1 end to end: 5/5 episodes, 4,020.6 px of the 4,032 px flagpole (`evals/pinn_realtime_eval.json`) |

### 1.6 Model-Free Baselines (`src/train_ppo_recurrent.py`, `src/train_ram.py`, `src/train_gnn.py`, etc.)

All 14 paradigms in the README are implemented. Benchmark results in `evals/benchmark_n30_merged.json` (n=30 stochastic + 10 deterministic, 14 models). No model clears the level in 100 k steps; best performers at 100 k: DrQ-v2, CURL, SPR in the 300–430 px range (stochastic mean); GNN (graph-based RAM) and RAM-PPO are working pipelines.

### 1.7 Experiment Tracking

- MLflow: `mlflow.db`, `mlruns.db`, `mlruns/` — all training runs logged
- TensorBoard: `tensorboard_logs/RecurrentPPO_0..20`
- `evals/`: JSON snapshots for every eval run

---

## 2. What Is Missing ❌

Listed in priority order for advancing the ML goal (clearing 1-1 or reaching a publishable result).

### P0 — Correctness blockers

| Gap | Impact | Suggested Fix |
|-----|--------|---------------|
| **Level-finish detection** | Agents cannot get the +100 reward for clearing; reward signal is truncated | ✅ Solved: Hardware flag isolated at `CLEAR_FLAG_ADDR = 0x020DCF27` via differential RAM search |
| **Absolute X ambiguity ±40 px** | Progress reward is correct (delta), but absolute position reporting in evals is off | ✅ Solved: Origin fix in `src/ram_env.py` (4o/4q); `ram_env` discovers Mario before fixing origin; `eval_pinn_realtime.py` reads `mario_B` directly with historical rounding; `evals/pinn_realtime_eval.json` reports `final_mario_px` / `max_mario_px` |
| **PPO policy collapses at ~336 px** | First Goomba on flat ground always kills the PPO agent (87 identical steps across 7/8 episodes → deterministic death loop) | Grounded contact mortality added to imagination + $25\times$ death loss weight in PINN |

### P1 — World model quality

| Gap | Impact | Suggested Fix |
|-----|--------|---------------|
| **Open-loop drift 1738 px over 44 frames** | MPC re-plans every step (mitigates this), but PPO relies on long imagined rollouts where the model is inaccurate | Increase transitions to 10k (currently 3500); add recurrent state in the PINN trunk (GRU head); or use an ensemble of 3 PINNs and take pessimistic lower bound |
| **Enemy dynamics unmodeled beyond relative position** | Enemies move; their relative position changes in ways the PINN doesn't capture well (logvar head marks this as high-uncertainty) | ✅ Solved: Dynamic enemy velocities `(vx, vy)` tracked in `RamState.enemies()` |
| **No contact/death prediction** | The PINN `continue` head is a binary; it doesn't distinguish death-by-enemy from death-by-pit | ⚠️ Partially solved: Explicit physical collision hitbox override in `PINNImaginationEnv` (`src/pinn_world_model.py:124-127`), but the `continue` head remains binary and does not learn cause-of-death |

### P2 — Planning gaps

| Gap | Impact | Suggested Fix |
|-----|--------|---------------|
| **Enemy-unaware A\* planner** | MPC consistently dies at ~1730 px where a Goomba blocks the approach to a pit edge | ✅ Solved: `enemy_threat_ahead` added to `TilemapAStar` |
| **Multi-level generalization** | Planner and RAM addresses only validated on 1-1 | Savestate not a blocker (`tools/enter_course.py`, states for 2-1–8-1 exist); tileset binding route closed (4f/4h: MUCD does not encode solidity). Remaining: no clear outside World 1 yet; courses refused at coverage guard (`C01_1` 26%, `G01_1` 28.4%); probe-based overrides + honest `UnsupportedCourse` refusal are in place |
| **JUMP_REACH_PX hardcoded** | 120 px works for most pits in 1-1 but breaks on wider pits in later worlds | ⚠️ Partially solved: `dynamic_jump_reach_px(vx)` exists (`src/tilemap_planner.py:34`) but hardcoded `JUMP_REACH_PX = 109.73` remains the default constructor value (`tilemap_planner.py:30,50`); doc says `t_air = 48 frames` but code default is `airborne_frames=52.0` |

### P3 — Documentation & reproducibility

| Gap | Impact | Suggested Fix |
|-----|--------|---------------|
| **No formal `docs/` entry for RAM reverse engineering** | The discovery methodology lives only in `ram_state.py` | ✅ Solved: Documented in [`docs/REVERSE_ENGINEERING_REPORT.md`](REVERSE_ENGINEERING_REPORT.md) |
| **Complete Gap Analysis & Roadmap** | Gaps across all pillars not centralized | ✅ Solved: Documented in [`docs/COMPREHENSIVE_GAP_ANALYSIS_AND_ROADMAP.md`](COMPREHENSIVE_GAP_ANALYSIS_AND_ROADMAP.md) |
| **`evals/benchmark_n30_merged.json` not described** | The eval protocol (n=30, seeds, det/stoch split) is in the JSON but not in any doc | ✅ Solved: Documented in [`docs/BENCHMARK_PROTOCOL.md`](BENCHMARK_PROTOCOL.md) |
| **No trained model cards** | Each `.zip`/`.pt` in `models/` has no metadata doc | ✅ Solved: Documented in [`docs/MODEL_CARDS.md`](MODEL_CARDS.md) |
| **README §14 (PINN) not linked to WORLD_MODEL_PINN.md** | The detailed doc exists but README only has a CLI snippet | ✅ Solved: Link added in `README.md:375` and `README.md:98` |

---

## 3. Current Research Status & Next Steps (ordered)

### Completed Milestones ✅
1. **Level-Finish Detection:** Hardware stage clear flag isolated at `CLEAR_FLAG_ADDR = 0x020DCF27` (transitions $0 \to 1$ on flagpole descent / castle entry).
2. **Dataset Export:** 191 stages parsed and normalized into [`out/dataset/levels.jsonl`](../out/dataset/levels.jsonl) and [`out/dataset/entities.csv`](../out/dataset/entities.csv) with corrected 20.12 fixed-point coordinate scaling.
3. **PINN & Policy Retraining:** $25\times$ death loss weighting + grounded physical collision hitbox mortality integrated into imagination.
4. **Player Jump State Machine Decomposition (Route B):** Executed 60 Hz single-frame telemetry captures on DeSmuME via [`tools/decompose_mario_jump.py`](../tools/decompose_mario_jump.py). Exact piecewise gravities ($g_{\text{ascent}} = 0.0919\text{ px/f}^2$ vs $g_{\text{descent}} = 0.2118\text{ px/f}^2$), launch velocity ($V_y = 3.7188\text{ px/f}$), apex hang time (3 frames @ frame 26), and maximum reach ($109.73\text{ px}$ @ $V_x = 2.136\text{ px/f}$) saved to [`out/dataset/physics_jump_state_machine.json`](../out/dataset/physics_jump_state_machine.json).
5. **Pipe Obstacle Anticipation in PPO+Reflex:** Integrated `TilemapAStar` terrain perception into `ReflexivePPOAgent`. Launch window anticipates obstacles $16-55\text{ px}$ ahead, ensuring takeoff before $X = 358\text{ px}$ for the 3-block pipe at $X = 384\text{ px}$. Ballistic trajectory reaches apex ($Y = 71.06\text{ px}$) over the pipe, eliminating wall collisions and advancing PPO+Reflex from $375.6\text{ px}$ to **$1,000.76\text{ px}$ mean** and **$1,139.46\text{ px}$ max**.
6. **Overcoming the 1,498 px Obstacle in MPC:** Extended CEM-MPC predictive rollout horizon to $H=24$ and implemented corridor-specific adaptive deceleration (`compound == "decelerate"` via Action 1: Walk Right) at cols 92–97 ($1472-1552\text{ px}$). Filtered non-hostile entities (MegaDrop, coins) and eliminated phantom row 0 ceiling markers in `Course.ground_cover`. CEM-MPC surpassed the $1,498\text{ px}$ barrier, reaching **$1,510.02\text{ px}$ max** ($1,268.16\text{ px}$ mean).
7. **Academic Documentation:** Added [`docs/REVERSE_ENGINEERING_REPORT.md`](REVERSE_ENGINEERING_REPORT.md), [`docs/MODEL_CARDS.md`](MODEL_CARDS.md), and [`docs/BENCHMARK_PROTOCOL.md`](BENCHMARK_PROTOCOL.md).

---

## 4. Empirical Evaluation Summary Table

| Controller | Mean Progress (px) | Max Progress (px) | Clear Rate | Notes |
|---|:---:|:---:|:---:|---|
| **CEM-MPC (PINN, n=3)** | **1,268.2** | **1,510.0** | **0.0%** | Overcomes 1498 px barrier via extended $H=24$ horizon & patrol deceleration |
| **PPO+Reflex (PINN, n=5)** | **1,000.8** | **1,139.5** | **0.0%** | 100% clearance of 384 px pipe via TilemapAStar takeoff before 358 px |
| **Reactive (Rule-based, n=5)** | **4,020.6** | **4,020.6** | **100%** | Clears to the flagpole; `evals/pinn_realtime_eval.json`. Supersedes the n=4 run (1,136.7 mean, 0%) that died at 1115-1203 px |
| **PPO Pure (Pixels, n=10 det)** | 120.8 | 120.8 | 0.0% | Deterministic Goomba death loop at entrance |
| **Goal (two numbers, not one)** | **4,032.0** | **4,256.0** | — | 4,032 px = flagpole sprite x, what `level_cleared` fires on (`eval_pinn_realtime.goal_px()`); 4,256 px = last solid tile column x16 (`TilemapAStar.goal_px()`). Quote the first for clears, the second for geometry |


---

## 4b. How much of the 1-1 clear is geometry, and how much is tuning

`BaseReflexController` carried seven hand-placed World 1-1 pixel bands: the
3870/3920/4020 endgame push, the 1450..1485 corridor step-up, the 2430..2630
floating-block skip, the `abs_px >= 3700` staircase hop, the 3480..3520 pit-9
leap suppression, `TilemapAStar.is_narrow_corridor` (cols 91..97) and
`SPAWN_ABS_PX = 88`. `--reflex profile` replaces all of them with values derived
from the ROM by `src/level_profile.py` (spawn from the entrance record, flagpole
from the sprite list, the end ramp from the surface profile, ledges and
isolated ?-blocks from column persistence). `tools/parity_reflex.py` runs both
arms:

| arm | steps | final px | reward | outcome |
| --- | --- | --- | --- | --- |
| `--reflex legacy` | 245 | 4,021 | 1663.087 | **cleared** |
| `--reflex profile` | 90 | 1,250 | 429.5542 | died (peak 1,250 px), **short of the flagpole** |

So the bands are load-bearing: with pure geometry the controller gets ~31% of the
way. The stall is not a reflex bug but a **geometry-fidelity** bug -- see 4c.
(Both rows are re-measured by `tools/parity_reflex.py`; the lineage is
nine-pit 209 / 1530.375, true-four-pit 235 / 1529.075, and since 4o's origin
fix -- coordinates in true mario_B -- 245 / 1663.087, with the profile arm
at 90 / 429.5542.)

## 4c. Walkable-surface fidelity fix (2026-10-11)

`Course.floor_row()` returned `min(rows)`, the *highest* ground-class tile in a
column. Ground-class objects also cover floating brick rows and platforms, so a
column's floor was reported up on the platform and the real ground underneath
disappeared. At cols 117..121 of A01_1 the profile claimed floor row 24 while the
rendered frame shows the walkable floor at row 30 with a brick row floating at 24
(`D:/tmp_verify/reality_1889.png` was captured to check this, not assumed).

Fixes:

- `Course.floor_row()` now returns the top of the **bottom contiguous** ground run
  -- the surface Mario walks on -- and `Course.platform_rows()` exposes the
  ground-class surfaces stacked above it separately.
- `TilemapAStar.surface` and `LevelProfile.surface` build from
  `course.ground_cover` (ground classes only). They used to walk
  `course.rects`, which absorbed bricks, ?-blocks and decor into the walkable
  profile -- that is what the hardcoded `2430..2630` reflex band was patching
  over, and why the dashboard's tile panel showed blocks that are not there.

Effect, measured: the pit census is unchanged (9 pits, widths 2-3, `feasible=True`
-- on the static classes as they stood then; the drop probe later measured five
of those nine as slope ramps, 4k/4l),
obstacle columns went 49 -> 60 (heights recomputed against the real floor), the
Rust/Python query parity still passes on all 37,281 sampled queries, and the
`reactive` arm still cleared 5/5 at 4,022.4 px at the time (4o later moved the reporting to true mario_B: 4,020.6)
(209 vs 210), which is why `evals/pinn_realtime_eval.json` was regenerated
(superseded by the 4l re-baseline at 235 steps, then by 4o's origin fix at 245 -- the true flagpole contact).
The pure-geometry arm is still short (1,730 px then; 1,728 on the true
profile), so parser fidelity was necessary but not sufficient: the remaining
gap is in the reflex rules themselves.

## 4d. Two formulas for Mario's absolute position (resolved)

`src/ram_env.py` reports `info["mario_px"] = st["mario_B"] / 4096`, while
`eval_pinn_realtime.py` and the controller use `SPAWN_ABS_PX + obs[0] * 512`.
They are not two approximations of one quantity: `mario_px` is **spawn-relative**
(it differs from the other by exactly the 88 px spawn constant -- 3,934.379 vs
4,022.379), so labelling it "px to the flagpole" against an absolute 4,032 px goal
was wrong by that offset. That was a bug in the dashboard recorder, found by
recording both values in the eval rather than by arguing about them.

Resolved: `run_episode` now writes `final_mario_px` / `max_mario_px` alongside the
existing fields (all previously published values verified unchanged), and
`tools/record_dashboard.py` uses the eval's quantity **and its stop rule**
(`abs_px >= goal - 16` or `level_cleared`). One trajectory, one number: the clip
and `evals/pinn_realtime_eval.json` both reported 209 steps and 4,022 px at the
time of writing; the 4l re-baseline moved both to 235 steps, and 4o's origin
fix to 245 steps and 4,021 px -- the first time the run actually reached
the flagpole (the stop rule's 88 px shift had been ending every run short
of it).

## 4e. Porting to another course: investigation results (2026-10-11)

World 8-1 was the target, so the geometry layer was measured against it. The
answer is worse than "8-1 needs a savestate": **the tile parser is valid for the
grass tileset family and degrades everywhere else.** `Course` now reports
`ground_coverage` (fraction of the course's own x extent that has ground-class
tiles) and `exit_found`, and `tools/tileset_probe.py` reproduces these numbers:

| course | rects | ground coverage | exit object found | plannable |
| --- | --- | --- | --- | --- |
| `A01_1` (1-1) | 197 | **92.1%** | flagpole @ 4,032 px | yes |
| `A02_1` (1-2) | 382 | 85.7% | **none** | refused |
| `A05_1` (1-5) | 170 | **11.0%** | flagpole @ 6,160 px | refused |
| `A08_1` (1-8) | 140 | **3.7%** | **none** | refused |

The level numbers in that table are the ones measured in 4j. When this section
was first written they were guessed as 2-1, 5-1 and 8-1, and the guess is what
the rest of the section then argued about; see 4j for the correction.

Why: `GROUND_OBJS` is a flat set of tile-object numbers, but those numbers are
**local to each course's tileset**. 1-1's floor is classes `0x0A/0x07/0x09/0x06/
0x08/0x00` in its tileset 16; `A08_1`'s terrain is `0x16/0x17/0x18` in *its*
tileset 16, and `0x16` never appears in 1-1 at all. The course header confirms the
theme difference: the 32-byte block at offset 112 carries `0x0190` for A01_1/
A02_1/A05_1 and `0x01f4` for A08_1.

A geometric shortcut was tried and **rejected by measurement**: classifying a
rect as floor-bearing when its bottom edge reaches row 28+ agrees with the class
list on only 62.4% of A01_1's columns and 20.1% of A05_1's, and on A08_1 it
invents an 86-column pit. It would have silently invalidated the published clear.

The authoritative solidity data is `BG_chk/*MainUnitChangeData.bin`, present in
the ROM but listed as undecoded in
[`REVERSE_ENGINEERING_REPORT.md`](REVERSE_ENGINEERING_REPORT.md). So the real
work is: decode the per-tileset behaviour table, key the ground classes by
tileset instead of by a global list, and add per-course exit-object types (`A02_1`
and `A08_1` end on something that is not a flagpole sprite, so `flag_x_px`
currently falls back to 4032 for them).

What is **not** a blocker: a savestate. `py-desmume` exposes
`emu.savestate.save_file()`, so a state at any reachable course can be created by
playing there once. And `LevelProfile` already derives its spawn from the
course's own entrance record (40 px for 8-1 vs 88 px for 1-1), so pointing the
pipeline at another course needs no code change -- it needs the parser to stop
lying, which is now enforced: `TilemapAStar` and `LevelProfile` raise
`UnsupportedCourse` instead of planning on 4% of a level.

The exit-object bug found while doing that scan: **sprite 32 is both the mid-level
checkpoint flag and the level exit**, and `Course` took `min()` over the candidates
while its own comment said "highest X". 17 of the 191 courses have two flagpole
sprites, so those reported the checkpoint as the goal (E02_1 claimed 416 px for a
level running to 1,680 px; G05_1 and B04_1 were off by ~3,700 px). Now `max()`,
which matches the comment and leaves A01_1 unchanged at 4,032 px (one candidate).

After the fix, **29 of 191** courses pass the guard, but the **main path (the `A`
courses) has only three**: `A01_1` (1-1), `A04_1` (1-4, 95% coverage, exit at
3,600 px) and `A07_3`. 1-4 is the natural transfer target: comparable length,
same tileset family, and it needs no parser work -- only a savestate, which
`emu.savestate.save_file()` can create by reaching the level in-game. (Reached
later the same way as 1-2; see 4i. The level numbers here are the corrected ones
from 4j.)

The exit is also constrained to ground the course actually has: a flagpole beyond
the parsed span is a parser gap, not a level end. That demotes `J10_1` (flagpole
at 4,144 px over 1,376 px of ground) from "plannable" to refused, which is where
the count of 30 became 29.

## 4g. Reaching another course headless: the two routes tried on 2026-10-11 both failed (4i supersedes this)

Tried 2026-10-11 on a **copy** of the ROM and the game save (the user's
`data/…sav` was verified unchanged by MD5 afterwards).

1. **Title -> adventure -> map stalls headless.** The boot reaches "Select a Game"
   and "Select a character" fine, then sits on the "Searching for Luigi..." card
   forever. It ignores A, START and the d-pad, and it ignores the stylus
   (`input.touch_set_pos` / `touch_release` exist and were tried at four points),
   while `emu.is_running()` reports True and the core keeps cycling frames. So the
   scene is running but not advancing -- something in that card's progression is
   not satisfied by the headless SoftRasterizer path.
2. **The in-level escape is structurally unavailable.** From the working 1-1
   savestate the pause menu opens correctly (Continue / Return to map / Options /
   Quit) but **"Return to map" is greyed out**, because that savestate was
   created from a direct level start -- there is no map to return to. The menu
   cursor therefore steps Continue -> Options, skipping the disabled item.

Consequence at the time: a savestate at another course needed either the game
booted to its map in the GUI (where the intro plays out normally) or the ARM9
route to start a course directly. **The first half of that sentence is now
obsolete** -- 4i finds a headless route to the map that uses neither the title
screen nor the menu. Item 2 was also only half true: "Return to map" is greyed
out from a state dropped straight into a level, but after a *clear* the same menu
offers it normally.
Verified side effect worth keeping: the pause menu rendered with the corrected
palette -- blue sky, red-shirted Mario -- which is the palette fix confirmed on
live game graphics rather than on my own compositor.

## 4f. `BG_chk/*MainUnitChangeData.bin` decoded (and the binding that is still missing)

Attempted 2026-10-11. What the file actually is:

- 48 real MUCD files, each **3072 bytes = 1536 u16**, laid out as **6 u16 per
  object number** (256 objects), i.e. three `(unit, model)` pairs per object --
  which is what "MainUnitChange" describes: for object N, replace unit X with
  model Y. `0x14` (question blocks in A01_1) is `(0,0),(0,0),(0,0)`; the ground
  classes carry real pairs, e.g. `0x0d -> (7,2048),(8,2048),(9,2048)`.
- **Retraction of an earlier reading in this same investigation:** I first wrote
  that the values are "only 0 or 1, so it is a boolean table". That was true of
  the one file I sampled (`GaKe`, 1518 zeros and 18 ones) and false in general --
  across the 48 files the domain is far larger (1024, 2048, 4096, 16384, 32768,
  45066, ...). The number gate caught it, not a re-read.
- What *does* correlate with terrain is coarser: **"this object number has at
  least one non-zero pair"**. On A01_1 that reproduces the hand-validated ground
  set exactly -- 246 of 267 columns, 92.1% coverage, 9 pits, widths [2 x8, 3 x1]
(the static-class census as it stood then; five of the nine are measured slope
ramps, 4k)
  -- and it separates A01_1's ground classes at 94.9% against a 64.1% baseline
  (`tools/mucd_probe.py`). **Read 4h before using this as a solidity source: it
  holds for the grass tileset and fails as a general rule.**

What is **not** solved, stated as a negative result so nobody re-pays for it: the
course -> tileset binding. It is not the ROM file ID (MUCD IDs are 262..302; the
course header carries 400/500 and 8/1/33/0 instead), and it cannot be recovered by
fitting geometry to tables: scoring all 48 tables against each course's own floor
signature gives a unique winner (margin > 0.05) for only **10 of 128** courses, and
courses sharing a header index disagree on the winner (2/5, 5/9, 1/4 ...). The
tables are similar enough that "best fit" is noise -- an earlier read of this
experiment that treated A08_1's `F1 = 0.989` under `SaBaKu` as a binding was
wrong; that table still leaves an 86-column hole in 8-1, so it is not 8-1's.

Consequences: 8-1 stays blocked, now with a precise reason -- the solidity source
is decoded, but picking *which* of the 48 tables a course uses needs the game's
internal tileset table, i.e. ARM9 work (the strings are not in the ARM9 because
files are referenced by ID, and the ARM9 is uncompressed -- `e7ffdeff` fill, not a
container). The practical alternative is per-course identification against
rendered frames, which needs a savestate at that course.

Note: `BG_chk/` also holds six 512-byte `ScrollControl_*.bin` files (UI scroll
flags, 256 u16 each). They are not MUCD tables and must be excluded from any
count of 48.

## 4h. The ARM9 route, attempted and closed with numbers

Authorized attempt to recover the course -> tileset binding from the executable.
Five approaches, all measured, all failing:

1. **Strings in the ARM9:** none. But note the method check -- control strings
   ("main", "NSMB", "Tile", "http") also return zero hits, so the earlier claim
   that "the ARM9 holds no tileset names" was made with a broken instrument. The
   names live only in the ROM's filename table (0x2548b6..0x254da4); the game
   opens files by ID.
2. **A static table of MUCD file IDs (259..313) in the ARM9:** scanned as u16 with
   stride 2, u32 low-16 with stride 4, u16 at +2 with stride 4, and runs of length
   >= 6 -- **zero runs found**.
3. **Header field as an index into the BG_chk file-ID order:** swept all 16 u16
   slots of the 32-byte level header. Best is 43.5% of courses reaching >= 70%
   ground coverage; nothing crossed 50%.
4. **Per-course geometric fit:** degenerate by construction -- six tables score
   J=1.000 against A01_1, and group fitting (positive *and* negative evidence over
   the 191 courses) gives median winner margins of 0.000.
5. **Local floor estimation** (median deepest row in a +-12 column window): lifts
   A01_1 to 95.9% but erases five of its nine real pits and collapses A05_1 to
   1.2%. Rejected.

The bound that settles it: choosing the **best possible table per course** (an
oracle) still leaves only **122 of 191** courses reaching 70% ground coverage. So
the flag reading from 4f -- "this object number has a non-zero (unit, model) pair,
therefore it is terrain" -- is **wrong as a general rule**. It reproduces A01_1's
validated ground exactly because the grass tileset's ground classes happen to be
the classes with unit-change entries there; it does not mean the table encodes
solidity.

Consequences, stated plainly:
- The binding is not the thing that unlocks other courses, because the tables do
  not carry the semantics I assumed. What is actually needed is the tile
  collision data (`BG_chk/*MainUnitChangeData.bin` is not it; the map16/tile
  behaviour lives in the tileset graphics and collision archives, `BG_ncg`,
  `BG_ncl`, `BG_pnl` and the `ARCHIVE/Dat_*.narc` files), plus a disassembler and
  the community's tileset-ID knowledge. Neither is available on this machine
  inside a reasonable budget.
- **4-1 does not need any of this.** `A04_1` already passes the geometry guard at
  95% coverage with a detected exit; its only blocker is a savestate (4g). So the
  transfer experiment is gated on the emulator path, not on reverse engineering.
- **The plannable `A0N_1` courses do not need any of this.** `A04_1` already
  passes the geometry guard at 95% coverage with a detected exit (4j re-reads
  which level that is); its only blocker was a savestate, and 4i removes that. So
  the transfer experiment was gated on the emulator path, not on reverse engineering.
- `tools/tileset_sweep.py`, `tileset_groups.py`, `tileset_posneg.py` and
  `mucd_bind.py` are kept with their success criteria in the docstrings, so this
  ground is not re-covered.

## 4i. The map is reachable headless: clearing a level returns you to it (2026-10-12)

The route is the game's own: **touch the flagpole, and the level ends by putting
Mario back on the World 1 map.** No title screen, no menu, no stylus, no save
editing. `tools/reach_map.py` automates it end to end:

1. drive the published reactive agent to the clear (219 env steps on the
   trajectory as it stood then, pre-4l; the re-baselined clear runs 235);
2. keep cycling. The clear animation, the castle walk and the fade take **540
   cycles (9 s of game time)**; the fade-to-black frame is the marker;
3. the map appears with 1-1 flagged and the next node unlocked, but it **slides in
   and drops d-pad input for ~30 s**: a RIGHT pressed too early is lost, and A then
   replays the level just cleared. The tool waits 1800 cycles before touching it,
   and compares the spawn it lands on against the one it started from, so a cursor
   that never moved is a warning rather than a silently wrong savestate;
4. `keypad_add_key(KEY_RIGHT)` moves the cursor one node east, `KEY_A` enters.

Re-running the finished tool reproduces the state the transfer experiment used:
`--node 1` enters with `mario_B = 40.0 px, mario_A = -416.0 px`, the same two
numbers measured from the manual run that produced `data/state_1-2.ds1`.
(Those states were made with the pre-4l agent; the shipped .ds1 files are
unchanged, and re-running today reproduces the sequence with the 4l clear's
slightly different timing.)

**What the map does not give: another world.** The world row (`w1..w8`) along the
bottom of the map screen is stylus-operated, and the stylus is inert in
`py-desmume` 0.0.9 -- seven attempts across two screens (four on the title card
earlier, three here on the world row and on two map nodes) all changed the frame
by 0.6-0.9, which is the ambient drift, while every button arm in the same
process produced a real response. The d-pad moves the cursor (+23.0), and **L and
R pan the camera** (+19-20, with the yellow edge arrows appearing) without
changing worlds: after eight R presses and four L presses the plate still reads
"World 1" and the marker is still on 1-1. So the map is walkable east inside
world 1 only.

That is also why a downloaded "everything unlocked" `.sav` would not have helped
here, and the reason is measured rather than assumed: the 1-2 unlock appeared on
the map **while `data/…sav` stayed byte-identical** (`b7458c3445fd2c9949aee7e312dd4de5`,
checked before and after). A savestate restores RAM, and RAM is what the map
reads, so an on-disk save is not in the loop at all -- and the only path that
would consume one is the title-screen boot, which is the route that never
advances headless. Where a full-completion save *does* pay off is in a GUI
emulator: one human session to boot, tap to world 8 and save a state, and every
tool in this repo can then run on it.

That save no longer has to be hunted for: `tools/make_save.py` builds it from
the user's own `.sav`. The unlock field was found by validating the ARDS
download `new-super-mario-bros.16742.dst` (500-byte header wrapping this exact
8 KB save format, 10/10 checksums from the ported routine) and diffing it
against the local save: **+0x6A per file slot, 8 x u16, one per world** --
`43 00` in every early-progress slot (all three local ones *and* the
download's own untouched slot 3) versus `ff 03 ff 03 ff 03 ff 03 ff 03 ff 07
ff 03 df 07` in both of its beaten slots. The C# editor's recipe had the
offset but the wrong value and one byte too many: its 17th write lands on the
slot sentinel (`d7`/`d0`) and clobbers it. The tool copies the 16 measured
bytes instead, mirrors them into both 4 KB blocks, and rewrites the
checksums; the emitted `data/save_unlocked.sav` verifies 10/10 with zero
bytes changed outside the field and its checksums. An earlier claim in the
tool's own docstring -- that this field "does not gate the world row" because
it is byte-identical across the three local slots -- was the wrong inference:
identical because all three slots were played to the same early point, which
is exactly what the untouched-vs-beaten contrast in the download disproves.
The GUI session this save was built for has since happened: one human run
produced savestates on the World 5-1 and World 8-1 plates (`.ds2`/`.ds3` under
`data/`), identified and put to work in 4j.

The route that stays inside the inputs that work is the game's own world
advance: clearing a world's last node hands you the next world's map, exactly the
mechanism 4i uses to reach world 1's map. It needs the earlier worlds marked
cleared in RAM first, which is a search this repo has not done.

Two things that looked like blockers were artefacts of my own inputs. Holding
START during the clear sequence opens the pause menu, which *pauses the sequence*:
an earlier attempt sampled every 5 s for 60 s and saw a frozen castle scene, and
a longer one with a button press every second reported the level "resuming" for
240 s before the fade. Both were the menu toggling, not the emulator stalling.
The sequence needs no input at all.

Everything ran on a copy of the ROM, save and savestate under `D:/tmp_verify/
playdir`; `data/…sav` was verified byte-identical afterwards
(`b7458c3445fd2c9949aee7e312dd4de5`, unchanged from the pre-experiment reading).
The emulator writes `.dsv`-style saves next to the ROM it opens, so the original
`data/` save was never a candidate for it.

## 4j. What the course file names mean, measured, and the first transfer experiment

**The naming guess that ran through 4e was wrong.** 4e labelled `A02_1` as 2-1,
`A05_1` as 5-1 and `A08_1` as 8-1 by pattern-matching the file names against how
NSMB's worlds are numbered on screen. Two courses were then identified from the
emulator itself, and that reading does not survive:

| course | how it was identified | level the HUD calls it |
| --- | --- | --- |
| `A01_1` | the shipped savestate; spawn 88 px = entrance 80 px + half a tile | 1-1 |
| `A02_1` | entered from the map node east of 1-1; HUD plate reads "W 1-2" | **1-2** |

`A02_1` is 1-2, not 2-1, so the digit is the level and the letter is the world --
which means every "8-1" statement in 4e was actually about **1-8**. Extrapolating
that rule past letter A was untested when this was written: it predicted the real
World 8-1 is `H01_1` (184 rects, 78.8% coverage, exit candidate at 3,920 px), a
file this repo had never examined. Confirmed below: the 8-1 savestate measures
as `H01_1` and the 5-1 one as `E01_1`, exactly where the rule puts them. The
3.7%-coverage finding stands; what was wrong is which level it describes.

Three independent measurements tie the running level to `A02_1`, because the HUD
plate alone is a pixel reading and I did not want to build on that:

1. **spawn**: RAM reports `mario_B / 4096 = 40.0` px; `entrances[0].x_px + 8 = 40`
   for `A02_1`. This also re-validates the half-tile offset constant on a second
   course, which is the first time it has been tested outside 1-1.
2. **surface rows**: holding RIGHT and logging Mario's `y` gives the top of the
   ground at ten columns (26, 26, 30, 29, 28, 27, 27, 27, 29, 29). Scoring all 191
   courses by how many of those rows appear as a rect top at that column ranks
   `A02_1` first at **9/10**, next best 7/10.
3. **one enemy**: RAM shows an object at x=746 px, y=-464 px. `A02_1` has a Goomba
   sprite at (752, 448) -- the 6 px is the walk, and the y matches the calibrated
   `object_y = -(sprite_y + 16)` exactly.

**First transfer result.** With a savestate for a course the pipeline had never
run on, both reflex arms were measured against it. Goal = 4,688 px, the last
parsed ground column, passed explicitly with `--goal-px` because `A02_1` has no
flagpole sprite (4e); the JSON records `goal_source: "operator --goal-px"` so the
metric cannot be misread as a level clear.

| arm | distance | of goal | steps | reward | outcome |
| --- | --- | --- | --- | --- | --- |
| `--reflex profile` (ROM-derived thresholds) | 1,759.8 px | 38% | 224 | 1637.6 | death |
| `--reflex legacy` (the 1-1 pixel bands) | **3,274.4 px** | **70%** | 313 | 3146.5 | death |

n=5 per arm (byte-identical within each arm). The lineage of these two rows
is the whole coordinate story in miniature. First pass (obs accidentally
absolute): profile 1,929.6 / legacy 3,314.4 -- the historical doubles.
4o's origin fix made the legacy reconstruction (spawn_mb + 88 + rel*512)
differ from the historical by ~0.04 px of float32 noise, and 1-2's pipe
window sits on exactly that knife-edge (call 38: identical inputs, one arm
leaps, the other runs): legacy fell to 722, profile to a 201 px stall.
**The resolution is in the code now: the legacy arm reads mario_B from
the RAM handle and reproduces the historical arithmetic bit for bit**
(`88 + float32(mb/4096/512)*512` -- the bands were calibrated against
the ROUNDED value, so the rounding is part of the tuning). The legacy
row is back at **3,274.4 px / 70%** (the historical trajectory minus the
spawn double-count: 3,314.4 - 40), and the profile row moved because 4p's
swim escalation freed its 201 px stall -- both rows are now the honest
measurements of the current stack. The old "bands transfer better"
ordering is back, by the way. The pit-vector reading correction below
stands regardless. Both historical trajectories died with
`pit=[1,1,1,1,1,1] ground=1` in the death diagnostic -- which this
section first read as "the pit features say ground
is missing ahead while the RAM says Mario is standing", and that reading was
**backwards**: `pit_ahead` returns 1.0 for *solid* and 0.0 for *pit*, so
`pit=[1,...]` said ground exists ahead, in agreement with `ground=1` and with
the level. There was no detection disagreement at the 1-2 deaths (4k's drop
probe later confirmed the parser has zero false floors on every probed course);
what actually failed was the leap itself, and 4k's autopsies name the pattern.

Reporting fix that came out of this: `run_episode` used to add the 1-1 spawn
constant (88 px) to every arm's distance. On a course whose spawn is at 40 px that
inflated the legacy arm by 48 px while the profile arm anchored on its own course,
so the two arms were not on the same ruler. The anchor is now derived from the
course for reporting (`spawn_anchor_px` in the JSON) while the legacy controller
keeps reasoning in 1-1 pixels, which is what makes it the legacy arm. For
`A01_1` the value is 88 px and the published numbers are unchanged -- re-verified
byte for byte by `tools/parity_reflex.py` after the edit.

**The GUI session the unlocked save was built for has happened, and it carried
the naming rule past letter A.** Two savestates came back from one human DeSmuME
session booted with `data/save_unlocked.sav` (4i): `.ds2`, saved from the plate
the HUD calls World 5-1, and `.ds3`, from World 8-1. The plates were not taken
on trust -- each state was identified from the emulator itself, with the
entrance formulas above and, where the entrance tied, the floor rows a
RIGHT-only walk actually stood on (`tools/identify_state.py`; boot frames
committed as `media/nsmb_w5_1_spawn.png` / `media/nsmb_w8_1_spawn.png`):

| savestate | entrance match over all 191 courses | walk-right floor rows | verdict |
| --- | --- | --- | --- |
| `.ds2` | spawn (56, -480) -> `E01_1` entrance[0] (48, 464), unique on the pair | 2/2 grounded samples, row 30 | **`E01_1` = 5-1** |
| `.ds3` | spawn (56, -736) -> `H01_1` entrance[0] and `F07_2` entrance[2], tied at (48, 720) | `H01_1` 4/4 at row 46; `F07_2` 1/4 | **`H01_1` = 8-1** |

Both courses parse with exits: `E01_1` flagpole at 4,576 px over 96.0% ground
coverage, `H01_1` at 3,920 px over 78.8%.

**Second and third transfers, n=5 per arm, re-measured in true mario_B by
4o's origin fix.** The historical numbers (measured while obs was
accidentally absolute, so every distance carried the course's spawn on top
of mario_B) are kept in the line below each row. Goal from the course's own
flagpole sprite.

| course | arm | distance | of goal | steps | reward | outcome |
| --- | --- | --- | --- | --- | --- | --- |
| 5-1 `E01_1` | profile | 998.2 px | 22% | 80 | 424.1 | death |
| 5-1 `E01_1` | legacy | 1,009.1 px | 22% | 101 | 428.5 | death |
| 8-1 `H01_1` | profile | 466.5 px | 12% | 43 | 159.6 | death |
| 8-1 `H01_1` | legacy | 565.0 px | 14% | 50 | 209.0 | death |

(4o shift: 5-1 profile 998.2 = the old 1,052.4 minus E01_1's spawn 56;
legacy 1,474.2 = 1,530.7 - 56; 8-1 profile 466.5 = 522.5 - 56... the 8-1
profile row moved further, to 466.5, because 4o also changed which arm the
probe diagnosis rides on that course. The lineage on 8-1 profile: 337 ->
522 -> 660 -> 621 -> 466.5 px.) 5-1's profile arm gets *less far* than the
pre-detection 1,718 because the spinner at x~944 (RAM type 247, measured
to kill on the first contact frame) is now seen and dodged -- the dodge
leap predates any landing check and lands in the adjacent pit, the same
policy failure the 4k autopsies measure directly. No arm has yet cleared
a course outside world 1, and the blocker is no longer detection: it is
the reflex's leap-on-threat policy on 5-1's platform complexes, measured
in 4k.

## 4k. The tile map measured against the game, and the reflex's real failure modes (2026-10-12)

**The drop probe: Mario as the solidity oracle.** The decomp names
`CollisionMgr::getSolidTileType(x, y)` but its body is not decompiled, the
ARM9 route is closed (4h), and the course->tileset binding is still unknown
(4f). `tools/probe_floor.py` takes the other end: py-desmume can *write*
memory, so the probe teleports Mario above every column and lets the game's
own physics answer. The teleport took three measured discoveries: the
player's x authority is `+0x5C` (the synced actor B at `+0x68` follows it;
writing B does nothing), y authority IS the A field, and after a mid-run
death the player object re-allocates -- a second type-0x1C copy keeps reading
spawn coordinates, so the live one is found by testing whose `+0x5C` write
moves its own `+0x68`. The camera cannot be teleported (enemy linking and
streaming follow it), so each drop pins until the y-write *engages* -- in
unloaded space the game holds the player at y=0 and nothing is measured.

784 columns probed across the three savestate courses (A01_1 251, E01_1 287,
H01_1 246; `out/floor_probe/*.json`):

- **Zero confirmed false floors.** Every candidate fall was a probe artifact
  or -- the interesting case -- a ride on a moving platform: E01_1 col 213
  "falls" at 0.2 px/frame for 200+ frames, i.e. it *lands* on the descending
  platform the static parser cannot see but whose existence agrees with its
  floor claim.
- **The real falsehood class is fake pits: slope ramps the flat class list
  misses.** On A01_1, **five of the nine published pits are ramps** (cols
  51-52, 59-60, 64-65, 147-148, 206-207: Mario lands on them at rows
  28-29.75; objects 0x48/0x49/0x4A/0x4B/0x35/0x66 of the grass tileset). On
  H01_1, three floor columns sit on kazan objects 0x15/0x16.
- **Hover is not floor**: a landing stable at the drop height is a cave roof
  or high structure (H01_1's ceiling produces 27 of them); a landing far
  below the dominant floor is a pit's inner shelf. Neither overrides the
  parser -- the walker cannot use either surface.
- **Corrections to this document's own earlier readings:** `pit_ahead`
  returns 1.0 for *solid* (fixed above, in 4j); and the 8-1 opening "pit at
  x~258" is not a pit -- it is a **crumbling-block bridge** (the probe lands
  on it in under 12 frames; a 32-frame stall collapses it, which is what
  killed the jump-less identification trace).

**The measured overrides.** `Course` now loads `out/floor_probe/<id>.json`
and prefers a confirmed measurement wherever it contradicts the static
classes (`floor_row`/`solid_columns`; see `_load_floor_probe` for exactly
what counts as confirmed). H01_1's three columns are active -- its pit list
shrinks accordingly. **A01_1's override was at first deliberately staged
off**: with the true four-pit profile the legacy arm died at 1,338 px and
`tools/parity_reflex.py` failed all four metrics, because the published
clear measurably depended on leaping the five phantom pits. 4l resolves
this: one policy rule (the leap landing cone) re-tunes the clear onto the
true profile at 235 steps (4o: 245), and A01_1's overrides are now on.

**Hostility measured by contact** (`tools/probe_hostility.py`): species names
are unreadable (the decomp's classes for those slots are stubs), but the game
can answer contact directly -- pin Mario onto the actor, drop onto it, or
walk there with the agent when it only links near the camera.

| RAM type | where | verdict |
| --- | --- | --- |
| 172, 184, 312 | `H01_1` | **hostile** -- all three stomp-vanish on contact |
| 247 | `E01_1` | **hostile** -- kills on the first contact frame |
| 180 | `E01_1` | harmless: 140 pinned contact frames and repeated drops |
| 258 | `E01_1` | no death in direct contact frames (leaper) |
| 291, 307 | `E01_1` | static foam at the level edge, deep below |

`HOSTILE_ENEMY_TYPES` gains 172/184/247/312; parity still passes because none
of those ids appear in 1-1's entity list.

**The autopsies: what actually kills the arms outside world 1.**

- *5-1* (trace, `E01_1` profile): the fatal leap is **enemy-triggered ~90 px
  before the pit** -- at takeoff the pit vector correctly reads all-solid;
  the full-height arc peaks, descends past the lip, and Mario lands *inside*
  the 4-tile pit, wedges into the east wall at row 35.6, and the kill line
  finishes him 7 steps later. The leap has no landing-safety check.
- *8-1* (trace, `H01_1`): he runs into the level's opening structure at
  x=281, the blocked-logic jumps **in place on the crumbling block**, and the
  block expires under him. The pit vector was never wrong once.

Both are reflex *policy* failures -- leap-on-threat and stall-at-wall -- with
the detections now measured true. Fixing them (a landing-safety rule during
descent, a no-stall rule over decaying floors) is the concrete open work this
section leaves behind, alongside the A01_1 re-tune above.

## 4l. The re-tune: one policy rule puts the 1-1 clear on true geometry (2026-10-12)

The 4k autopsy of the 1,338 px death named the mechanism exactly: the
goomba-triggered leap took off from a *slope*, the slope-launched apex
measured 83 px (vs the 71 px flat calibration), the touchdown ran ~55 px past
the nominal 109.73 px reach, and the arc came down inside the real pit at
cols 76-77, wedging into the far wall. The published clear had survived the
same stretch because the phantom-pit leap at col 63 repositioned it first --
a dependency on the parser's lie.

The fix is one rule, and it is physics rather than pixel bands: **a
threat-triggered leap is a commitment of ~110-165 px with no way back, so it
is only made when the landing cone has ground.** `_threat_landing_unsafe`
queries the planner's own `surface_ahead` over cols `reach-1 .. reach+4`
(the +4 is the measured overshoot; a cone that only covers the nominal arc
is a cone the game has already been measured to miss) and when the cone
overlaps a pit the reflex hops in place instead -- the threat passes
underneath, and the pit's own edge-leap, which is never suppressed, clears
both. A no-stall retreat burst (straight to the [LEFT x4, RUN x8, JUMP]
escalation instead of in-place jumps) is added for the profile arm only,
where measured stall floors expire; the legacy arm keeps the published
escalation.

The measured result, all on the true four-pit profile (`A01_1.json` now
`apply_overrides: true`):

- **legacy clears again, 5/5** -- 235 steps in the historical coordinates
  (was 209 on the phantom-pit profile), reward 1529.075, pixel bands
  untouched: the clear generalizes to true geometry through the one
  physics rule.
- profile (pure geometry, no bands): 159 steps, died at 1,816 px -- the
  bands remain load-bearing for the full clear, unchanged story.
- `evals/pinn_realtime_eval.json` re-published from that n=5 run;
  `tools/parity_reflex.py` passes against it (legacy == published, all five
  metrics); `tools/export_geometry.py` regenerated the Rust planner's
  geometry with the true pit census (4 pits, 4 takeoffs); the README clip
  re-recorded on the new trajectory (31 s, 1,880 frames = 235 steps x 8).
  4o's origin fix later re-measured the same trajectory in true mario_B:
  245 steps, 4,020.6 px -- the first contact with the flagpole itself.
- The transfer table in 4j picks up the same rules: 8-1 profile 522 -> 660
  px (the no-stall retreat); 5-1 unchanged at its measured failure points.

What is still open after this: the 5-1 leap-on-threat policy around its
platform complexes (the 4k autopsies; the cone rule cannot help a mid-air
goomba collision inside a safe cone), and a clear outside world 1 by any
arm -- 4m takes the second half further than any prior session.

## 4m. The world row, unlocked from RAM: every world's 1-1 is now a savestate (2026-10-12)

The stylus was the last input the headless pipeline could not produce, and
it gated exactly one thing: the world row on the map (`4i`). The py-desmume
touch API is measured dead twice (ambient drift only, four points, both
coordinate scales, per-frame holds and full tap cycles), but 4k's
memory-write capability made the question moot from the game's own side.

**What the A-press actually commits is the running course's area index** --
a u16 at 0x020980D4 with a copy at 0x0209B59E, whose value is the course
file's position among the 191. Calibrated on the four savestates this repo
already had and verified against the ROM's own file order: A01_1=0,
A02_1=3, E01_1=88, H01_1=152. Writing the Stage group/id/area globals alone
(the decomp's `Stage::stageID` at 0x02098054, DebugScene's route) through the
fade still loaded 1-1 -- the map commits early, and the load reads whatever
stands at scene-create, so `tools/enter_course.py` re-writes the course id
and the Stage globals (for the intro plate) after **every** cycle of the
transition. The last writer wins, and the tool is the last writer.

Five new savestates made this way and identified by entrance + walk trace
(`data/state_*.ds1`, gitignored like every state): **2-1 `B01_1`, 3-1
`C01_1`, 4-1 `D01_1`, 6-1 `F01_1`, 7-1 `G01_1`**. The identification of the
4-1/7-1 states ties at one grounded sample (D01_1 vs F03_1, D07_1 vs
G01_1); the injected, calibrated area index is the stronger evidence there.

Static parse: `F01_1` 83.5% coverage with a real flagpole at 4,080 px (the
drop probe then measured 253 of 256 columns in agreement, zero real
falsehoods -- the kaigan tileset parses); `B01_1` 78.2% but no flagpole
sprite (goal passed explicitly at 4,208 px, the last parsed ground column --
the 4j protocol), with four measured slope columns overriding; `C01_1`
(1.6%), `D01_1` (48.6%, under the 60% guard) and `G01_1` (21.6%) stay
refused pending their own probes.

Two more hostile actor types measured by contact (`tools/probe_hostility.py`,
stomp-vanish on touch): 38/158, the `F01_1` fish that killed the first 6-1
run at 609 px, and 232, `B01_1`'s desert crawler. `HOSTILE_ENEMY_TYPES`
carries them; parity still passes (none appear in 1-1's entity list).

**First measurements on the new states** (n=5 per arm, same protocol):

| course | arm | distance | of goal | steps | reward | outcome |
| --- | --- | --- | --- | --- | --- | --- |
| 2-1 `B01_1` | profile | 862.7 px | 21% | 39 | 234.4 | death |
| 2-1 `B01_1` | legacy | 862.7 px | 21% | 39 | 234.4 | death |
| 4-1 `D01_1` | profile | 505.0 px | 12% | 43 | 391.8 | death |
| 4-1 `D01_1` | legacy | 505.0 px | 12% | 43 | 391.8 | death |
| 6-1 `F01_1` | profile | 1,136.3 px | 28% | 84 | 488.9 | death |
| 6-1 `F01_1` | legacy | 1,929.0 px | 47% | 1200 | 844.5 | **trunc (alive)** |

**4-1 unlocked by measurement.** `D01_1` parsed at 48.6% (the ice tileset's
objects are all unknown to the flat class list) and the drop probe measured
**142 missed floor columns**; `require_plannable` now counts the
probe-corrected columns, so 4-1 sits at **85.1% measured coverage** and is
plannable. `G01_1` gained 134 the same way but stays honestly refused at
28.4% -- the cloud level's gaps are real gaps. `C01_1`'s probe cannot even
bound itself: its static parse is 1.6% and its "flagpole at 272 px" is a
false sprite the tooling had not met before. Two more hostile types
measured on the ice: 135 and 311, both stomp-vanish.

The 4-1 rows die in a **pocket trap** -- or so the trace autopsy read:
Mario frozen at x=505 with the koopa patrolling in place 12 px ahead
(its dx constant for a dozen steps), the fall landing on it. The 4r wall
oracle retracts the "pinned between sprite solids" half: all 63 cells
of cols 29..35 x rows 20..28 are free, so whatever froze the run was
not a solid -- the same self-inflicted-retreat class as 2-1. (This
autopsy first said "on a floor that expires"; 4p's crumble census
retracted that reading for the 2-1 and 6-1 tiles, and D01_1 col 31 is
untested -- the claim is kept only for the 8-1 bridge, which measures
17 frames.) That autopsy
killed the fifth rule candidate for the mid-air contact class: an
approach-direction hop (hop over walkers whose dx is closing, leap over
everything else) was implemented with a first-sighting wait and its own
landing cone, and was **neutral on its target** -- the koopa there never
approaches, it patrols -- so it was reverted too. The class ledger now
reads: level-diff row overrides (broke 1-1 parity), no-wall escape jump
(F01_1 1,601-alive -> 1,250-dead), behind-solid retreat guard (same
regression via the parser's own gaps), stomp_hop (same regression), and
approach-direction hop (neutral: the D01_1 killer never approaches). All
five with their measurements live in the code comments where each rule
used to be.

The conclusion the five share: **the remaining deaths are terrain
semantics, not decision timing** -- sprite solids, expiring floors,
patrolling-in-place enemies -- and bgdat-plus-rules cannot carry them.
The measured way forward is the probe itself: the drop probe already
LANDS on sprite tops (B01_1 col 39's 20.35 came from the probe, not the
parser), so a sprite-solids map derived from probe-vs-bgdat disagreement
is the infrastructure this needs next, and it is derivable from data this
repo already collects.

The reflex therefore stands at its measured-best state: parity passes, the
1-1 clear holds on true geometry, and the named open problems are: (a)
terrain semantics -- sprite solids, expiring floors, patrol-in-place
pockets (measured on four courses: E01_1, B01_1, D01_1, F01_1's stall),
with the probe-derived sprite-solids map as the named next
infrastructure, (b) the 6-1 cliff stall, (c) C01_1's parse.

The 6-1 rows, re-measured in true mario_B by 4o, with 4p's swim
escalation, and with 4q's RAM-exact legacy coordinate: the legacy arm
crossed the water wall at x=1481 that had stalled it for two sessions,
passed the second structure at x=1856 (the knife-edge flipped in its
favor when the legacy abs went RAM-exact -- 1,856-death became a pass),
and now reaches **1,929 px (47%), ALIVE at the 1200-step cap** -- both
the deepest reach and the only surviving arm outside world 1. The
profile arm is unchanged at 1,136.3: it dies before the water.

The 2-1 death: this autopsy was retracted twice, and the wall oracle
(4r) is what settled it -- there is no wall at all at x=613.7. The
original reading ("sprite-solid geometry, x frozen even airborne") was
the retreat queue's own LEFT holds read as a wall; plain RIGHT walks
straight through the spot, off the ledge edge (which the parser reports
as row-27 floor while the walker stands at row 21 -- the level-diff
class), down to the base floor and on to 665. The reflex dies there
because the blocked response knocks it off the edge into the 232
crawler's patrol. (The ledge itself does NOT expire -- 4p's crumble
census held it 1,465+ frames under a fresh stand. With 4q's RAM-exact
legacy coordinate both arms now die at the same 613.7 px spot.)

## 4n. What a working controller cost, by era (2026-10-12)

`mlflow.db` holds every finished training run this repo ever made (34 of
them), and the current learned stack writes its own durations into
`evals/*.json`. Measured:

| era | what was trained | wall time | result |
| --- | --- | --- | --- |
| 2026-06 (visual RL) | RecurrentPPO, one run | 1,376 min (22.9 h) | no clear |
| | IMPALA-CURL, one run | 1,135 min (18.9 h) | no clear |
| | pixel PPO 100k steps, 7 arms | ~48-50 min each | no clear (pixel PPO: 120.8 px) |
| 2026-09-06 (world models) | GA/CMA world models + ICM PPO | 44-101 min each | no clear |
| 2026-09-10 (RAM era) | ram_ppo / ram_geo / gnn, 100k steps | 45.4 / 46.1 / 47.8 min | no clear |
| 2026-09-23 (PINN) | world model Stage 1 | **52.6 s** (3,500 transitions, 43,440 params) | enables 418 steps/s imagination |
| | Dyna-PPO Stage 2, 200k steps | **478 s (8.0 min)** at 418 steps/s | no clear (learned arms peak at 1,510 px) |
| 2026-10 (measured reflex) | nothing trains | **0** | **1-1 cleared 5/5; 6-1 alive at 50%** |

Total finished training wall time in mlflow: **56.0 h across 34 runs -- and
not one of those hours produced a clear.** The clear came from measurement
instead of gradient. The per-course bring-up cost today is one-time
calibration, roughly half an hour per course: `enter_course` (~3 min),
`identify_state` (~2 min), a floor probe (~10-15 min), hostility oracles
(~2 min per type), and the n=5 eval (~3-8 min at ~20 fps headless) -- and
the reflex itself iterates in minutes, not epochs.

## 4o. obs is absolute, not relative: the origin bug, measured (2026-10-12)

Building the sprite-solids map (4m's named next step) needed the reflex to
know Mario's own row. The first attempt diagnosed a block 22 columns east of
where he stood -- and the chase exposed a bug that has been in
`ram_env._observe` since the beginning:

**On the first poll after a fresh load the player object is not yet in the
entity list, `poll()` returns None, and the origin silently becomes 0.** From
the second poll on, `obs[0]*512` equals `mario_B` exactly -- measured: second
poll 2.0 px with `mario_B` 90.0 on 1-1, and the pocket at B01_1 reporting
973.7 in "abs_px" while the true pocket sits at 613.7. The docstring's
"relative to episode reset" has never been true.

What shipped on top of it, all measured:

**The re-baseline, done.** The origin fix is in (`ram_env` discovers the
player before fixing the origin), the 1-1 clear is re-published in true
mario_B -- **245 steps, 4,020.6 px, 5/5, the first trajectory that actually
reaches the flagpole** -- parity passes against it, the clip is re-recorded
(33 s), and every table row above is re-measured in the true coordinate.

Reproducing the historical trajectories under the fix required keeping
legacy's coordinate at `mario_B + 88` explicitly (`_abs_px`: the course's
own spawn mario_B + 88 + relative), because the bands and the tuned planner
windows were calibrated in the shifted space. On 1-1 and B01_1 the
reproduction is byte-identical (verified by per-call action logs); on
A02_1 it is **not**, and the reason is worth keeping: obs is float32, and
the two arithmetics (`88 + mb` vs `base + 88 + rel`) reconstruct mario_B
with ~0.04 px of different rounding -- invisible everywhere except where a
decision sits on a comparison boundary. The 1-2 legacy run has exactly one
such knife-edge (its pipe window at call 38, measured: identical inputs,
one arm leaps, the other runs), and the flipped branch compounds into a
completely different trajectory: the historical "3,314.4 px" became an
honest 722.0 px. The 1-2 rows in 4j are the new measurements; the
knife-edge is why they moved, and it means the historical cross-course
comparisons (the old "legacy transfers better at 71%") were standing on a
comparison that a float rounding can flip. The rows that could be
reproduced byte-for-byte (1-1, 5-1, 8-1, B01_1) carry their lineage in
the tables; the ones that could not, say so.

One further trap from the fix, on record: `_mario_pos()` returns `(A, B)` =
(y, x); unpacking it as (x, y) puts the y in the x origin and sends the
legacy arm into the pipe at 71 steps -- caught by the per-call action logs
before it could pollute anything.

**The probe-solids map, landed before the re-baseline:** `Course` exposes
`probe_surfaces`/`probe_surface` (the full measured map) and
`LevelProfile.probe_rise_ahead`; the profile arm's blocked branch diagnoses
via the probe -- a surface 0.5..1.5 rows above -> hop onto it, 1.5..4 ->
leap at it, measured ground with nothing above -> ride it, unprobed ->
retreat. At B01_1's pocket it fires a hop onto the sprite block at col 39
(probe surface 20.35) instead of the blind retreat -- the hop goes up, and
the pocket still kills him because it is walled both ways on an expiring
floor, which is terrain semantics, not detection.

## 4p. The swim rule: the water wall crossed (2026-10-13)

The 6-1 stall that two sessions called a "cliff" is **water**. The drop
probe's whole cols 112+ block of F01_1 reads "hover at row 6" -- Mario
floats to the surface and bobs there; the seabed he actually walks (row
22 at the stall) is invisible to a probe that floats first. The trace at
the stall told the rest: x frozen at exactly 1,481 even airborne with
right held, a wall at col 93 spanning rows 9-21 per the rects -- whose
top sits BELOW the water surface (row 6), so the route is over the top,
through the water, swimming.

The water mechanics, measured before writing any rule:

- **A in water is a stroke: a fixed ~67 px impulse per PRESS, not per
  hold.** Holding A through the ascent (clearing armed) changed nothing:
  the stroke peaks at the same height and the sink starts while A is
  still down. A 16-JUMP queue = one stroke and 15 no-ops.
- **The sink between strokes is ~40 px per 8-frame step**, so the game's
  own swim rhythm is press/release alternation: each fresh press = a new
  67 px stroke, each release = a short sink, net ~+48 px per pair.
- The reflex's blocked events DO fire on the seabed (vy==0 at the
  touch-downs), roughly every 7 steps -- the stall was never the stuck
  counter's fault; it was that the retreat queue cannot cross water.

**The rule:** on the third blocked event at the same spot (~8 px), the
reflex escalates to `[JUMP, RUN, RUN] x 16` -- press, two glide steps
while the rise plays out, press again at the apex -- with
right and dash held throughout, ~576 frames, enough to climb from the
seabed to the surface and over any wall whose top sits below it.
Parity-safe by measurement: the published 1-1 clear logs ZERO blocked
events, so the escalation cannot fire on it -- and `tools/parity_reflex.py`
passes unchanged.

Measured result on 6-1 legacy: the endless 1,481 px retreat loop became
a crossing -- stroke-climb to the surface, over the wall at col 93, down
the far bank, running east at row 30, a leapt fish, and a new death at
**1,856 px (45%)**: past the water the level's next obstacle is a dry
shaft (the fall hits terminal velocity -- air, not water) that the
parser's ground classes do not see. The deepest any arm has gone outside
world 1, and the first water wall ever crossed. The other courses:
5-1's legacy stall now runs the escalation once (118 -> 132 steps, same
1,474 px death spot); 2-1, 4-1 and 8-1 are byte-identical.

Also from this pass: `tools/probe_floor.py` no longer lets a false
flagpole bound the probe -- C01_1's claimed "flagpole at 272 px" on a
1.6%-coverage parse first inverted the range (cols 53..18, zero
probed), then, with rect-span bounds, truncated it to 18 columns of a
280-column level. The bounds now come from the ROM's rect span, and the
flag cap requires the parse to validate its own flagpole (coverage
>= 30%). C01_1's probe still measures only 16 floors before 47 columns
of unmeasurable drops (its water and its enemies), so World 3-1 stays
honestly refused at 26% measured coverage.

**The crumble census, and two retractions.** With three putative
"expiring floors" on record (the 8-1 bridge, the 2-1 pocket ledge, the
6-1 rock at x=1856), a timer was measured on each: pin ABOVE the tile
(no contact), release, let Mario land on it fresh, and count frames
from touchdown to the fall -- pinning ON the tile contaminates the
reading (the timer runs while touched; released, the 2-1 and 6-1 tiles
read as already gone until the no-contact protocol fixed it).

| tile | fresh-touchdown hold | verdict |
| --- | --- | --- |
| `H01_1` col 16 (8-1's opening bridge) | **17 frames, then falls** | a real crumbling tile |
| `B01_1` col 38 (the 2-1 pocket ledge) | **1,465+ frames, still standing** | does NOT expire |
| `F01_1` col 116 (the 6-1 rock past the water) | **1,485+ frames, still standing** | does NOT expire |

So the 4m autopsy line "standing on a floor that expires (~28 frames)"
is wrong twice: neither the 2-1 ledge nor the 6-1 rock is a crumbling
tile. What makes the in-run falls happen there is still unnamed -- both
falls occur during reflex burst actions at those spots (the 2-1 retreat
leap, the 6-1 swim escalation), on tiles that hold indefinitely under a
plain fresh stand, so the cause is interaction-specific (platform flip
phase, the stroke's forward drift off a tile edge), not a tile timer.
The only measured crumbling tile in the project is the 8-1 bridge -- a
17-frame hold from touchdown, which is the number any stand-and-act
rule would have to beat.

## 4q. The knife-edge, closed: legacy reads mario_B from the RAM (2026-10-13)

4o ended with the float32 knife-edge as a measured fact: reconstructing
the legacy coordinate (mario_B + 88) from the relative obs differs from
the historical value by ~0.04 px, and A02_1's pipe window sits on exactly
one such boundary -- identical inputs, one arm leaps, the other runs, and
the 1-2 legacy row fell from the historical 70% to 15% on the flip.

The closure is a two-line change in principle: the eval harness passes
its RamState handle into the reactive controllers, and the legacy
`_abs_px` reads mario_B directly -- **but with the historical rounding
reproduced, not removed**: `88 + float32(mb/4096/512)*512`, because the
bands were calibrated against the ROUNDED value through the old absolute
obs, so the rounding is part of the tuning, not noise on it. Verified
three ways: the published 1-1 clear reproduces byte-for-byte (parity
passes); the 1-2 legacy row returns to **3,274.4 px / 70%** -- the
historical trajectory, same death, same steps (313 vs the original 311;
the two-step delta is the rule set that grew since: the cone rule and
the swim escalation did not exist when 311 was measured); and the other
courses re-measured honestly rather than reproduced -- the knife-edges
there flip as they flip (5-1 legacy moved 1,474.2 -> 996.4 -> 1,009.1
across the rhythm change: different trajectories under the same rules,
because the escalation and the cone rule fire at moments the historical
run never reached).

**The frontier moved with it.** 6-1's legacy arm, which died at the
second structure at 1,856 px in 4p, now passes it under the exact
coordinate and stands at **1,929 px (47%), alive at the 1200-step cap**
-- the deepest reach outside world 1, and the only arm still alive
there. The profile arm is untouched by all of this (it never used the
legacy coordinate).

## 4r. The wall oracle: both "sprite-solid pockets" retracted, the 6-1 wall mapped (2026-10-13)

The drop probe answers "where can Mario land"; nothing answered "can
Mario's body occupy this cell and move east through it" -- which is the
question the three remaining blockers actually need. `tools/probe_walls.py`
answers it: teleport to each (col, row), pin until the write engages,
release, hold RIGHT for 90 frames, measure the x advance. Frozen x = a
cell solid to Mario at that height; the reading is validated by the
plain-walk control below. Three regions probed:

**The 2-1 pocket: there is no wall.** Every cell of cols 37..40 x rows
20..23 is free (17-65 px of advance). And a control walk -- teleport to
col 35, hold plain RIGHT, log every 15 frames -- walks straight through
the "pocket": 568 -> 614, where Mario walks off the edge of the ledge
the parser reports as row-27 floor while he stands at row 21 (the
level-diff class again), falls to the base floor, and continues east to
665. The 4m autopsy's "sprite-solid wall, x frozen even airborne" was
the retreat queue's own LEFT holds read as a wall -- the third retraction
of that autopsy. The reflex dies there because the blocked response
(leap + retreat) knocks it off the ledge edge into the 232 crawler's
patrol, not because anything blocks east.

**The 4-1 pocket: also no wall.** All 63 cells of cols 29..35 x rows
20..28 are free. The "pinned between sprite solids in both directions"
reading does not survive the oracle either; whatever froze the run at
x=505, it was not a solid -- the same self-inflicted-retreat class.

**The 6-1 wall: real, and now mapped.** The bgdat claimed col 122 rows
20-31; the oracle measures two walls -- col 120 blocking rows 22..26
and col 123 blocking rows 14..19 -- with the rows 20..21 corridor
between them FREE at every probed column (cols 118..125). The route past
the stall is that corridor, 5 rows above the seabed where the run
stands. A direct stroke test at the stall cell (teleport, press A in the
measured burst rhythm, 16 cycles) settles why the burst cannot take it:
**each press bobs 10 px and sinks back, cycle after cycle** -- not the
67 px stroke that carried the 1856 crossing 65 px west. The stall cell
is a dry pocket under the water region (NSMB DS water is region-based;
rows 6..20 are wet, the corridor is not): there, A is neither jump nor
stroke. Three burst rhythms were measured against it -- press every 16,
24 and 32 frames -- all neutral; the terrain wins, and the run correctly
stands at 1,929 px alive at the cap. The frontier is now oracle-mapped,
not just traced.

## 4s. The reward ledger: one reward, two arms (2026-10-13)

The project's original goal was a vision-only agent; along the way a parallel
RAM arm grew and the two drifted apart in reward. They no longer. As of this
section the **reward is the same on both arms and reads only the RAM**;
the only thing that differs between the arms is the observation. The ledger,
so no future re-measure has to rediscover it:

**Reward systems that exist in the codebase:**

1. **The unified RAM reward** (`MarioRamEnv.step`, src/ram_env.py; and
   `MarioNdsEnv` with `reward_mode="ram"`, src/env.py): +2.0 per pixel of
   camera-RAM progress, -0.05 per step, **-15 on death** (RAM lives), **+100
   on level clear** (RAM clear flag / flagpole contact), **-50 on stage-clock
   timeout**. This is the reward of every current training and every eval.
2. **The optical-flow reward** (`MarioNdsEnv` with `reward_mode="flow"`, the
   June-era default): Farneback median background flow + Mario's absdiff
   displacement - 0.05. Retained **only** to reproduce the June-era mlflow
   runs and the n30 benchmark rows, which were measured under it; it is no
   longer the default anywhere.
3. **The imagination shaping** (`ShapingConfig`, src/pinn_world_model.py):
   what the Dyna-PPO/CEM planner optimizes *inside the world model* --
   learned reward + 4.0*cam_delta - 3.5*enemy-proximity (Gaussian, 48/42 px
   sigmas) - 1.5*pit-in-next-column - 0.02/step - pessimism*enemy-uncertainty,
   with a grounded -15 on predicted lethal collisions (stomps exempt). This
   shapes planning, not the emulator reward.
4. **ICM curiosity** (`--use-icm`, src/train_ppo_recurrent.py): an intrinsic
   bonus logged separately as intrinsic vs extrinsic; an experiment knob,
   not part of the unified design.

**The change (this section):** `MarioNdsEnv`'s default flipped
`reward_mode="flow"` -> `"ram"`, plus a RAM clear-flag finish check (+100)
so that death, clear and progress are all RAM-derived on the visual arm too.
The visual arm's terminal templates (Bowser-silhouette death, ORB
finish/timeout) remain active as redundant detectors, as they always were.

**The two arms, as they now stand:**

| arm | observation | reward |
|---|---|---|
| RAM (`MarioRamEnv`) | 23 floats: Mario (x,y,vx,vy,ground,lives,time,cam), 3 enemies (dx,dy,type), 6 pit flags | unified RAM reward |
| vision (`MarioNdsEnv`) | top screen, grayscale, 84x84x1 | **the same unified RAM reward** |

Which is exactly the design brief restated: a CV agent must learn from pixels
what the RAM arm reads directly -- the scenario, the enemies, Mario's
position -- while progress and death mean the same thing on both arms, so
their numbers are directly comparable.

## 4t. The water oracle: warm-only strokes, and six swim-rule candidates measured and reverted (2026-10-14)

The wall oracle (4r) mapped the 6-1 stall's solids but left the water
itself unmeasured -- whether the rows 20-21 corridor is wet, and whether
any rhythm can climb out of the dry pocket, were claims from the bgdat
parse, not measurements. `tools/probe_water.py` measures it: teleport to
each (col,row), release, 120 frames of no-input float, then one clean A
press, classifying every cell by what the physics does. The sweep
(cols 114..126 x rows 18..27, 130 cells) and the follow-ups retracted
three beliefs and produced the attempt history below.

**Retraction one: NSMB water sinks, it does not buoy.** The oracle's
first classifier assumed "wet = floats up" and read every fall as
"dry-air" -- wrong on both ends. The measured falls are 0.1-1.5 px/f
(a dry free-fall covers ~500 px in the same 120 frames), i.e. the whole
region is water-drag, and the 'hover 6' of the floor probe is the
surface bob, not mid-water buoyancy. Mario in deep NSMB water sinks at
~1-2.5 px/f unless he strokes; only the surface band holds him.

**Retraction two: the swim state is warm-only, and the 4r stall test
was a cold artifact.** Teleporting into the water leaves Mario with the
drag but no stroke -- a cold A press mid-water does nothing, and from
the floor it gives only the 24 px contact hop. In-run, where he entered
the water by falling through its surface, strokes chain fine (the
in-run trace shows +vy 0.90 and 0.77 on presses 16-24 frames apart),
and the stall ladder climbs +82 px off the seabed. So the 4r reading
"each press bobs 10 px at the stall, A is neither jump nor stroke
there" overstated the barrier: the stall IS swimmable-warm; what it
lacks is enough climb.

**What the region actually is (all measured):** an underwater ledge at
rows 18-19 spanning cols 116..122, walkable east at 17-49 px/90 f until
the col-123 wall (rows 14-19) blocks it; the corridor rows 20-21 free
and cold-swimmable east (106-133 px/90 f); the seabed floor at rows
26-27 with a hole to row 29 at cols 117-118; and embedded solids (stand,
A does nothing) at cols 121-126 rows 22-27 -- the body under the
corridor. The stall pocket floor sits at the col-120 wall's west face.

**The six candidates, with their numbers.** The route the map suggests
is the surface band (rows 6-8, over both walls); the reflex needed (a)
to know it is in water, (b) a stroke cadence that climbs, (c) not to
fire on dry jumps over water. Six measured attempts:

1. **Ungated cadence** (airborne over a hover column -> stroke every
   32 f): **276 px, dead at step 25** -- the cadence broke a dry
   start-section jump over a hover column; a fish (type 38) killed him
   at point-blank.
2. **Two-tick latch** (two in-band vy samples): **632 px stall** -- a
   dry arc's apex spends two steps in the drag band with a small
   derivative; the false latch steered him onto a walkway-block loop.
3. **Three-near-constant latch, 32 f cadence:** **1,929 px, byte-equal**
   -- never engaged: every water moment happens inside queued burst
   sequences, which return before the state update runs.
4. **State update moved before the queue, 16 f cadence:** **1,433 px,
   dead at step 101** -- engaged at the fall-in, floundered (the deep
   stroke band is net-zero: stroke +40 vs sink -2.5x16), drifted west
   onto a ledge, hopped off its edge and died to the fish at 1,369 px.
5. **Depth gate (stroke only above obs1 0.25, else the old RUN):**
   **1,929 px** -- the stall's ladder apex (obs1 0.252) is above the
   line, but the three-sample latch arms two steps after the apex, when
   obs1 has sunk below it; the cadence inherited a floor-level Mario.
6. **Latch armed at the blocked event itself, ladder cut to [JUMP,RUN]*8
   so it drains at the apex:** **1,929 px** -- and this one is
   trace-proven to work as designed: the cadence fires JUMP at the apex
   (obs1 0.252, a=4 in the log) and holds him at altitude. It still
   cannot leave: the sustained stroke band is net-zero, the +82 px
   apex is the measured ceiling of every rhythm tried, the drift at
   altitude is west (against the wall faces), and the exits -- the
   surface band at obs1 0.64, the col-123 wall top at 0.39 -- are far
   above the ceiling.

All six were reverted; the parity gate passes byte-for-byte and the 6-1
legacy arm re-measured at HEAD gives the same aggregate row (1,929 px /
47% / trunc / 844.5). What remains committed is the measurement, not
the rule: the oracle, its sweep JSON, the rows 18-21 walk/swim JSON,
and `Course.water_columns`/`is_water_course` as documented data (218/256
hover columns on F01_1 = 0.85; D01_1 0.008; A01_1 0.078 -- the marking
that would gate any future rule, with the engagement history that says
how sharp its gates must be).

**The blocker, restated with the new numbers.** From the stall floor the
reflex's action set has a measured climb ceiling of +82 px (row ~18.4);
every exit of the pocket is higher (corridor walls, ledge faces, the
col-123 wall at row 14, the surface at row 6). The ceiling exists
because floor-contact presses jump full-height but mid-water strokes
only break even against the drag. A surface entry exists in the level --
the western walkway falls enter shallow, one stroke from the surface --
but the seabed route, which is the only route the reflex's rules ever
follow, never passes one. Crossing 6-1 therefore needs route knowledge
(enter the water shallow, swim the surface over both walls) or
collision/behavior RE beyond what the oracles can measure -- the same
named blocker as 4r, now with the numbers that close the
"just climb out" branch for good.

Also from this pass: the committed `evals/transfer_6_1_legacy.json` was
regenerated at HEAD -- the 2177afe burst-rhythm change ([JUMP,RUN,RUN]
-> [JUMP,RUN,RUN,RUN]) had never been re-measured into the file. The
aggregate row is unchanged (1,929.0 / 47 / 1200 / 844.5 / trunc); the
per-episode stall-phase detail moved (final_mario_px 1,897 -> 1,929 at
the truncation step, the phase of the stall loop the two rhythms leave
him in).

## 4u. The dashboard's enemy panel: two measured bugs, fixed (2026-10-14)

The README's inline GIF showed enemy detection nobody could trust:
enemies that did not exist, and enemies that appeared or vanished
whenever Mario jumped or fell. Both were real, and both are in
`tools/record_dashboard.py`'s panel, not in the agent.

**Bug one, the anchor.** The obs dy of each enemy slot is relative to
Mario's *live* y, but the panel anchored the drawn tile to Mario's row
on the *static parsed surface* -- so every jump translated all enemy
tiles by the jump height and pushed them out of the 10-row window.
Fix: anchor to Mario's real row, derived from obs[1] (the origin-fixed
relative y) against the spawn column's standing row; Mario himself is
now drawn at his real row too, so the panel shows the jump instead of
pretending he never leaves the floor. Measured on the re-recorded clip:
across 82 frames with enemies visible during >8 px vertical Mario
transitions, Mario shifts a mean 20.1 px while the enemy tiles shift
3.2 px (their own walking) -- decoupled, where the old panel moved them
in lockstep.

**Bug two, the class.** The 3 obs slots hold whatever the entity linked
list holds, and the panel painted every nonzero type red as "enemy". A
census of the published clear trajectory (`_last_enemies`, 245 steps)
finds ten distinct types in the slots: Goomba 160 and Koopa 163 -- the
only measured-hostile ones -- plus coin 157, star coin 159, powerup 44,
and 76/187/49/43/307 (items, decorations, effects). Eight of ten were
phantom "enemies", including the star coins on the end staircase the
GIF actually displays. Fix: hostile types (the `HOSTILE_ENEMY_TYPES`
set the hostility oracle measured) draw red; every other real entity
draws a neutral violet "entity" tile; entities pushed out of the
window by a jump tick the panel edge with a solid marker (a thin AA
outline dissolves in the GIF's palette compression -- the first
attempt at the tick re-introduced the vanishing).

The clip was re-recorded: same trajectory byte-for-byte (the panel
draws after the agent decides), 245 steps, 4,021 px, clear flag raised,
33 s, preview reaching the final frame; the gate stays 48/48.

## 4v. The wall-map gate, and the blocked-response hypothesis refuted by trace (2026-10-14)

The 4r autopsy blamed the 2-1/4-1 deaths on the blocked *response* --
"the retreat/leap knocks Mario off the ledge edge into the crawler's
patrol". This section implemented the obvious fix for that story and
measured it, and the trace refutes the story instead: the deaths
preempt the blocked machinery entirely.

**The map first.** B01_1's pocket sweep was re-run for the committed
record (cols 36..42 x rows 19..28, 70 cells): the pocket itself is
free -- 17-81 px of x advance per cell across cols 36-40, all rows --
and the one real solid in the window is a block at col 41, rows 24-27.
`Course` now loads every `out/wall_probe/<id>*.json` into a
`(col,row) -> free` map with a `wall_probe_free()` query.

**The gate.** When the 3-stuck blocked event fires and the wall oracle
has measured Mario's own cell FREE, there is no wall to answer: the
response is plain RUN (the 4r control walk strolls through the pocket,
off the ledge, down to the base floor and east). Capped at three
overrides before the normal chain takes over; silent on any unprobed
cell; parity-safe twice over (1-1 logs zero blocked events AND has no
probe data). Testing caught a semantics inversion before it shipped --
the loader stored `blocked` under a getter named `free`, which would
have fired the rule exactly at the 6-1 stall wall and stayed silent in
the pocket.

**Measured: byte-identical everywhere.** 2-1: 613.704 px, 4-1: 505.0,
6-1: 1,928.999755859375, 1-2: 3,274.374755859375 -- the gate never
changes an outcome, because on 2-1 and 4-1 **no blocked event ever
fires**: the death arrives first.

**The traces.** On 2-1 Mario freezes at 613.7 for three ground steps
(stuck 0->2, plain RUN held, x frozen) while the Pokey (type 232, the
log's rounded slot type "1") sits at dx 12 px, dy -94 px -- *below the
ledge, outside the +-55 px threat window, invisible to the reflex*.
The first hit lands on the fourth step (vy +0.7: the damage launch),
stuck resets because he is airborne, he falls with x still frozen, and
the second hit kills him at step 29. On 4-1 the Koopa (163) is at dx
12, dy -12 -- *inside* the window -- and its body is the wall: x
pinned at 505, the reflex's answer to a close hostile is leap, and
with no run-up room every leap is a vertical bounce; he bounces
against the Koopa for ten-plus steps until contact kills him.

**The 4r refinement.** The response-is-the-killer reading is refuted
for both pockets: the freeze plus enemy contact kills before any
response can misfire. Two distinct mechanisms share the outcome:
an out-of-window enemy below (2-1) and an in-window enemy whose body
blocks x while point-blank leaps bounce off it (4-1).

**The tenth candidate, vetoed by parity before implementation.** The
obvious rule -- count consecutive point-blank hostile steps
(dx < 24, |dy| < 55) and answer with retreat-then-leap once the count
says the situation is not resolving -- cannot be thresholded: the
published 1-1 clear itself contains a **16-step consecutive
point-blank run** (steps 219-234, a goomba at dx 15 that the published
trajectory survived by resolving, likely stomping) -- longer than the
4-1 fatal bounce. No counter separates the surviving case from the
fatal one; what would is per-frame hitbox measurement of the stomp
contact (did the 1-1 bounces land on the goomba's head while the 4-1
arcs pass beside the Koopa?) -- a new named blocker, not a guess. The
gate stays (measured neutral, structurally safe, and the correct
answer for any freeze that does reach the machinery); the candidate
does not ship.

## 4w. The contact oracle: stomp zones measured, three questions closed (2026-10-14)

The 4v veto left a named blocker: no counter could separate 1-1's
survived 16-step point-blank run from 4-1's fatal bounce, and the
separating measurement was the contact geometry itself.
`tools/probe_contact.py` is that measurement: per (dx, dy) offset it
teleports Mario near the target actor, releases a pure vertical fall,
and classifies the frame-by-frame outcome -- stomp (the actor vanishes
or changes type), damage-death, damage-hit (a vy spike with the actor
unchanged), or miss. Three tool bugs were caught by their own output
before any map was trusted: a sign error that teleported Mario INTO
the ground under the actor (`ey - dy` under the mario_A convention
where more negative is lower); a plain-RIGHT pre-phase that 4-1's
opening step pins at x 105 (the working-state walk now drives the
reactive agent itself); and a first-link working state on 4-1 that sat
in a trailing hostile's range, painting every cell with that enemy's
hit (the walk now stops only at a CLEAN link -- no other hostile
within 120 px).

**Goomba (A01_1, 27 cells).** The stomp window, in teleport
coordinates: Mario falling with the goomba 0-24 px to his WEST
(dx -24..0) stomps; 24-32 px west is the lateral catch (damage);
east of it the goomba walks away (miss). Which closes the 1-1
question: the published clear's survived point-blank run (steps
219-234) held the goomba at obs dx ~15 -- the goomba EAST of Mario,
i.e. Mario west of it by 15 px -- **inside the stomp window**. The
published trajectory survived by stomping, exactly as hypothesized.

**Koopa (D01_1, 27 cells).** Every fall in the window stomps --
27/27, dx -32..+32 at every probed height, the actor converting to
its shell form. So 4-1's fatal bounce cannot be a stomp-geometry
problem, and the in-run trace shows why it never got the chance: the
pocket holds **two Koopas ~65 px apart**. The reflex leaps over the
first (the trace shows the arc passing it at contact dx 0.3, landing
past it), lands pinned against the second's body, and from a
standstill every leap is a vertical hop; the first contact is
lateral (damage, the Super-to-small shrink), the following arcs
phase through during the invincibility frames (the trace's dy sign
flip -- he falls PAST the Koopa with no collision), and when they
expire the next contact kills. The fix 4-1 needs is spacing -- land
short of the second Koopa, deal with them one at a time -- a
two-step plan the reactive layer has no arithmetic for (leap length
vs the gap between hostiles).

**Pokey (B01_1, 42 cells across two sweeps).** The first map read
"damage-hit" everywhere and the verbose cell looked stomp-class -- a
+0.81 vy bounce with no knockback and no death. **That reading was
wrong, and the double-fall test retracts it.** The entity's A anchors
the Pokey's BASE, not its center, so the first sweep's dy 16..48
teleported Mario INSIDE the 48 px body, and the "no knockback" was the
artifact of landing pinned on top of it, not a stomp signature. The
double-fall (land, keep bouncing, no reload): the first contact hops
(the Super-to-small shrink), the i-frames expire around frame ~90,
and the next contact KILLS at frame 107 -- two damage hits, zero
segment pops, zero new entities in the census, the entity's A never
moves. The head-height re-sweep (dy 48..80 above the base, dx -16..16)
completes the map: damage at dx -16..+8, miss beyond -- **no stomp
window exists for this Pokey; it is unstompable from above**, a
stationary spiky hazard (its B never moved in any trace). The in-run
implication stands sharpened: the 2-1 ledge edge is guarded by a
fixed, unstompable enemy below it, and the only measured safe
interaction is to not be there -- the 4r control walk fell past it
(the dx +16.. miss column) while the reflex froze at the edge long
enough to take the hits.

**What the numbers support.** Not one of the three courses yields a
clean rule candidate: 1-1 already stomps (no change needed), 4-1 needs
spacing arithmetic (a named capability gap, not a threshold), 2-1
needs a below-ledge threat window whose parity grep already showed
would fire 12 times on the published clear. The blocker statement
stands sharpened: the reactive layer can see every enemy and stomp
geometry is generous (falls onto a Koopa always kill it), but it
cannot COUNT enemies ahead or plan two contacts -- that is planner
work, the same frontier the CEM-MPC's imagination shaping was built
for and never measured against these pockets.

**The planner hypothesis, measured the same day.** The pure CEM-MPC
(`--controller mpc --no-reflex`, the first time it has ever run on
the pockets): on 4-1 it does what the reflex cannot -- episodes 3
and 4 plan PAST the double-Koopa gate to **606 px**, past both
Koopas, dying instead to the NEXT contact class (the 135 at dx 14)
-- but the other three episodes die at 429/457 before the gate, so
the mean (505.3) only equals the reactive's hard 505, at 134 ms/act
and with cross-episode variance the deterministic reflex does not
have. On 2-1 it is simply worse: a 360 px stall, alive at the
1,200-step cap, west of everything the reactive clears. The
capability is real -- a planner CAN count enemies and route past a
gate that pins the reflex -- but this planner's enemy imagination
does not yet survive a second contact, and its variance costs more
than its planning earns. Both evals are committed
(`evals/transfer_4_1_mpc_noreflex.json`,
`evals/transfer_2_1_mpc_noreflex.json`) as the baseline any future
planner work on these pockets has to beat.

## 4x. The ledge guard crossed: 2-1 moves 613.7 -> 862.7 px (2026-10-14)

The 4w Pokey retraction opened this: the 2-1 ledge edge is guarded by
a stationary, unstompable-from-above enemy sitting below it, outside
the +-55 px threat window and invisible to every existing rule. Three
candidates measured, two shipped as one rule.

**Eleventh candidate (leap at the freeze): fires and is IGNORED.** At
the first frozen step with a below-window hostile in the slots
(dx 0..40, dy -55..-120), commit the leap. The rule fired exactly as
designed (a=4 at stuck 1, 2, 3 in the trace) and changed nothing: the
spot at 613.7 is a game HOLD that suppresses x AND jumps -- three
JUMP presses bobbed Mario 3 px with vy pinned at 0.0000 -- then the
fall off the edge drops him nearly vertically through the guard's
measured damage column (dx -16..+8). Once grabbed, no action-level
answer exists. The 4r control walk crossed the same edge at speed
without being held, so the escape is to cross AIRBORNE, before the
hold engages.

**Twelfth candidate (the early leap, wall-probe-gated).** Widen the
window (dx <= 75) so the leap commits while the guard is still ~50+ px
ahead -- the 109 px running leap then carries over the lip and
touches down past the guard, inside the measured miss column. The
first gate (the course object) broke parity instantly: the object is
always set, so the rule fired on 1-1's own below-window moments and
died at 2,912 px. The second gate (the probe data itself, empty on
1-1) passed parity -- and the measurement: **2-1 613.7 -> 862.7 px on
BOTH arms, +249 px, the largest single transfer gain the project has
measured** -- but 6-1 regressed 1,929 -> 1,481: the wider window
caught one of the water region's fish below the seabed walk and broke
the 4p crossing.

**Thirteenth refinement (the type gate): the shipped rule.** The 4w
contact maps carry exactly the semantics needed: an enemy qualifies
only if ITS OWN measured map has no stomp window anywhere -- 2-1's
Pokey (232: damage at every probed offset) qualifies; goombas and
Koopas (measured stompable) and 6-1's fish (unmapped on F01_1) do
not. `Course.contact_unstompable` loads the verdicts; the rule now
fires only for an unstompable-mapped hostile in the below-ledge
window, on a wall-probed course, on the ground. Battery: **2-1
862.7/862.7 (both arms, the frontier's new number, committed to the
transfer evals), 6-1 restored to 1,929 byte-identical, 4-1 unchanged
at 505, 1-1/1-2/5-1/8-1 structurally silent (no probe or contact data
-- the code path cannot execute), parity PASS.** The 2-1 run now dies
at 862.7 to the next hazard class (type 54 at contact, with 43/44 in
the slots) -- a new frontier with a new autopsy waiting.

The rule's full condition stack, for whoever tunes it next:
`wall_probe_cells` non-empty (probed course) AND on_ground AND a
hostile slot with `type in contact_unstompable` AND 0 <= dx <= 75 AND
-120 <= dy <= -55 -> JUMP. Every clause is a measured retracement:
drop the type gate and 6-1 loses 448 px; drop the data gate and 1-1's
parity dies at 2,912 px; fire at the freeze instead of on approach
and the hold ignores you.

## 4y. The floating blocks: invisible in the panel, and why the camera cannot verify them (2026-10-14)

The dashboard's tile grid showed neither the destructible brick rows nor
the yellow ?-blocks the gameplay screen displays right next to it. Two
causes, one fix, and a camera discovery worth recording.

**Cause one: the profile is a scalar.** The panel drew `surface[c]` --
one walkable-top number per column -- so anything FLOATING above the
surface did not exist in it. The old "isolated rise" outline only
caught single-column peaks, and those peaks are the blocks the parser
ABSORBED into the surface: `GROUND_OBJS` includes 0x2C and 0x30 -- the
bgdat classes behind the floating ?/brick rows (obj 0x30 sits at
exactly cols 154/157/160 row 26, the trio src/eval_pinn_realtime.py's
legacy bands call "the floating question/brick blocks"; obj 0x14's
23-tile cluster spans cols 54-78 rows 24-26). So the ?-blocks appeared
as outlined floor spikes and the non-absorbed brick clusters appeared
as nothing at all.

**The fix:** the panel now draws the parser's floating block rects
(rows 20..27, at most 3x3, the absorbed 0x2C/0x30 included over the
floor fill) as one warm `?/brick` class -- 61 rects, 14 classes on
A01_1. Verified in the re-recorded clip: BLOCKOBJ pixels present in the
panel across all three stretches of the run (peaks 827/1,441/863 px,
7-12 blocks on screen at once). The planner's reading is unchanged --
the panel is visualization; the profile the code reasons over is the
parser's, absorption and all.

**The camera discovery, three attempts deep.** Distinguishing ? from
brick by rendered hue needed screen sampling, and the sampler kept
reading sky: the mario_A row mapping was validated against the floor,
but the X mapping never was -- the floor exists at every x. The
decisive test: the camera X **does not follow teleported Mario**. Write
attempts get reverted by the game's follow logic every frame, and
after a teleport the camera stays at the spawn's view (read back:
458 px while Mario stood at 2,400 px) -- the deadzone comment on
CAM_ADDRS said as much. So on-screen sampling of arbitrary positions
is not reachable with the current RAM map; the ?/brick hue split
stays unmeasured and the panel honestly draws one class. What would
measure it: walking (not teleporting) the camera into place, or
extracting the tileset graphics from the ROM with ndspy.

## 4z. The entity classes, identified by position (2026-10-14)

Two viewer complaints, both real, both about the violet "entity"
class: a fence drawn as an item, and the interactable blocks grouped
with the coins. The fix is a POSITIONAL census: log every linked
entity's (type, world-column) along the published 1-1 clear, then
match types to the bgdat feature positions. Geometric identification,
no naming table needed (OBJ_TYPES turns out to be wrong for at least
one class):

| type | world columns | measured identity |
|---|---|---|
| 43 | 22.5, 38.5, 95.5, 141.5, 229.5 -- five static positions matching the row-28 scenery rects | **the fences** |
| 157 | one static row, cols 238.5-242.5 (93 step-sightings, 5 columns) | **the coin row over the final staircase** |
| 159 | 123.0, 184.5 | **the star coins** |
| 44 | slides 22.5 -> 28.7 in half-column steps | **the moving platform** -- OBJ_TYPES' "Powerup" name is wrong |
| 187 | one static actor at col 153, beside the 154-160 block trio | **the interactable-block actor** |
| 49 / 76 / 307 | 252 (the flagpole), 86, 0 | flag, scenery, spawn-time fx |

The panel now draws: hostile red, coins gold, the 187 block-actor with
the 4y warm block class, fences/flag/fx gray scenery, the moving
platform as solid-cyan fill, and anything unknown keeps the honest
violet fallback. Verified in the re-recorded clip: gray scenery in
all three stretches (130/81/168 px), gold coins peaking at the
staircase row (312 px), warm blocks throughout -- and ZERO violet
pixels anywhere: every entity type on the published trajectory is now
classified by census, not by name.

## 4aa. The RE attempt, and the stall premises it reversed (2026-10-14)

Asked whether agent-driven reverse engineering (REA-class tooling)
could close the game's remaining gaps, the first attempt ran without
Ghidra: extract the ARM9 (ndspy header parse, 465 KB at 0x02000000),
parse the overlay table (135 overlays), and scan every binary for the
known RAM anchors. Result: absolute addresses never appear -- DS code
reaches EWRAM through a base register, and the physics constants are
not stored as plain u32 literals either (the fall-terminal
0xFFFFC000 appears exactly once, in overlay 10 at 0x0210AC48, in a
neighborhood that disassembles as garbage in both ARM and Thumb --
the same wall 4h hit, now with better instruments). The exact needles
for a future Ghidra pass are measured and on record: held-jump v0
**0x3980** (14720), fall terminal **0xFFFFC000** (-16384), held-jump
gravity **0x100/frame**, water arc gravity ~**0x2C0/frame** (from the
fine trace's vy decay).

**The behavioral pivot paid instead -- three premise reversals:**

1. **The 4r reading "A is neither jump nor stroke at the stall cell"
   is REFUTED.** A single A-press at the cold-teleported stall cell
   (col 120.5, row 26.5, settled 40+ frames) gives the FULL impulse
   every time: YVEL +0x3980 -- the same v0 as the 1-1 ground jump. The
   measured "10 px bobs" were the burst-rhythm protocol (32-frame
   press spacing with RIGHT held), not a property of the cell.
2. **The "24 px bob" of the 4t water sweep was the full jump too**:
   at v0 0.898 px/f and g 0.0156 the standstill apex is 25.8 px. The
   water-probe's bob column and the jump column were the same
   phenomenon at two sampling resolutions.
3. **The warm-only swim state is demoted.** The cold stall presses
   stroke fine; the chaining curve at the stall (presses at 8/16/24/
   32/48/64-frame gaps) shows NO cooldown at 32+ frames -- every
   press full. The warm/cold distinction the 4t/4w/4x chain was built
   on remains unexplained only for the in-run chained-stroke rises
   (vy +0.9/+0.77 at 16-24-frame spacing inside a live run), but it
   gates nothing at the stall cell. The RAM diff's warm-only candidates
   (+0x2B0 = 0xC0210000, the 0x0CF00000 family at +0x98/+0xB0/+0x204,
   +0x2EC) are recorded but unconfirmed -- the write-test was mooted
   by reversal #1 (the baseline already strokes, so there is no
   off-state to switch on).

**What does not move: the stall ceiling.** The no-keys 24-frame
cadence is a stable LIMIT CYCLE (rows 24.75 -> 25.78, apex +1.2 rows,
no cumulative climb); the 16-frame RIGHT-held ladder remains the best
measured climb (row ~21.4, the 4x +82 px). The 6-1 frontier stands at
1,929 with its blocker restated: no tested rhythm converts chained
presses into a sustained climb at the stall cell, and the surface
band stays ~190 px above the best apex. The static route's next step
is now sharply defined and cheap to aim: a real decompiler pass over
overlay 10 with the exact constants above as anchors.

**The static hunt's terminal state, the next day.** The full-filesystem
sweep (ARM9 + all 135 overlays, uncompressed per ndspy and entropy
checks) for the measured constants in every cheap encoding -- u32
words, float32, ARM MOV/CMP immediate encodings with corrected
rotations, pool-reference hunts in both ARM and Thumb, phase-bootstrapped
disassembly around each hit -- ends where 4h ended, now with better
instruments: three scattered, pool-unreferenced 0xFFFFC000 words
(overlays 10, 81), a 0x6D00 and a 0xE60 sitting in byte
neighborhoods that decode as neither clean ARM nor clean Thumb, and
zero LDR references to any of them. The constants either come from
data archives (NARC) or are computed, and no scalar scan will find
the physics function. The route that remains is the one 4aa named: a
real decompiler pass (Ghidra needs a Java install on this machine;
radare2 is the lighter alternative) over overlays 10 and 128 with the
banked needles -- and the behavioral oracles stay the cheap judge of
anything it claims.

**The tooling checkpoint (day three of the static route).** Rizin
0.9.1 (portable, no admin needed) is installed under the temp area;
`tools/extract_overlays.py` now regenerates `out/re/arm9.bin` and all
135 overlay binaries with their RAM bases (`out/re/overlay_table.json`
carries the table). The overlay loading model is measured from the
table: the small modules all share the 0x020DE360 slot (mutually
exclusive, swapped in and out), overlay 10 -- the 190 KB main engine
-- loads into that same slot, overlay 128 (181 KB) into 0x020C2020.
Neither big overlay begins with code (ov128 opens with 64 zero bytes),
so the entry convention lives in the ARM9's overlay dispatcher. The
immediate next steps, in order: find the dispatcher's call (the
FS_LoadOverlay path in arm9.bin, then the init-jump convention), seed
rizin's `aaa` from real entry points, and hunt the physics function
through the mario-object field accesses ([obj+0x5C/0x60/0x74/0x78],
whose ARM-mode encodings the Thumb-only scan missed -- the ARM9
itself showed two such clusters at 0x02012C4E and 0x02026EAA worth
disassembling first). The needles stay banked: v0 0x3980, terminal
0xFFFFC000, held-jump g 0x100, stroke 0x6D00, water g ~0x2C0.

**The static hunt's honest close (the rizin pass).** The ARM-mode field
scan across arm9 + overlays returns zero clusters -- because the
engine IS the ARM9 (853 Thumb field accesses, the overlays are mostly
per-level data). Rizin's full `aaa` on the raw 465 KB Thumb blob
does not finish inside 30 minutes (no symbols, no entry map), and
targeted function analysis from PUSH-prologue heuristics drifts
across the interleaved data. The finding that closes the scalar
route for good: the physics constants are COMPUTED, not stored --
the measured jump v0 0x3980 appears as a u32 in exactly zero of the
136 binaries (the full-FS needle scan), so the game derives it
(base + form/state bonuses); no constant scan can find the physics
function. What remains is the real-tooling pass -- Ghidra with a
Java install (the winget UAC needs the user's admin prompt) or an
entry-point archaeology project in rizin through the ARM9's init
sequence -- and the behavioral oracles stay the cheap judge of
whatever either claims.

## 4ab. The 863 autopsy: a floating bar, and the rear blind spot (2026-10-14)

The 4x crossing put 2-1's death at 862.7 px "to the next hazard class
(type 54 at contact)". The autopsy traces it fully:

**The trace.** The approach runs east and lands on the base floor at
col 53.9 (863 px); x then freezes for 11+ steps (stuck 0->2 cycling,
blocked events answering with the leap). The leap is VERTICAL -- x
pinned at 0.981844 through the whole arc, the same hold-class x-pin as
the 613.7 ledge -- rising +48 px and falling back, over and over. The
killer (54) arrives from BEHIND during the freeze: contact at dx -5,
dy -14, and the death line reads it with the fence (43) and platform
(44) classes in the slots. The final arc falls below the floor
reference and the episode ends at step 39.

**The geometry (wall probe, 48 cells, cols 51..58 x rows 23..28,
committed as out/wall_probe/B01_1_step863.json).** The only solids in
the window besides the base floor: **a one-tile bar at row 25, cols
53..55** (pin-never-engaged on all three cells; rows 24 and 26 above
and below it FREE). The floor probe's "elevated floor, row 24" at
those columns is markers landing ON TOP of the bar. The parser sees
none of it: its floor is 27 across the window and the pit flags read
solid throughout. A floating bar, head-height to a runner on the base
floor, invisible to every layer the planner reads.

**The verdict: a named blocker, not a rule -- twice over.**

1. *The bar.* The blocked machinery cannot answer it: the leap that
   would clear (horizontal travel onto the bar's top or past it) is
   exactly what the hold-class pin forbids, and the parser has no row-
   25 surface for the profile arm's hop-onto-it branch to find.
2. *The rear blind spot.* The 54 killed from dx = -5 -- behind
   Mario -- and the threat window is one-sided (`enemy_lo=0,
   enemy_hi=75`: nothing behind is ever watched). But a watch-behind
   rule is parity-dead by measurement: the PUBLISHED 1-1 clear contains
   **91 hostile-behind-within-75px moments** (goombas trail him the
   whole run -- at dx -1 for the steps 218-234 tail-ride) and survives
   every one of them by outrunning the pursuit; any rear-window rule
   fires on the published trajectory and breaks parity. The rear
   threat is only lethal COMBINED with a freeze, and the frozen-plus-
   rear shape also fires at 1-1's own steps 218-234.

2-1's gate count is now three, all measured: the 613.7 ledge guard
(crossed, 4x), the 863 floating bar (parser-blind, hold-pinned), and
the rear pursuit that only a freeze makes lethal.

## 4ac. The asymmetric actor-critic, measured: 88 -> 1,250 px deterministic (2026-10-14)

The A/B the 4s reward unification was built for, run at 200k steps
each, 12 envs, same seed, same code -- the CRITIC's input is the only
difference (`src/train_asymmetric_ac.py --value-input {pixels,ram}`).
The policy is a NatureCNN over the June-pipeline 84x84 grayscale in
BOTH arms: a pure CV policy. The reward: the unified RAM reward.
Training: 33.2 min per arm at ~100 steps/s (the SubprocVecEnv
configuration the June runs used, scaled from 4 to 12 envs on this
32-thread machine).

| arm (200k steps) | det mean px | stoch mean px | stoch max px | ep_rew_mean (train) |
|---|---|---|---|---|
| pixel-critic (baseline) | 88.0 | 782.4 | 1,081.7 | ~294-300 |
| **ram-critic (asymmetric)** | **1,250.5** | **1,360.0** | **2,610.9** | ~504-515 |

Readings, in order of importance:

1. **The asymmetry is worth 14x deterministic** (88 vs 1,250.5) and
   +74% on the stochastic mean -- the Pinto et al. mechanism, measured:
   the pixel-critic drowns the advantage in variance exactly when the
   CNN is still learning to see, and its deterministic policy collapses
   to a run-the-clock-at-88px loop (the baseline's det episodes run
   1,000 steps without moving; the ram arm's die at 89 steps, running).
2. **The ram arm's deterministic policy dies at 1,250.5 px -- the exact
   number of the reflex's profile arm** (4b, the known geometry wall at
   the 1,250 px stairs): a pure pixel policy, 200k steps, no reflex
   help, converging to the same wall the rule-based controller hits is
   convergent evidence both are seeing the same level.
3. **The stoch max 2,610.9 px is the deepest run by any learned policy
   in the repo's history** -- past the June pixel PPO (120.8), past the
   reflex-assisted ppo+reflex (1,139), past the CEM-MPC (1,510). At
   n=3 the max is a sample, not a mean; the honest headline is the
   deterministic wall and the +70% training-return gap, and the full
   n=30 benchmark remains future work.

Two operational findings, both now in AGENTS.md and the code comments:
DeSmuME initialized AFTER torch is loaded access-violates (the eval
harness has always constructed the env before models for this reason);
and the training pair ran detached (`Start-Process` + log polling) --
the 30-minute in-session bash cap is the real ceiling for any run.

Artifacts: `models/ac_pixels_200k.zip` + `models/ac_ram_200k.zip`
(19-20 MB each), `out/ac_{pixels,ram}_200k.log`, the eval tool
`src/eval_asymmetric_ac.py`, the env `src/dual_obs_env.py`.

## 5. File Index

```
src/
  env.py                 Visual (CNN) environment — pixel-based obs, optical flow reward
  ram_env.py             RAM environment — 17-dim vector obs, no CNN
  ram_state.py           RAM address constants + RamState reader (validated EUR)
  course.py              ROM level parser (ndspy) — sprites, bgdat, entrances
  tilemap_planner.py     A* global planner from tile geometry
  pinn_world_model.py    Hard-residual PINN + CEM-MPC controller
  train_pinn_world_model.py  Stage 1: collect transitions + train PINN
  train_pinn_policy.py   Stage 2: Dyna-PPO in imagination
  eval_pinn_realtime.py  Stage 3: real-time eval (mpc / ppo / reactive)
  train_ram.py           RAM-PPO baseline
  train_gnn.py           Graph NN baseline
  train_ppo_recurrent.py RecurrentPPO (all augmentation variants)
  evaluate_benchmark.py  n=30 benchmark runner
  run_pipeline.py        Full pipeline orchestrator
  ...

models/
  pinn_nsmb.pt           PINN world model (43 440 params, MSE 0.056)
  pinn_pool.npz          3500 real transitions
  pinn_policy.zip        Dyna-PPO policy (200 k imagination steps)
  ram_ppo_100k.zip       RAM-PPO 100 k
  ram_geo_100k.zip       RAM + geometry features 100 k
  gnn_100k.zip           GNN 100 k
  ppo_icm_100k.zip       PPO + ICM 100 k
  ppo_icm_ae_100k.zip    PPO + ICM + Autoencoder 100 k
  ...

evals/
  pinn_worldmodel_metrics.json   PINN train/test metrics
  pinn_policy_metrics.json       Dyna-PPO training summary
  pinn_realtime_mpc.json         n=8 MPC real-time eval
  pinn_realtime_ppo.json         n=8 PPO real-time eval
  pinn_realtime_reactive.json    n=8 reactive real-time eval
  pinn_realtime_eval.json        n=4 reactive (PINN obs)
  transfer_1_2_profile.json      1-2, ROM-derived reflex thresholds (4j)
  transfer_1_2_legacy.json       1-2, the 1-1 pixel bands (4j control arm)
  transfer_5_1_profile.json      5-1 (E01_1), ROM-derived thresholds (4j)
  transfer_5_1_legacy.json       5-1, the 1-1 pixel bands (4j)
  transfer_8_1_profile.json      8-1 (H01_1), ROM-derived thresholds (4j)
  transfer_8_1_legacy.json       8-1, the 1-1 pixel bands (4j)
  transfer_2_1_profile.json      2-1 (B01_1), the headless-entered desert (4m)
  transfer_2_1_legacy.json       2-1 control arm (4m)
  transfer_4_1_profile.json      4-1 (D01_1), unlocked by its 142 probe floors (4m)
  transfer_4_1_legacy.json       4-1 control arm (4m)
  transfer_6_1_profile.json     6-1 (F01_1), first arm to survive to the
                                 step cap outside world 1 (4m)
  transfer_6_1_legacy.json       6-1 control arm: 50%, alive at the cap (4m)
  benchmark_n30_merged.json      n=30 stochastic + n=10 det, 14 models

tools/
  export_levels.py               Batch parser for all 191 levels
  build_dataset.py               Emits entities.csv and levels.jsonl
  extract_profiles.py            Scrapes 342 actor profiles from CC0 decomp
  reach_map.py                   Clear 1-1 -> World 1 map -> savestate of any
                                 unlocked node (4i)
  identify_state.py              Which course a savestate is in: entrance match
                                 over all 191, then floor rows on a walk-right
                                 trace (4j)
  probe_floor.py                 The drop oracle: teleport Mario per column and
                                 let the game's physics report the true floor
                                 (4k); emits out/floor_probe/*.json
  probe_hostility.py             Behavioral hostility: pin/drop/walk onto an
                                 actor type and measure death/stomp/nothing (4k)
  probe_walls.py                 The wall oracle: teleport per (col,row) cell,
                                 hold RIGHT, measure the x advance -- a 2D
                                 solidity map of what actually blocks (4r);
                                 emits out/wall_probe/*.json
  enter_course.py                Enter ANY course headless: holds the running
                                 course id through the map's A-press fade, so
                                 the stylus-only world row is no longer a gate
                                 (4m); produces the savestate directly
  parity_reflex.py               Legacy arm must equal the published 1-1 eval
  parity_planner.py              Rust sidecar vs Python, 37,281 queries
  record_dashboard.py            The README clip and its measured caption
  check_readme_numbers.py        Gate: every number quoted in README.md
  make_save.py                   Build an NSMB DS .sav with the world row
                                 unlocked: the 16 bytes a verified 100% ARDS
                                 download carries at +0x6A, mirrored, checksums
                                 rewritten 10/10
  tileset_probe.py               Per-course ground coverage (4e table)
  mucd_probe.py / mucd_bind.py / mucd_groups.py   4f investigation
  tileset_groups.py / tileset_posneg.py / tileset_sweep.py   4h investigation

out/dataset/
  levels.jsonl                   191 levels with full entity tables (0.70 MB)
  entities.csv                   5311 entities with corrected pixel coords (0.28 MB)
  object_ids.json                342 actor profile mappings
  physics.json                   20.12 fixed-point engine constants and laws
  meta.json                      Dataset schema specification

out/floor_probe/
  A01_1.json / E01_1.json / H01_1.json   measured per-column floors from the
                                 drop probe; Course prefers their confirmed
                                 contradictions (4k). A01_1's carries
                                 apply_overrides=false -- see 4k for why the
                                 published clear keeps the static profile.
  B01_1.json / F01_1.json        the two headless-entered courses that parse:
                                 B01_1 gains four slope columns, F01_1 agrees
                                 with the parser on all 256 (4m)
  D01_1.json                     142 measured floor columns -- what unlocked
                                 4-1 at 85.1% measured coverage (4m)
  G01_1.json / C01_1.json        134 floors found on G01_1 (still honestly
                                 refused at 28.4%: the cloud gaps are real);
                                 C01_1: 16 floors measured, 47 columns
                                 unmeasurable (water + enemies), 26% coverage

out/wall_probe/
  D01_1_pocket.json            the 4-1 pocket: all 63 cells free -- the
                                  "pinned between sprite solids" reading
                                  retracted (4r)
  B01_1_pocket.json             the 2-1 pocket, re-measured for the record
                                  (4v): cols 36-40 free at every probed row
                                  (17-81 px advance), the window's one real
                                  solid at col 41 rows 24-27; the in-run
                                  freeze at 613.7 is enemy contact, not
                                  geometry
  F01_1_wall122.json            the 6-1 wall region: col 120 rows 22-26
                                  and col 123 rows 14-19, the rows 20-21
                                  corridor free -- the route past 1,929 is
                                  mapped, and the dry-pocket stroke test
                                  explains why no rhythm crosses it
  F01_1_row19.json               the rows 18-21 walk/swim map: the
                                  underwater ledge walkable east to the
                                  col-123 wall, the corridor
                                  cold-swimmable 106-133 px/90 f (4t)

out/water_probe/
  F01_1.json                     the swim-state oracle sweep (130 cells,
                                  cols 114-126 x rows 18-27): water drag
                                  everywhere (falls 0.1-1.5 px/f), floors,
                                  the row-19 ledge, the hole, embedded
                                  solids -- and the cold-teleport no-A
                                  cells that exposed the warm-only swim
                                  state (4t)

tools/
  probe_water.py                 the water oracle: teleport per cell,
                                  float phase + one clean A press, drag
                                  and stroke response measured (4t)
  probe_contact.py               the contact oracle: teleport-release
                                  vertical falls around an actor, the
                                  stomp/damage zone per type measured
                                  (4w); the working-state walk drives the
                                  reactive agent to a CLEAN first link

out/contact_probe/
  A01_1_160.json                 the goomba map: stomp window = falls
                                  with the goomba 0-24 px west of Mario;
                                  lateral west is damage, east walks off
  D01_1_163.json                 the Koopa map: 27/27 stomps -- every
                                  fall in the window shells it; 4-1's
                                  death is the double-Koopa spacing trap,
                                  not stomp geometry
  B01_1_232.json                 the Pokey body sweep: A anchors its
                                  BASE, so dy 16-48 teleports INSIDE the
                                  48px body -- damage at every measured
                                  offset (the first "stomp-class bounce"
                                  reading is retracted in 4w by the
                                  double-fall test)
  B01_1_232_head.json            the head-height re-sweep: falls from
                                  48-80px above its base damage at
                                  dx -16..+8, miss beyond -- NO stomp
                                  window: the 2-1 Pokey is unstompable
                                  from above, a stationary hazard

docs/
  WORLD_MODEL_PINN.md                         Deep-dive: PINN methodology
  PROJECT_STATUS.md                           This file
  REVERSE_ENGINEERING_REPORT.md               Comprehensive NSMB DS reverse engineering report
  COMPREHENSIVE_GAP_ANALYSIS_AND_ROADMAP.md   Master gap analysis across RE and ML pillars
```
