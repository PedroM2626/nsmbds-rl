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
| All 382 course files parsed | ✅ | ndspy `NintendoDSRom`, 2449-file FS, manifest at `tools/fs_manifest.json` |
| JSON export for all levels | ✅ | 191 levels exported (`out/dataset/levels.jsonl`, `entities.csv` via `tools/build_dataset.py`) |
| Enemy-type ID mapping | ✅ (85%) | 342 profile slots extracted from CC0 decomp into `out/dataset/object_ids.json`, loaded by `src/course.py` |
| Level-finish tile detection | ✅ | Flagpole detected at tile 252 (`4032.0 px`) across course sprites |

### 1.3 Tilemap A\* Planner (`src/tilemap_planner.py`)

| Component | Status | Notes |
|-----------|--------|-------|
| Ground/pit mask from `_bgdat.bin` | ✅ | `GROUND_OBJS` includes all solid surfaces; real 1-1 pits detected at width 2-3 tiles |
| Run-jump reachability | ✅ | Dynamic ballistic reach: $X_{reach} = v_x \cdot t_{airborne}$ ($t_{air} = 48\text{ frames}$) |
| A\* over tile columns | ✅ | Global layer; feeds takeoff column to CEM-MPC |
| Enemy-aware planning | ✅ | `enemy_threat_ahead()` checks takeoff/landing corridors; triggers `dash_leap` or `retreat_wait` |
| Multi-level support | ⚠️ Not tested | Only validated on 1-1 |

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
| — Reactive | ❌ | Mean 484 px, max 561 px; truncation at 1000 steps |
| — MPC | ❌ | Mean 1269 px, max 1730 px; best so far but never clears pit cluster at ~1730 px |
| — Reactive (PINN obs) | ❌ | Mean 1137 px, max 1203 px; consistent Goomba deaths |
| Level clear | ❌ **Not achieved** | Goal: 4256 px |

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
| **Absolute X ambiguity ±40 px** | Progress reward is correct (delta), but absolute position reporting in evals is off | Fix: anchor `B_OFFSET` once; compare `0x021C1828` spawn value with known entrance coordinate |
| **PPO policy collapses at ~336 px** | First Goomba on flat ground always kills the PPO agent (87 identical steps across 7/8 episodes → deterministic death loop) | Grounded contact mortality added to imagination + $25\times$ death loss weight in PINN |

### P1 — World model quality

| Gap | Impact | Suggested Fix |
|-----|--------|---------------|
| **Open-loop drift 1738 px over 44 frames** | MPC re-plans every step (mitigates this), but PPO relies on long imagined rollouts where the model is inaccurate | Increase transitions to 10k (currently 3500); add recurrent state in the PINN trunk (GRU head); or use an ensemble of 3 PINNs and take pessimistic lower bound |
| **Enemy dynamics unmodeled beyond relative position** | Enemies move; their relative position changes in ways the PINN doesn't capture well (logvar head marks this as high-uncertainty) | ✅ Solved: Dynamic enemy velocities `(vx, vy)` tracked in `RamState.enemies()` |
| **No contact/death prediction** | The PINN `continue` head is a binary; it doesn't distinguish death-by-enemy from death-by-pit | ✅ Solved: Explicit physical collision hitbox check in `PINNImaginationEnv` |

### P2 — Planning gaps

| Gap | Impact | Suggested Fix |
|-----|--------|---------------|
| **Enemy-unaware A\* planner** | MPC consistently dies at ~1730 px where a Goomba blocks the approach to a pit edge | ✅ Solved: `enemy_threat_ahead` added to `TilemapAStar` |
| **Multi-level generalization** | Planner and RAM addresses only validated on 1-1 | Run `course.py` batch export on all 382 levels → JSON; verify calibration on 1-2 and 2-1 |
| **JUMP_REACH_PX hardcoded** | 120 px works for most pits in 1-1 but breaks on wider pits in later worlds | ✅ Solved: Dynamic reach $X_{\text{reach}}(v_x) = v_x \cdot t_{\text{air}}$ ($t_{\text{air}} = 48\text{ frames}$) |

### P3 — Documentation & reproducibility

| Gap | Impact | Suggested Fix |
|-----|--------|---------------|
| **No formal `docs/` entry for RAM reverse engineering** | The discovery methodology lives only in `ram_state.py` | ✅ Solved: Documented in [`docs/REVERSE_ENGINEERING_REPORT.md`](REVERSE_ENGINEERING_REPORT.md) |
| **Complete Gap Analysis & Roadmap** | Gaps across all pillars not centralized | ✅ Solved: Documented in [`docs/COMPREHENSIVE_GAP_ANALYSIS_AND_ROADMAP.md`](COMPREHENSIVE_GAP_ANALYSIS_AND_ROADMAP.md) |
| **`evals/benchmark_n30_merged.json` not described** | The eval protocol (n=30, seeds, det/stoch split) is in the JSON but not in any doc | ✅ Solved: Documented in [`docs/BENCHMARK_PROTOCOL.md`](BENCHMARK_PROTOCOL.md) |
| **No trained model cards** | Each `.zip`/`.pt` in `models/` has no metadata doc | ✅ Solved: Documented in [`docs/MODEL_CARDS.md`](MODEL_CARDS.md) |
| **README §14 (PINN) not linked to WORLD_MODEL_PINN.md** | The detailed doc exists but README only has a CLI snippet | Add link in README |

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
| **Reactive (Rule-based, n=5)** | **4,022.4** | **4,022.4** | **100%** | Clears to the flagpole; `evals/pinn_realtime_eval.json`. Supersedes the n=4 run (1,136.7 mean, 0%) that died at 1115-1203 px |
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
| `--reflex legacy` | 209 | 4,022 | 1530.375 | **cleared** |
| `--reflex profile` | 109 | 1,819 | 623.6911 | died (peak 1,730 px), **short of the flagpole** |

So the bands are load-bearing: with pure geometry the controller gets ~43% of the
way. The stall is not a reflex bug but a **geometry-fidelity** bug -- see 4c.

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

Effect, measured: the pit census is unchanged (9 pits, widths 2-3, `feasible=True`),
obstacle columns went 49 -> 60 (heights recomputed against the real floor), the
Rust/Python query parity still passes on all 37,281 sampled queries, and the
`reactive` arm still clears 5/5 at 4,022.4 px -- one step shorter than before
(209 vs 210), which is why `evals/pinn_realtime_eval.json` was regenerated. The
pure-geometry arm is still short (1,730 px), so parser fidelity was necessary but
not sufficient: the remaining gap is in the reflex rules themselves.

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
and `evals/pinn_realtime_eval.json` both report 209 steps and 4,022 px.

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

1. drive the published reactive agent to the clear (219 env steps, same trajectory
   as `evals/pinn_realtime_eval.json`);
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
| `--reflex profile` (ROM-derived thresholds) | 1,929.6 px | 41% | 151 | 1770.1 | death |
| `--reflex legacy` (the 1-1 pixel bands) | **3,314.4 px** | **71%** | 311 | 3146.6 | death |

n=1 per arm, so this is a measurement of one trajectory each, not an estimate of
a rate. What it says: **the 1-1 bands transfer better than the geometry-derived
thresholds do.** On 1-1 the ordering is the same way round (legacy clears, profile
peaks at 1,818 px), so the reflex arms are consistent in *which* one is stronger;
what changed is that the gap narrowed from "clears vs 45%" to "71% vs 41%" on a
level neither was tuned for. Both died with `pit=[1,1,1,1,1,1] ground=1` in the
death diagnostic -- the pit features say ground is missing ahead while the RAM
says Mario is standing, which is the coverage gap of 4e showing up as a bad
decision rather than as a refusal.

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

**Second and third transfers, same protocol as the first: n=1 per arm, goal
from the course's own flagpole sprite.**

| course | arm | distance | of goal | steps | reward | outcome |
| --- | --- | --- | --- | --- | --- | --- |
| 5-1 `E01_1` | profile | 1,717.8 px | 38% | 119 | 752.4 | death |
| 5-1 `E01_1` | legacy | 1,701.5 px | 37% | 127 | 745.0 | death |
| 8-1 `H01_1` | profile | 337.0 px | 9% | 30 | 68.0 | death |
| 8-1 `H01_1` | legacy | 337.0 px | 9% | 30 | 68.0 | death |

The 1-2 transfer's ordering -- the 1-1 bands ahead of the ROM-derived
thresholds -- does not reproduce on 5-1: the arms die 16 px apart, with the same
`ground=1 pit=[1,1,1,1,1,1]` disagreement the 1-2 deaths showed, so on this
course the two reflexes are equally insufficient. On 8-1 neither arm survives
the opening 30 steps: both fall into the pit around x=258 px that the jump-less
identification trace falls into too, with byte-identical death diagnostics --
and unlike 1-2 and 5-1, this one is internally consistent (`ground=0`,
mid-fall). No arm has yet cleared a course outside world 1; the reactive
controller's measured world row is now 1-1 (100%), 1-2 (71%, legacy), 5-1
(38%), 8-1 (9%).

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

docs/
  WORLD_MODEL_PINN.md                         Deep-dive: PINN methodology
  PROJECT_STATUS.md                           This file
  REVERSE_ENGINEERING_REPORT.md               Comprehensive NSMB DS reverse engineering report
  COMPREHENSIVE_GAP_ANALYSIS_AND_ROADMAP.md   Master gap analysis across RE and ML pillars
```
