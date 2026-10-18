# AGENTS.md — operational facts for anyone (human or agent) working here

## Environment

- **Python**: use `.venv\Scripts\python.exe` — the system Python lacks
  `gymnasium` and friends. Everything below assumes it.
- **GPU**: torch +cu124 builds (see `requirements.txt`, pinned and verified;
  the +cu124 wheels live on the PyTorch index, not PyPI).
- **Emulator**: py-desmume 0.9.12 over DeSmuME, headless.

## The two gates (run both before every commit)

```
.venv\Scripts\python.exe tools\check_readme_numbers.py   # 48/48
.venv\Scripts\python.exe tools\parity_reflex.py          # byte-for-byte vs evals/pinn_realtime_eval.json
```

- The parity gate reproduces the **published 1-1 clear exactly**
  (245 steps, reward 1663.087402343757, final 4020.6328125). Any reflex
  change must keep it passing — the legacy arm is a frozen contract.
- The numbers gate re-derives every README/PROJECT_STATUS number from the
  eval JSONs; never type a number, regenerate the eval.

## Hard constraints (each one bit someone)

- **One DeSmuME per process.** A second init in the same process
  access-violates. Parallel envs go through `SubprocVecEnv` (one per
  subprocess) — that is fine; two `MarioRamEnv`s in one process is not.
  **And DeSmuME must be initialized BEFORE torch is loaded** — a
  `PPO.load` followed by an env construction access-violates at init
  (the eval harness has always built the env first for this reason;
  measured again in 4ac).
- **Teleporting does not move the camera.** The camera X follows the game's
  own logic (it does not follow `mario_B` writes; reads back 458 px with
  Mario at 2,400 px). Screen sampling of arbitrary positions is unreachable
  with the current RAM map.
- **Savestates restore RAM, not the on-disk save.** A `.sav` only matters at
  title-screen boot, which never advances headless.
- **REFLEX_LOG=filename** env var enables per-call action logging in the
  eval harness (the source of truth for autopsies; trace tools that don't
  pass `ram=` to the agent diverge from the eval trajectory).

## How work is done here

1. **Measure, then write.** The oracles in `tools/` (floor, wall, water,
   contact, hostility) answer questions by teleporting and watching the
   game. Output JSONs live in `out/<oracle>/` and get committed.
2. **Every claim carries a number** that a gate or a committed JSON backs.
3. **Retractions are written down** — PROJECT_STATUS sections 4a..4ab are
   the measured history; corrections land as new sections, old sections
   stay as era records. Never silently rewrite a retracted claim.
4. **Reward**: unified RAM reward on both arms (4s); flow is the legacy
   opt-in (`reward_mode="flow"`) for reproducing June-era numbers.
5. **Commit protocol**: gates green first, then commit; push to
  `origin main` (the remote answers as `mario-ds-rl` but lives at
  `nsmbds-rl` — the redirect works, the URL rename is pending).

## Where things live

- `docs/PROJECT_STATUS.md` — the living lab notebook; sections 4a-4ab.
- `src/eval_pinn_realtime.py` — the reflex controller + eval harness.
- `out/floor_probe|wall_probe|water_probe|contact_probe/` — measured data.
- `models/`, `evals/` — checkpoints and gated results.
- Long trainings: launch detached (`Start-Process`) and poll; in-session
  bash calls cap at ~30 min.
