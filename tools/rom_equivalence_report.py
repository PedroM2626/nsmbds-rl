"""Compare evaluations replayed on ROM-built states with the published ones (4ba).

  .venv\\Scripts\\python.exe tools\\eval_rebench.py --arm rppo_curl --which best \\
        --state data\\built_1-1.ds1 --out evals\\rebench_rom\\eval_rppo_curl_best.json   # per eval
  .venv\\Scripts\\python.exe tools\\rom_equivalence_report.py

Writes evals/rom_state_equivalence.json: for every v2 re-benchmark evaluation,
whether all 3 argmax + 30 sampled episodes are identical on the ROM-built 1-1
state, and both summaries where they are not. The per-course contract results
are those printed by `tools/make_states.py --verify` (src/cold_boot.py table).
"""
from __future__ import annotations

import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

KEYS = ("max_x_px", "final_x_px", "steps", "termination", "return")
# `tools/make_states.py --verify`, 2026-11-01: published evaluation replayed
# (all episodes) on each ROM-built course state.
COURSES = {
    "1-1": ("exact", "pinn_realtime_eval.json parity + eval_genetic_w11.json, 5/5 episodes identical"),
    "2-1": ("exact", "eval_2_1_clear.json, 5/5 identical"),
    "4-1": ("exact", "eval_4_1_clear.json, 5/5 identical"),
    "5-1": ("exact", "eval_5_1_clear.json, 5/5 identical"),
    "6-1": ("exact", "eval_6_1_clear.json, 5/5 identical"),
    "8-1": ("exact", "eval_8_1_clear.json, 5/5 identical"),
    "1-2": ("outcome", "eval_1_2_clear.json: same 326 steps, goal, final X 4038.77; total reward 4029.47 vs 4026.47"),
    "7-1": ("outcome", "eval_special_71_full_clear.json: same 809-step clear, reward 2259.55; enemy 221 1 px off from step 27"),
    "3-1": ("differs", "eval_special_31_full_clear.json: route dies at step 183 (731.4 px) vs clear at 557"),
    "3-1_main": ("differs", "eval_special_31_main_clear.json: route dies at step 90 vs clear at 497"),
}


def episodes(d):
    return d["det"]["episodes"] + (d["stoch"]["episodes"] if d["stoch"] else [])


def summary(d):
    s = d["stoch"]
    det = d["det"]["episodes"][0]
    return {"det_max_x_px": det["max_x_px"], "det_termination": det["termination"],
            "det_steps": det["steps"],
            "stoch_mean_max_x_px": s["mean_max_x_px"] if s else None,
            "stoch_sd_max_x_px": s["sd_max_x_px"] if s else None,
            "stoch_clears": s["clears"] if s else None}


def main():
    built_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "evals", "rebench_rom")
    rows = {}
    for p in sorted(glob.glob(os.path.join(ROOT, "evals", "rebench", "eval_*.json"))):
        q = os.path.join(built_dir, os.path.basename(p))
        if not os.path.exists(q):
            continue
        a, b = json.load(open(q)), json.load(open(p))
        ea, eb = episodes(a), episodes(b)
        n_diff = sum(any(x[k] != y[k] for k in KEYS) for x, y in zip(ea, eb)) + abs(len(ea) - len(eb))
        rows[b["name"]] = {"obs": b["obs"], "episodes": len(eb), "differing_episodes": n_diff,
                           "identical": n_diff == 0}
        if n_diff:
            rows[b["name"]]["reference_state"] = summary(b)
            rows[b["name"]]["rom_built_state"] = summary(a)
    from eval_provenance import collect
    import paths
    out = {"schema": 1, "courses": {c: {"status": s, "detail": d} for c, (s, d) in COURSES.items()},
           "rebench_1_1": rows,
           "rebench_identical": sum(r["identical"] for r in rows.values()), "rebench_total": len(rows),
           "provenance": collect(rom=paths.ROM_REL, state=None,
                                 extra={"built_dir": os.path.relpath(built_dir, ROOT).replace("\\", "/")})}
    path = os.path.join(ROOT, "evals", "rom_state_equivalence.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"{out['rebench_identical']}/{out['rebench_total']} re-benchmark evaluations identical -> {path}")
    for n, r in rows.items():
        if not r["identical"]:
            print(f"  {n}: {r['differing_episodes']}/{r['episodes']} episodes differ")


if __name__ == "__main__":
    main()
