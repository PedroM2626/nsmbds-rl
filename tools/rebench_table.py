"""Merge the v2 re-benchmark evals into one JSON + a Markdown table (4az).

Reads evals/rebench/eval_*.json and *_training_log.json, writes
evals/benchmark_v2_unified.json and prints the table used in the README.
--check verifies that the README block between the rebench-table markers is
exactly the generated table (tests/test_rebench.py runs it as a gate).

  .venv\\Scripts\\python.exe tools\\rebench_table.py
  .venv\\Scripts\\python.exe tools\\rebench_table.py --check
"""
from __future__ import annotations

import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import rebench_common as rc  # noqa: E402

FLAG_PX = 4032.0
README = os.path.join(os.path.dirname(HERE), "README.md")
BEGIN, END = "<!-- rebench-table:begin -->", "<!-- rebench-table:end -->"
LABELS = {
    "rppo_pure": "RecurrentPPO CnnLstm (= June \"PPO Pure\" / \"Recurrent PPO\")",
    "rppo_ae": "RecurrentPPO + frozen Autoencoder",
    "rppo_curl": "RecurrentPPO + CURL",
    "rppo_hybrid": "RecurrentPPO + Hybrid AE+CURL",
    "rppo_spr": "RecurrentPPO + SPR",
    "rppo_drq": "RecurrentPPO + DrQ random shifts",
    "rppo_icm": "RecurrentPPO + ICM",
    "rppo_icm_ae": "RecurrentPPO + ICM + Autoencoder",
    "rppo_impala_curl": "RecurrentPPO IMPALA + CURL",
    "rtrf_impala_spr": "Causal Transformer IMPALA + SPR",
    "ppo_impala": "PPO ImpalaCNN (feedforward)",
    "ppo_naturecnn": "PPO NatureCNN (feedforward)",
    "ppo_ram17": "PPO RAM-only MLP (17-d)",
    "ppo_ram23": "PPO RAM + Geometry MLP (23-d)",
    "ppo_gnn": "PPO GNN MeanMPNN (entity graph)",
    "wm_ga": "World Models V+M+C, GA",
    "wm_cma": "World Models V+M+C, sep-CMA-ES",
    "ref_genetic": "Neuroevolution MLP (4ao, ref.)",
    "ref_iqn": "IQN (4ap, ref.)",
    "ref_rainbow": "Rainbow DQN (4aq, ref.)",
    "ref_ppo_simple": "PPO RAM MLP (4ar, ref.)",
    "ref_visual_simple": "PPO NatureCNN (4as, ref.)",
    "ref_visual_impala": "PPO IMPALA (4at, ref.)",
    "ref_visual_genetic": "Visual GA CNN (4at, ref.)",
}
OBS = {"pixels": "pixels", "ram17": "RAM 17", "ram23": "RAM 23", "graph": "graph"}


def load(path):
    with open(path) as f:
        return json.load(f)


def build():
    """Return (markdown table, merged JSON dict) from the committed eval files."""
    evals = {}
    for p in sorted(glob.glob(os.path.join(rc.EVAL_DIR, "eval_*.json"))):
        d = load(p)
        evals[d["name"]] = {k: v for k, v in d.items() if k not in ("det", "stoch")}
        evals[d["name"]]["det"] = {k: v for k, v in d["det"].items() if k != "episodes"}
        evals[d["name"]]["det"]["episode0"] = d["det"]["episodes"][0]
        evals[d["name"]]["stoch"] = ({k: v for k, v in d["stoch"].items() if k != "episodes"}
                                     if d["stoch"] else None)
        evals[d["name"]]["source"] = p.replace("\\", "/")
    train = {}
    for p in sorted(glob.glob(os.path.join(rc.EVAL_DIR, "*_training_log.json"))):
        d = load(p)
        key = os.path.basename(p)[: -len("_training_log.json")]
        train[key] = {k: d.get(k) for k in
                           ("total_steps", "episodes", "best_max_x_px", "best_step",
                            "first_clear_step", "wall_min", "elapsed_min", "policy_params",
                            "params", "algo")}
        train[key]["source"] = p.replace("\\", "/")

    rows = []
    arms = list(rc.ARMS) + ["wm_ga", "wm_cma"]
    for arm in arms:
        for which in ("best", "final"):
            name = arm if arm.startswith("wm_") else f"{arm}_{which}"
            if arm.startswith("wm_") and which == "final":
                continue
            if name in evals:
                rows.append((arm, which, evals[name]))
        for seed in (0, 1):   # preserved runs that diverged before 100k steps (4az)
            if f"{arm}_seed{seed}_best" in evals:
                rows.append((f"{arm}_seed{seed}", f"best (seed {seed})",
                             evals[f"{arm}_seed{seed}_best"]))
    for name in LABELS:
        if name.startswith("ref_") and name in evals:
            rows.append((name, "ref", evals[name]))

    def fmt_px(v):
        return f"{v:,.1f}"

    lines = ["| Arm | Obs | Ckpt | Train best x (px) | Det x (px) / end | "
             "Stoch n=30 mean ± sd [95% CI] | Stoch max | Stoch clears |",
             "| :--- | :---: | :---: | ---: | :--- | :--- | ---: | ---: |"]
    for arm, which, e in rows:
        t = train.get(arm, {})
        tb = fmt_px(t["best_max_x_px"]) if t.get("best_max_x_px") is not None else "—"
        d0 = e["det"]["episode0"]
        det = f"{fmt_px(d0['max_x_px'])} / {d0['termination']}"
        s = e["stoch"]
        if s:
            ci = f" [±{s['ci95_halfwidth_px']:,.0f}]" if s.get("ci95_halfwidth_px") else ""
            st = f"{fmt_px(s['mean_max_x_px'])} ± {s['sd_max_x_px']:,.1f}{ci}"
            smax, sclr = fmt_px(s["max_max_x_px"]), f"{s['clears']}/{s['n']}"
        else:
            st, smax, sclr = "n/a (argmax)", "—", "—"
        obs = OBS.get(e["obs"], e["obs"])
        label = LABELS.get(arm) or LABELS.get(arm.split("_seed")[0], arm)
        lines.append(f"| {label} | {obs} | {which} | {tb} | {det} | {st} | {smax} | {sclr} |")
    table = "\n".join(lines)

    from eval_provenance import collect
    out = {"schema": 1, "protocol": next(iter(evals.values()))["protocol"] if evals else None,
           "flag_px": FLAG_PX, "evals": evals, "training": train}
    return table, out


def readme_block():
    body = open(README, encoding="utf-8").read()
    if BEGIN not in body or END not in body:
        return None
    return body.split(BEGIN, 1)[1].split(END, 1)[0].strip()


def main():
    table, out = build()
    if "--check" in sys.argv[1:]:
        ok = readme_block() == table
        print("README rebench table matches the eval JSONs" if ok
              else "README rebench table DIFFERS from the eval JSONs -- regenerate it")
        sys.exit(0 if ok else 1)
    from eval_provenance import collect
    out["provenance"] = collect(rom=rc.paths.ROM_REL, state=rc.paths.STATE_REL)
    with open(os.path.join("evals", "benchmark_v2_unified.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(table)


if __name__ == "__main__":
    main()
