"""Gate: every number quoted in the README showcase block must equal what the
recorded run actually displayed (out/dashboard_run.json) or what the code says.

README prose rots silently -- this project has already been burned by a typed
figure surviving CI. Run it after regenerating the clip.

Usage:
    python tools/check_readme_numbers.py [--with-parity]
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

README = os.path.join(ROOT, "README.md")
RUN = os.path.join(ROOT, "out", "dashboard_run.json")


def doc_checks(readme_text):
    """Cells in the results prose and in docs/ must equal the committed evals
    and the ROM geometry -- these are the numbers that rotted once already."""
    out = []
    from tilemap_planner import TilemapAStar
    p = TilemapAStar(rom_path=os.path.join(
        ROOT, "data", "0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds"))
    n_pits = len(p.pits)
    widest = max(w for (_, _, w) in p.pit_widths)

    def grab(path, pattern, expected, label):
        body = open(path, encoding="utf-8").read()
        m = re.search(pattern, body)
        got = m.group(1) if m else None
        out.append((label, expected, got,
                    got is not None and got.replace(",", "") == str(expected)))

    rp = os.path.join(ROOT, "README.md")
    wm = os.path.join(ROOT, "docs", "WORLD_MODEL_PINN.md")
    grab(rp, r"\((\d+) pits, max (\d+) tiles", n_pits, "README pit count")
    grab(rp, r"pits, max (\d+) tiles", widest, "README widest pit tiles")
    import eval_pinn_realtime as ev
    grab(rp, r"flagpole is at ([\d,]+) px", ev.goal_px(), "README flagpole px")
    grab(rp, r"solid tile column ends at ([\d,]+) px", p.goal_px(), "README last-column px")
    grab(wm, r"(\d+) pits, the widest (\d+) tiles", n_pits, "PINN doc pit count")
    grab(wm, r"the widest (\d+) tiles", widest, "PINN doc widest pit tiles")

    m = re.search(r"!\[[^\]]*\]\((media/[A-Za-z0-9_]+\.gif)\)", readme_text)
    gif = os.path.join(ROOT, m.group(1)) if m else None
    exists = bool(gif) and os.path.exists(gif)
    out.append(("README inline gif referenced and present", "media/*.gif on disk",
                m.group(1) if m else None, exists))

    for name, fname, pattern, label in (
            ("pinn_realtime_eval.json", "reactive", r"reactive.? reaches \*{0,2}([\d,]+) px",
             "README reactive mean px"),
            ("pinn_realtime_mpc.json", "mpc", r"CEM-MPC peaks at ([\d,]+) px",
             "README CEM-MPC max px"),
            ("pinn_realtime_ppo_reflex.json", "ppo+reflex", r"PPO\+Reflex at ([\d,]+) px",
             "README ppo+reflex max px")):
        path = os.path.join(ROOT, "evals", name)
        if not os.path.exists(path):
            out.append((label, "eval json present", "missing " + name, False))
            continue
        agg = json.load(open(path, encoding="utf-8"))["aggregate"]
        key = "mean_progress_px" if fname == "reactive" else "max_progress_px"
        grab(rp, pattern, round(agg[key]), label)

    ev_path = os.path.join(ROOT, "evals", "pinn_realtime_eval.json")
    if os.path.exists(ev_path):
        agg = json.load(open(ev_path, encoding="utf-8"))["aggregate"]
        m = re.search(r"in (\d+)/(\d+) episodes", readme_text)
        n_ep = int(agg["episodes"])
        cleared = round(agg["clear_rate"] * n_ep)
        out.append(("README reactive episodes cleared", f"{cleared}/{n_ep}",
                    f"{m.group(1)}/{m.group(2)}" if m else None,
                    bool(m) and (m.group(1), m.group(2)) == (str(cleared), str(n_ep))))
        # the PROJECT_STATUS summary row is the cell that rotted once already
        ps = open(os.path.join(ROOT, "docs", "PROJECT_STATUS.md"), encoding="utf-8").read()
        row = re.search(r"Reactive \(Rule-based, n=(\d+)\)\*\*\s*\|\s*\*\*([\d.,]+)\*\*"
                        r"\s*\|\s*\*\*([\d.,]+)\*\*\s*\|\s*\*\*(\d+)%", ps)
        want = (str(n_ep), f"{agg['mean_progress_px']:,.1f}",
                f"{agg['max_progress_px']:,.1f}", f"{round(agg['clear_rate'] * 100)}")
        got = row.groups() if row else None
        out.append(("PROJECT_STATUS reactive row (n, mean, max, clear%)",
                    want, got, got == want))
    return out


def main():
    ap_ok = "--with-parity" in sys.argv
    if not os.path.exists(RUN):
        print(f"missing {RUN}: run tools/record_dashboard.py first", file=sys.stderr)
        return 2
    run = json.load(open(RUN, encoding="utf-8"))
    text = open(README, encoding="utf-8").read()
    block = text.split("### World 1-1", 1)
    if len(block) != 2:
        print("README showcase block heading not found", file=sys.stderr)
        return 2
    block = block[1].split("## Contents", 1)[0]

    from ram_env import OBS_DIM, GEO_DIM
    import eval_pinn_realtime as ev
    flagpole = ev.goal_px()

    checks = []

    def want(pattern, expected, label):
        m = re.search(pattern, block)
        got = m.group(1) if m else None
        ok = got is not None and str(got).replace(",", "") == str(expected)
        checks.append((label, expected, got, ok))

    want(r"\*\*(\d+) steps", run["steps"], "steps")
    m = re.search(r"(\d+)\s+px\s+of\s+(\d+)\s+px", block)
    checks.append(("distance px / goal px",
                   f"{run['abs_px']} px of {run['goal_px']} px",
                   f"{m.group(1)} px of {m.group(2)} px" if m else None,
                   bool(m) and m.group(1) == str(run["abs_px"])
                   and m.group(2) == str(run["goal_px"])))
    want(r"\((\d+) s,", run["seconds"], "clip duration s")
    want(r"(\d+×\d+)", f"{run['width']}×{run['height']}", "frame size")
    want(r"the (\d+) px flagpole", flagpole, "flagpole goal px")
    want(r"the (\d+)-float observation", OBS_DIM + GEO_DIM, "observation width")
    want(r"`media/(nsmb_dashboard_[a-z]+\.mp4)`", run["video"], "video filename")

    if ap_ok:
        r = subprocess.run([sys.executable, os.path.join(HERE, "parity_planner.py")],
                           capture_output=True, text=True)
        ok = "PASS" in r.stdout
        checks.append(("Rust/Python planner parity", "PASS",
                       "PASS" if ok else r.stdout.strip().splitlines()[-1:], ok))

    checks += doc_checks(text)

    bad = [c for c in checks if not c[3]]
    for label, exp, got, ok in checks:
        print(f"{'ok  ' if ok else 'FAIL'} {label}: README={got!r} expected={exp!r}")
    if bad:
        print(f"\n{len(bad)} README number(s) do not match the run", file=sys.stderr)
        return 1
    print(f"\nall {len(checks)} README numbers match")
    return 0


if __name__ == "__main__":
    sys.exit(main())
