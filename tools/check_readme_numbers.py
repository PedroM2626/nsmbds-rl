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

    # docs/PROJECT_STATUS.md 4b: the reflex arms table must equal the last
    # tools/parity_reflex.py run, not a number someone typed in.
    rp_arms = os.path.join(ROOT, "out", "reflex_parity.json")
    ps = open(os.path.join(ROOT, "docs", "PROJECT_STATUS.md"), encoding="utf-8").read()
    if os.path.exists(rp_arms):
        arms = json.load(open(rp_arms, encoding="utf-8"))
        for arm in ("legacy", "profile"):
            a = arms[arm]
            row = re.search(r"`--reflex " + arm + r"` \| (\d+) \| ([\d,]+) \| ([\d.]+)", ps)
            want = (str(a["steps"]), f"{a['final_abs_px']:,.0f}", f"{a['total_reward']:.7g}")
            got = row.groups() if row else None
            ok = bool(row) and got[0] == want[0] and got[1] == want[1] and got[2] == want[2]
            out.append((f"PROJECT_STATUS 4b {arm} row", want, got, ok))
    else:
        out.append(("PROJECT_STATUS 4b reflex rows", "out/reflex_parity.json",
                    "missing (run tools/parity_reflex.py)", False))
    # docs/PROJECT_STATUS.md 4e: the per-course coverage table must equal the parser
    from course import Course
    for course_id, short in (("course/A01_1.bin", "A01_1"), ("course/A02_1.bin", "A02_1"),
                             ("course/A05_1.bin", "A05_1"), ("course/A08_1.bin", "A08_1")):
        c = Course(name=course_id, rom_path=os.path.join(
            ROOT, "data", "0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds"))
        want = f"{c.ground_coverage * 100:.1f}%"
        row = re.search(r"\|\s*`" + short + r"`[^|]*\|\s*(\d+)\s*\|\s*\**([\d.]+)%\**\s*\|\s*([^|]+)\|", ps)
        got = (row.group(2) + "%", row.group(3).strip()) if row else None
        # the exit cell is prose ("flagpole @ 4,032 px" / "**none**"), so check the
        # claim it makes rather than an exact string
        says_none = got is not None and "none" in got[1].lower().replace("*", "")
        ok = bool(row) and got[0] == want and (says_none == (not c.exit_found))
        out.append((f"PROJECT_STATUS 4e {short} coverage/exit", (want, "none" if not c.exit_found else "an exit object"),
                    got, ok))
    # docs/PROJECT_STATUS.md 4j: the first transfer table, cell by cell against the
    # two eval JSONs it was copied from.
    row_re = (r"\|\s*`--reflex (\w+)`[^|]*\|\s*\**([\d,]+\.\d) px\**\s*\|"
              r"\s*\**(\d+)%\**\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*(\w+)\s*\|")
    found = {m.group(1): m.groups()[1:] for m in re.finditer(row_re, ps)}
    for arm in ("profile", "legacy"):
        p = os.path.join(ROOT, "evals", f"transfer_1_2_{arm}.json")
        if not os.path.exists(p):
            out.append((f"PROJECT_STATUS 4j {arm} row", "evals/transfer_1_2_%s.json" % arm,
                        "missing", False))
            continue
        a = json.load(open(p, encoding="utf-8"))["aggregate"]
        e = json.load(open(p, encoding="utf-8"))["episodes"][0]
        goal = a["goal_abs_px"]
        exp = (f"{a['mean_progress_px']:,.1f}",
               f"{100.0 * a['mean_progress_px'] / goal:.0f}",
               str(e["steps"]), f"{e['total_reward']:.1f}", e["termination"])
        got = found.get(arm)
        out.append((f"PROJECT_STATUS 4j {arm} row", exp, got, got == exp))
    # docs/PROJECT_STATUS.md 4j: the world 5/8 transfer table, cell by cell against
    # the four eval JSONs it was copied from.
    for tag, cid in (("5-1", "E01_1"), ("8-1", "H01_1")):
        for arm in ("profile", "legacy"):
            p = os.path.join(ROOT, "evals",
                             f"transfer_{tag.replace('-', '_')}_{arm}.json")
            if not os.path.exists(p):
                out.append((f"PROJECT_STATUS 4j {tag} {arm} row", p, "missing", False))
                continue
            d = json.load(open(p, encoding="utf-8"))
            a, e = d["aggregate"], d["episodes"][0]
            goal = a["goal_abs_px"]
            exp = (f"{a['mean_progress_px']:,.1f}",
                   f"{100.0 * a['mean_progress_px'] / goal:.0f}",
                   str(e["steps"]), f"{e['total_reward']:.1f}", e["termination"])
            row = re.search(r"\|\s*" + tag + r" `" + cid + r"`\s*\|\s*" + arm +
                           r"\s*\|\s*\**([\d,]+\.\d) px\**\s*\|\s*\**(\d+)%\**"
                           r"\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*(\w+)", ps)
            got = row.groups() if row else None
            out.append((f"PROJECT_STATUS 4j {tag} {arm} row", exp, got, got == exp))
    # docs/PROJECT_STATUS.md 4m: the worlds 2/6 table, same cell-by-cell gate
    for tag, cid in (("2-1", "B01_1"), ("6-1", "F01_1")):
        for arm in ("profile", "legacy"):
            p = os.path.join(ROOT, "evals",
                             f"transfer_{tag.replace('-', '_')}_{arm}.json")
            if not os.path.exists(p):
                out.append((f"PROJECT_STATUS 4m {tag} {arm} row", p, "missing", False))
                continue
            d = json.load(open(p, encoding="utf-8"))
            a, e = d["aggregate"], d["episodes"][0]
            goal = a["goal_abs_px"]
            exp = (f"{a['mean_progress_px']:,.1f}",
                   f"{100.0 * a['mean_progress_px'] / goal:.0f}",
                   str(e["steps"]), f"{e['total_reward']:.1f}", e["termination"])
            row = re.search(r"\|\s*" + tag + r" `" + cid + r"`\s*\|\s*" + arm +
                            r"\s*\|\s*\**([\d,]+\.\d) px\**\s*\|\s*\**(\d+)%\**"
                            r"\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*\*?\*?(\w+)", ps)
            got = row.groups() if row else None
            out.append((f"PROJECT_STATUS 4m {tag} {arm} row", exp, got, got == exp))
    # docs/PROJECT_STATUS.md 4f: the MUCD claims are recomputed from the ROM, not
    # remembered from a notebook.
    try:
        import struct
        import ndspy.rom as _R
        from course import Course
        _rom = _R.NintendoDSRom(open(os.path.join(
            ROOT, "data", "0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds"),
            "rb").read())
        _bg = _rom.filenames.subfolder('BG_chk')
        _names = [n for n in _bg.files if n.endswith('MainUnitChangeData.bin')]
        claims = [("4f MUCD file count", "48", str(len(_names)), len(_names) == 48)]
        _sizes = {len(_rom.getFileByName('BG_chk/' + n)) for n in _names}
        claims.append(("4f MUCD file size", "3072", str(sorted(_sizes)), _sizes == {3072}))
        # the table is 6 u16 per object number (3 (unit, model) pairs), so 256*6
        # slots; assert that reading is at least self-consistent across files
        _slots = {len(struct.unpack('<1536H', bytes(_rom.getFileByName('BG_chk/' + n))))
                  for n in _names}
        claims.append(("4f slots per file (256 objects x 6)", "1536", str(sorted(_slots)),
                       _slots == {1536}))
        shiro = struct.unpack('<1536H', bytes(_rom.getFileByName(
            'BG_chk/ShiRoMainUnitChangeData.bin')))
        solid = {o for o in range(256) if any(shiro[o * 6:o * 6 + 6])}
        cA = Course(name="course/A01_1.bin", rom_path=os.path.join(
            ROOT, "data", "0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds"))
        cols = set()
        for r in cA.rects:
            if r["obj"] in solid:
                cols.update(range(r["tx"], r["tx"] + r["w"]))
        claims.append(("4f A01_1 columns under 'has any entry'", "246", str(len(cols)),
                       len(cols) == 246))
        # and the negative result that killed the boolean reading
        _dom = set()
        for n in _names:
            _dom |= set(struct.unpack('<1536H', bytes(_rom.getFileByName('BG_chk/' + n))))
        claims.append(("4f value domain is NOT boolean", "not {0,1}",
                       f"{len(_dom)} distinct", len(_dom) > 2))
        for label, exp, got, ok in claims:
            out.append((label, exp, got, ok))
        for label, exp, got, ok in claims:
            out.append((label, exp, got, ok))
    except Exception as e:  # noqa: BLE001 - a skipped claim must be visible, not silent
        out.append(("4f MUCD claims", "recomputed", f"skipped: {e}", False))
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

    if ap_ok:
        # slow census: the "N of 191 courses pass the guard" claim in 4e
        from course import Course
        ps_text = open(os.path.join(ROOT, "docs", "PROJECT_STATUS.md"),
                       encoding="utf-8").read()
        rom_path = os.path.join(ROOT, "data",
                                "0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds")
        names = json.load(open(os.path.join(ROOT, "out", "nsmb_levels.json"),
                               encoding="utf-8"))["levels"].keys()
        n_pl = sum(1 for nm in names
                   if (lambda c: c.ground_coverage >= 0.6 and c.exit_found)(
                       Course(name=nm, rom_path=rom_path)))
        m = re.search(r"\*\*(\d+) of 191\*\* courses pass the guard", ps_text)
        checks.append(("PROJECT_STATUS 4e plannable count", str(n_pl),
                       m.group(1) if m else None,
                       bool(m) and m.group(1) == str(n_pl)))

    if "gif_seconds" in run:
        checks.append(("inline preview reaches the final frame", True,
                       run.get("gif_reaches_end"), bool(run.get("gif_reaches_end"))))

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
