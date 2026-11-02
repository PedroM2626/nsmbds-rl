"""Calibrate a ROM-built savestate against a reference one (PROJECT_STATUS 4ba).

Needs the reference savestate, so only the maintainer runs it; its output is
the handful of constants (save frame, counter values) that go into
src/cold_boot.py, after which nobody needs the reference.

  .venv\\Scripts\\python.exe tools\\calibrate_state.py --course-id B01_1 \\
        --reference data\\state_2-1.ds1 --lo 1200 --n 500

It boots the ROM, enters the course the way tools/enter_course.py does
(course-id writes held through the transition), saves every frame of the
window, and reports the frames whose reset observation (MarioRamEnv, geo)
equals the reference's -- ignoring the stage timer, then exactly -- plus the
global words that still differ at the best frame.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

GLOBALS = (0x02098000, 0x0209F000)   # the static game globals (lives, camera, counters)


def gen(args):
    import cold_boot
    from enter_course import course_area_index, stage_globals_of
    writes = cold_boot.course_writes(course_area_index(args.course_id), *stage_globals_of(args.course_id))
    writes += [tuple(int(v, 0) for v in w.split(":")) for w in args.write]
    steps = cold_boot._TO_WORLD1_MAP + [("key", "A"), ("hold", args.lo, writes),
                                        ("series", args.n, writes)]
    cold_boot.run_script(steps, series_dir=args.work)


def scan(args):
    import numpy as np
    from ram_env import MarioRamEnv
    env = MarioRamEnv(state_path=args.reference, geo=True)
    ref, _ = env.reset()
    mem = env.emu.memory

    def words():
        return np.array([mem.read(a, a, 4, False) for a in range(GLOBALS[0], GLOBALS[1], 4)],
                        dtype=np.int64)
    env.emu.savestate.load_file(args.reference)
    ref_words = words()
    rows = []
    for k in range(args.n):
        env.state_path = os.path.join(args.work, f"f{k}.ds1")
        obs, _ = env.reset()
        d = np.abs(obs - ref)
        no_timer = np.delete(d, 6)
        rows.append((k, float(no_timer.max()), float(obs[6] - ref[6])))
    exact = [r for r in rows if r[1] < 1e-6 and abs(r[2]) < 1e-6]
    near = sorted((r for r in rows if r[1] < 1e-6), key=lambda r: abs(r[2]))[:5]
    print("reference reset obs:", np.round(ref, 4).tolist())
    print("frames equal except timer (closest timer first):",
          [(args.lo + k, round(t, 6)) for k, _, t in near])
    print("frames exactly equal:", [args.lo + k for k, _, _ in exact])
    best = exact[0][0] if exact else (near[0][0] if near else None)
    out = {"course_id": args.course_id, "reference": args.reference, "lo": args.lo,
           "exact": [args.lo + k for k, _, _ in exact],
           "near": [(args.lo + k, t) for k, _, t in near], "global_diffs": []}
    if best is not None:
        env.emu.savestate.load_file(os.path.join(args.work, f"f{best}.ds1"))
        w = words()
        for i in np.nonzero(w != ref_words)[0]:
            out["global_diffs"].append((hex(GLOBALS[0] + 4 * int(i)), int(ref_words[i]), int(w[i])))
        print(f"global words differing at frame {args.lo + best}:")
        for a, r, c in out["global_diffs"]:
            print(f"  {a}: reference {r}  built {c}")
    if args.json:
        json.dump(out, open(args.json, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--course-id", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--lo", type=int, default=1200, help="first frame after A to save")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--work")
    ap.add_argument("--json")
    ap.add_argument("--write", action="append", default=[],
                    help="extra held write addr:size:value, e.g. 0x0209DC00:1:99 (lives)")
    ap.add_argument("--phase", choices=["gen", "scan"])
    args = ap.parse_args()
    if args.phase == "gen":
        return gen(args)
    if args.phase == "scan":
        return scan(args)
    args.work = args.work or tempfile.mkdtemp(prefix="calib_")
    base = [sys.executable, __file__, "--course-id", args.course_id, "--reference", args.reference,
            "--lo", str(args.lo), "--n", str(args.n), "--work", args.work]
    for w in args.write:
        base += ["--write", w]
    subprocess.run(base + ["--phase", "gen"], check=True)   # one DeSmuME per process
    subprocess.run(base + ["--phase", "scan"] + (["--json", args.json] if args.json else []),
                   check=True)


if __name__ == "__main__":
    main()
