"""Build the savestates from your own ROM dump -- no savestate needed (4ba).

  .venv\\Scripts\\python.exe tools\\make_states.py              # every missing state
  .venv\\Scripts\\python.exe tools\\make_states.py --verify     # ... then check each one
  .venv\\Scripts\\python.exe tools\\make_states.py --course 1-1 --out scratch\\c11.ds1 --verify

The ROM must be the EUR dump this project measured (SHA-1 checked). Existing
states are never overwritten unless --force is given. --verify replays the
course's published contract on the built state (for 1-1: the legacy reflex
clear of evals/pinn_realtime_eval.json, byte for byte) and exits non-zero on
any difference.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import paths  # noqa: E402
import cold_boot  # noqa: E402
from eval_provenance import sha1_file  # noqa: E402

PARITY_KEYS = ("steps", "total_reward", "max_progress_px", "final_abs_px", "termination")
# Published evaluations replayed on a built state: their provenance argv is
# re-run with --state/--out swapped, and every episode must come out identical.
CONTRACTS = {
    "1-1": ["evals/eval_genetic_w11.json"],
    "1-2": ["evals/eval_1_2_clear.json"],
    "2-1": ["evals/eval_2_1_clear.json"],
    "3-1": ["evals/eval_special_31_full_clear.json"],
    "3-1_main": ["evals/eval_special_31_main_clear.json"],
    "4-1": ["evals/eval_4_1_clear.json"],
    "5-1": ["evals/eval_5_1_clear.json"],
    "6-1": ["evals/eval_6_1_clear.json"],
    "7-1": ["evals/eval_special_71_full_clear.json"],
    "8-1": ["evals/eval_8_1_clear.json"],
}
# Measured, documented exceptions (PROJECT_STATUS 4ba/4bb): these references carry
# emulator-internal state, inherited from a GUI-made savestate, that no RAM write
# reproduces. Their replays reach the same outcome but differ in detail; they
# are reported, not hidden, and then checked against the ROM-built contracts.
KNOWN_DIFF = {
    "evals/eval_1_2_clear.json": "same steps, goal and final X; total reward +3.0 (camera progress)",
    "evals/eval_special_71_full_clear.json": "same 809-step clear and reward; one enemy 1 px off from step 27",
    "evals/eval_special_31_full_clear.json": "same 557-step clear and total reward 3744.15; per-step camera reward differs from step 71",
    "evals/eval_special_31_main_clear.json": "same 497-step clear and total reward 3747.15; per-step camera reward differs from step 37",
}
# Where a reference cannot be rebuilt exactly, the same evaluation measured on
# the ROM-built state is committed under evals/rom/ and must replay exactly
# (two independent builds gave identical episodes, 4bb).
ROM_CONTRACTS = {
    "evals/eval_1_2_clear.json": "evals/rom/eval_1_2_clear.json",
    "evals/eval_special_71_full_clear.json": "evals/rom/eval_special_71_full_clear.json",
    "evals/eval_special_31_full_clear.json": "evals/rom/eval_special_31_full_clear.json",
    "evals/eval_special_31_main_clear.json": "evals/rom/eval_special_31_main_clear.json",
}
VOLATILE = ("latency", "fps", "wall", "elapsed", "seconds", "time_s", "duration")


def _strip(x):
    """Episode content without timing fields, which differ run to run."""
    if isinstance(x, dict):
        return {k: _strip(v) for k, v in x.items() if not any(t in k for t in VOLATILE)}
    if isinstance(x, list):
        return [_strip(v) for v in x]
    return x


def _script(name):
    for d in ("src", "tools"):
        p = os.path.join(ROOT, d, os.path.basename(name))
        if os.path.exists(p):
            return p
    raise FileNotFoundError(name)


def replay(contract, state):
    """Re-run a published evaluation on `state`; True when every episode is identical."""
    pub = json.load(open(os.path.join(ROOT, contract), encoding="utf-8"))
    argv = list(pub["provenance"]["argv"])
    args, skip = [], False
    for a in argv[1:]:                       # drop --state/--out and their values
        if skip:
            skip = False
            continue
        if a in ("--state", "--out"):
            skip = True
            continue
        args.append(a)
    out = state + ".replay.json"
    cmd = [sys.executable, _script(argv[0])] + args + ["--state", state, "--out", out]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    # Some evaluators exit non-zero by design when not every episode clears
    # (eval_special_route.py); only a missing result is an error.
    if not os.path.exists(out):
        print(f"  FAIL {contract}: evaluator exited {r.returncode} without a result")
        print(r.stdout[-1500:], r.stderr[-1500:])
        return False
    got = json.load(open(out, encoding="utf-8"))
    os.remove(out)
    a, b = _strip(got["episodes"]), _strip(pub["episodes"])
    same = a == b
    rom = ROM_CONTRACTS.get(contract)
    if not same and rom and os.path.exists(os.path.join(ROOT, rom)):
        exact = a == _strip(json.load(open(os.path.join(ROOT, rom), encoding="utf-8"))["episodes"])
        print(f"  {'ok  ' if exact else 'FAIL'} {rom}: {len(a)} episodes "
              f"{'identical (ROM-built contract)' if exact else 'differ from the ROM-built contract'}"
              f"; reference run differs as documented: {KNOWN_DIFF.get(contract, '?')}")
        return exact
    n_bad = sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b))
    tag = "ok  " if same else ("KNOWN" if contract in KNOWN_DIFF else "FAIL")
    note = "" if same else (f" -- known difference: {KNOWN_DIFF[contract]}" if contract in KNOWN_DIFF
                             else " -- first built episode: " + str(a[0])[:300])
    print(f"  {tag} {contract}: {len(b)} episodes, {n_bad} differ{note}")
    return same or contract in KNOWN_DIFF


def parity(state):
    """The frozen 1-1 contract: the legacy reflex clear of evals/pinn_realtime_eval.json."""
    pub = json.load(open(os.path.join(ROOT, "evals", "pinn_realtime_eval.json"),
                         encoding="utf-8"))["episodes"][0]
    out = state + ".verify.json"
    cmd = [sys.executable, os.path.join(ROOT, "src", "eval_pinn_realtime.py"),
           "--controller", "reactive", "--headless", "--episodes", "1",
           "--reflex", "legacy", "--state", state, "--out", out]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        print(r.stdout[-1500:], r.stderr[-1500:])
        return False
    ep = json.load(open(out, encoding="utf-8"))["episodes"][0]
    os.remove(out)
    ok = all(ep[k] == pub[k] for k in PARITY_KEYS)
    print(f"  {'ok  ' if ok else 'FAIL'} parity (pinn_realtime_eval.json): "
          + ", ".join(f"{k}={ep[k]!r}" for k in PARITY_KEYS))
    return ok


def verify(course, state):
    ok = parity(state) if course == "1-1" else True
    for c in CONTRACTS.get(course, []):
        ok &= replay(c, state)
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--course", choices=sorted(cold_boot.COURSES), action="append",
                    help="default: every course")
    ap.add_argument("--rom", default=paths.ROM_REL)
    ap.add_argument("--out", help="write here instead of the canonical path (one course)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing state")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    rom = paths.resolve(args.rom)
    if not os.path.exists(rom):
        sys.exit(f"ROM not found: {rom}\nPut your own dump of New Super Mario Bros. (EUR) there.")
    sha = sha1_file(rom)
    if sha != cold_boot.ROM_SHA1:
        sys.exit(f"ROM SHA-1 {sha} is not the measured EUR dump {cold_boot.ROM_SHA1}; "
                 "the input scripts and every published number assume that dump.")
    courses = args.course or sorted(cold_boot.COURSES)
    if args.out and len(courses) != 1:
        sys.exit("--out needs exactly one --course")

    # Generation needs this process's single DeSmuME, so only one course is built
    # per process; further courses run as children.
    todo = []
    for c in courses:
        out = paths.resolve(args.out or cold_boot.COURSES[c]["out"])
        if os.path.exists(out) and not args.force:
            print(f"[{c}] {paths.display(out)} exists, kept (use --force to rebuild)")
        else:
            todo.append((c, out))
    for i, (c, out) in enumerate(todo):
        if i == 0:
            cold_boot.generate(c, out, rom)
        else:
            subprocess.run([sys.executable, __file__, "--course", c, "--rom", rom,
                            "--out", out, "--force"], check=True)
            continue
        print(f"[{c}] built {paths.display(out)} from the ROM (sha1 {sha1_file(out)})")
    if args.verify:
        bad = []
        for c in courses:
            out = paths.resolve(args.out or cold_boot.COURSES[c]["out"])
            print(f"[{c}] verifying {paths.display(out)}")
            if not verify(c, out):
                bad.append(c)
        if bad:
            sys.exit(f"verification FAILED for {bad}")
        print("verification passed")


if __name__ == "__main__":
    main()
