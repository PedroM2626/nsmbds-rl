"""Drop-in Rust backend for the geometry queries of TilemapAStar.

Implements exactly the four methods src/eval_pinn_realtime.py calls, backed by
rust/nsmb_planner.exe (a persistent process speaking a one-line protocol).
tools/parity_planner.py is the gate: it proves these answers are identical to
tilemap_planner.py before this class is used anywhere.

Usage:
    from rust_planner import RustPlanner
    planner = RustPlanner()                 # exports geometry if missing
    dist = planner.pipe_ahead(abs_px, look_tiles=5, min_height_tiles=2)
"""
from __future__ import annotations

import os
import subprocess
import threading

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEOM = os.path.join(ROOT, "out", "nsmb_geometry.bin")
EXE = os.path.join(ROOT, "rust", "nsmb_planner.exe")

TILE = 16


class RustPlannerError(RuntimeError):
    pass


class RustPlanner:
    """Line-protocol client exposing the TilemapAStar query surface."""

    def __init__(self, geom=GEOM, exe=EXE):
        for path, how in ((exe, "run rust/build.sh"), (geom, "run tools/export_geometry.py")):
            if not os.path.exists(path):
                raise RustPlannerError(f"missing {path}: {how}")
        cmd = [exe, geom, "serve"]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      bufsize=1, text=True)
        self._lock = threading.Lock()
        self._min_col, self._max_col, self._reach, self._feasible, self._goal = self._summary()

    # -- protocol -----------------------------------------------------------
    def _ask(self, line: str) -> str:
        with self._lock:
            self._proc.stdin.write(line + "\n")
            self._proc.stdin.flush()
            out = self._proc.stdout.readline()
            if not out:
                raise RustPlannerError("planner process died mid-query")
            return out.strip()

    def _summary(self):
        parts = self._ask("G").split()
        if parts[0] != "G":
            raise RustPlannerError(f"bad geometry summary: {parts}")
        return int(parts[1]), int(parts[2]), int(parts[3]), parts[4] == "true", int(parts[5])

    def close(self):
        try:
            self._ask("quit")
        except Exception:
            pass
        try:
            self._proc.wait(timeout=5)
        except Exception:
            self._proc.kill()

    # -- TilemapAStar surface ------------------------------------------------
    def goal_px(self) -> int:
        return self._goal

    @property
    def reach(self) -> int:
        return self._reach

    @property
    def feasible(self) -> bool:
        return self._feasible

    def surface_ahead(self, mario_abs_px, look_tiles=4):
        """(cur_row, [(col, row_or_None), ...]) -- None means a pit."""
        parts = self._ask(f"S {float(mario_abs_px):.6f} {int(look_tiles)}").split()
        cur = None if parts[1] == "NA" else int(parts[1])
        prof = []
        for tok in parts[2:]:
            c, r = tok.split(":")
            prof.append((int(c), None if r == "NA" else int(r)))
        return cur, prof

    def wall_ahead(self, mario_abs_px, look_tiles=3, min_wall_tiles=3) -> int:
        return int(self._ask(f"W {float(mario_abs_px):.6f} {int(look_tiles)} "
                             f"{int(min_wall_tiles)}").split()[1])

    def pipe_ahead(self, mario_abs_px, look_tiles=5, min_height_tiles=2):
        v = self._ask(f"P {float(mario_abs_px):.6f} {int(look_tiles)} "
                      f"{int(min_height_tiles)}").split()[1]
        return None if v == "NA" else float(v)

    def is_narrow_corridor(self, mario_abs_px) -> bool:
        return self._ask(f"N {float(mario_abs_px):.6f}").split()[1] == "1"


if __name__ == "__main__":
    import time

    p = RustPlanner()
    print(f"goal={p.goal_px()} reach={p.reach} feasible={p.feasible}")
    N = 2000
    t0 = time.perf_counter()
    for i in range(N):
        px = float(i % 4000)
        p.pipe_ahead(px, look_tiles=5, min_height_tiles=2)
        p.surface_ahead(px, look_tiles=5)
        p.wall_ahead(px, look_tiles=6, min_wall_tiles=2)
        p.is_narrow_corridor(px)
    dt = time.perf_counter() - t0
    print(f"{4*N} roundtrips in {dt:.2f}s -> {4*N/dt:.0} queries/s over the pipe "
          f"({dt/N*1000:.3f} ms per controller step)")
