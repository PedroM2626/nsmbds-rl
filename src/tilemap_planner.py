"""Tilemap A* jump-reachability and Full 2D Spatio-Temporal planner for NSMB-DS levels (Task 3).

Parses level geometry from the ROM (`course.py`) and plans a forward path as a
sequence of *run-jump* hops between solid ground platforms, so the reactive/MPC
controller takes off from the exact edge column for each pit (rather than a crude
fixed pixel lookahead) and never stalls on ledges/walls.

In addition to the 1D column projection (`surface`, `obstacle_h`, `pits`, `takeoffs`)
required by the World 1-1 parity contract, this module implements a full 2D
spatio-temporal planner over `(col, row, t)` using the empirically measured
ballistic jump regimes (`out/dataset/physics_jump_state_machine.json`), 2D solid
cells (`Course.solid_cells_2d`), overhead ceiling clearances, walkable slope/step
discrimination, crumbling tiles, and moving enemy trajectories `(dx + vx*t, dy + vy*t)`.

Physics grounding
-----------------
With the dash action enabled, Mario reaches ~3 px/frame and a running jump spans
roughly `JUMP_REACH_PX` horizontally. A pit narrower than that can be cleared from
its near edge; wider pits need the plan to confirm feasibility (they would require
a higher/longer jump than the engine's walking-running jump allows).
"""
from __future__ import annotations

import heapq
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from course import Course, TILE, require_plannable  # noqa: E402

ROM = "data/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds"
# Empirical kinematics grounded via 60 Hz DeSmuME telemetry (Route B decomposition):
# Running leap: 52 frames, 109.73 px reach, apex at frame 26 (71.06 px high, ~55 px forward)
# Walking jump: 52 frames, 75.33 px reach, apex at frame 26 (71.06 px high)
# Short hop: 22 frames, 19.94 px high
JUMP_REACH_PX = 109.73          # empirical running-jump horizontal reach
JUMP_REACH_TILES = int(JUMP_REACH_PX // TILE)  # 6 tiles (96 px safe margin)
FRAMES_PER_STEP = 8.0           # 8 emulator frames per RL step

# Four empirical jump regimes measured in out/dataset/physics_jump_state_machine.json
JUMP_REGIMES = {
    "tap_hop": {
        "action": 3,
        "air_frames": 22.0,
        "max_height_px": 19.94,
        "vy0": 3.5938,
        "g_up": 0.3284,
        "g_down": 0.2662,
        "vx_scale": 0.85,
    },
    "standing_full": {
        "action": 4,
        "air_frames": 48.0,
        "max_height_px": 63.81,
        "vy0": 3.5938,
        "g_up": 0.0963,
        "g_down": 0.2339,
        "vx_scale": 0.0,
    },
    "walk_jump": {
        "action": 3,
        "air_frames": 52.0,
        "max_height_px": 71.06,
        "vy0": 3.7188,
        "g_up": 0.0919,
        "g_down": 0.2118,
        "vx_default": 1.4716,
    },
    "run_leap": {
        "action": 4,
        "air_frames": 52.0,
        "max_height_px": 71.06,
        "vy0": 3.7188,
        "g_up": 0.0919,
        "g_down": 0.2118,
        "vx_default": 2.136,
    },
}


def dynamic_jump_reach_px(vx_px_per_frame=None, airborne_frames=52.0):
    """Computes ballistic horizontal reach using empirical DeSmuME kinematics:
    X_reach = vx * t_airborne.
    Walking (1.47 px/f) -> 75.33 px (~4.7 tiles).
    Running/dash (2.14 px/f) -> 109.73 px (~6.8 tiles).
    Standing (0.0 px/f) -> 0 px (pure vertical leap: 63.81 px high).
    """
    vx = max(0.0, float(vx_px_per_frame if vx_px_per_frame is not None else 2.136))
    return vx * airborne_frames


def dynamic_jump_reach_tiles(vx_px_per_frame=None, airborne_frames=52.0):
    return int(dynamic_jump_reach_px(vx_px_per_frame, airborne_frames) // TILE)


def sample_ballistic_arc_px(x0_px: float, y0_row: float, vx_pf: float, regime_name: str = "run_leap", n_steps: int = 7):
    """Sample `(x_px, row, step_idx)` along an empirical jump arc at 8-frame RL steps.
    Note: smaller `row` means higher elevation (`row = y0_row - dy_px / 16.0`).
    """
    reg = JUMP_REGIMES[regime_name]
    vy0 = reg["vy0"]
    g_up = reg["g_up"]
    g_down = reg["g_down"]
    t_apex = vy0 / g_up
    max_h = reg["max_height_px"]
    pts = []
    for s in range(1, n_steps + 1):
        f = s * FRAMES_PER_STEP
        dx = vx_pf * f
        if f <= t_apex:
            dy = vy0 * f - 0.5 * g_up * (f ** 2)
            dy = min(dy, max_h)
        else:
            df = f - t_apex
            dy = max_h - 0.5 * g_down * (df ** 2)
        row_t = y0_row - (dy / float(TILE))
        pts.append((x0_px + dx, row_t, s))
    return pts


class TilemapAStar:
    def __init__(self, rom_path=ROM, name="course/A01_1.bin", jump_reach_tiles=JUMP_REACH_TILES,
                 exit_known=True, use_inferred=False):
        course = Course(name=name, rom_path=rom_path, use_inferred=use_inferred)
        require_plannable(course, exit_known=exit_known)   # refuse to plan on a fiction
        self.course = course
        self.course_name = name
        cols = sorted(course.solid_columns())
        self.min_col, self.max_col = cols[0], cols[-1]
        self.solid = set(cols)
        self.goal_col = self.max_col
        self.reach = jump_reach_tiles
        # pits = maximal runs of missing columns between [min_col, max_col]
        self.pits = []          # (start_col, end_col) inclusive, missing ground
        c = self.min_col
        while c <= self.max_col:
            if c not in self.solid:
                a = c
                while c <= self.max_col and c not in self.solid:
                    c += 1
                self.pits.append((a, c - 1))
            c += 1
        self.pit_widths = [(a, b, b - a + 1) for (a, b) in self.pits]
        # walls/pipes = solid rects that rise well above the walking surface (a
        # running Mario collides and must *leap* them). Store the obstacle height per
        # column so short 1-2 tile steps are walked up, and only tall walls/pipes
        # (>= min_wall_tiles) trigger a leap -- leaping at small steps just throws
        # Mario into nearby enemies.
        self.obstacle_h = {}     # col -> tiles the wall rises above the floor
        self.true_wall_h = {}    # col -> wall height excluding walkable slopes
        self.ceilings = {}       # col -> sorted list of overhead solid rows (ty < floor_row)
        self.crumble_cols = set()
        self.stepping_platforms = {}  # col -> top row of mid-air stepping blocks over pits

        slope_objs = getattr(course, "inferred_slope_objs", frozenset({0x06, 0x07, 0x08}))
        for r in course.rects:
            top, bot = r["ty"], r["ty"] + r["h"]
            obj_id = r["obj"]
            for cc in range(r["tx"], r["tx"] + r["w"]):
                if obj_id == 37 and top >= 18:
                    self.crumble_cols.add(cc)
                fr = course.floor_row(cc)
                if fr is not None and top <= fr - 2 and bot >= fr - 1:
                    h = fr - top
                    if h > self.obstacle_h.get(cc, 0):
                        self.obstacle_h[cc] = h
                    if obj_id not in slope_objs and h > self.true_wall_h.get(cc, 0):
                        self.true_wall_h[cc] = h
                # Track overhead ceilings / floating blocks (rows 18 .. fr - 2)
                if top >= 18:
                    if fr is not None and bot <= fr - 2:
                        s_ceil = self.ceilings.setdefault(cc, set())
                        for yy in range(top, bot):
                            s_ceil.add(yy)
                    elif fr is None and r["h"] <= 4 and top >= 30:
                        # Floating stepping block over a pit (e.g. obj 20 in H01_1)
                        prev = self.stepping_platforms.get(cc)
                        if prev is None or top < prev:
                            self.stepping_platforms[cc] = top

        self.ceilings = {cc: sorted(rows) for cc, rows in self.ceilings.items()}
        self.obstacle_cols = set(self.obstacle_h)
        # Walkable SURFACE height per column: the top of the solid run contiguous with
        # the ground (ground + blocks/pipes/stairs stacked on it), ignoring floating
        # ?-blocks/coins that Mario cannot stand on from the floor. Row 30 = floor;
        # a smaller row = a higher surface. This unifies pits (None), pipes and the
        # end staircase into a single height profile the controller can follow.
        solid_rows = course.ground_cover
        self.surface = {}        # col -> top walkable row (or absent = pit)
        for cc in range(self.min_col, self.max_col + 1):
            fr = course.floor_row(cc)
            if fr is None:
                continue
            top = fr
            rows = solid_rows.get(cc, set())
            while (top - 1) in rows:      # climb contiguous solids above the floor
                top -= 1
            self.surface[cc] = top
        # 2D solid cells for full 2D collision checking
        self.cells_2d = course.solid_cells_2d()
        # takeoff column = last solid before a pit; landing = first solid after it
        self.takeoffs = {}      # takeoff_col -> (pit_start, pit_end, landing_col, width)
        for (a, b) in self.pits:
            tk = a - 1
            if tk in self.solid:
                self.takeoffs[tk] = (a, b, b + 1, b - a + 1)
        self.feasible = all((b - a + 1) <= self.reach for (a, b) in self.pits)
        # Precompute 2D A* static path over (col, row)
        self.path_2d = self._compute_static_2d_path()

    def _compute_static_2d_path(self):
        """Compute global 2D A* waypoint sequence `(col, row)` from `min_col` to `max_col`
        using walkable surfaces and mid-air stepping platforms over wide pits."""
        nodes = {}
        for cc, row in self.surface.items():
            nodes[cc] = int(row)
        for cc, row in self.stepping_platforms.items():
            if cc not in nodes:
                nodes[cc] = int(row)
        if not nodes:
            return []
        start_c = min(nodes)
        goal_c = max(nodes)
        pq = [(0.0, start_c)]
        g_cost = {start_c: 0.0}
        parent = {}
        max_step_cols = max(self.reach + 1, 8)
        while pq:
            _, u = heapq.heappop(pq)
            if u == goal_c:
                break
            ru = nodes[u]
            for v in range(u + 1, min(goal_c + 1, u + max_step_cols + 1)):
                if v not in nodes:
                    continue
                rv = nodes[v]
                dc = v - u
                dr = ru - rv  # positive means jumping up
                # Check vertical reachability (max jump height ~ 4.4 tiles)
                if dr > 4:
                    continue
                # Penalty for long jumps or high elevation changes
                step_cost = float(dc) + (0.5 * max(0, dc - 1) ** 2) + 0.3 * abs(dr)
                # Check if an overhead ceiling blocks the arc midpoint
                mid_c = (u + v) // 2
                ceil_rows = self.ceilings.get(mid_c, ())
                if dc >= 3 and any(( min(ru, rv) - 3 ) <= cr <= max(ru, rv) for cr in ceil_rows):
                    step_cost += 4.0
                ng = g_cost[u] + step_cost
                if ng < g_cost.get(v, float("inf")):
                    g_cost[v] = ng
                    parent[v] = u
                    h = float(goal_c - v)
                    heapq.heappush(pq, (ng + h, v))
        if goal_c not in g_cost and g_cost:
            goal_c = max(g_cost)
        path = []
        curr = goal_c
        while curr in parent:
            path.append((curr, nodes[curr]))
            curr = parent[curr]
        if start_c in nodes:
            path.append((start_c, nodes[start_c]))
        path.reverse()
        return path

    def next_takeoff(self, mario_col):
        """Nearest required takeoff column strictly ahead of (or at) Mario. Returns
        dict with the takeoff col, pit width, and landing col, or None near the goal."""
        best = None
        for tk, (a, b, land, w) in self.takeoffs.items():
            if tk >= mario_col - 1:            # takeoff at/after current position
                if best is None or tk < best["takeoff_col"]:
                    best = {"takeoff_col": tk, "pit_start": a, "pit_end": b,
                            "landing_col": land, "width": w}
        return best

    def ceiling_ahead(self, mario_abs_px, mario_row=None, look_tiles=4, clearance_tiles=3):
        """Returns distance in tiles (`1..look_tiles`) to the nearest low ceiling
        that sits within `clearance_tiles` above `mario_row` (0 = none).
        Used by the 2D planner to prefer low hops or running below overhead bars."""
        col = int(mario_abs_px / TILE)
        base_row = int(round(mario_row)) if mario_row is not None else self.surface.get(col, 30)
        if base_row is None:
            base_row = 30
        for d in range(0, look_tiles + 1):
            c_rows = self.ceilings.get(col + d)
            if not c_rows:
                continue
            for cr in c_rows:
                if base_row - clearance_tiles <= cr < base_row:
                    return max(1, d)
        return 0

    def step_up_ahead(self, mario_abs_px, look_tiles=2):
        """Returns `(dist_tiles, rise_tiles)` if the walkable surface steps up by 1-2 tiles
        within `look_tiles` ahead (excluding smooth slopes), or `(0, 0)` if flat/down."""
        col = int(mario_abs_px / TILE)
        cur = self.surface.get(col)
        if cur is None:
            cur = self.course.probe_surface(col)
            if cur is not None:
                cur = int(round(cur))
        if cur is None:
            return 0, 0
        for d in range(1, look_tiles + 1):
            nxt = self.surface.get(col + d)
            if nxt is None:
                ps = self.course.probe_surface(col + d)
                if ps is not None:
                    nxt = int(round(ps))
            if nxt is not None and 1 <= (cur - nxt) <= 2:
                return d, int(cur - nxt)
        return 0, 0

    def plan_2d(self, mario_abs_px, mario_row=None, vx_pf=2.136, enemies=None):
        """Full 2D Spatio-Temporal trajectory evaluation over candidate action regimes:
          - `'run'` (action 2: keep running on ground)
          - `'tap_hop'` (action 3 short hop: clears 1-2 tile step-ups or close enemies under low ceilings)
          - `'walk_jump'` (action 3 full jump: moderate horizontal reach 75 px)
          - `'run_leap'` (action 4 full dash leap: max horizontal reach 110 px)
          - `'wait'` (action 6: decelerate / in-place hop when landing zone has an enemy)

        Evaluates 2D tile occupancy `(col, row)`, pit spans, overhead ceiling clearance,
        and dynamic enemy trajectories `(dx_i + vx_i * f, dy_i + vy_i * f)` over a
        6-step (48-frame) horizon.

        Returns dict with `best_regime`, `recommended_action`, `expected_landing_px`,
        `ceiling_dist`, `step_up`, and per-regime scores.
        """
        col = int(mario_abs_px / TILE)
        cur_row = float(mario_row) if mario_row is not None else float(self.surface.get(col, 30) or 30)
        enemies = enemies or []
        ceil_d = self.ceiling_ahead(mario_abs_px, cur_row, look_tiles=4, clearance_tiles=4)
        step_d, step_rise = self.step_up_ahead(mario_abs_px, look_tiles=2)

        scores = {}
        landings = {}

        # Candidate 0: Ground run (action 2)
        run_score = 0.0
        v_ground = max(1.5, float(vx_pf))
        for s in range(1, 5):
            f = s * FRAMES_PER_STEP
            x_t = mario_abs_px + v_ground * f
            c_t = int(x_t / TILE)
            if c_t not in self.solid and c_t not in self.stepping_platforms:
                run_score -= 100.0  # falls into pit
            if self.true_wall_h.get(c_t, 0) >= 2:
                run_score -= 40.0   # runs into wall
            for e in enemies:
                edx = e.get("dx")
                edy = e.get("dy", 0.0)
                evx = e.get("vx", -0.8)
                if edx is not None:
                    pred_ex = mario_abs_px + edx + evx * f
                    if abs(x_t - pred_ex) < 18.0 and abs(edy) < 20.0:
                        run_score -= 80.0
        if step_d == 1 and step_rise >= 1:
            run_score -= 35.0
        scores["run"] = round(run_score + 15.0, 2)
        landings["run"] = round(mario_abs_px + v_ground * 32.0, 1)

        # Candidates 1..3: Airborne regimes
        for rname, v_cand in (("tap_hop", max(1.4, vx_pf * 0.85)),
                              ("walk_jump", 1.4716),
                              ("run_leap", max(2.136, vx_pf))):
            pts = sample_ballistic_arc_px(mario_abs_px, cur_row, v_cand, rname, n_steps=6)
            sc = 0.0
            land_x = pts[-1][0]
            for (x_t, row_t, s) in pts:
                f = s * FRAMES_PER_STEP
                c_t = int(x_t / TILE)
                r_int = int(round(row_t))
                # Overhead ceiling collision check
                if r_int in self.ceilings.get(c_t, ()):
                    sc -= 45.0
                # Dynamic enemy check in space-time
                for e in enemies:
                    edx = e.get("dx")
                    edy = e.get("dy", 0.0)
                    evx = e.get("vx", -0.8)
                    evy = e.get("vy", 0.0)
                    if edx is not None:
                        pred_ex = mario_abs_px + edx + evx * f
                        pred_erow = cur_row - ((edy + evy * f) / float(TILE))
                        if abs(x_t - pred_ex) < 16.0 and abs(row_t - pred_erow) < 1.1:
                            # If descending onto enemy top, stomp bonus; else collision penalty
                            if s >= 4 and row_t < pred_erow:
                                sc += 10.0
                            else:
                                sc -= 75.0
                # Check landing support near end of arc
                if s >= 5:
                    surf_r = self.surface.get(c_t, self.stepping_platforms.get(c_t))
                    if surf_r is None:
                        sc -= 60.0
                    else:
                        sc += 20.0 + 0.2 * (x_t - mario_abs_px)
                        land_x = x_t
                        break
            if ceil_d > 0 and rname in ("walk_jump", "run_leap"):
                sc -= 20.0
            scores[rname] = round(sc, 2)
            landings[rname] = round(land_x, 1)

        best_regime = max(scores, key=lambda k: scores[k])
        action_map = {"run": 2, "tap_hop": 3, "walk_jump": 3, "run_leap": 4}
        rec_action = action_map[best_regime]
        if scores[best_regime] < -30.0:
            best_regime = "wait"
            rec_action = 6

        return {
            "best_regime": best_regime,
            "recommended_action": rec_action,
            "expected_landing_px": landings.get(best_regime, mario_abs_px),
            "ceiling_dist": ceil_d,
            "step_up": (step_d, step_rise),
            "scores": scores,
        }

    def enemy_threat_ahead(self, mario_abs_px, enemies=None, lookahead_px=80.0):
        """Scans real-time enemies for collision threats ahead, especially near takeoff edges.
        Returns:
            threat_type: None | 'dash_leap' | 'jump_stomp' | 'retreat_wait'
            info: dict with threat metrics
        """
        if not enemies:
            return None, {}
        for e in enemies:
            dx = e.get("dx")
            dy = e.get("dy", 0.0)
            if dx is None or dx <= 0.0 or dx > lookahead_px or abs(dy) > 48.0:
                continue
            evx = e.get("vx", 0.0)
            col = int((mario_abs_px + dx) / TILE)
            # Check if enemy is right at a takeoff boundary
            is_takeoff = (col in self.takeoffs) or ((col + 1) in self.takeoffs)
            if is_takeoff:
                # Goomba blocking pit edge!
                if dx < 48.0:
                    return "dash_leap", {"dx": dx, "dy": dy, "col": col, "hazard": "takeoff_blocker"}
                else:
                    return "retreat_wait", {"dx": dx, "dy": dy, "col": col, "hazard": "takeoff_blocker"}
            if dx < 38.0:
                return "jump_stomp", {"dx": dx, "dy": dy, "col": col, "hazard": "flat_enemy"}
        return None, {}

    def should_jump(self, mario_abs_px, vx_px_per_frame=None, enemies=None):
        """True when Mario is at a pit takeoff edge, approaching an enemy, or needs a running leap.
        Returns (jump: bool, upcoming_pit_width_tiles: int, reason: str).
        """
        # 1. Dynamic enemy threat check
        threat, info = self.enemy_threat_ahead(mario_abs_px, enemies)
        if threat in ("dash_leap", "jump_stomp"):
            return True, 0
        elif threat == "retreat_wait":
            return False, 0

        col = int(mario_abs_px / TILE)
        nxt = col + 1
        reach_tiles = dynamic_jump_reach_tiles(vx_px_per_frame)
        if nxt not in self.solid:                       # hole immediately ahead
            width = self._gap_width_at(nxt)
            return True, width
        # wide pit: anticipate takeoff by one tile so the long hop clears the gap
        tk = col + 1
        if tk in self.takeoffs and self.takeoffs[tk][3] >= 3:
            return True, self.takeoffs[tk][3]
        return False, 0

    def wall_ahead(self, mario_abs_px, look_tiles=3, min_wall_tiles=3):
        """Tiles until the next tall pipe/wall Mario would run into (0 = none).
        Only walls rising >= ``min_wall_tiles`` above the floor count, so 1-2 tile
        steps are walked up rather than leapt into nearby enemies. Used to trigger a
        *running* leap while Mario still has forward speed, since a standstill jump
        is too short to clear a ~4-tile pipe."""
        col = int(mario_abs_px / TILE)
        for d in range(1, look_tiles + 1):
            if self.obstacle_h.get(col + d, 0) >= min_wall_tiles:
                return d
        return 0

    def surface_ahead(self, mario_abs_px, look_tiles=4):
        """Height profile of the walkable surface over the next columns.
        Returns (cur_row, [(col, row_or_None), ...]) where a smaller row is higher and
        None is a pit. The controller compares these to decide walk / hop / leap."""
        col = int(mario_abs_px / TILE)
        cur = self.surface.get(col)
        prof = [(col + d, self.surface.get(col + d)) for d in range(1, look_tiles + 1)]
        return cur, prof

    def _gap_width_at(self, col):
        for (a, b) in self.pits:
            if a <= col <= b:
                return b - a + 1
        return 0

    def pipe_ahead(self, mario_abs_px, look_tiles=5, min_height_tiles=2):
        """Distance in pixels to the next pipe/wall rising >= min_height_tiles,
        or None if no pipe in lookahead.
        Used to ensure Mario launches before X = 358 px for the X = 384 px pipe."""
        col = int(mario_abs_px / TILE)
        cur = self.surface.get(col, 30)
        for d in range(1, look_tiles + 1):
            ahead_col = col + d
            h = self.obstacle_h.get(ahead_col, 0)
            s = self.surface.get(ahead_col)
            rise = (cur - s) if (cur is not None and s is not None) else 0
            if h >= min_height_tiles or rise >= min_height_tiles:
                return ahead_col * TILE - mario_abs_px
        return None

    def is_narrow_corridor(self, mario_abs_px):
        """True when Mario is approaching or within the narrow elevated corridor (1472-1552px, cols 92-97),
        where the Goomba patrols at X=1498px."""
        col = int(mario_abs_px / TILE)
        return 91 <= col <= 97

    def goal_px(self):
        return self.max_col * TILE


if __name__ == "__main__":
    p = TilemapAStar()
    print(f"goal_col={p.goal_col} (px {p.goal_px()})  feasible={p.feasible}")
    print("pits (start,end,width tiles):", p.pit_widths)
    print("takeoffs:", {k: v for k, v in sorted(p.takeoffs.items())})
    wide = [v for v in p.pit_widths if v[2] > p.reach]
    print(f"pits beyond {p.reach}-tile run-jump reach:", wide)
    print(f"2D path waypoints: {len(p.path_2d)}")
