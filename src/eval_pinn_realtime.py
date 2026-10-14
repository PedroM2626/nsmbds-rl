"""Stage 3: deploy the PINN-world-model agent in REAL TIME on the emulated
New Super Mario Bros. DS console and measure whether it clears the level.

Two controllers (both consume the *same* RAM observation, zero-shot):
  * mpc : closed-loop CEM-MPC planning inside the hard-residual PINN (no policy
          training; re-plans every executed step from the true state).
  * ppo : amortised Dyna-PPO policy trained inside imagination (<1 ms/frame).

Rendering streams the emulator's top screen through an OpenCV window so the run
is directly watchable; use --headless for batched metric collection.

Usage
-----
    # watch the MPC agent play live
    python src/eval_pinn_realtime.py --controller mpc --episodes 1 --render
    # headless, 5 episodes, metrics to evals/pinn_realtime_eval.json
    python src/eval_pinn_realtime.py --controller ppo --episodes 5

Output
------
    evals/pinn_realtime_eval.json
"""
import argparse
import contextlib
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ram_env import MarioRamEnv  # noqa: E402
from pinn_world_model import MarioPINNWorldModel, CEMMPController, N_ACTIONS, ShapingConfig  # noqa: E402
from course import Course, TILE  # noqa: E402
from tilemap_planner import TilemapAStar, dynamic_jump_reach_tiles  # noqa: E402

ROM = "data/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds"
STATE = "data/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).ds1"
SPAWN_ABS_PX = 88.0  # Mario object B at entrance (see src/course.py calibration)
FRAMESKIP = 8
FRAME_DT = FRAMESKIP / 60.0  # wall-clock a single env step represents (s)

# Hostile entities in RAM that require avoidance/stomp (ignores coins, items, moving platforms)
# 160: Goomba, 163: Koopa, 164: Paratroopa, 31: Piranha, 54: Spiney, 68: Buzzy, 69: DryBones, 81: HammerBro
# Measured with the behavioral hostility oracles (pin-contact, standing-bait,
# agent-walk-touch -- see 4k): 172/184/312 patrol H01_1 (ledge walker, floor
# patrol, ceiling-corner sitter -- all stomp-vanish on contact); 247 is a small
# spinner in E01_1 that kills on the first contact frame; 258 is a leaper that
# arcs through the standing body zone and killed a motionless Mario at frame
# 160 (its first full cycle -- the initial 3-frame touch was too thin a
# negative, corrected by standing-bait); 38/158 patrol F01_1 and both
# stomp-vanish on contact; 232 is B01_1's desert crawler, stomp-vanish;
# 135/311 patrol D01_1 (ice level), stomp-vanish. Measured NOT hostile and
# left out: 180 (140 pinned
# contact frames + repeated drops, harmless), 291/307 (static foam high at
# the level edge).
HOSTILE_ENEMY_TYPES = {160, 163, 164, 31, 54, 68, 69, 81, 38, 135, 158, 172, 184, 232, 247, 258, 311, 312}

def is_hostile_enemy(et):
    return int(round(float(et))) in HOSTILE_ENEMY_TYPES


@contextlib.contextmanager
def _silence_c_stdout(enabled=True):
    if not enabled:
        yield
        return
    try:
        out_fd = sys.stdout.fileno()
        saved = os.dup(out_fd)
        devnull = os.open(os.devnull, os.WRONLY)
        sys.stdout.flush(); os.dup2(devnull, out_fd); os.close(devnull)
        yield
        sys.stdout.flush(); os.dup2(saved, out_fd); os.close(saved)
    except Exception:
        yield


def goal_px(name="course/A01_1.bin"):
    """Absolute pixel X of the course's exit (a flagpole contact in World 1-1)."""
    try:
        c = Course(name=name, rom_path=ROM)
        flag = next((s['x_px'] for s in c.sprites if s['type'] == 32), None)
        if flag is not None:
            return flag
        return max(c.ground_cover.keys()) * TILE
    except Exception:
        return 4032


class MPCAgent:
    name = "mpc"

    def __init__(self, model, device, horizon, n_cand, n_iter, seed, cfg=None):
        self.ctrl = CEMMPController(model, horizon=horizon, n_candidates=n_cand,
                                    n_iters=n_iter, device=device, seed=seed, cfg=cfg)
        self._prev = None

    def act(self, obs):
        a, self._prev = self.ctrl.plan(obs, best_prev=self._prev)
        return a

    def reset(self):
        self._prev = None


class BaseReflexController:
    """Unified physical reflex controller for NSMB-DS World 1-1.

    Grounded in Route B empirical kinematics (60 Hz DeSmuME telemetry):
      - Running leap: Action 4 (Right+X+A), reach 109.73 px, apex 71.06 px @ f26.
      - Walking hop: Action 2 (Right+A), reach 75.33 px, apex 19.94 px @ f11.
      - Dash drive: Action 3 (Right+X), horizontal velocity ~2.14 px/f.
      - In-place hop: Action 7 (A), pure vertical lift to clear close threats.
      - Retreat unstick: Action 5 (Left).

    reflex_mode "legacy" keeps the World 1-1 pixel bands that produced the
    published clear. "profile" replaces them with values derived from the ROM by
    src/level_profile.py, which is what any other course would need. Both are
    reachable from the CLI so the two can be compared instead of asserted.
    """
    HOP = 2    # right + jump (walking hop: climbs 1-tile staircase steps, avoids close frontal contact)
    RUN = 3    # right + dash (maintain dash speed / airborne momentum)
    JUMP = 4   # right + dash + jump (running leap: reach 109.73px, apex 71.06px @ frame 26)
    LEFT = 5   # retreat / unstick
    IN_PLACE_HOP = 7  # vertical jump in place

    reflex_mode = "legacy"
    profile = None
    # The wall oracle's course handle (4v), set per course by the eval
    # entrypoint for BOTH arms. None or an unprobed cell = the rule is
    # silent and every existing response stands.
    wall_probe_course = None

    def _abs_px(self, obs):
        """Mario's absolute stage x.

        Profile mode: true mario_B (spawn + relative), so every geometry
        query lands on the column Mario actually stands in.

        Legacy mode: the historical coordinate the pixel bands and planner
        windows were tuned in -- mario_B + 88. When they were tuned,
        obs[0] was accidentally absolute (the ram_env origin bug, 4o), so
        SPAWN_ABS_PX + obs[0]*512 computed 88 + mario_B on EVERY course,
        whatever that course's own spawn was. Reproducing that coordinate
        through the relative obs (spawn_mb + 88 + rel*512) reconstructs
        mario_B with ~0.04 px of float32 noise -- invisible until a
        decision sits on a comparison boundary, and A02_1's pipe window
        sits on exactly one (4o: identical inputs, one arm leaps and the
        other runs). When the RAM handle is available the raw mario_B is
        read instead and the historical value is exact: the 1-2
        knife-edge reproduces and the historical trajectory comes back.
        """
        if self.reflex_mode == "profile" and self.profile is not None:
            return self.profile.abs_px(float(obs[0]))
        if self.ram is not None:
            mb = self.ram.poll()["mario_B"]
            if mb is not None:
                # Bit-exact historical arithmetic: the tuning era computed
                # 88 + obs0*512 with obs0 the env's float32 obs value --
                # float32(mb/4096/512) -- so the bands were calibrated
                # against the ROUNDED mario_B, and knife-edge decisions
                # (A02_1's pipe window) flip on exactly that rounding.
                v = np.float32(mb / 4096.0 / 512.0)
                return 88.0 + float(v) * 512.0
        base = (self.profile.spawn_px if self.profile is not None
                else SPAWN_ABS_PX)
        return base + 88.0 + obs[0] * 512.0

    def _surface_threat(self, obs):
        if self.reflex_mode == "profile" and self.profile is not None:
            return self._surface_threat_profile(obs)
        return self._surface_threat_legacy(obs)

    def __init__(self, planner=None, enemy_lo=0.0, enemy_hi=75.0, enemy_dy=55.0, pit_lookahead=2, enabled=True, ram=None):
        self.planner = planner
        self.ram = ram
        self.enemy_lo = enemy_lo
        self.enemy_hi = enemy_hi
        self.enemy_dy = enemy_dy
        self.pit_lookahead = pit_lookahead
        self.enabled = enabled
        self.clearing = False
        self._last_x = None
        self._stuck = 0
        self._blocked = 0
        self._queue = []
        self._blocked_positions = []  # abs px of every blocked event

    def reset(self):
        self.clearing = False
        self._last_x = None
        self._stuck = 0
        self._blocked = 0
        self._queue = []
        self._blocked_positions = []

    def _surface_threat_legacy(self, obs):
        """Geometry decisions for pits, pipes, ledges, and the end staircase."""
        if self.pit_lookahead > 0 and any(obs[17 + c] < 0.5 for c in range(self.pit_lookahead)):
            return "leap"  # pit ahead -> commit running leap

        if self.planner is not None:
            # _abs_px, not the raw formula: legacy's coordinate is
            # mario_B + 88 on every course (see _abs_px), and with the
            # origin-fixed (relative) obs the raw SPAWN_ABS_PX + obs[0]*512
            # would read mario_B -- 88 px west of every band below.
            abs_px = self._abs_px(obs)

            # Endgame flagpole push (X >= 3870 px)
            if abs_px >= 3870.0:
                if abs_px < 3920.0:
                    return "leap"   # leap off the top of the staircase
                elif abs_px < 4020.0:
                    return "run"    # sprint into flagpole

            # Step up gently onto col 93 (row 28) before the narrow corridor
            if 1450.0 <= abs_px <= 1485.0:
                return "hop"

            # Pipe anticipation: running leap launches 16-66px before pipe face (launches at 335px for 400px pipe)
            pipe_dist = self.planner.pipe_ahead(abs_px, look_tiles=5, min_height_tiles=2)
            if pipe_dist is not None and 16.0 <= pipe_dist <= 66.0:
                return "leap"

            cur, prof = self.planner.surface_ahead(abs_px, look_tiles=5)
            if cur is not None:
                for _c, row in prof:
                    if row is None:
                        continue
                    rise = cur - row
                    dist_px = _c * TILE - abs_px
                    if rise >= 2 and 16.0 <= dist_px <= 55.0:
                        # Ignore floating question/brick blocks that have clear floor underneath (Cols 154-160, X=2430-2630px)
                        if 2430.0 <= _c * TILE <= 2630.0:
                            continue
                        return "leap"  # tall wall/pipe -> running leap
                    elif rise == 1 and 0.0 <= dist_px <= 24.0 and abs_px >= 3700.0:
                        return "hop"   # End staircase: climb 1-tile steps via walking hop
        return None

    def _surface_threat_profile(self, obs):
        """Same decisions, every threshold derived from the ROM course data."""
        p = self.profile
        if self.pit_lookahead > 0 and any(obs[17 + c] < 0.5 for c in range(self.pit_lookahead)):
            return "leap"

        abs_px = self._abs_px(obs)

        # End staircase: take off from the last columns of the ramp, then sprint
        # the flat ground up to the flag. Legacy used 3870/3920/4020.
        if p.ramp_leap_from is not None and abs_px >= p.ramp_leap_from:
            if abs_px <= p.ramp_leap_to:
                return "leap"
            elif abs_px < p.run_to_px:
                return "run"

        # A 2-tile step that keeps rising is a lip to climb onto, not a wall to
        # leap. Legacy used the 1450..1485 band.
        if p.ledge_ahead(abs_px, look_tiles=3) is not None:
            return "hop"

        if self.planner is not None:
            cur, prof = self.planner.surface_ahead(abs_px, look_tiles=5)
            if cur is not None:
                for _c, row in prof:
                    if row is None:
                        continue
                    # Isolated single-column rises are stacked ?-blocks Mario runs
                    # under; only persistent rises are pipes/stairs. Legacy skipped
                    # them with the 2430..2630 band.
                    if _c not in p.persistent_rise_cols:
                        continue
                    rise = cur - row
                    dist_px = _c * TILE - abs_px
                    if rise >= 2 and 16.0 <= dist_px <= 55.0:
                        return "leap"
                    elif rise == 1 and 0.0 <= dist_px <= 24.0:
                        return "hop"   # 1-tile steps, e.g. the end staircase
        return None

    def _in_corridor(self, abs_px):
        """Legacy: a hardcoded column window for the 1-1 elevated corridor.
        Profile: the approach to any derived 2-tile ledge."""
        if self.reflex_mode == "profile" and self.profile is not None:
            return self.profile.ledge_ahead(abs_px, look_tiles=6) is not None
        return self.planner.is_narrow_corridor(abs_px)

    def _compound_threat(self, obs):
        """Perception for complex compound hazards (enemies in tight corridors or near walls)."""
        if self.planner is None:
            return None
        abs_px = self._abs_px(obs)

        # Narrow elevated corridor: legacy cols 92-97, profile the derived ledge
        if self._in_corridor(abs_px):
            for k in range(3):
                dx = obs[8 + 3 * k] * 256.0
                dy = obs[9 + 3 * k] * 256.0
                et = obs[10 + 3 * k] * 256.0
                if is_hostile_enemy(et) and abs(dy) < self.enemy_dy:
                    if 0.0 <= dx <= 90.0:
                        # On elevated platform (row 26): running leap clears the foot of ridge and Goomba
                        return "leap"

        # Enemies approaching near walls/pipes
        wall_d = self.planner.wall_ahead(abs_px, look_tiles=6, min_wall_tiles=2)
        if wall_d > 0:
            wall_px = wall_d * 16.0
            for k in range(3):
                dx = obs[8 + 3 * k] * 256.0
                dy = obs[9 + 3 * k] * 256.0
                et = obs[10 + 3 * k] * 256.0
                if is_hostile_enemy(et) and abs(dy) < self.enemy_dy:
                    if 0.0 <= dx < wall_px + 24.0:
                        if 38.0 <= dx <= 75.0:
                            return "leap"
                        elif dx > 75.0:
                            return "decelerate"
                        elif 0.0 <= dx < 38.0:
                            return "hop"
        return None

    def _threat(self, obs):
        """Hostile enemy threat detection based on empirical ballistic clearance."""
        abs_px = self._abs_px(obs) if self.planner is not None else 0.0
        # The mid-air lateral contact class was chased with five rule
        # candidates, all measured and all reverted or neutral; the autopsy
        # that settled it is D01_1's pocket: the koopa there PATROLS IN
        # PLACE (dx constant at 12 px for a dozen steps) while Mario is
        # pinned between sprite solids on an expiring floor -- approach
        # detection has nothing to see, and the death is the pocket class,
        # not a leap timing problem. What the class needs is terrain
        # semantics (sprite solids, expiring floors), which bgdat+rules
        # cannot carry.
        for k in range(3):
            dx = obs[8 + 3 * k] * 256.0
            dy = obs[9 + 3 * k] * 256.0
            et = obs[10 + 3 * k] * 256.0
            if is_hostile_enemy(et) and abs(dy) < self.enemy_dy:
                # Approaching a pit takeoff: suppress the early leap so Mario
                # commits to the long hop instead of stalling short of the gap.
                # Legacy used the 3480..3520 band (pit 9 of 1-1); profile asks the
                # geometry whether a takeoff edge is ahead.
                if self.reflex_mode == "profile" and self.profile is not None:
                    near_takeoff = self.profile.near_takeoff(abs_px)
                else:
                    near_takeoff = 3480.0 <= abs_px <= 3520.0
                hi = 55.0 if near_takeoff else self.enemy_hi
                if 0.0 <= dx <= hi:
                    return "leap"   # Running leap: clears apex > 50px high or stomps
        return None

    def _threat_landing_unsafe(self, obs):
        """True when the cone a threat-leap would land in overlaps a pit.

        The cone is reach-1 .. reach+4 columns ahead: far wider than the
        nominal 109.73 px reach on purpose, because measured leaps keep
        landing past it -- off an E01_1 slope the touchdown ran ~48 px long,
        and off A01_1's slope the apex hit 83 px (vs the 71 px calibration)
        and the touchdown ~55 px past reach. A cone that only covers the
        nominal arc is a cone the game has already been measured to miss.

        Reads the surface through the planner's own surface_ahead so both
        backends (TilemapAStar, RustPlanner) answer identically.
        """
        if self.planner is None:
            return False
        abs_px = self._abs_px(obs)
        col = int(abs_px // 16)
        reach = dynamic_jump_reach_tiles()
        lo, hi = col + reach - 1, col + reach + 4
        try:
            _, prof = self.planner.surface_ahead(abs_px, look_tiles=hi - col + 1)
        except Exception:
            return False
        return any(r is None and lo <= c <= hi for c, r in prof)

    def reflex_act(self, obs):
        r = self._reflex_act_impl(obs)
        import os as _os
        _log = _os.environ.get("REFLEX_LOG")
        if _log:
            with open(_log, "a") as f:
                f.write(f"obs0={obs[0]:.6f} obs1={obs[1]:.6f} vy={obs[3]:.4f} "
                        f"g={obs[4]:.0f} a={r} stuck={self._stuck} "
                        f"q={len(self._queue)} "
                        f"en={obs[8]:.3f},{obs[9]:.3f},{obs[10]:.0f};"
                        f"{obs[11]:.3f},{obs[12]:.3f},{obs[13]:.0f} "
                        f"pit={[int(x * 100) for x in obs[17:23]]}\n")
        return r

    def _reflex_act_impl(self, obs):
        on_ground = obs[4] > 0.5
        vy = obs[3]

        # Generic blocked detection (driving right but not advancing)
        if self._last_x is not None and on_ground:
            self._stuck = self._stuck + 1 if obs[0] - self._last_x < 0.0004 else 0
        else:
            self._stuck = 0
        self._last_x = obs[0]

        if self._queue:
            return self._queue.pop(0)

        # Airborne momentum preservation:
        # During ascent (vy > 0): hold JUMP to maintain low gravity (0.096 px/f^2) and achieve full 71px apex.
        # During descent (vy <= 0): hold RUN to preserve dash speed while releasing KEY_A.
        # If touching down with an enemy immediately ahead (dx <= 40px),
        # return JUMP to execute an instant touchdown chain-leap!
        if not on_ground:
            if vy <= 0.0:
                for k in range(3):
                    edx = obs[8 + 3 * k] * 256.0
                    edy = obs[9 + 3 * k] * 256.0
                    et = obs[10 + 3 * k] * 256.0
                    if is_hostile_enemy(et) and 0.0 <= edx <= 40.0 and abs(edy) < self.enemy_dy:
                        self.clearing = True
                        return self.JUMP
                return self.RUN
            if self.clearing:
                return self.JUMP
            return self.RUN

        # THE LEDGE GUARD (4x, candidates eleven and twelve). Measured on
        # 2-1: the Pokey below the ledge edge is stationary, unstompable
        # from above (the double-fall test, 4w), and its damage column is
        # dx -16..+8. The eleventh candidate fired the leap AT the freeze
        # (stuck >= 1) and was measured ignored: the spot at 613.7 is a
        # game hold that suppresses x AND jumps (three JUMP presses -> a
        # 3 px bob, vy pinned at 0.0000), so any action-level answer is
        # too late once grabbed. The 4r control walk crossed the same
        # edge at speed without being held -- the escape is to cross it
        # AIRBORNE, committing the leap while the below-ledge hostile is
        # still far (dx <= 75: the leap's 109 px of travel then carries
        # over the lip and touches down ~50+ px past the guard, inside
        # the measured miss column). Gated to wall-probed courses -- the
        # course OBJECT is always set, so the gate must be the probe DATA
        # itself (empty on 1-1/1-2/5-1/8-1: parity structural). The first
        # version gated on the object and broke parity at once, dying at
        # 2,912 px inside the wider dx<=75 window; the published clear also
        # logs zero below-window-while-frozen moments (grep of the
        # 245-step log: zero). The below-ledge window is dx 0..75, dy
        # -55..-120 -- below the +-55 threat window, hence invisible to
        # every other rule. And the TYPE must have a measured contact
        # map with NO stomp window (the 4w oracle: 2-1's Pokey is
        # unstompable from above; goombas and Koopas are not): the first
        # version fired on 6-1's fish too and knocked that frontier from
        # 1,929 back to 1,481 -- stompable and unmapped enemies keep
        # their ordinary handling.
        if self.wall_probe_course is not None \
                and self.wall_probe_course.wall_probe_cells and on_ground:
            for k in range(3):
                edx = obs[8 + 3 * k] * 256.0
                edy = obs[9 + 3 * k] * 256.0
                et = obs[10 + 3 * k] * 256.0
                if is_hostile_enemy(et) \
                        and int(round(et)) in self.wall_probe_course.contact_unstompable \
                        and 0.0 <= edx <= 75.0 and -120.0 <= edy <= -55.0:
                    self.clearing = True
                    return self.JUMP

        self.clearing = False

        blocked = self._stuck >= 3
        if blocked:
            self._stuck = 0
            self._blocked += 1
            here = self._abs_px(obs)
            self._blocked_positions.append(here)
            near = sum(1 for p in self._blocked_positions[-6:] if abs(p - here) < 8.0)
            # THE WALL-MAP GATE (4v). The 4r autopsy retracted the
            # "sprite-solid wall" reading at both remaining pockets: every
            # cell there measured FREE (B01_1 cols 36-40: 17-81 px of x
            # advance; D01_1 cols 29-35: all 63 cells), and a plain-RIGHT
            # control walk strolls through the 2-1 pocket, off the ledge
            # edge, down to the base floor and east to 665. What kills the
            # run at those freezes is the blocked response itself -- the
            # 1st/2nd JUMP leaps off the ledge edge into the 232 crawler's
            # patrol, the 3rd+ retreat walks back into it. So when the
            # oracle has measured Mario's own cell FREE, there is no wall
            # to answer: keep driving. The override is capped at three
            # events (the _blocked counter is left alone, so the fourth
            # event falls through to the normal chain and its escalation)
            # -- a freeze the map does not explain still gets every
            # existing response. Parity is safe twice over: the published
            # 1-1 clear logs ZERO blocked events, and 1-1 has no wall-probe
            # data at all, so the gate is structurally dead there. On 6-1
            # the stall cell IS measured blocked (col 120, rows 22-26), so
            # the frontier's behavior is untouched.
            if self.wall_probe_course is not None and self._blocked <= 3:
                try:
                    if self.reflex_mode == "profile" and self.profile is not None:
                        cpx = self.profile.abs_px(float(obs[0]))
                        crow = int(self.profile.mario_row(obs[1]))
                    elif self.ram is not None:
                        st = self.ram.poll()
                        if st.get("mario_B") is None:
                            raise ValueError("no mario in entity list")
                        cpx = st["mario_B"] / 4096.0
                        crow = int(-(st["mario_A"] or 0) / 4096.0 / 16.0)
                    else:
                        raise ValueError("no coordinate source")
                    if self.wall_probe_course.wall_probe_free(int(cpx // 16), crow):
                        return self.RUN
                except Exception:
                    pass
            if near >= 3:
                # The third blocked event at the same spot: neither the
                # in-place jump nor the retreat queue is solving this.
                # Measured on F01_1 (World 6-1, the water mid-section):
                # the retreat loop is ENDLESS there -- walk left, run right,
                # hop, sink, block again -- because the level expects
                # SWIMMING, and A in water is a stroke -- a fixed ~67 px
                # impulse per PRESS, not per hold (measured: holding A
                # through the ascent adds nothing; the stroke peaks and the
                # sink starts while A is still down). The 4p burst pressed
                # every 16 and every 24 frames and the LATER presses were
                # swallowed -- the first stroke lifted ~3 rows and the rest
                # did nothing, which smells like a stroke cooldown. The
                # 6-1 wall map (out/wall_probe/F01_1_wall122.json) says
                # the passage is the rows 20-21 corridor between the
                # col-120 wall (rows 22-26) and the col-123 wall (rows
                # 14-19): two more rows of climb and the burst passes. The
                # rhythm below presses every 32 frames -- slower than the
                # measured swallow point, so every stroke should take.
                # Parity-safe by measurement: the published 1-1 clear logs
                # ZERO blocked events, so this escalation cannot fire on it.
                self._blocked = 0
                self._blocked_positions = []
                self._queue.clear()
                self._queue = ([self.JUMP, self.RUN, self.RUN, self.RUN] * 16)
                self.clearing = True
                return self._queue.pop(0)
            if self.reflex_mode == "profile" and self.profile is not None:
                # The probe-derived sprite-solids diagnosis. The bgdat parser
                # cannot see sprite terrain, and the five reverted rules of
                # 4m all failed on exactly that; but the drop probe DID
                # measure those surfaces (B01_1's block at 20.35, the
                # crumbling bridges, the ice ledges). When x freezes, ask the
                # measured map what is standing in front of Mario:
                #   a surface 0.5..1.5 rows above him -> hop onto it
                #   a surface 1.5..4 rows above him   -> leap at it
                #   nothing measured above, ground ahead -> ride it (a
                #     moving platform: run, do not jump)
                # Only an unprobed or pit answer falls back to the retreat.
                #
                # Coordinates: obs is spawn-relative in 512 px units (the
                # origin fix made that true), so mario_B is
                # spawn_mb + relative, course px is mario_B minus the
                # half-sprite, and the row comes from the profile.
                mb_px = self.profile.spawn_px + obs[0] * 512.0
                course_px = mb_px - 8.0
                my_row = self.profile.mario_row(obs[1])
                rise = self.profile.probe_rise_ahead(course_px, my_row)
                if rise is not None:
                    self._blocked = 0
                    if rise[1] <= 1.5:
                        return self.HOP       # one-row sprite step: land on it
                    return self.JUMP          # taller sprite wall: leap at it
                above = self.profile.probe_rise_ahead(
                    course_px, my_row, look_tiles=4, lo=-99.0, hi=0.0)
                knows = any(self.profile.course.probe_surface(
                    int(course_px // 16) + d) is not None for d in range(1, 5))
                if above is None and knows:
                    self._blocked = 0
                    return self.RUN  # measured ground, nothing above: ride the platform
                self._blocked = 0
                self._queue = [self.LEFT] * 4 + [self.RUN] * 8 + [self.JUMP]
            elif self._blocked >= 3:
                self._blocked = 0
                self._queue = [self.LEFT] * 4 + [self.RUN] * 8 + [self.JUMP]
            else:
                return self.JUMP

        surf = self._surface_threat(obs)
        compound = self._compound_threat(obs)
        enemy = self._threat(obs)

        if compound == "decelerate":
            return 1  # Action 1: Walk Right (time enemy cycle)
        elif compound == "leap" or enemy == "leap" or surf == "leap":
            if surf != "leap" and self._threat_landing_unsafe(obs):
                # An enemy-triggered leap is a commitment: ~110 px of travel
                # with no way back, and the flat-ground calibration
                # understates it -- leaping off a slope measured an 83 px apex
                # (vs 71) and a ~165 px touchdown, 55 px past the nominal
                # reach, which is how A01_1's true-geometry run wedged into
                # the pit at cols 76-77 (4k autopsy class). When the landing
                # cone overlaps a pit, hop in place instead: the threat
                # passes underneath and the pit's own edge-leap (never
                # suppressed) clears both.
                return self.IN_PLACE_HOP
            self.clearing = True
            return self.JUMP
        elif compound == "hop" or surf == "hop":
            return self.HOP  # Action 2: Right + Jump (climbs 1-tile step)
        elif surf == "run":
            return self.RUN

        self._blocked = 0
        return None


class ReflexiveMPCAgent(BaseReflexController):
    """Reflex layer + model-based planner (reference S10.33 'Reflexive MPC')."""
    name = "mpc+reflex"

    def __init__(self, model, device, horizon, n_cand, n_iter, seed, cfg=None,
                 enemy_lo=0.0, enemy_hi=75.0, enemy_dy=55.0, pit_lookahead=2, enabled=True,
                 pit_enemy_suppress=0.0, planner=None):
        super().__init__(planner=planner, enemy_lo=enemy_lo, enemy_hi=enemy_hi,
                         enemy_dy=enemy_dy, pit_lookahead=pit_lookahead, enabled=enabled)
        self.mpc = MPCAgent(model, device, horizon, n_cand, n_iter, seed, cfg)

    def act(self, obs):
        if self.enabled:
            ref_a = self.reflex_act(obs)
            if ref_a is not None:
                return ref_a
        return self.mpc.act(obs)

    def reset(self):
        super().reset()
        self.mpc.reset()


class ReactiveAgent(BaseReflexController):
    """Ablation: hazard reflex + continuous rightward drive, no MPC."""
    name = "reactive"

    def __init__(self, *a, **k):
        planner = k.pop("planner", None)
        enemy_lo = k.pop("enemy_lo", 0.0)
        enemy_hi = k.pop("enemy_hi", 75.0)
        enemy_dy = k.pop("enemy_dy", 55.0)
        pit_lookahead = k.pop("pit_lookahead", 2)
        enabled = k.pop("enabled", True)
        ram = k.pop("ram", None)
        super().__init__(planner=planner, enemy_lo=enemy_lo,
                         enemy_hi=enemy_hi, enemy_dy=enemy_dy,
                         pit_lookahead=pit_lookahead, enabled=enabled, ram=ram)

    def act(self, obs):
        if self.enabled:
            ref_a = self.reflex_act(obs)
            if ref_a is not None:
                return ref_a
        return self.RUN

    def reset(self):
        super().reset()


class PPOAgent:
    name = "ppo"

    def __init__(self, policy_path, device):
        from stable_baselines3 import PPO
        self.model = PPO.load(policy_path, device=device)

    def act(self, obs):
        a, _ = self.model.predict(obs[None], deterministic=True)
        return int(a[0])

    def reset(self):
        pass


class ReflexivePPOAgent(BaseReflexController):
    """Hybrid PPO + safety reflex with TilemapAStar terrain perception."""
    name = "ppo+reflex"

    def __init__(self, policy_path, device, planner=None,
                 enemy_lo=0.0, enemy_hi=75.0, enemy_dy=55.0, pit_lookahead=2, enabled=True):
        super().__init__(planner=planner, enemy_lo=enemy_lo, enemy_hi=enemy_hi,
                         enemy_dy=enemy_dy, pit_lookahead=pit_lookahead, enabled=enabled)
        from stable_baselines3 import PPO
        self.model = PPO.load(policy_path, device=device)

    def act(self, obs):
        if self.enabled:
            ref_a = self.reflex_act(obs)
            if ref_a is not None:
                return ref_a
        a, _ = self.model.predict(obs[None], deterministic=True)
        return int(a[0])

    def reset(self):
        super().reset()


def grab_top_screen(emu):
    buf = np.array(emu.display_buffer_as_rgbx(), dtype=np.uint8).reshape(384, 256, 4)
    # The buffer is already BGR-native (sky reads [244,160,54] by channel index and
    # the 4th byte is always 0), so reversing it swaps red and blue -- that is what
    # rendered Mario blue and the sky orange in the recorded clips.
    return buf[:192, :, :3].copy()  # top screen, BGR


def render_frame(emu, label, speed, writer=None):
    try:
        import cv2
        top = grab_top_screen(emu)
        top = cv2.resize(top, (256 * 2, 192 * 2), interpolation=cv2.INTER_NEAREST)
        cv2.putText(top, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        if writer is not None:
            writer.write(top)
        cv2.imshow("PINN NSMB-DS Agent (real-time)", top)
        cv2.waitKey(max(1, int(1000 * FRAME_DT / speed)))
    except Exception as e:  # pragma: no cover
        print("render disabled:", e)


def run_episode(env, agent, goal, render, speed, max_steps, verbose, writer=None,
                 spawn_px=SPAWN_ABS_PX):
    obs, _ = env.reset()
    agent.reset()
    total_r, steps = 0.0, 0
    max_rel_px = 0.0
    # Two coordinate systems coexist in this repo: the controller's
    # SPAWN_ABS_PX + obs[0]*512 (what the agent reasons with) and the env's
    # RAM-derived info["mario_px"] (what the stage actually reports). They
    # disagree by ~10 steps' worth of distance, so both are recorded instead of
    # one being quietly relabelled as the other.
    last_mario_px = 0.0
    max_mario_px = 0.0
    latencies = []
    cause, finished = "step_limit", False
    death_diag = ""
    t_wall = time.time()
    with _silence_c_stdout(enabled=not render):
        for steps in range(1, max_steps + 1):
            t0 = time.time()
            action = agent.act(obs)
            latencies.append((time.time() - t0) * 1000.0)
            obs, r, term, trunc, info = env.step(action)
            total_r += r
            max_rel_px = max(max_rel_px, obs[0] * 512.0)
            last_mario_px = float(info.get("mario_px", last_mario_px))
            max_mario_px = max(max_mario_px, last_mario_px)
            abs_px = spawn_px + obs[0] * 512.0
            if render or writer is not None:
                render_frame(env.emu, f"step {steps} | x={abs_px:.0f}px | goal {goal}px",
                             speed, writer)
            if abs_px >= (goal - 16.0) or info.get("level_cleared"):
                finished, cause = True, "goal"
                break
            if term:
                cause = "death" if obs[6] > 0.02 else "timeout"
                if cause == "death":
                    ex = [round(float(obs[8 + 3 * k]) * 256) for k in range(3)]
                    et = [round(float(obs[10 + 3 * k]) * 256) for k in range(3)]
                    death_diag = (f"abs {spawn_px + obs[0]*512:.0f}px vy={obs[3]:.2f} "
                                  f"ground={obs[4]:.0f} pit={[int(x) for x in obs[17:23]]} "
                                  f"enemy_dx_px={ex} enemy_type={et}")
                break
            if trunc:
                cause = "trunc"
                break
    wall = time.time() - t_wall
    return {
        "steps": steps, "total_reward": float(total_r),
        "max_progress_px": float(max_rel_px), "final_abs_px": float(spawn_px + max_rel_px),
        "final_mario_px": last_mario_px, "max_mario_px": max_mario_px,
        "reached_goal": bool(finished), "termination": cause,
        "mean_control_latency_ms": float(np.mean(latencies)),
        "achieved_env_fps": float(steps / max(wall, 1e-9)),
        "death_diag": death_diag,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", choices=["mpc", "ppo", "ppo+reflex", "reactive"], default="mpc")
    ap.add_argument("--model", default="models/pinn_nsmb.pt")
    ap.add_argument("--policy", default="models/pinn_policy.zip")
    ap.add_argument("--rom", default=ROM); ap.add_argument("--state", default=STATE)
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--horizon", type=int, default=24, help="CEM-MPC rollout horizon (extended to 24 for corridor/landing anticipation)")
    ap.add_argument("--n-cand", type=int, default=160)
    ap.add_argument("--n-iter", type=int, default=3)
    ap.add_argument("--speed", type=float, default=2.0, help="wall-clock realtime multiplier when rendering")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--record", type=str, default="", help="write gameplay .avi to this path")
    ap.add_argument("--record-episode", type=int, default=1, help="which episode (1-based) to record")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--goal-px", type=int, default=0, help="override goal abs px (0=auto from ROM)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--w-progress", type=float, default=2.0)
    ap.add_argument("--w-enemy", type=float, default=0.0)
    ap.add_argument("--w-pit", type=float, default=0.0)
    ap.add_argument("--no-reflex", action="store_true", help="disable hazard reflex")
    ap.add_argument("--enemy-lo", type=float, default=0.0, help="minimum hazard distance (0.0 detects all frontal proximity)")
    ap.add_argument("--enemy-hi", type=float, default=75.0, help="maximum frontal hazard distance (75px allows 20-26 frame ballistic clearance)")
    ap.add_argument("--enemy-dy", type=float, default=55.0)
    ap.add_argument("--pit-lookahead", type=int, default=2, help="lookahead tiles for pit takeoff (2 tiles = 32px safe runway)")
    ap.add_argument("--w-uncertainty", type=float, default=0.6,
                    help="pessimism on unpredictable enemy motion (probabilistic head)")
    ap.add_argument("--pit-enemy-suppress", type=float, default=0.0,
                    help="skip pit-jump if an enemy is within this many px (0=always jump)")
    ap.add_argument("--planner", choices=["python", "rust"], default="python",
                    help="geometry backend for wall/pipe queries (rust = rust/nsmb_planner.exe)")
    ap.add_argument("--reflex", choices=["legacy", "profile"], default="legacy",
                    help="legacy = the World 1-1 pixel bands that produced the published "
                         "clear; profile = thresholds derived from the ROM course data")
    ap.add_argument("--course", default="course/A01_1.bin",
                    help="course parsed by src/course.py: the planner, the geometry "
                         "features, the goal and --reflex profile all read it")
    ap.add_argument("--out", default="evals/pinn_realtime_eval.json",
                    help="metrics json; point elsewhere to keep published evals intact")
    args = ap.parse_args()

    render = args.render and not args.headless
    device = args.device
    goal = args.goal_px or goal_px(args.course)
    # An explicit --goal-px also becomes the profile's goal: a course with no
    # flagpole sprite would otherwise get thresholds derived around the 4032 px
    # fallback, which is not its exit.
    override = int(args.goal_px) if args.goal_px else None
    # The reporting anchor is the course's own entrance, not 1-1's, so both
    # reflex arms are scored on the same ruler even though the legacy one still
    # reasons in 1-1 pixels. For course/A01_1 this is 88 px, i.e. unchanged.
    course_obj = Course(name=args.course, rom_path=args.rom)
    spawn_anchor = course_obj.entrances[0]["x_px"] + TILE / 2.0
    # The wall oracle's handle for the blocked-response gate (4v) -- BOTH
    # arms: on courses without probe data the gate is structurally silent.
    BaseReflexController.wall_probe_course = course_obj
    if args.planner == "rust":
        from rust_planner import RustPlanner
        planner = RustPlanner()          # rust/nsmb_planner.exe over a line protocol
    else:
        # ROM geometry -> wall/pipe perception, on the same course the agent runs in
        planner = TilemapAStar(rom_path=args.rom, name=args.course,
                               exit_known=override is None)
    if args.reflex == "profile":
        from level_profile import LevelProfile
        BaseReflexController.reflex_mode = "profile"
        BaseReflexController.profile = LevelProfile(args.rom, args.course, exit_px=override)
        print("[pinn-eval] reflex profile: " + BaseReflexController.profile.summary())
    print(f"[pinn-eval] controller={args.controller} device={device} goal_abs_px={goal} "
          f"planner={args.planner} reflex={args.reflex}")

    if args.controller == "mpc":
        ck = torch.load(args.model, map_location=device, weights_only=False)
        model = MarioPINNWorldModel(hid=ck["hid"]).to(device); model.load_state_dict(ck["state_dict"]); model.eval()
        cfg = ShapingConfig(progress=args.w_progress, enemy=args.w_enemy, pit=args.w_pit,
                            uncertainty=args.w_uncertainty)
        make_agent = lambda: ReflexiveMPCAgent(model, device, args.horizon, args.n_cand,
                                               args.n_iter, args.seed, cfg,
                                               enemy_lo=args.enemy_lo, enemy_hi=args.enemy_hi,
                                               enemy_dy=args.enemy_dy, pit_lookahead=args.pit_lookahead,
                                               pit_enemy_suppress=args.pit_enemy_suppress,
                                               planner=planner, enabled=not args.no_reflex)
    elif args.controller == "reactive":
        make_agent = lambda: ReactiveAgent(
            None, device, args.horizon, args.n_cand, args.n_iter, args.seed,
            enemy_lo=args.enemy_lo, enemy_hi=args.enemy_hi, enemy_dy=args.enemy_dy,
            pit_lookahead=args.pit_lookahead, pit_enemy_suppress=args.pit_enemy_suppress,
            planner=planner, enabled=not args.no_reflex, ram=env.ram)
    elif args.controller == "ppo+reflex":
        make_agent = lambda: ReflexivePPOAgent(
            args.policy, device, planner=planner,
            enemy_lo=args.enemy_lo, enemy_hi=args.enemy_hi, enemy_dy=args.enemy_dy)
    else:
        make_agent = lambda: PPOAgent(args.policy, device)

    env = MarioRamEnv(rom_path=args.rom, state_path=args.state, geo=True,
                      course_name=args.course, max_steps=args.max_steps)
    writer = None
    if args.record:
        import cv2
        os.makedirs(os.path.dirname(args.record) or ".", exist_ok=True)
        writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"MJPG"), 30, (512, 384))
    results = []
    for ep in range(args.episodes):
        agent = make_agent()
        w = writer if (writer is not None and ep + 1 == args.record_episode) else None
        res = run_episode(env, agent, goal, render, args.speed, args.max_steps,
                          verbose=True, writer=w, spawn_px=spawn_anchor)
        res["episode"] = ep + 1
        results.append(res)
        flag = "LEVEL CLEARED" if res["reached_goal"] else "did not finish"
        print(f"[pinn-eval] ep {ep+1}: progress {res['final_abs_px']:.0f}px / {goal}px "
              f"({100*res['final_abs_px']/goal:.0f}%) | steps {res['steps']} | "
              f"reward {res['total_reward']:.1f} | {res['termination']} | "
              f"{res['achieved_env_fps']:.1f} fps | {res['mean_control_latency_ms']:.1f} ms/act -> {flag}")
        if res.get("death_diag"):
            print(f"           death: {res['death_diag']}")
    env.close()
    if writer is not None:
        writer.release()
        print(f"[pinn-eval] recorded -> {args.record}")
    if render:
        import cv2; cv2.destroyAllWindows()

    agg = {
        "controller": args.controller, "episodes": args.episodes, "goal_abs_px": goal,
        "goal_source": "operator --goal-px" if override else "course flag sprite",
        "course": args.course, "state": args.state, "reflex": args.reflex,
        "spawn_anchor_px": spawn_anchor,
        "planner": args.planner,
        "clear_rate": float(np.mean([r["reached_goal"] for r in results])),
        "mean_progress_px": float(np.mean([r["final_abs_px"] for r in results])),
        "max_progress_px": float(np.max([r["final_abs_px"] for r in results])),
        "mean_reward": float(np.mean([r["total_reward"] for r in results])),
        "mean_fps": float(np.mean([r["achieved_env_fps"] for r in results])),
        "mean_latency_ms": float(np.mean([r["mean_control_latency_ms"] for r in results])),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out = {"aggregate": agg, "episodes": results}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("[pinn-eval] aggregate:", json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
