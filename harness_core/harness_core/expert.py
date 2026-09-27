"""Scripted, force-guided expert for wire routing and connector insertion.

It acts through exactly the same 5-D action space as a learned policy (see
``actions.py``), so its rollouts are directly usable as demonstrations. All
force regulation is done in an outer loop on the measured wrench: the inner
compliance controller turns position leads into bounded contact forces.

Per fork i:
    pick   choose the arc length whose wire, held taut from the last fixation,
           reaches just past fork i; guarded touch-down of the pads on the board,
           back off, close. The finger orientation is chosen so that the held wire
           can later be turned to run away from the fixation (no hairpin at the
           fingers) without driving wrist 3 into its limits
    carry  lift and sweep around the fixation point with the wire nearly taut,
           so it passes over the fork instead of dragging across posts
    lower  descend beyond the fork, moving away when the wire is slack and giving
           way when the tension rises (tension = horizontal pull on the gripper)
    seat   ramp the tension, centre the wire over the slot from perception, wiggle
           until it snaps past the barbed jaws; release and verify
Connector:
    pull it over by its wire if it stands on its end, move it if it lies next to a
    fixture, grasp it, approach the holder from behind so the wire leaves through
    the back slot, guarded descent, spiral search at constant force, press with a
    small wiggle, release and verify (averaged perception).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Generator, Optional

import numpy as np

from .actions import ActionSpec, gripper_action_from_opening
from .config import CellConfig
from .geometry import polyline_arclength, polyline_point_at, wrap_angle
from .perception import arclength_near, cable_crossing_in_fork

Obs = Dict[str, np.ndarray]


@dataclass
class ExpertParams:
    hover: float = 0.06            # TCP height above the board before picking
    transit_clearance: float = 0.08   # gripper height above the prong tops while carrying
    beyond: float = 0.05           # how far beyond the fork the wire is held while seating
    pick_margin: float = 0.006     # slack on top of the lifted, taut wire length
    touch_force: float = 1.5       # N, touch-down detection
    grasp_backoff: float = 0.0015  # m above the touch-down height
    tension_low: float = 2.5
    tension_high: float = 10.0
    tension_descend: float = 6.0
    press_height: float = 0.014    # TCP height above the board while seating
    min_beyond: float = 0.03
    max_beyond: float = 0.16
    seat_time: float = 5.0
    speed_fast: float = 0.16
    speed_slow: float = 0.03
    yaw_speed: float = 1.5
    spiral_growth: float = 0.0007  # m per revolution
    spiral_period: float = 1.0     # s per revolution
    spiral_max_radius: float = 0.005
    insert_force: float = 3.0
    approach_offset: float = 0.05  # m behind the pocket where the connector is lowered
    press_force: float = 6.0
    max_route_attempts: int = 3


class HarnessExpert:
    def __init__(self, cfg: CellConfig, spec: Optional[ActionSpec] = None,
                 params: Optional[ExpertParams] = None, dt: Optional[float] = None,
                 verbose: bool = False, skip_connector: bool = False):
        self.cfg = cfg
        self.spec = spec or ActionSpec()
        self.p = params or ExpertParams()
        self.dt = dt or cfg.sim.policy_dt
        self.verbose = verbose
        self.skip_connector = skip_connector
        self.reset()

    # ------------------------------------------------------------ public API
    def reset(self) -> None:
        self.phase = "start"
        self.current_fork = 0
        self._relocations = 0
        self._tips = 0
        self.done = False
        self.failed = False
        self.fail_reason = ""
        self.log = []
        self._grip = -1.0
        self._gen = self._program()
        next(self._gen)

    def step(self, obs: Obs) -> np.ndarray:
        if self.done:
            return self._hold()
        try:
            a = self._gen.send(obs)
        except StopIteration:
            self.done = True
            a = self._hold()
        return np.asarray(a, dtype=float)

    # --------------------------------------------------------------- helpers
    def _say(self, msg: str) -> None:
        self.log.append((self._t, msg))
        if self.verbose:
            print(f"[expert t={self._t:6.2f}] {msg}")

    def _hold(self) -> np.ndarray:
        return np.array([0.0, 0.0, 0.0, 0.0, self._grip])

    def _action_toward(self, obs: Obs, goal_pos, goal_yaw=None, speed: float = 0.1,
                       yaw_speed: Optional[float] = None) -> np.ndarray:
        s = self.spec
        a = np.zeros(5)
        d = np.asarray(goal_pos, dtype=float) - obs["target_pos"]
        step = speed * self.dt
        n = np.linalg.norm(d)
        if n > step:
            d = d * step / n
        a[:3] = d / s.max_dpos
        if goal_yaw is not None:
            ys = (yaw_speed or self.p.yaw_speed) * self.dt
            dy = float(np.clip(wrap_angle(goal_yaw - float(obs["target_yaw"][0])), -ys, ys))
            a[3] = dy / s.max_dyaw
        a[4] = self._grip
        return np.clip(a, -1.0, 1.0)

    def _goto(self, goal_pos, goal_yaw=None, speed=None, tol=0.002, yaw_tol=0.03,
              timeout=8.0, settle=0.0, max_force: Optional[float] = None) -> Generator:
        speed = speed or self.p.speed_fast
        obs = self._obs
        t_end = self._t + timeout
        t_ok = None
        while True:
            if max_force is not None and np.linalg.norm(obs["wrench"][:3]) > max_force:
                self._say(f"force limit {np.linalg.norm(obs['wrench'][:3]):.1f} N during move")
                return False
            err = np.linalg.norm(np.asarray(goal_pos) - obs["tcp_pos"])
            yerr = 0.0 if goal_yaw is None else abs(wrap_angle(goal_yaw - float(obs["tcp_yaw"][0])))
            if err < tol and yerr < yaw_tol:
                t_ok = self._t if t_ok is None else t_ok
                if self._t - t_ok >= settle:
                    return True
            else:
                t_ok = None
            if self._t > t_end:
                return False
            obs = yield self._action_toward(obs, goal_pos, goal_yaw, speed)

    def _wait(self, duration: float) -> Generator:
        t_end = self._t + duration
        obs = self._obs
        while self._t < t_end:
            obs = yield self._action_toward(obs, obs["target_pos"], None, 0.0)
        return True

    def _set_grip(self, g: float, duration: float = 0.4) -> Generator:
        self._grip = float(g)
        yield from self._wait(duration)

    @property
    def _obs(self) -> Obs:
        return self.__dict__.get("_last_obs")

    @property
    def _t(self) -> float:
        o = self.__dict__.get("_last_obs")
        return float(o["time"][0]) if o is not None else 0.0

    def _board_z(self) -> float:
        return float(self._obs["board_z"][0])

    def _q6_after(self, yaw_target: float, yaw_from: Optional[float] = None,
                  q6_from: Optional[float] = None) -> float:
        """Predicted wrist-3 angle after rotating the (vertical) tool to yaw_target the
        short way. For a downward tool, yaw = q1 - q6 + const."""
        if yaw_from is None:
            yaw_from = float(self._obs["tcp_yaw"][0])
        if q6_from is None:
            q6_from = float(self._obs["q"][5])
        return q6_from - wrap_angle(yaw_target - yaw_from)

    def _closest_yaw(self, yaw: float, ref: float) -> float:
        """yaw or yaw + pi (parallel gripper symmetry): the one closer to ref unless that
        would drive wrist 3 towards its +-2pi limit."""
        cands = sorted([wrap_angle(yaw), wrap_angle(yaw + math.pi)],
                       key=lambda c: abs(wrap_angle(c - ref)))
        for c in cands:
            if abs(self._q6_after(c)) < 4.2:
                return c
        return min(cands, key=lambda c: abs(self._q6_after(c)))

    def _pick_yaw(self, wire_yaw: float, carry_yaw: float) -> float:
        """Gripper yaw for grasping a wire whose tangent (towards the free end) points
        along wire_yaw, when that tangent must point along carry_yaw once the wire is
        carried. The rotation between the two is fixed by the wire, so of the two
        (symmetric) finger orientations take the one that keeps wrist 3 clear of its
        limits through pick and carry, then the one that needs less rotation now."""
        rot = wrap_angle(carry_yaw - wire_yaw)
        yaw_now = float(self._obs["tcp_yaw"][0])
        best, best_key = wire_yaw, None
        for alpha in (0.0, math.pi):
            yp = wrap_angle(wire_yaw + alpha)
            q6p = self._q6_after(yp)
            key = (max(abs(q6p), abs(q6p - rot)) > 4.2, abs(wrap_angle(yp - yaw_now)))
            if best_key is None or key < best_key:
                best, best_key = yp, key
        return best

    def _held_wire_yaw_offset(self, s_grasp: float) -> float:
        """0 or pi: direction of the held wire's tangent (towards the free end) relative
        to the gripper x-axis, from the perceived cable."""
        _, t = polyline_point_at(self._obs["cable"], s_grasp)
        rel = wrap_angle(math.atan2(t[1], t[0]) - float(self._obs["tcp_yaw"][0]))
        return 0.0 if abs(rel) < 0.5 * math.pi else math.pi

    def _tension(self, u: np.ndarray) -> float:
        """Horizontal wire tension pulling the gripper back along -u."""
        f = self._obs["wrench"][:3]
        return float(-(f[0] * u[0] + f[1] * u[1]))

    def _carry_over(self, P: np.ndarray, ell: float, D_final: float, phi_final: float, yaw,
                    skip=()) -> Generator:
        """Carry a grasped wire whose other end is fixed at P (length ell between P and
        the gripper) to the point at horizontal distance D_final / azimuth phi_final
        from P, keeping the wire nearly taut and lifted so it cannot drag across and
        snag on fork posts. First lift, then extend radially, then swing around P."""
        bz = self._board_z()
        wire_drop = 0.0045
        ell_eff = ell - 0.012
        z_max = bz + self.cfg.fork.post_height + self.cfg.fork.prong_height + 0.14
        z_min = bz + self.cfg.fork.post_height + self.cfg.fork.prong_height + 0.03

        def height(D: float) -> float:
            h = P[2] + math.sqrt(max(ell_eff * ell_eff - D * D, 0.0)) + wire_drop
            return float(np.clip(h, z_min, z_max))

        g0 = self._obs["tcp_pos"].copy()
        rel = g0[:2] - P[:2]
        D0 = float(np.linalg.norm(rel))
        phi0 = math.atan2(rel[1], rel[0]) if D0 > 1e-3 else phi_final
        self.phase = "route_lift"
        if not (yield from self._goto(np.array([g0[0], g0[1], height(D0)]), None, speed=0.10,
                                      tol=0.008, timeout=5.0, max_force=18.0)):
            return False
        self.phase = "route_transit"
        pts = []
        for D in np.linspace(D0, D_final, max(2, int(abs(D_final - D0) / 0.02) + 1))[1:]:
            pts.append(np.array([P[0] + D * math.cos(phi0), P[1] + D * math.sin(phi0), height(D)]))
        dphi = wrap_angle(phi_final - phi0)
        n_rot = max(2, int(abs(dphi) * D_final / 0.02) + 1)
        for k in range(1, n_rot + 1):
            phi = phi0 + dphi * k / n_rot
            pts.append(np.array([P[0] + D_final * math.cos(phi), P[1] + D_final * math.sin(phi), height(D_final)]))
        for k, w in enumerate(pts):
            last = k == len(pts) - 1
            ok = yield from self._goto(w, yaw, speed=0.12, tol=0.004 if last else 0.012,
                                       yaw_tol=0.05 if last else 3.0, timeout=4.0, max_force=18.0)
            if not ok and not last:
                return False
        return True

    # ------------------------------------------------------------- program
    def _program(self) -> Generator:
        obs = yield None                     # primed by reset(); first send() delivers obs
        self._last_obs = obs
        # Wrap the generator so every received observation is recorded.
        gen = self._task()
        a = next(gen)
        while True:
            obs = yield a
            self._last_obs = obs
            try:
                a = gen.send(obs)
            except StopIteration:
                return

    def _task(self) -> Generator:
        n_forks = len(self._obs["forks"])
        self.phase = "init"
        self._grip = -1.0
        yield from self._wait(0.2)
        for i in range(n_forks):
            self.current_fork = i
            ok = False
            for attempt in range(self.p.max_route_attempts):
                self._say(f"fork {i}: attempt {attempt + 1}")
                ok = yield from self._route_fork(i, attempt)
                if ok:
                    break
                if self._obs["protective_stop"][0] > 0.5:
                    break
            if not ok:
                self.failed = True
                self.fail_reason = f"could not route fork {i}"
                self._say(self.fail_reason)
                yield from self._retreat()
                return
        self.current_fork = -1
        if self.skip_connector:
            yield from self._retreat()
            return
        ok = False
        for attempt in range(4):
            ok = yield from self._insert_connector()
            if ok or self._obs["protective_stop"][0] > 0.5:
                break
        if not ok:
            self.failed = True
            self.fail_reason = "connector insertion failed"
            self._say(self.fail_reason)
        yield from self._retreat()

    def _retreat(self) -> Generator:
        self.phase = "retreat"
        self._grip = -1.0
        obs = self._obs
        up = obs["tcp_pos"].copy()
        up[2] = max(up[2], self._board_z() + 0.12)
        yield from self._goto(up, None, tol=0.01, timeout=4.0)
        home = np.array([0.45, 0.0, 0.20])
        yield from self._goto(home, self._closest_yaw(0.0, float(self._obs["tcp_yaw"][0])), tol=0.01, timeout=6.0)
        self.phase = "done"

    # ------------------------------------------------------------ wire pick
    def _fixation(self, i: int):
        """Point (3D) and arc length where the wire is currently fixed before fork i."""
        obs = self._obs
        cable = obs["cable"]
        if i == 0:
            p = obs["anchor_pos"]
        else:
            fx, fy, fz, fyaw = obs["forks"][i - 1]
            p = np.array([fx, fy, fz + self.cfg.fork.post_height + self.cfg.wire.radius])
        return p, arclength_near(cable, p)

    def _choose_pick_arclength(self, s_ideal: float, s_min: float, s_max: float,
                               forward_first: bool = False) -> Optional[float]:
        """Arc length closest to s_ideal (within [s_min, s_max]) whose wire point is clear of
        the fixtures, so the fingers do not land on a fork, the holder or the clamp.
        ``forward_first`` searches further along the wire before going back, so a retry
        that asks for more wire actually gets more wire."""
        cable = self._obs["cable"]
        obstacles = [f[:2] for f in self._obs["forks"]]
        obstacles.append(self._obs["holder_pos"][:2])
        obstacles.append(self._obs["anchor_pos"][:2])
        if s_max < s_min:
            return None
        offsets = [0.0]
        if forward_first:
            offsets += [0.01 * k for k in range(1, 26)] + [-0.01 * k for k in range(1, 11)]
        else:
            for k in range(1, 11):
                offsets += [0.01 * k, -0.01 * k]
        bz = self._board_z()
        on_board_limit = bz + 3.0 * self.cfg.wire.radius
        fallback = None
        for off in offsets:
            s = s_ideal + off
            if s < s_min or s > s_max:
                continue
            p, _ = polyline_point_at(cable, s)
            if all(np.linalg.norm(p[:2] - o) > 0.05 for o in obstacles):
                if p[2] < on_board_limit:
                    return s               # clear of fixtures and lying on the board
                if fallback is None:
                    fallback = s           # clear, but draped over something
        if fallback is not None:
            return fallback
        return float(np.clip(s_ideal, s_min, s_max))

    def _pick_wire(self, s_pick: float, yaw_ref: float, carry_yaw: Optional[float] = None) -> Generator:
        """Grasp the wire at arc length s_pick. With carry_yaw, the finger orientation is
        chosen for the rotation that later turns the held wire along carry_yaw.
        Returns True on a successful grasp."""
        p_cfg = self.p
        cable = self._obs["cable"]
        s_total = polyline_arclength(cable)[-1]
        s_pick = float(np.clip(s_pick, 0.05, s_total - 0.05))
        p_w, t_w = polyline_point_at(cable, s_pick)
        wire_yaw = math.atan2(t_w[1], t_w[0])
        if carry_yaw is None:
            yaw = self._closest_yaw(wire_yaw, yaw_ref)
        else:
            yaw = self._pick_yaw(wire_yaw, carry_yaw)
        bz = self._board_z()
        self.phase = "pick_approach"
        self._say(f"pick wire at s={s_pick:.3f} -> ({p_w[0]:.3f}, {p_w[1]:.3f}, {p_w[2] - bz:.3f})")
        self._grip = gripper_action_from_opening(0.034, self.spec.max_opening)
        above = np.array([p_w[0], p_w[1], bz + p_cfg.hover])
        if not (yield from self._goto(above, yaw, tol=0.003, timeout=8.0, settle=0.15)):
            return False
        # re-read the wire position (it may have moved) and descend
        cable = self._obs["cable"]
        p_w, _ = polyline_point_at(cable, s_pick)
        self.phase = "pick_descend"
        z_wire = p_w[2]
        on_board = z_wire < bz + 2.0 * self.cfg.wire.radius
        if on_board:
            # guarded touch-down: pads land on the board on either side of the wire
            goal = np.array([p_w[0], p_w[1], bz + 0.002])
            obs = self._obs
            z_touch = None
            t_end = self._t + 6.0
            f0 = float(obs["wrench"][2])          # remove any residual sensor offset
            while self._t < t_end:
                if obs["wrench"][2] - f0 > p_cfg.touch_force:
                    z_touch = obs["tcp_pos"][2]
                    break
                near = obs["tcp_pos"][2] < bz + 0.018
                obs = yield self._action_toward(obs, goal, yaw, 0.01 if near else p_cfg.speed_slow)
            if z_touch is None:
                self._say("no touch-down detected")
                return False
            z_grasp = z_touch + p_cfg.grasp_backoff
        else:
            z_grasp = z_wire + 0.0045
        yield from self._goto(np.array([p_w[0], p_w[1], z_grasp]), yaw, speed=p_cfg.speed_slow,
                              tol=0.0015, timeout=2.0)
        self.phase = "pick_close"
        yield from self._set_grip(1.0, 0.5)
        opening = float(self._obs["gripper"][0])
        d = 2.0 * self.cfg.wire.radius
        if not (0.5 * d < opening < d + 0.003):
            self._say(f"grasp failed (opening {opening * 1000:.1f} mm)")
            yield from self._set_grip(-1.0, 0.3)
            up = self._obs["tcp_pos"].copy()
            up[2] = bz + p_cfg.hover
            yield from self._goto(up, None, tol=0.005, timeout=3.0)
            return False
        return True

    # ----------------------------------------------------------- route fork
    def _route_fork(self, i: int, attempt: int = 0, pick_offset: Optional[float] = None) -> Generator:
        """Route the wire into fork i. ``pick_offset`` (m along the wire) overrides the
        default per-attempt shift of the pick point."""
        cfg, p_cfg = self.cfg, self.p
        obs = self._obs
        bz = self._board_z()
        fx, fy, fz, fyaw = obs["forks"][i]
        f_xy = np.array([fx, fy])
        p_fix, s_fix = self._fixation(i)
        # The taut wire runs straight from its last fixation through the fork, so the
        # gripper is placed on the extension of that line (not along the fork axis).
        L = float(np.linalg.norm(f_xy - p_fix[:2]))
        u = (f_xy - p_fix[:2]) / max(L, 1e-6)
        z_top = bz + cfg.fork.post_height + cfg.fork.prong_height
        z_transit = z_top + p_cfg.transit_clearance
        wire_drop = 0.0045                       # wire axis below the TCP when grasped
        # Wire length (fixation -> gripper) we would like to hold: enough to carry it,
        # taut, over the fork and `beyond` past it at transit height ...
        ell_ideal = math.hypot(L + p_cfg.beyond, (z_transit - wire_drop) - p_fix[2]) + p_cfg.pick_margin
        # ... and the minimum that still reaches 2.5 cm past the fork at the lowest carry height
        rise_min = (z_top + 0.03 - wire_drop) - p_fix[2]
        ell_min = math.hypot(L + 0.025, rise_min) + p_cfg.pick_margin
        s_total = polyline_arclength(obs["cable"])[-1]
        shift = 0.02 * attempt if pick_offset is None else float(pick_offset)
        s_pick = self._choose_pick_arclength(s_fix + ell_ideal + shift,
                                             s_fix + ell_min, s_total - 0.06,
                                             forward_first=shift > 0.0)
        if s_pick is None:
            self._say(f"fork {i}: no reachable pick point on the wire")
            return False
        route_yaw = math.atan2(u[1], u[0])
        n = np.array([-u[1], u[0]])
        yaw_ref = self._closest_yaw(route_yaw, float(obs["tcp_yaw"][0]))
        if not (yield from self._pick_wire(s_pick, yaw_ref, carry_yaw=route_yaw)):
            return False

        # carry the wire over the fork on a taut-wire sweep around the fixation point.
        # Turn the gripper so the held wire continues *away* from the fixation: the
        # other finger orientation would fold it into a hairpin that eats the slack.
        yaw = wrap_angle(route_yaw - self._held_wire_yaw_offset(s_pick))
        if abs(self._q6_after(yaw)) > 5.6:
            yaw = self._closest_yaw(route_yaw, float(self._obs["tcp_yaw"][0]))
        ell = s_pick - s_fix
        ell_eff = ell - 0.012
        D_final = min(L + p_cfg.beyond, math.sqrt(max(ell_eff ** 2 - rise_min ** 2, 0.0)))
        D_final = max(D_final, L + 0.02)
        if not (yield from self._carry_over(p_fix, ell, D_final, route_yaw, yaw)):
            self._say("carry-over aborted")
            yield from self._set_grip(-1.0, 0.3)
            up = self._obs["tcp_pos"].copy()
            up[2] = max(up[2], z_transit)
            yield from self._goto(up, None, tol=0.01, timeout=3.0)
            return False

        # lower the held wire beyond the fork while keeping it taut: take up slack by
        # moving away from the fork, give way when the tension gets high
        self.phase = "route_descend"
        z_press = bz + p_cfg.press_height
        d = D_final - L
        z_goal = float(self._obs["tcp_pos"][2])
        obs = self._obs
        t_end = self._t + 12.0
        while self._t < t_end:
            T = self._tension(u)
            if T < p_cfg.tension_low:
                d += 0.0015
            elif T > p_cfg.tension_descend:
                d -= 0.001
            d = float(np.clip(d, p_cfg.min_beyond, p_cfg.max_beyond))
            above_fork = obs["tcp_pos"][2] > z_top + 0.015
            if above_fork or T > 0.5 * p_cfg.tension_low or d >= p_cfg.max_beyond - 1e-6:
                z_goal = max(z_goal - 0.0015, z_press)
            q_xy = f_xy + d * u
            if abs(obs["tcp_pos"][2] - z_press) < 0.002:
                break
            obs = yield self._action_toward(obs, np.array([q_xy[0], q_xy[1], z_goal]), yaw, speed=0.06)

        # seat: ramp the tension (and wiggle a little) until the wire snaps past the lips
        self.phase = "route_seat"
        lat = 0.0
        t0 = self._t
        routed_since = None
        obs = self._obs
        while self._t - t0 < p_cfg.seat_time:
            chk = cable_crossing_in_fork(obs["cable"], obs["forks"][i], cfg.fork, bz)
            if chk["inside"]:
                routed_since = self._t if routed_since is None else routed_since
                if self._t - routed_since > 0.25:
                    break
            else:
                routed_since = None
            tau = self._t - t0
            ramp = min(1.0, tau / (0.6 * p_cfg.seat_time))
            T_ref = p_cfg.tension_low + ramp * (p_cfg.tension_high - p_cfg.tension_low)
            T = self._tension(u)
            d = float(np.clip(d + np.clip(0.0004 * (T_ref - T), -0.001, 0.001),
                              p_cfg.min_beyond, p_cfg.max_beyond))
            # centre the wire over the slot: the perceived lateral offset of the wire at
            # the fork is fed back into a lateral shift of the gripper
            if np.isfinite(chk["y"]) and chk["z"] > cfg.fork.post_height + cfg.fork.prong_height - 0.01:
                y_fork = math.cos(fyaw) * n[1] - math.sin(fyaw) * n[0]   # fork y-axis . n
                lat = float(np.clip(lat - 0.3 * chk["y"] * np.sign(y_fork if abs(y_fork) > 1e-3 else 1.0),
                                    -0.012, 0.012))
            wiggle = 0.0 if tau < 1.0 else 0.003 * math.sin(2.0 * math.pi * 1.2 * tau)
            q_xy = f_xy + d * u + (lat + wiggle) * n
            obs = yield self._action_toward(obs, np.array([q_xy[0], q_xy[1], z_press]), yaw, speed=0.04)
        chk = cable_crossing_in_fork(self._obs["cable"], self._obs["forks"][i], cfg.fork, bz)
        self._say(f"fork {i}: seat check inside={chk['inside']} y={chk['y'] * 1000:.1f}mm "
                  f"z={chk['z'] * 1000:.1f}mm tension={self._tension(u):.1f}N")

        # release and back off upwards
        self.phase = "route_release"
        yield from self._set_grip(-1.0, 0.35)
        tcp = self._obs["tcp_pos"].copy()
        yield from self._goto(np.array([tcp[0], tcp[1], z_transit]), None, speed=0.08, tol=0.005, timeout=3.0)
        chk = cable_crossing_in_fork(self._obs["cable"], self._obs["forks"][i], cfg.fork, bz)
        if not chk["inside"]:
            self._say(f"fork {i}: wire not retained after release")
            return False
        self._say(f"fork {i}: routed")
        return True

    # ----------------------------------------------------------- connector
    def _connector_ok_to_grasp(self) -> bool:
        """Lying flat and no fixture inside the footprint of the open fingers."""
        obs = self._obs
        cp, cR = obs["connector_pos"], obs["connector_rot"]
        if abs(cR[2, 0]) > 0.5:
            return False
        yaw_c = math.atan2(cR[1, 0], cR[0, 0])
        c, s_ = math.cos(yaw_c), math.sin(yaw_c)
        # fixtures as discs (centre, radius); fingers close across the connector (gripper y)
        discs = [(f[:2], 0.017) for f in obs["forks"]] + [(obs["anchor_pos"][:2], 0.025)]
        for centre, rad in discs:
            d = centre - cp[:2]
            lx, ly = c * d[0] + s_ * d[1], -s_ * d[0] + c * d[1]
            if abs(lx) < 0.013 + rad and abs(ly) < 0.036 + rad:
                return False
        return True

    def _staging_point(self) -> np.ndarray:
        """Free spot near the holder where a badly placed connector is laid down. It must
        be reachable with the free wire length left after the last fork."""
        obs = self._obs
        hp = obs["holder_pos"][:2]
        yaw_h = float(obs["holder_yaw"][0])
        n_h = np.array([-math.sin(yaw_h), math.cos(yaw_h)])
        x_h = np.array([math.cos(yaw_h), math.sin(yaw_h)])
        last = obs["forks"][-1]
        cable = obs["cable"]
        s_total = polyline_arclength(cable)[-1]
        s_last = arclength_near(cable, np.array([last[0], last[1], last[2] + self.cfg.fork.post_height]))
        free = s_total - s_last
        obstacles = [f[:2] for f in obs["forks"]] + [obs["anchor_pos"][:2], hp]
        cands = []
        for lat in (0.07, -0.07, 0.05, -0.05):
            for back in (0.0, -0.03, 0.03):
                cands.append(hp + lat * n_h + back * x_h)
        def score(c):
            clear = min(np.linalg.norm(c - o) for o in obstacles)
            reach = free - (np.linalg.norm(c - last[:2]) + 0.09)
            return (reach > 0.0, min(clear, 0.06), reach)
        return max(cands, key=score)

    def _tip_connector(self) -> Generator:
        """A connector standing on its end is pulled over by its wire: grasp the wire a
        few cm from it and move towards the last fork while lowering, so the connector
        tips over and ends up lying flat with the wire leaving towards the fork."""
        self.phase = "connector_tip"
        bz = self._board_z()
        cable = self._obs["cable"]
        s_total = polyline_arclength(cable)[-1]
        last = self._obs["forks"][-1]
        s_last = arclength_near(cable, np.array([last[0], last[1], last[2] + self.cfg.fork.post_height]))
        s_pick = self._choose_pick_arclength(s_total - 0.05, min(s_last + 0.06, s_total - 0.03), s_total - 0.03)
        if s_pick is None:
            return False
        self._say("connector stands on its end -> pull it over")
        if not (yield from self._pick_wire(s_pick, float(self._obs["tcp_yaw"][0]))):
            return False
        self.phase = "connector_tip"
        d = last[:2] - self._obs["connector_pos"][:2]
        d = d / max(float(np.linalg.norm(d)), 1e-6)
        tcp = self._obs["tcp_pos"].copy()
        goal = np.array([tcp[0] + 0.06 * d[0], tcp[1] + 0.06 * d[1], bz + 0.03])
        yield from self._goto(goal, None, speed=0.04, tol=0.006, timeout=4.0, max_force=10.0)
        yield from self._set_grip(-1.0, 0.4)
        tcp = self._obs["tcp_pos"].copy()
        yield from self._goto(np.array([tcp[0], tcp[1], bz + 0.09]), None, speed=0.06, tol=0.008, timeout=3.0)
        yield from self._wait(0.3)
        return True

    def _relocate_connector(self) -> Generator:
        """Pick the wire a few cm behind the connector, carry the connector to a free
        spot and drag it while lowering so that it ends up lying flat."""
        self.phase = "connector_relocate"
        bz = self._board_z()
        cable = self._obs["cable"]
        s_total = polyline_arclength(cable)[-1]
        last = self._obs["forks"][-1]
        s_last = arclength_near(cable, np.array([last[0], last[1], last[2] + self.cfg.fork.post_height]))
        target = self._staging_point()
        self._say(f"relocating connector to ({target[0]:.3f}, {target[1]:.3f})")
        # grasp close to the connector but leave enough free wire after the last fork so
        # lifting does not pull the wire out of it
        s_pick = self._choose_pick_arclength(s_total - 0.05, min(s_last + 0.10, s_total - 0.03), s_total - 0.03)
        if s_pick is None:
            return False
        if not (yield from self._pick_wire(s_pick, float(self._obs["tcp_yaw"][0]))):
            return False
        self.phase = "connector_relocate"
        # lift only as high as the free wire after the last fork allows (taut geometry)
        tcp = self._obs["tcp_pos"].copy()
        ell_free = s_pick - s_last
        D = float(np.linalg.norm(tcp[:2] - last[:2]))
        z_slot = last[2] + self.cfg.fork.post_height + self.cfg.wire.radius
        z_taut = z_slot + math.sqrt(max(ell_free ** 2 - D ** 2, 0.0)) - 0.015
        z_carry = float(np.clip(z_taut, bz + 0.035, bz + self.cfg.fork.post_height + self.cfg.fork.prong_height + 0.03))
        if not (yield from self._goto(np.array([tcp[0], tcp[1], z_carry]), None, speed=0.08, tol=0.008,
                                      timeout=4.0, max_force=8.0)):
            yield from self._set_grip(-1.0, 0.4)
            return False
        drag = target - self._obs["tcp_pos"][:2]
        drag = drag / max(np.linalg.norm(drag), 1e-6)
        start = target - 0.035 * drag
        yield from self._goto(np.array([start[0], start[1], z_carry]), None, speed=0.10, tol=0.008,
                              timeout=5.0, max_force=8.0)
        # lower while moving on: the hanging connector touches down and tips over
        end = target + 0.035 * drag
        yield from self._goto(np.array([end[0], end[1], bz + 0.04]), None, speed=0.04, tol=0.006,
                              timeout=4.0, max_force=8.0)
        yield from self._set_grip(-1.0, 0.4)
        tcp = self._obs["tcp_pos"].copy()
        yield from self._goto(np.array([tcp[0], tcp[1], z_carry]), None, speed=0.06, tol=0.008, timeout=3.0)
        yield from self._wait(0.3)
        return True

    def _insert_connector(self, auto_recover: bool = True) -> Generator:
        cfg, p_cfg = self.cfg, self.p
        c = cfg.connector
        bz = self._board_z()
        if auto_recover and abs(self._obs["connector_rot"][2, 0]) > 0.5 and self._tips < 2:
            self._tips += 1
            yield from self._tip_connector()
        if auto_recover and not self._connector_ok_to_grasp() and self._relocations < 2:
            self._relocations += 1
            yield from self._relocate_connector()
        if abs(self._obs["connector_rot"][2, 0]) > 0.5:
            self._say("connector is not lying flat")
            return False
        obs = self._obs
        cp, cR = obs["connector_pos"], obs["connector_rot"]
        yaw_c = math.atan2(cR[1, 0], cR[0, 0])
        yaw_h = float(obs["holder_yaw"][0])

        # grasp with gripper x along the connector axis; pick the flip that keeps the
        # wrist furthest from its limits over grasp and placement
        def wrist_cost(delta: float) -> float:
            q6_grasp = self._q6_after(wrap_angle(yaw_c + delta))
            q6_place = self._q6_after(wrap_angle(yaw_h + delta), wrap_angle(yaw_c + delta), q6_grasp)
            return max(abs(q6_grasp), abs(q6_place))
        delta = min((0.0, math.pi), key=wrist_cost)
        grasp_yaw = wrap_angle(yaw_c + delta)
        self.phase = "connector_approach"
        self._say(f"connector at ({cp[0]:.3f}, {cp[1]:.3f}), yaw {yaw_c:.2f}; holder yaw {yaw_h:.2f}")
        self._grip = gripper_action_from_opening(0.034, self.spec.max_opening)
        near_fork = any(np.linalg.norm(cp[:2] - f[:2]) < 0.08 for f in obs["forks"])
        z_hover = bz + (cfg.fork.post_height + cfg.fork.prong_height + 0.035 if near_fork else p_cfg.hover)
        tcp = obs["tcp_pos"].copy()
        if tcp[2] < z_hover:
            yield from self._goto(np.array([tcp[0], tcp[1], z_hover]), None, speed=0.08, tol=0.006, timeout=3.0)
        if not (yield from self._goto(np.array([cp[0], cp[1], z_hover]), grasp_yaw, tol=0.003,
                                      timeout=8.0, settle=0.15, max_force=10.0)):
            self._say("connector approach failed")
            return False
        cp = self._obs["connector_pos"]
        self.phase = "connector_grasp"
        z_grasp = cp[2] + 0.5 * c.height - 0.004
        if not (yield from self._goto(np.array([cp[0], cp[1], z_grasp]), grasp_yaw, speed=p_cfg.speed_slow,
                                      tol=0.0015, timeout=3.0, max_force=6.0)):
            self._say("connector grasp descent blocked (fingers on a fixture?)")
            up = self._obs["tcp_pos"].copy()
            up[2] = z_hover
            yield from self._goto(up, None, tol=0.005, timeout=3.0)
            if auto_recover and self._relocations < 2:
                self._relocations += 1
                yield from self._relocate_connector()
            return False
        yield from self._set_grip(1.0, 0.5)
        opening = float(self._obs["gripper"][0])
        if not (c.width - 0.004 < opening < c.width + 0.003):
            self._say(f"connector grasp failed (opening {opening * 1000:.1f} mm)")
            yield from self._set_grip(-1.0, 0.3)
            up = self._obs["tcp_pos"].copy()
            up[2] = z_hover
            yield from self._goto(up, None, tol=0.005, timeout=3.0)
            return False

        c_h = self.cfg.holder
        wall_top = bz + c_h.floor_height + c_h.end_wall_height
        self.phase = "connector_transit"
        tcp = self._obs["tcp_pos"].copy()
        yield from self._goto(np.array([tcp[0], tcp[1], max(z_hover, wall_top + 0.05)]), None,
                              speed=0.08, tol=0.005, timeout=3.0)
        # in-hand pose of the connector (a real cell would get this from a camera)
        obs = self._obs
        R_g = obs["tcp_rot"]
        off = R_g.T @ (obs["connector_pos"] - obs["tcp_pos"])       # in the gripper frame
        yaw_off = wrap_angle(math.atan2(obs["connector_rot"][1, 0], obs["connector_rot"][0, 0])
                             - float(obs["tcp_yaw"][0]))
        tcp_above_center = float(obs["tcp_pos"][2] - obs["connector_pos"][2])
        place_yaw = wrap_angle(yaw_h - yaw_off)

        def tcp_xy_for(conn_xy: np.ndarray) -> np.ndarray:
            c_, s_ = math.cos(place_yaw), math.sin(place_yaw)
            o = np.array([c_ * off[0] - s_ * off[1], s_ * off[0] + c_ * off[1]])
            return conn_xy - o

        hp = obs["holder_pos"]                 # noisy estimate of the seat centre
        x_h = np.array([math.cos(yaw_h), math.sin(yaw_h)])
        z_low = wall_top + 0.004 + 0.5 * c.height + tcp_above_center
        z_hover2 = z_low + 0.03
        # come in from behind the pocket (the side the wire leaves through) and slide
        # forward, so the wire is dragged straight out through the back slot
        pre = tcp_xy_for(hp[:2] - p_cfg.approach_offset * x_h)
        if not (yield from self._goto(np.array([pre[0], pre[1], z_hover2]), place_yaw, speed=0.10,
                                      tol=0.004, yaw_tol=0.03, timeout=8.0, settle=0.1)):
            self._say("pre-insert approach timeout")
        yield from self._goto(np.array([pre[0], pre[1], z_low]), place_yaw, speed=0.04, tol=0.003, timeout=3.0)
        self.phase = "connector_align"
        goal_xy = tcp_xy_for(hp[:2])
        if not (yield from self._goto(np.array([goal_xy[0], goal_xy[1], z_low]), place_yaw, speed=0.03,
                                      tol=0.002, yaw_tol=0.02, timeout=6.0, settle=0.2, max_force=12.0)):
            self._say("holder alignment timeout")

        # guarded descent, then force-controlled spiral search until the connector
        # drops to seat height, then press
        self.phase = "connector_descend"
        z_seated_tcp = hp[2] + tcp_above_center
        k_z = self.cfg.controller.kp_lin / self.cfg.controller.kf_lin
        obs = self._obs
        t_end = self._t + 5.0
        while self._t < t_end:
            if obs["tcp_pos"][2] < z_seated_tcp + 0.0015:
                break
            if obs["wrench"][2] > p_cfg.insert_force * 0.7:     # touched the holder rim
                break
            obs = yield self._action_toward(obs, np.array([goal_xy[0], goal_xy[1], z_seated_tcp - 0.004]),
                                            place_yaw, speed=0.015)
        seated_now = lambda: self._obs["tcp_pos"][2] < z_seated_tcp + 0.0015
        if not seated_now():
            self.phase = "connector_search"
            self._say("contact above the seat -> spiral search")
            t0 = self._t
            obs = self._obs
            while self._t - t0 < 15.0 and not seated_now():
                tau = self._t - t0
                theta = 2.0 * math.pi * tau / p_cfg.spiral_period
                r = min(p_cfg.spiral_growth * theta / (2.0 * math.pi), p_cfg.spiral_max_radius)
                z_ref = float(obs["tcp_pos"][2])
                goal = np.array([goal_xy[0] + r * math.cos(theta), goal_xy[1] + r * math.sin(theta),
                                 z_ref - p_cfg.insert_force / k_z])
                dyaw = 0.03 * math.sin(2.0 * math.pi * tau / 0.7)
                obs = yield self._action_toward(obs, goal, place_yaw + dyaw, speed=0.03)
            if not seated_now():
                self._say("spiral search did not reach seat height")
        self.phase = "connector_press"
        xy = self._obs["tcp_pos"][:2].copy()
        yield from self._goto(np.array([xy[0], xy[1], z_seated_tcp - p_cfg.press_force / k_z]), place_yaw,
                              speed=0.02, tol=0.0025, timeout=1.5)
        # press with a small wiggle so a slightly tilted connector settles into the pocket
        t0 = self._t
        obs = self._obs
        while self._t - t0 < 1.2 and not (obs["tcp_pos"][2] < z_seated_tcp + 0.0005):
            tau = self._t - t0
            wig = 0.0008 * np.array([math.sin(2 * math.pi * 2.0 * tau), math.cos(2 * math.pi * 2.0 * tau)])
            goal = np.array([xy[0] + wig[0], xy[1] + wig[1], z_seated_tcp - p_cfg.press_force / k_z])
            obs = yield self._action_toward(obs, goal, place_yaw + 0.03 * math.sin(2 * math.pi * 3.0 * tau), speed=0.03)
        yield from self._wait(0.3)
        self.phase = "connector_release"
        yield from self._set_grip(-1.0, 0.4)
        tcp = self._obs["tcp_pos"].copy()
        yield from self._goto(np.array([tcp[0], tcp[1], tcp[2] + 0.05]), None, speed=0.06, tol=0.005, timeout=3.0)
        # average a few perception samples before judging (poses are noisy)
        cps, hps = [], []
        obs = self._obs
        for _ in range(8):
            cps.append(obs["connector_pos"].copy())
            hps.append(obs["holder_pos"].copy())
            obs = yield self._action_toward(obs, obs["target_pos"], None, 0.0)
        cp, hp = np.mean(cps, axis=0), np.mean(hps, axis=0)
        cR = self._obs["connector_rot"]
        yaw_err = wrap_angle(math.atan2(cR[1, 0], cR[0, 0]) - yaw_h)
        e_xy = cp[:2] - hp[:2]
        e_ax = float(e_xy @ x_h)                              # along the holder axis
        e_lat = float(e_xy @ np.array([-x_h[1], x_h[0]]))     # across it
        tilt = math.asin(min(1.0, abs(float(cR[2, 0]))))      # long axis out of the board plane
        seated = (abs(cp[2] - hp[2]) < 0.002 and abs(e_ax) < 0.0035 and abs(e_lat) < 0.002
                  and abs(yaw_err) < 0.15 and tilt < 0.12)
        self._say(f"connector seated={seated} (dz={1000 * (cp[2] - hp[2]):.1f} mm, yaw err {yaw_err:.2f}, "
                  f"tilt {math.degrees(tilt):.0f} deg)")
        return bool(seated)
