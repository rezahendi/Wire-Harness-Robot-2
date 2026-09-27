"""Isaac Sim (PhysX) implementation of the benchmark rigs.

Mirrors ``harness_core.rigs_mujoco.MujocoRigs`` method for method and returns the same
traces, so ``harness_bench`` scores both engines with the same analysis code.

Run it with Isaac Sim's interpreter (or a Python 3.11 venv with the isaacsim pip
package), from the repository's ``isaac`` directory:

    python -m harness_isaac.run_benchmarks --out results/isaac.json

Notes on the modelling choices are in ``scene_usd.py``; the short version is that
PhysX has no Cosserat rod, so the wire is a capsule chain whose joint drives carry the
bending stiffness EI / L_segment.
"""

from __future__ import annotations

import math
import time
from typing import Dict, Optional, Sequence

import numpy as np

from harness_core import benchmarks as B
from harness_core.config import CellConfig

from . import compat
from . import scene_usd as S


class IsaacRigs:
    name = "isaac"

    def __init__(self, cfg: Optional[CellConfig] = None, bend_scale: float = 1.0,
                 friction_scale: float = 1.0, headless: bool = True):
        self.cfg = cfg.copy() if cfg is not None else CellConfig()
        self.bend_scale = bend_scale
        self.friction_scale = friction_scale
        compat.start_app(headless=headless)          # must precede any pxr/omni import
        self._World = compat.world_class()
        self._RigidView = compat.rigid_prim_view()
        self.world = None

    # ------------------------------------------------------------- helpers
    @property
    def bend(self) -> float:
        return self.cfg.wire.bend_modulus * self.bend_scale

    @property
    def twist(self) -> float:
        return self.cfg.wire.twist_modulus * self.bend_scale

    def properties(self) -> Dict[str, float]:
        return B.wire_properties(self.cfg.wire.radius, self.cfg.wire.density, self.bend)

    def version(self) -> Dict[str, str]:
        v = "unknown"
        try:
            import isaacsim                                   # noqa: F401
            v = getattr(isaacsim, "__version__", "unknown")
        except Exception:
            pass
        return {"engine": "isaac", "version": str(v), "integrator": "PhysX TGS",
                "cable": "capsule chain with D6 joint drives (k = EI / L_segment)"}

    def _new_world(self, timestep: Optional[float] = None):
        dt = timestep or self.cfg.sim.timestep
        if self.world is not None:
            self.world.clear_instance()
        self.world = self._World(stage_units_in_meters=1.0, physics_dt=dt, rendering_dt=dt)
        stage = self.world.stage
        S.setup_stage(stage, dt)
        self.material = S.physics_material(
            stage, "/World/materials/wire",
            static_friction=self.cfg.wire.friction * self.friction_scale,
            dynamic_friction=self.cfg.wire.friction * self.friction_scale)
        return stage

    def _wire(self, stage, n: int, seg: float, start, root: str = "/World/wire") -> Dict:
        p = self.properties()
        return S.add_wire(stage, root, n, seg, self.cfg.wire.radius, self.cfg.wire.density,
                          p["EI"], self.twist, self.cfg.wire.joint_damping, start,
                          material=self.material)

    def _points(self, wire: Dict) -> np.ndarray:
        """Segment origins plus the far end of the last segment, like cable_points()."""
        view = self._RigidView(prim_paths_expr=wire["segments"][0].rsplit("/", 1)[0] + "/segment_*")
        pos, quat = view.get_world_poses()
        pos = np.asarray(pos, dtype=float)
        q = np.asarray(quat, dtype=float)[-1]            # (w, x, y, z)
        w, x, y, z = q
        x_axis = np.array([1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)])
        return np.vstack([pos, pos[-1] + x_axis * wire["segment_length"]])

    def _step(self, n: int) -> None:
        for _ in range(n):
            self.world.step(render=False)

    # ---------------------------------------------------------- cantilever
    def cantilever(self, length: float, segment_length: float, spec: B.CantileverSpec,
                   record: bool = False, timestep: Optional[float] = None) -> Dict:
        n = max(2, int(round(length / segment_length)))
        seg = length / n
        h = spec.clamp_height
        stage = self._new_world(timestep)
        wire = self._wire(stage, n, seg, (0.0, 0.0, h))
        S.fix_to_world(stage, "/World/clamp_joint", wire["segments"][0])
        self.world.reset()
        dt = self.world.get_physics_dt()
        steps = int(spec.settle_time / dt)
        sample = max(1, int(round(0.002 / dt)))
        t_hist, tip_hist = [], []
        for k in range(steps):
            self.world.step(render=False)
            if record and k % sample == 0:
                t_hist.append(k * dt)
                tip_hist.append(self._points(wire)[-1, 2])
        pts = self._points(wire)
        out = {"points": pts, "clamp": np.array([0.0, 0.0, h]),
               "settled_speed": 0.0, "finite": bool(np.all(np.isfinite(pts))),
               "n_segments": n, "segment_length": seg, "length": length}
        if record:
            out["t"] = np.array(t_hist)
            out["tip_z"] = np.array(tip_hist)
        return out

    # ----------------------------------------------------------------- sag
    def sag(self, length: float, span: float, segment_length: float, spec: B.SagSpec) -> Dict:
        n = max(3, int(round(length / segment_length)))
        seg = length / n
        h = spec.clamp_height
        stage = self._new_world()
        # start on a circular arc through both supports so the ends are already close
        arc = _arc_points(length, span, n, (0.0, 0.0, h))
        wire = self._wire(stage, n, seg, (0.0, 0.0, h))
        _place_chain(stage, wire["segments"], arc)
        S.pin_to_world(stage, "/World/pin_a", wire["segments"][0], (0.0, 0.0, h))
        S.pin_to_world(stage, "/World/pin_b", wire["segments"][-1], (span, 0.0, h),
                       local_pos=(seg, 0.0, 0.0))
        self.world.reset()
        self._step(int(spec.settle_time / self.world.get_physics_dt()))
        pts = self._points(wire)
        return {"points": pts, "span": float(np.linalg.norm(pts[-1, :2] - pts[0, :2])),
                "nominal_span": span, "length": length, "settled_speed": 0.0,
                "finite": bool(np.all(np.isfinite(pts)))}

    # --------------------------------------------------------------- swing
    def swing(self, spec: B.SwingSpec) -> Dict:
        res = self.cantilever(spec.length, spec.segment_length,
                              B.CantileverSpec(clamp_height=spec.clamp_height,
                                               settle_time=spec.record_time), record=True)
        res["spec"] = "swing"
        return res

    # ---------------------------------------------------------------- snap
    def snap(self, spec: B.SnapSpec, lateral_offset: float = 0.0) -> Dict:
        cfg = self.cfg
        f = cfg.fork
        n = max(4, int(round(spec.wire_length / spec.segment_length)))
        seg = spec.wire_length / n
        x_fork = 0.03
        slot_z = f.post_height + cfg.wire.radius
        z_high = f.post_height + f.prong_height + 0.012
        stage = self._new_world()
        S.add_ground_box(stage, "/World/ground", (0.8, 0.8, 0.04), (0.0, 0.0, -0.02), self.material)
        S.add_fork(stage, "/World/fork", cfg, (x_fork, lateral_offset, 0.0), self.material)
        hand = S.add_prismatic_hand(stage, "/World/hand", "Z", (0.0, 0.0, z_high), 0.2,
                                    stiffness=4000.0, damping=120.0, lower=-0.3, upper=0.3)
        wire = self._wire(stage, n, seg, (0.0, 0.0, z_high))
        S.fix_to_world(stage, "/World/hand_wire_joint", wire["segments"][0])  # replaced below
        _reparent_fixed_joint(stage, "/World/hand_wire_joint", hand["body"], wire["segments"][0])
        self.world.reset()
        dt = self.world.get_physics_dt()
        self._step(int(0.5 / dt))
        i_fork = min(n - 1, max(0, int(round(x_fork / seg))))
        hand_view = self._RigidView(prim_paths_expr=hand["body"])

        t_hist, z_hist, f_hist, ph_hist = [], [], [], []
        k_drive = hand["drive_stiffness"]

        def hand_z() -> float:
            pos, _ = hand_view.get_world_poses()
            return float(np.asarray(pos, dtype=float).reshape(-1, 3)[0, 2])

        def run(target_z: float, speed: float, phase: int):
            start = hand_z()
            steps = max(1, int(abs(target_z - start) / max(speed, 1e-6) / dt))
            for k in range(steps):
                target = start + (target_z - start) * (k + 1) / steps
                _set_drive_target(stage, hand["joint"], target - z_high)
                self.world.step(render=False)
                pts = self._points(wire)
                # force from the drive's deflection: F = k (target - actual)
                f_hist.append(k_drive * (target - hand_z()))
                t_hist.append(len(t_hist) * dt)
                z_hist.append(float(pts[i_fork, 2]))
                ph_hist.append(phase)

        run(slot_z, spec.press_speed, 0)
        self._step(int(spec.hold_time / dt))
        wire_z_seated = float(self._points(wire)[i_fork, 2])
        run(z_high, spec.pull_speed, 1)
        self._step(int(0.3 / dt))
        pts = self._points(wire)
        retained = bool(pts[i_fork, 2] < f.post_height + f.prong_height - 0.004)
        return {"t": np.array(t_hist), "z": np.array(z_hist), "fz": np.array(f_hist),
                "phase": np.array(ph_hist), "retained": retained,
                "wire_z_seated": wire_z_seated, "slot_z": slot_z,
                "lip_z": f.post_height + f.prong_height, "lip_radius": f.lip_radius,
                "wire_radius": cfg.wire.radius, "lateral_offset": lateral_offset,
                "finite": bool(np.all(np.isfinite(pts)))}

    # --------------------------------------------------------------- slide
    def slide(self, spec: B.SlideSpec) -> Dict:
        cfg = self.cfg
        n = max(4, int(round(spec.wire_length / spec.segment_length)))
        seg = spec.wire_length / n
        r = cfg.wire.radius
        stage = self._new_world()
        S.add_ground_box(stage, "/World/board", (0.8, 0.4, 0.04), (0.1, 0.0, -0.02), self.material)
        hand = S.add_prismatic_hand(stage, "/World/hand", "X", (0.0, 0.0, r + 0.004), 0.2,
                                    stiffness=4000.0, damping=120.0, lower=-0.1, upper=0.6)
        wire = self._wire(stage, n, seg, (0.0, 0.0, r + 0.004))
        S.fix_to_world(stage, "/World/hand_wire_joint", wire["segments"][0])
        _reparent_fixed_joint(stage, "/World/hand_wire_joint", hand["body"], wire["segments"][0])
        self.world.reset()
        dt = self.world.get_physics_dt()
        self._step(int(spec.settle_time / dt))
        hand_view = self._RigidView(prim_paths_expr=hand["body"])
        steps = max(1, int(spec.distance / spec.speed / dt))
        t_hist, fx_hist = [], []
        for k in range(steps):
            target = spec.distance * (k + 1) / steps
            _set_drive_target(stage, hand["joint"], target)
            self.world.step(render=False)
            pos, _ = hand_view.get_world_poses()
            x = float(np.asarray(pos, dtype=float).reshape(-1, 3)[0, 0])
            fx_hist.append(hand["drive_stiffness"] * (target - x))
            t_hist.append(k * dt)
        wire_mass = math.pi * r * r * spec.wire_length * cfg.wire.density
        # PhysX contact forces would need contact reporters; the wire's own weight minus
        # the part carried by the hand is a good enough normal load for the ratio
        normal = 0.6 * wire_mass * B.GRAVITY
        return {"t": np.array(t_hist), "fx": np.array(fx_hist), "fn": np.array([]),
                "normal_force": normal, "wire_weight": wire_mass * B.GRAVITY,
                "nominal_friction": cfg.wire.friction * self.friction_scale,
                "finite": bool(np.all(np.isfinite(fx_hist)))}

    # ----------------------------------------------------------- step rate
    def step_rate(self, spec: B.StepRateSpec) -> Dict:
        out: Dict[str, float] = {}
        for n_seg in (12, 24, 48):
            stage = self._new_world()
            wire = self._wire(stage, n_seg, 0.015, (0.0, 0.0, 0.5))
            S.fix_to_world(stage, "/World/clamp_joint", wire["segments"][0])
            self.world.reset()
            self._step(200)
            dt = self.world.get_physics_dt()
            steps = int(0.5 / dt)
            t0 = time.perf_counter()
            self._step(steps)
            out[f"cable{n_seg}_physics_steps_per_s"] = float(steps / (time.perf_counter() - t0))
        return {"metrics": out}

    # ----------------------------------------------------------- stability
    def stability(self, spec: B.StabilitySpec) -> Dict:
        results = {}
        for dt in spec.timesteps:
            try:
                res = self.cantilever(spec.length, spec.segment_length,
                                      B.CantileverSpec(settle_time=spec.settle_time),
                                      record=True, timestep=dt)
                tip, t = res.get("tip_z", np.array([])), res.get("t", np.array([]))
                last = tip[t > t[-1] - 1.0] if len(t) else np.array([])
                drop = float(res["clamp"][2] - np.mean(last)) if len(last) else float("nan")
                results[dt] = {"finite": bool(res["finite"]) and bool(np.isfinite(drop)),
                               "swing": float(np.std(last)) if len(last) else float("nan"),
                               "tip_drop": drop}
            except Exception as exc:
                results[dt] = {"finite": False, "swing": float("nan"),
                               "tip_drop": float("nan"), "error": repr(exc)}
        return results


# --------------------------------------------------------------- utilities
def _set_drive_target(stage, joint_path: str, value: float) -> None:
    from pxr import UsdPhysics
    joint = stage.GetPrimAtPath(joint_path)
    drive = UsdPhysics.DriveAPI.Get(joint, "linear")
    drive.GetTargetPositionAttr().Set(float(value))


def _reparent_fixed_joint(stage, joint_path: str, body0: str, body1: str) -> None:
    from pxr import UsdPhysics
    joint = UsdPhysics.FixedJoint.Get(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([body0])
    joint.CreateBody1Rel().SetTargets([body1])


def _arc_points(length: float, span: float, n: int, start) -> np.ndarray:
    """Positions of n segment origins along a circular arc of the given chord."""
    if span >= length:
        return np.array([[start[0] + i * length / n, start[1], start[2]] for i in range(n)])
    lo, hi = 1e-6, math.pi * 1.999
    for _ in range(100):
        phi = 0.5 * (lo + hi)
        chord = length * 2.0 * math.sin(phi / 2.0) / phi
        if chord > span:
            lo = phi
        else:
            hi = phi
    phi = 0.5 * (lo + hi)
    R = length / phi
    pts = []
    for i in range(n):
        a = -phi / 2.0 + phi * i / n
        pts.append([start[0] + R * (math.sin(a) + math.sin(phi / 2.0)),
                    start[1],
                    start[2] - R * (math.cos(a) - math.cos(phi / 2.0))])
    return np.array(pts)


def _place_chain(stage, paths: Sequence[str], points: np.ndarray) -> None:
    """Move the segment bodies onto the given polyline before the sim starts."""
    from pxr import Gf, UsdGeom
    for path, p, nxt in zip(paths, points, list(points[1:]) + [points[-1]]):
        prim = stage.GetPrimAtPath(path)
        xf = UsdGeom.Xformable(prim)
        ops = {op.GetOpName(): op for op in xf.GetOrderedXformOps()}
        t = ops.get("xformOp:translate")
        if t is None:
            t = xf.AddTranslateOp()
        t.Set(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])))
        d = np.asarray(nxt, dtype=float) - np.asarray(p, dtype=float)
        if np.linalg.norm(d) > 1e-9:
            d = d / np.linalg.norm(d)
            axis = np.cross([1.0, 0.0, 0.0], d)
            angle = math.degrees(math.acos(float(np.clip(d[0], -1.0, 1.0))))
            if np.linalg.norm(axis) > 1e-9:
                r = ops.get("xformOp:rotateXYZ")
                q = ops.get("xformOp:orient")
                if q is None and r is None:
                    q = xf.AddOrientOp()
                if q is not None:
                    axis = axis / np.linalg.norm(axis)
                    s = math.sin(math.radians(angle) / 2.0)
                    q.Set(Gf.Quatf(math.cos(math.radians(angle) / 2.0),
                                   Gf.Vec3f(*(axis * s))))
