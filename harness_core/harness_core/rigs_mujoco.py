"""MuJoCo implementation of the cross-simulator benchmark rigs (see benchmarks.py).

Each method builds a small scene, drives it and returns a trace (numpy arrays). All
metrics are computed in ``benchmarks.py`` so another engine only has to produce the
same traces. The wire is the same Cosserat-rod cable as in the cell, with the same
material parameters, contact parameters and solver settings.
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional

import numpy as np
import mujoco

from . import benchmarks as B
from .config import CellConfig
from .scene import ENV, SOLIMP, SOLREF, WIRE, _col, _f, _v


def _materials() -> str:
    return """  <asset>
    <material name="wire" rgba="0.96 0.55 0.10 1"/>
    <material name="fork" rgba="0.16 0.36 0.78 1"/>
    <material name="lip" rgba="0.98 0.82 0.18 1"/>
    <material name="board" rgba="0.80 0.69 0.52 1"/>
  </asset>
"""


def _wire_chain(cfg: CellConfig, n: int, seg: float, pin_root: bool,
                prefix: str = "wire", friction_scale: float = 1.0) -> str:
    """Chain of capsule segments with ball joints and the cable plugin, along +x."""
    w = cfg.wire
    r = w.radius
    seg_mass = math.pi * r * r * seg * w.density
    col = _col(WIRE)
    parts: List[str] = []
    for i in range(n):
        pos = "0 0 0" if i == 0 else f"{_f(seg)} 0 0"
        joint = ""
        if i > 0 or pin_root:
            joint = (f'<joint name="{prefix}_j{i}" type="ball" damping="{_f(w.joint_damping)}" '
                     f'armature="{_f(w.joint_armature)}"/>')
        parts.append(
            f'<body name="{prefix}_{i}" pos="{pos}">{joint}'
            f'<geom name="{prefix}_g{i}" type="capsule" size="{_v(r, seg / 2)}" pos="{_f(seg / 2)} 0 0" '
            f'quat="0.707107 0 -0.707107 0" mass="{_f(seg_mass)}" {col} condim="3" '
            f'friction="{_f(w.friction * friction_scale)} 0.005 0.0001" material="wire"/>'
            f'<plugin instance="wire"/>')
    parts.append(f'<site name="{prefix}_end" pos="{_f(seg)} 0 0" size="0.002" group="5"/>')
    return "".join(parts) + "</body>" * n


def _rig_xml(cfg: CellConfig, world: str, timestep: float, bend: float, twist: float,
             equality: str = "", actuator: str = "", sensor: str = "") -> str:
    return f"""<mujoco model="rig">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{_f(timestep)}" integrator="implicitfast" cone="elliptic" impratio="10"/>
  <size memory="64M"/>
  <extension>
    <plugin plugin="mujoco.elasticity.cable">
      <instance name="wire">
        <config key="twist" value="{_f(twist)}"/>
        <config key="bend" value="{_f(bend)}"/>
        <config key="vmax" value="0"/>
      </instance>
    </plugin>
  </extension>
{_materials()}  <default>
    <geom solref="{SOLREF}" solimp="{SOLIMP}"/>
    <default class="env">
      <geom {_col(ENV)} friction="0.35 0.005 0.0001" group="0"/>
    </default>
  </default>
  <worldbody>
{world}  </worldbody>
  <equality>{equality}</equality>
  <actuator>{actuator}</actuator>
  <sensor>{sensor}</sensor>
</mujoco>
"""


class MujocoRigs:
    """Benchmark rigs in MuJoCo. One instance per configuration."""

    name = "mujoco"

    def __init__(self, cfg: Optional[CellConfig] = None, bend_scale: float = 1.0,
                 friction_scale: float = 1.0):
        self.cfg = cfg.copy() if cfg is not None else CellConfig()
        self.bend_scale = bend_scale
        self.friction_scale = friction_scale

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
        return {"engine": "mujoco", "version": mujoco.__version__,
                "integrator": "implicitfast", "cone": "elliptic",
                "cable": "mujoco.elasticity.cable (Cosserat rod plugin)"}

    @staticmethod
    def _chain_points(model, data, n: int, seg: float, prefix: str = "wire") -> np.ndarray:
        bid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}_{i}") for i in range(n)]
        pts = data.xpos[bid].copy()
        end = data.xpos[bid[-1]] + data.xmat[bid[-1]].reshape(3, 3)[:, 0] * seg
        return np.vstack([pts, end])

    @staticmethod
    def _max_speed(data) -> float:
        return float(np.max(np.abs(data.qvel))) if data.qvel.size else 0.0

    # ---------------------------------------------------------- cantilever
    def cantilever(self, length: float, segment_length: float,
                   spec: B.CantileverSpec, record: bool = False) -> Dict:
        n = max(2, int(round(length / segment_length)))
        seg = length / n
        h = spec.clamp_height
        world = (f'    <body name="clamp" pos="0 0 {_f(h)}">\n'
                 f'      <geom name="clamp_g" class="env" type="box" size="0.012 0.012 0.012" pos="-0.014 0 0"/>\n'
                 f'      {_wire_chain(self.cfg, n, seg, pin_root=False, friction_scale=self.friction_scale)}\n'
                 f'    </body>\n')
        xml = _rig_xml(self.cfg, world, self.cfg.sim.timestep, self.bend, self.twist)
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        t_hist, tip_hist = [], []
        n_steps = int(spec.settle_time / model.opt.timestep)
        sample = max(1, int(round(0.002 / model.opt.timestep)))
        for k in range(n_steps):
            mujoco.mj_step(model, data)
            if record and k % sample == 0:
                pts = self._chain_points(model, data, n, seg)
                t_hist.append(data.time)
                tip_hist.append(pts[-1, 2])
            if not record and k % 200 == 0 and k > 0 and self._max_speed(data) < spec.still_speed:
                break
        pts = self._chain_points(model, data, n, seg)
        out = {
            "points": pts,
            "clamp": np.array([0.0, 0.0, h]),
            "settled_speed": self._max_speed(data),
            "finite": bool(np.all(np.isfinite(pts))),
            "n_segments": n,
            "segment_length": seg,
            "length": length,
        }
        if record:
            out["t"] = np.array(t_hist)
            out["tip_z"] = np.array(tip_hist)
        return out

    # ----------------------------------------------------------------- sag
    def sag(self, length: float, span: float, segment_length: float, spec: B.SagSpec) -> Dict:
        n = max(3, int(round(length / segment_length)))
        seg = length / n
        h = spec.clamp_height
        world = (f'    <site name="support_b" pos="{_v(span, 0, h)}" size="0.002" group="5"/>\n'
                 f'    <body name="clamp" pos="0 0 {_f(h)}">\n'
                 f'      {_wire_chain(self.cfg, n, seg, pin_root=True, friction_scale=self.friction_scale)}\n'
                 f'    </body>\n')
        equality = '\n    <connect name="far_end" site1="wire_end" site2="support_b" solref="0.02 1"/>\n  '
        xml = _rig_xml(self.cfg, world, self.cfg.sim.timestep, self.bend, self.twist, equality=equality)
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        # start from a circular arc through both supports so the constraint only has to
        # close a small gap instead of yanking a straight wire into place
        self._set_arc(model, data, n, seg, span)
        mujoco.mj_forward(model, data)
        for k in range(int(spec.settle_time / model.opt.timestep)):
            mujoco.mj_step(model, data)
            if k % 200 == 0 and k > 0 and self._max_speed(data) < 2e-3:
                break
        pts = self._chain_points(model, data, n, seg)
        return {
            "points": pts,
            "span": float(np.linalg.norm(pts[-1, :2] - pts[0, :2])),
            "nominal_span": span,
            "length": length,
            "settled_speed": self._max_speed(data),
            "finite": bool(np.all(np.isfinite(pts))),
        }

    def _set_arc(self, model, data, n: int, seg: float, span: float) -> None:
        """Bend the chain into a circular arc of the given chord (sagging downwards)."""
        length = n * seg
        if span >= length:
            return
        lo, hi = 1e-6, math.pi * 1.999
        for _ in range(100):                      # chord = 2 R sin(phi/2), length = R phi
            phi = 0.5 * (lo + hi)
            chord = length * 2.0 * math.sin(phi / 2.0) / phi
            if chord > span:
                lo = phi
            else:
                hi = phi
        phi = 0.5 * (lo + hi)
        dphi = phi / n
        # rotate each joint by -dphi about the local y axis, and start tangent at -phi/2
        q = np.array([math.cos(dphi / 2), 0.0, -math.sin(dphi / 2), 0.0])
        q0 = np.array([math.cos(-phi / 4), 0.0, -math.sin(-phi / 4), 0.0])
        for i in range(n):
            adr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"wire_j{i}")]
            data.qpos[adr:adr + 4] = q0 if i == 0 else q

    # --------------------------------------------------------------- swing
    def swing(self, spec: B.SwingSpec) -> Dict:
        res = self.cantilever(spec.length, spec.segment_length,
                              B.CantileverSpec(clamp_height=spec.clamp_height,
                                               settle_time=spec.record_time),
                              record=True)
        res["spec"] = "swing"
        return res

    # ---------------------------------------------------------------- snap
    def snap(self, spec: B.SnapSpec, lateral_offset: float = 0.0) -> Dict:
        """Press a gripped wire down into a fork and pull it back out.

        This mirrors what the robot does: a rigid hand holds the wire a few cm from
        the fork, presses it past the barbed jaws, releases the load and pulls up
        again. Measured: the vertical force at the hand through both events.
        """
        cfg = self.cfg
        f = cfg.fork
        n = max(4, int(round(spec.wire_length / spec.segment_length)))
        seg = spec.wire_length / n
        x_fork = 0.03                                 # fork this far along the wire
        slot_z = f.post_height + cfg.wire.radius      # wire centre height when seated
        z_high = f.post_height + f.prong_height + 0.012
        z_press = slot_z - spec.press_depth * 0.0     # target: the slot bottom
        fork = _fork_body(cfg, "fork0").replace('pos="0 0 0"', f'pos="{_v(x_fork, lateral_offset, 0)}"', 1)
        world = (
            f'    <geom name="ground" class="env" type="box" size="0.4 0.4 0.02" pos="0 0 -0.02" material="board"/>\n'
            f'{fork}'
            f'    <body name="hand" pos="{_v(0, 0, z_high)}">\n'
            f'      <joint name="hand_z" type="slide" axis="0 0 1" range="-0.05 0.3" damping="2"/>\n'
            f'      <inertial pos="0 0 0" mass="0.2" diaginertia="1e-4 1e-4 1e-4"/>\n'
            f'      <site name="hand_site" pos="0 0 0" size="0.002" group="5"/>\n'
            f'      {_wire_chain(cfg, n, seg, pin_root=False, friction_scale=self.friction_scale)}\n'
            f'    </body>\n')
        actuator = ('\n    <general name="hand_act" joint="hand_z" gaintype="fixed" biastype="affine" '
                    'gainprm="4000" biasprm="0 -4000 -120" ctrlrange="-0.3 0.3"/>\n  ')
        sensor = '\n    <force name="hand_force" site="hand_site"/>\n  '
        xml = _rig_xml(cfg, world, cfg.sim.timestep, self.bend, self.twist,
                       actuator=actuator, sensor=sensor)
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        dt = model.opt.timestep
        wire_bid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"wire_{i}") for i in range(n)]
        hand_qadr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hand_z")]
        i_fork = min(n - 1, max(0, int(round(x_fork / seg))))   # wire body nearest the fork

        for _ in range(int(0.5 / dt)):               # settle, then take the force baseline
            mujoco.mj_step(model, data)
        f0 = float(data.sensordata[2])
        t_hist, z_hist, f_hist, phase_hist = [], [], [], []

        def run(target_offset: float, speed: float, phase: int):
            start = float(data.qpos[hand_qadr])
            steps = max(1, int(abs(target_offset - start) / max(speed, 1e-6) / dt))
            for k in range(steps):
                data.ctrl[0] = start + (target_offset - start) * (k + 1) / steps
                mujoco.mj_step(model, data)
                t_hist.append(data.time)
                z_hist.append(float(data.xpos[wire_bid[i_fork]][2]))
                # force the fork/wire applies to the hand along +z (positive = pushing back)
                f_hist.append(-(float(data.sensordata[2]) - f0))
                phase_hist.append(phase)

        run(z_press - z_high, spec.press_speed, 0)   # press the wire into the slot
        for _ in range(int(spec.hold_time / dt)):
            mujoco.mj_step(model, data)
        wire_z_seated = float(data.xpos[wire_bid[i_fork]][2])
        run(0.0, spec.pull_speed, 1)                 # pull straight back up
        for _ in range(int(0.3 / dt)):
            mujoco.mj_step(model, data)
        wire_after = data.xpos[wire_bid][:, 2]
        # the fork kept the wire if it is still down at the slot after the hand left
        retained = bool(data.xpos[wire_bid[i_fork]][2] < f.post_height + f.prong_height - 0.004)
        return {
            "t": np.array(t_hist),
            "z": np.array(z_hist),
            "fz": np.array(f_hist),
            "phase": np.array(phase_hist),
            "retained": retained,
            "wire_z_seated": wire_z_seated,
            "slot_z": slot_z,
            "lip_z": f.post_height + f.prong_height,
            "lip_radius": f.lip_radius,
            "wire_radius": cfg.wire.radius,
            "lateral_offset": lateral_offset,
            "finite": bool(np.all(np.isfinite(wire_after))),
        }

    # --------------------------------------------------------------- slide
    def slide(self, spec: B.SlideSpec) -> Dict:
        """Pull a wire lying on the board along its own axis; drag force -> friction.

        The hand holds the wire a few mm clear of the board so that the held end does
        not get squeezed between a rigid hand and a rigid board; the rest of the wire
        rests under its own weight. The normal load is measured from the contacts
        rather than assumed, so the friction coefficient is a real ratio.
        """
        cfg = self.cfg
        n = max(4, int(round(spec.wire_length / spec.segment_length)))
        seg = spec.wire_length / n
        r = cfg.wire.radius
        world = (
            f'    <geom name="board" class="env" type="box" size="0.4 0.2 0.02" pos="0.1 0 -0.02" material="board"/>\n'
            f'    <body name="hand" pos="{_v(0, 0, r + 0.004)}">\n'
            f'      <joint name="hand_x" type="slide" axis="1 0 0" range="-0.05 0.5" damping="2"/>\n'
            f'      <inertial pos="0 0 0" mass="0.2" diaginertia="1e-4 1e-4 1e-4"/>\n'
            f'      <site name="hand_site" pos="0 0 0" size="0.002" group="5"/>\n'
            f'      {_wire_chain(cfg, n, seg, pin_root=False, friction_scale=self.friction_scale)}\n'
            f'    </body>\n')
        actuator = ('\n    <general name="hand_act" joint="hand_x" gaintype="fixed" biastype="affine" '
                    'gainprm="4000" biasprm="0 -4000 -120" ctrlrange="-0.1 0.6"/>\n  ')
        sensor = '\n    <force name="hand_force" site="hand_site"/>\n  '
        xml = _rig_xml(cfg, world, cfg.sim.timestep, self.bend, self.twist,
                       actuator=actuator, sensor=sensor)
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        dt = model.opt.timestep
        for _ in range(int(spec.settle_time / dt)):
            mujoco.mj_step(model, data)
        f0 = float(data.sensordata[0])
        steps = max(1, int(spec.distance / spec.speed / dt))
        board_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "board")
        t_hist, fx_hist, fn_hist = [], [], []
        f6 = np.zeros(6)
        for k in range(steps):
            data.ctrl[0] = spec.distance * (k + 1) / steps
            mujoco.mj_step(model, data)
            normal = 0.0
            for i in range(data.ncon):
                c = data.contact[i]
                if board_gid in (c.geom1, c.geom2):
                    mujoco.mj_contactForce(model, data, i, f6)
                    normal += abs(float(f6[0]))
            t_hist.append(data.time)
            fx_hist.append(float(data.sensordata[0]) - f0)   # drag force applied to the hand
            fn_hist.append(normal)
        wire_mass = math.pi * r * r * spec.wire_length * cfg.wire.density
        fn = np.array(fn_hist)
        return {
            "t": np.array(t_hist),
            "fx": np.array(fx_hist),
            "fn": fn,
            "normal_force": float(np.mean(fn[len(fn) // 3:])) if len(fn) else 0.0,
            "wire_weight": wire_mass * B.GRAVITY,
            "nominal_friction": cfg.wire.friction * self.friction_scale,
            "finite": bool(np.all(np.isfinite(fx_hist))),
        }

    # ----------------------------------------------------------- step rate
    def step_rate(self, spec: B.StepRateSpec) -> Dict:
        from .cell import HarnessCell
        out: Dict[str, float] = {}
        cell = HarnessCell(self.cfg, seed=0, randomize=False)
        dt = self.cfg.sim.control_dt
        n = int(spec.seconds_of_sim / dt)
        t0 = time.perf_counter()
        for _ in range(n):
            cell.sim.step()
        wall = time.perf_counter() - t0
        out["cell_realtime_factor"] = float(spec.seconds_of_sim / wall)
        out["cell_physics_steps_per_s"] = float(n * cell.sim.n_substeps / wall)
        out["cell_wire_segments"] = float(cell.sim.instance.n_segments)
        # cable-only scaling: cost per physics step against the number of segments
        for n_seg in (12, 24, 48):
            length = n_seg * 0.015
            res = self._time_chain(length, 0.015, seconds=0.5)
            out[f"cable{n_seg}_physics_steps_per_s"] = res
        return {"metrics": out}

    def _time_chain(self, length: float, seg_len: float, seconds: float) -> float:
        n = max(2, int(round(length / seg_len)))
        seg = length / n
        world = (f'    <body name="clamp" pos="0 0 0.5">\n'
                 f'      {_wire_chain(self.cfg, n, seg, pin_root=False)}\n'
                 f'    </body>\n')
        xml = _rig_xml(self.cfg, world, self.cfg.sim.timestep, self.bend, self.twist)
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        steps = int(seconds / model.opt.timestep)
        for _ in range(200):
            mujoco.mj_step(model, data)
        t0 = time.perf_counter()
        for _ in range(steps):
            mujoco.mj_step(model, data)
        return float(steps / (time.perf_counter() - t0))

    # ----------------------------------------------------------- stability
    def stability(self, spec: B.StabilitySpec) -> Dict:
        results = {}
        base_dt = self.cfg.sim.timestep
        for dt in spec.timesteps:
            cfg = self.cfg.copy()
            cfg.sim.timestep = dt
            rig = MujocoRigs(cfg, self.bend_scale, self.friction_scale)
            try:
                res = rig.cantilever(spec.length, spec.segment_length,
                                     B.CantileverSpec(settle_time=spec.settle_time), record=True)
                tip, t = res.get("tip_z", np.array([])), res.get("t", np.array([]))
                last = tip[t > t[-1] - 1.0] if len(t) else np.array([])   # average out the swing
                tip_drop = float(res["clamp"][2] - np.mean(last)) if len(last) else float("nan")
                results[dt] = {"finite": bool(res["finite"]) and bool(np.isfinite(tip_drop)),
                               "swing": float(np.std(last)) if len(last) else float("nan"),
                               "tip_drop": tip_drop}
            except Exception as exc:                     # a diverged model can fail to step
                results[dt] = {"finite": False, "swing": float("nan"),
                               "tip_drop": float("nan"), "error": repr(exc)}
        self.cfg.sim.timestep = base_dt
        return results


def _fork_body(cfg: CellConfig, name: str) -> str:
    """One fork, identical to the cell's, expressed relative to its parent body."""
    from types import SimpleNamespace
    from .layout import Pose2
    from .scene import _fork
    xml = _fork(cfg, SimpleNamespace(board_z=0.0), 0, Pose2(0.0, 0.0, 0.0))
    return xml.replace('<body name="fork0"', f'<body name="{name}"')
