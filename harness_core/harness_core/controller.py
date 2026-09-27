"""Cartesian compliance (admittance) controller for a velocity-controlled arm.

This is the kind of controller that runs at 500 Hz on a UR5e (force_mode) or in
FZI's ``cartesian_compliance_controller``: a stiff position-controlled robot is
made compliant by feeding the measured contact wrench back into the commanded
TCP velocity.

For every axis i (world-aligned task frame, [x y z rx ry rz]):

    v_i = kp_i * e_i + kf_i * (F_ext_i + F_des_i)      (compliant motion axis)
    v_i =              kf_i * (F_ext_i + F_des_i)      (force axis, selection_i = 1)

F_ext is the wrench the environment applies to the tool (measured, filtered,
dead-banded) and F_des the wrench the tool should apply to the environment, so
in steady contact F_ext = -F_des. On motion axes the pair (kp, kf) behaves like
a spring of stiffness kp / kf around the target pose.

The resulting TCP twist is mapped to joint velocities with damped least squares.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .config import ControllerParams
from .geometry import rotvec_from_mat
from .ur_kinematics import URKinematics


@dataclass
class ComplianceTarget:
    position: np.ndarray                      # TCP target position (world)
    rotation: np.ndarray                      # TCP target orientation (world, 3x3)
    wrench: np.ndarray = field(default_factory=lambda: np.zeros(6))      # desired wrench on env (world)
    selection: np.ndarray = field(default_factory=lambda: np.zeros(6))   # 1 = pure force axis
    stiffness_scale: np.ndarray = field(default_factory=lambda: np.ones(6))
    max_lin_vel: Optional[float] = None       # optional per-target speed limit


class ComplianceController:
    def __init__(self, kin: URKinematics, params: Optional[ControllerParams] = None, dt: float = 0.002):
        self.kin = kin
        self.p = params or ControllerParams()
        self.dt = dt
        self.reset()

    def reset(self) -> None:
        self.v_prev = np.zeros(6)
        self.w_filt = None
        self.protective_stop = False
        self.stop_reason = ""
        self.last_twist = np.zeros(6)
        self.last_wrench_world = np.zeros(6)

    # ------------------------------------------------------------------
    def tcp_wrench_world(self, q: np.ndarray, wrench_sensor: np.ndarray,
                         sensor_offset: float) -> np.ndarray:
        """Convert a tool0-axes wrench measured at the sensor face into a world-frame
        wrench about the TCP point."""
        T_tcp = self.kin.fk(q, tcp=True)
        R = T_tcp[:3, :3]
        f_w = R @ wrench_sensor[:3]
        t_w = R @ wrench_sensor[3:]
        # sensor point lies (tcp_offset - sensor_offset) behind the TCP along tool z
        r = -R[:, 2] * (self.kin.tcp_offset - sensor_offset)     # p_sensor - p_tcp
        t_tcp = t_w + np.cross(r, f_w)
        return np.concatenate([f_w, t_tcp])

    def _filter(self, w: np.ndarray) -> np.ndarray:
        if self.w_filt is None:
            self.w_filt = w.copy()
        else:
            a = 1.0 - np.exp(-2.0 * np.pi * self.p.wrench_filter_hz * self.dt)
            self.w_filt = self.w_filt + a * (w - self.w_filt)
        return self.w_filt

    @staticmethod
    def _deadband(v: np.ndarray, band: float) -> np.ndarray:
        n = np.linalg.norm(v)
        if n <= band:
            return np.zeros_like(v)
        return v * (1.0 - band / n)

    def update_from_sensor(self, q: np.ndarray, wrench_sensor: np.ndarray, sensor_offset: float,
                           target: ComplianceTarget) -> np.ndarray:
        """Control step from a raw (tool0-axes, sensor-point) wrench, one kinematics pass."""
        T, J = self.kin.fk_jacobian(q, tcp=True)
        R = T[:3, :3]
        f_w = R @ wrench_sensor[:3]
        r = -R[:, 2] * (self.kin.tcp_offset - sensor_offset)
        t_w = R @ wrench_sensor[3:] + np.array([r[1] * f_w[2] - r[2] * f_w[1],
                                                 r[2] * f_w[0] - r[0] * f_w[2],
                                                 r[0] * f_w[1] - r[1] * f_w[0]])
        return self.update(q, np.concatenate([f_w, t_w]), target, T=T, J=J)

    def update(self, q: np.ndarray, wrench_world_tcp: np.ndarray,
               target: ComplianceTarget, T: Optional[np.ndarray] = None,
               J: Optional[np.ndarray] = None) -> np.ndarray:
        """One control step. Returns joint velocity command (rad/s)."""
        p = self.p
        w = self._filter(np.asarray(wrench_world_tcp, dtype=float))
        self.last_wrench_world = w.copy()
        if (np.linalg.norm(w[:3]) > p.protective_stop_force
                or np.linalg.norm(w[3:]) > p.protective_stop_torque):
            if not self.protective_stop:
                self.stop_reason = (f"protective stop: |F|={np.linalg.norm(w[:3]):.1f} N, "
                                    f"|T|={np.linalg.norm(w[3:]):.2f} Nm")
            self.protective_stop = True
        if self.protective_stop:
            self.v_prev = np.zeros(6)
            self.last_twist = np.zeros(6)
            return np.zeros(6)

        if T is None or J is None:
            T, J = self.kin.fk_jacobian(q, tcp=True)
        e = np.zeros(6)
        e[:3] = target.position - T[:3, 3]
        e[3:] = rotvec_from_mat(target.rotation @ T[:3, :3].T)
        # saturate the pose error so a far-away target cannot build huge forces
        n_lin = np.linalg.norm(e[:3])
        if n_lin > 0.05:
            e[:3] *= 0.05 / n_lin
        n_rot = np.linalg.norm(e[3:])
        if n_rot > 0.5:
            e[3:] *= 0.5 / n_rot

        f_ext = np.concatenate([self._deadband(w[:3], p.force_deadband),
                                self._deadband(w[3:], p.torque_deadband)])
        kp = np.array([p.kp_lin] * 3 + [p.kp_rot] * 3) * target.stiffness_scale
        kf = np.array([p.kf_lin] * 3 + [p.kf_rot] * 3)
        sel = np.asarray(target.selection, dtype=float)
        v = (1.0 - sel) * kp * e + kf * (f_ext + np.asarray(target.wrench, dtype=float))

        # velocity and acceleration limits
        vmax = p.max_lin_vel if target.max_lin_vel is None else min(p.max_lin_vel, target.max_lin_vel)
        nv = np.linalg.norm(v[:3])
        if nv > vmax:
            v[:3] *= vmax / nv
        nw = np.linalg.norm(v[3:])
        if nw > p.max_rot_vel:
            v[3:] *= p.max_rot_vel / nw
        dv = v - self.v_prev
        max_dv_lin = p.max_lin_acc * self.dt
        max_dv_rot = p.max_rot_acc * self.dt
        n = np.linalg.norm(dv[:3])
        if n > max_dv_lin:
            dv[:3] *= max_dv_lin / n
        n = np.linalg.norm(dv[3:])
        if n > max_dv_rot:
            dv[3:] *= max_dv_rot / n
        v = self.v_prev + dv
        self.v_prev = v
        self.last_twist = v.copy()

        lam2 = p.damping_lambda ** 2
        qd = J.T @ np.linalg.solve(J @ J.T + lam2 * np.eye(6), v)
        return qd

    def clear_protective_stop(self) -> None:
        self.protective_stop = False
        self.stop_reason = ""
        self.v_prev = np.zeros(6)
