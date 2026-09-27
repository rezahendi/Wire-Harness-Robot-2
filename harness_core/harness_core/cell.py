"""The robot cell as seen by a task-level program: simulated hardware + the
500 Hz compliance controller that would run on the robot controller.

Task code (the scripted expert, a learned policy, or the ROS task node) only
sets Cartesian compliance targets and gripper commands and reads observations.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .config import CellConfig
from .controller import ComplianceController, ComplianceTarget
from .geometry import tool_down_rotation, tool_yaw
from .layout import CellInstance
from .sim import HarnessSim


def quintic(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return s * s * s * (10 - 15 * s + 6 * s * s)


class HarnessCell:
    def __init__(self, cfg: Optional[CellConfig] = None, seed: Optional[int] = None,
                 randomize: Optional[bool] = None, instance: Optional[CellInstance] = None):
        self.cfg = cfg.copy() if cfg is not None else CellConfig()
        self.sim = HarnessSim(self.cfg, seed=seed, randomize=randomize, instance=instance)
        self.kin = self.sim.kin
        self.ctrl = ComplianceController(self.kin, self.cfg.controller, self.cfg.sim.control_dt)
        self._after_reset()

    def reset(self, seed: Optional[int] = None, randomize: Optional[bool] = None,
              instance: Optional[CellInstance] = None) -> None:
        self.sim.reset(seed=seed, randomize=randomize, instance=instance)
        self._after_reset()

    def _after_reset(self) -> None:
        self.ctrl.reset()
        self.sim.zero_ft()
        p, R = self.sim.tcp_pose()
        self.target = ComplianceTarget(position=p, rotation=R)
        self.mode = "compliance"
        self._traj = None
        self._jtraj = None
        self._vel_cmd = np.zeros(6)
        self.gripper_cmd = self.sim.gripper_opening()

    # ------------------------------------------------------------ commands
    def set_pose_target(self, position: np.ndarray, rotation: Optional[np.ndarray] = None,
                        yaw: Optional[float] = None, wrench: Optional[np.ndarray] = None,
                        selection: Optional[np.ndarray] = None,
                        stiffness_scale: Optional[np.ndarray] = None,
                        max_lin_vel: Optional[float] = None) -> None:
        if rotation is None:
            rotation = tool_down_rotation(yaw) if yaw is not None else self.target.rotation
        self.target = ComplianceTarget(
            position=np.asarray(position, dtype=float).copy(),
            rotation=np.asarray(rotation, dtype=float).copy(),
            wrench=np.zeros(6) if wrench is None else np.asarray(wrench, dtype=float).copy(),
            selection=np.zeros(6) if selection is None else np.asarray(selection, dtype=float).copy(),
            stiffness_scale=np.ones(6) if stiffness_scale is None else np.asarray(stiffness_scale, dtype=float).copy(),
            max_lin_vel=max_lin_vel)
        self.mode = "compliance"

    def set_gripper(self, opening: float) -> None:
        self.gripper_cmd = float(opening)
        self.sim.set_gripper(opening)

    def move_joints(self, q_goal: np.ndarray, duration: Optional[float] = None) -> None:
        """Joint-space move with a quintic profile."""
        q0 = self.sim.q
        q_goal = np.asarray(q_goal, dtype=float)
        if duration is None:
            duration = max(0.5, float(np.max(np.abs(q_goal - q0))) / 0.8)
        self._traj = (self.sim.time, duration, q0, q_goal)
        self.mode = "joint"

    def follow_joint_trajectory(self, times: np.ndarray, positions: np.ndarray,
                                velocities: Optional[np.ndarray] = None) -> None:
        """Follow a joint trajectory (times relative to now, like a FollowJointTrajectory
        goal) with piecewise cubic Hermite interpolation starting from the current state."""
        times = np.asarray(times, dtype=float).reshape(-1)
        positions = np.asarray(positions, dtype=float).reshape(-1, 6)
        t = np.concatenate([[0.0], times])
        q = np.vstack([self.sim.q, positions])
        if velocities is not None and len(velocities) == len(times):
            v = np.vstack([self.sim.qd, np.asarray(velocities, dtype=float).reshape(-1, 6)])
        else:
            v = np.zeros_like(q)
            for k in range(1, len(q) - 1):
                h0, h1 = max(t[k] - t[k - 1], 1e-6), max(t[k + 1] - t[k], 1e-6)
                v[k] = 0.5 * ((q[k] - q[k - 1]) / h0 + (q[k + 1] - q[k]) / h1)
        self._jtraj = (self.sim.time, t, q, v)
        self.mode = "trajectory"

    def set_joint_velocity_command(self, qd: np.ndarray) -> None:
        """Direct joint velocity mode (like forward_velocity_controller)."""
        self._vel_cmd = np.asarray(qd, dtype=float).reshape(6).copy()
        self.mode = "velocity"

    def trajectory_done(self) -> bool:
        return self._traj is None and self._jtraj is None

    def _sample_jtraj(self, now: float):
        t0, t, q, v = self._jtraj
        tau = now - t0
        if tau >= t[-1]:
            return q[-1], np.zeros(6), True
        k = int(np.searchsorted(t, tau, side="right") - 1)
        k = min(max(k, 0), len(t) - 2)
        h = max(t[k + 1] - t[k], 1e-6)
        s = (tau - t[k]) / h
        h00, h10, h01, h11 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s, -2 * s**3 + 3 * s**2, s**3 - s**2
        d00, d10, d01, d11 = 6 * s**2 - 6 * s, 3 * s**2 - 4 * s + 1, -6 * s**2 + 6 * s, 3 * s**2 - 2 * s
        pos = h00 * q[k] + h10 * h * v[k] + h01 * q[k + 1] + h11 * h * v[k + 1]
        vel = (d00 * q[k] + d10 * h * v[k] + d01 * q[k + 1] + d11 * h * v[k + 1]) / h
        return pos, vel, False

    def hold_current_pose(self) -> None:
        """Switch to compliance mode holding the current TCP pose."""
        p, R = self.sim.tcp_pose()
        self.target = ComplianceTarget(position=p, rotation=R)
        self.mode = "compliance"
        self._traj = None
        self._jtraj = None
        self.ctrl.reset()

    # ---------------------------------------------------------------- step
    def step(self, n: int = 1) -> None:
        sim = self.sim
        for _ in range(n):
            if self.mode == "joint" and self._traj is not None:
                t0, T, q0, q1 = self._traj
                s = (sim.time - t0) / T
                sim.set_joint_position(q0 + (q1 - q0) * quintic(s))
                if s >= 1.0:
                    self.hold_current_pose()
            elif self.mode == "trajectory" and self._jtraj is not None:
                q, qd, done = self._sample_jtraj(sim.time)
                sim.set_joint_position(q, qd)
                if done:
                    self.hold_current_pose()
            elif self.mode == "velocity":
                sim.set_joint_velocity(self._vel_cmd)
                # keep the F/T filter and protective stop logic alive
                self.ctrl._filter(self.ctrl.tcp_wrench_world(sim.q, sim.ft_wrench(noise=True),
                                                             self.cfg.robot.ft_thickness))
            else:
                qd = self.ctrl.update_from_sensor(sim.q, sim.ft_wrench(noise=True),
                                                  self.cfg.robot.ft_thickness, self.target)
                sim.set_joint_velocity(qd)
            sim.step()

    def step_time(self, seconds: float) -> None:
        self.step(max(1, int(round(seconds / self.cfg.sim.control_dt))))

    # ---------------------------------------------------------- observation
    def wrench_world(self) -> np.ndarray:
        """Last filtered external wrench at the TCP (world frame) seen by the controller."""
        return self.ctrl.last_wrench_world.copy()

    def observe(self, rng: Optional[np.random.Generator] = None, noisy: bool = True) -> Dict[str, np.ndarray]:
        """Everything a task-level program may use (perception outputs are noisy)."""
        sim = self.sim
        rng = rng if rng is not None else sim.rng
        noise = self.cfg.noise
        p, R = sim.tcp_pose()
        cable = sim.cable_points()
        hp, hR = sim.holder_seat_pose()
        cp, cR = sim.connector_pose()
        if noisy:
            cable = cable + rng.normal(0.0, noise.cable_point_std, cable.shape)
            hp = hp + np.concatenate([rng.normal(0.0, noise.holder_pose_std, 2), [0.0]])
            cp = cp + rng.normal(0.0, noise.cable_point_std, 3)
        forks = []
        for i in range(sim.n_forks):
            fp, fR = sim.fork_pose(i)
            forks.append(np.array([fp[0], fp[1], fp[2], tool_yaw(fR)]))
        return {
            "time": np.array([sim.time]),
            "q": sim.q,
            "qd": sim.qd,
            "tcp_pos": p,
            "tcp_rot": R,
            "tcp_yaw": np.array([tool_yaw(R)]),
            "wrench": self.wrench_world(),
            "gripper": np.array([sim.gripper_opening()]),
            "cable": cable,
            "connector_pos": cp,
            "connector_rot": cR,
            "forks": np.array(forks),
            "holder_pos": hp,
            "holder_yaw": np.array([tool_yaw(hR)]),
            "anchor_pos": sim.data.site_xpos[sim.sid["anchor"]].copy(),
            "board_z": np.array([sim.instance.board_z]),
            "protective_stop": np.array([float(self.ctrl.protective_stop)]),
        }
