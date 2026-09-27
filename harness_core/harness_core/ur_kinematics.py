"""UR5e kinematics from the published UR DH parameters (numpy only).

Frame conventions follow ``ur_description`` so joint values, ``base_link`` and
``tool0`` mean the same thing as on a real UR5e driven by ``ur_robot_driver``:

* ``base_link``: ROS base frame of the robot.
* ``base``: UR controller base frame = ``base_link`` rotated by pi about z.
* DH chain starts at ``base``; the last DH frame is ``tool0`` (z out of the flange).
* ``tcp``: point between the gripper finger pads, ``tcp_offset`` along tool0 z.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .config import RobotParams
from .geometry import homogeneous, rot_z, rotvec_from_mat


def dh_transform(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    """Standard DH: Rz(theta) Tz(d) Tx(a) Rx(alpha)."""
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0.0, sa, ca, d],
        [0.0, 0.0, 0.0, 1.0],
    ])


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


class URKinematics:
    def __init__(self, robot: Optional[RobotParams] = None,
                 world_T_base_link: Optional[np.ndarray] = None):
        robot = robot or RobotParams()
        self.d = np.asarray(robot.dh_d, dtype=float)
        self.a = np.asarray(robot.dh_a, dtype=float)
        self.alpha = np.asarray(robot.dh_alpha, dtype=float)
        self.tcp_offset = float(robot.tcp_offset)
        self.world_T_base_link = np.eye(4) if world_T_base_link is None else np.asarray(world_T_base_link)
        self.base_link_T_base = homogeneous(rot_z(np.pi), np.zeros(3))
        self.world_T_base = self.world_T_base_link @ self.base_link_T_base
        self.tool0_T_tcp = homogeneous(np.eye(3), np.array([0.0, 0.0, self.tcp_offset]))
        self.lower = -np.asarray(robot.joint_limits, dtype=float)
        self.upper = np.asarray(robot.joint_limits, dtype=float)

    # ------------------------------------------------------------------ FK
    def frames(self, q: Sequence[float]) -> List[np.ndarray]:
        """World poses of DH frames 0..6 (frame 0 = ``base``, frame 6 = ``tool0``)."""
        T = self.world_T_base.copy()
        out = [T.copy()]
        for i in range(6):
            T = T @ dh_transform(q[i], self.d[i], self.a[i], self.alpha[i])
            out.append(T.copy())
        return out

    def fk(self, q: Sequence[float], tcp: bool = True) -> np.ndarray:
        T = self.frames(q)[-1]
        return T @ self.tool0_T_tcp if tcp else T

    # ------------------------------------------------------------ Jacobian
    def jacobian(self, q: Sequence[float], tcp: bool = True) -> np.ndarray:
        """Geometric Jacobian in the world frame: [v; w] = J qdot (v of the TCP/tool0 point)."""
        return self.fk_jacobian(q, tcp)[1]

    def fk_jacobian(self, q: Sequence[float], tcp: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """End pose and geometric Jacobian from a single forward pass."""
        frames = self.frames(q)
        T_end = frames[-1] @ self.tool0_T_tcp if tcp else frames[-1]
        p_end = T_end[:3, 3]
        J = np.zeros((6, 6))
        for i in range(6):
            z = frames[i][:3, 2]
            J[:3, i] = _cross(z, p_end - frames[i][:3, 3])
            J[3:, i] = z
        return T_end, J

    # ------------------------------------------------------------------ IK
    def ik(self, T_target: np.ndarray, q_init: Sequence[float], tcp: bool = True,
           iters: int = 200, tol: float = 1e-6, damping: float = 1e-3) -> np.ndarray:
        """Damped least-squares IK from an initial guess (stays in the same branch)."""
        q = np.array(q_init, dtype=float)
        for _ in range(iters):
            T = self.fk(q, tcp)
            err = np.concatenate([T_target[:3, 3] - T[:3, 3],
                                  rotvec_from_mat(T_target[:3, :3] @ T[:3, :3].T)])
            if np.linalg.norm(err) < tol:
                break
            J = self.jacobian(q, tcp)
            dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), err)
            q = np.clip(q + dq, self.lower, self.upper)
        return q
