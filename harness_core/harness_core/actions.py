"""Policy-level action interface shared by the expert, learned policies and ROS.

Action (5,) in [-1, 1]:
    a[0:3]  TCP position increment (world x, y, z), scaled by ``max_dpos`` per step
    a[3]    TCP yaw increment, scaled by ``max_dyaw`` per step (tool stays vertical)
    a[4]    gripper: -1 fully open ... +1 closed with full grip force

The interface integrates increments into a Cartesian compliance target. The
target may lead the measured TCP by at most ``max_lead`` so pushing into
contact produces a bounded force (≈ stiffness · lead) instead of winding up.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import tool_down_rotation, wrap_angle


@dataclass
class ActionSpec:
    max_dpos: float = 0.01       # m per policy step (0.2 m/s at 20 Hz)
    max_dyaw: float = 0.15       # rad per policy step
    max_lead: float = 0.02       # m, anti wind-up between target and TCP (~60 N at contact)
    max_yaw_lead: float = 0.35
    max_opening: float = 0.06
    z_min_above_board: float = -0.005
    workspace_min: tuple = (0.15, -0.55, 0.0)
    workspace_max: tuple = (0.80, 0.55, 0.45)


def gripper_opening_from_action(g: float, max_opening: float) -> float:
    g = float(np.clip(g, -1.0, 1.0))
    return max_opening * (1.0 - g) / 2.0


def gripper_action_from_opening(opening: float, max_opening: float) -> float:
    return float(np.clip(1.0 - 2.0 * opening / max_opening, -1.0, 1.0))


class ActionInterface:
    def __init__(self, spec: ActionSpec, board_z: float):
        self.spec = spec
        self.board_z = board_z
        self.target_pos = np.zeros(3)
        self.target_yaw = 0.0
        self.gripper = 0.0

    def reset(self, tcp_pos: np.ndarray, tcp_yaw: float, gripper_opening: float) -> None:
        self.target_pos = np.asarray(tcp_pos, dtype=float).copy()
        self.target_yaw = float(tcp_yaw)
        self.gripper = gripper_opening

    def apply(self, action: np.ndarray, tcp_pos: np.ndarray, tcp_yaw: float) -> None:
        s = self.spec
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        tp = self.target_pos + a[:3] * s.max_dpos
        lead = np.clip(tp - tcp_pos, -s.max_lead, s.max_lead)
        tp = tcp_pos + lead
        tp = np.clip(tp, np.array(s.workspace_min), np.array(s.workspace_max))
        tp[2] = max(tp[2], self.board_z + s.z_min_above_board)
        self.target_pos = tp
        yaw = self.target_yaw + a[3] * s.max_dyaw
        dyaw = wrap_angle(yaw - tcp_yaw)
        dyaw = float(np.clip(dyaw, -s.max_yaw_lead, s.max_yaw_lead))
        self.target_yaw = tcp_yaw + dyaw
        self.gripper = gripper_opening_from_action(a[4], s.max_opening)

    @property
    def target_rotation(self) -> np.ndarray:
        return tool_down_rotation(self.target_yaw)
