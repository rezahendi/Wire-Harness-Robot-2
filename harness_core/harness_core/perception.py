"""Geometric checks on (possibly noisy) perceived cable keypoints.

On the real cell these would run on the output of a cable tracker; in simulation
they run on the noisy ground-truth keypoints published by the simulator.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from .config import ForkParams
from .geometry import closest_point_on_polyline, polyline_arclength, rot_z


def cable_crossing_in_fork(cable: np.ndarray, fork_pose: np.ndarray, fork: ForkParams,
                           board_z: float, y_margin: float = 0.002) -> Dict[str, float]:
    """Crossing of the cable with the plane through the fork slot.

    fork_pose = (x, y, z_board_top, yaw). Returns local y / z (above the board)
    of the crossing closest to the slot and whether that crossing lies inside
    the slot below the lips.
    """
    x, y, z, yaw = fork_pose
    R = rot_z(yaw)
    local = (cable - np.array([x, y, z])) @ R          # fork frame
    s_all = polyline_arclength(cable)
    best = None
    for k in range(len(local) - 1):
        a, b = local[k], local[k + 1]
        if a[0] * b[0] <= 0.0 and abs(b[0] - a[0]) > 1e-9:
            t = -a[0] / (b[0] - a[0])
            p = a + t * (b - a)
            score = abs(p[1]) + max(0.0, p[2] - (fork.post_height + fork.prong_height))
            if best is None or score < best[0]:
                best = (score, p, s_all[k] + t * (s_all[k + 1] - s_all[k]))
    if best is None:
        return {"inside": False, "y": np.inf, "z": np.inf, "s": np.nan}
    _, p, s = best
    lip_bottom = fork.post_height + fork.prong_height - 2.0 * fork.lip_radius
    inside = (abs(p[1]) < fork.slot_width / 2 + y_margin
              and fork.post_height - 0.003 < p[2] < lip_bottom + 0.0015)
    return {"inside": bool(inside), "y": float(p[1]), "z": float(p[2]), "s": float(s)}


def arclength_near(cable: np.ndarray, point: np.ndarray) -> float:
    """Arc length of the cable point closest to `point` (3D)."""
    _, s, _, _ = closest_point_on_polyline(cable, np.asarray(point, dtype=float))
    return float(s)
