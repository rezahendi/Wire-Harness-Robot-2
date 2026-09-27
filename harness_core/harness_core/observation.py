"""Flat observation vector shared by the Gymnasium env and the ROS policy runner.

The same function turns the observation dict produced by ``HarnessCell.observe()``
(simulation) or assembled from ROS topics (``harness_task``) into the vector a
learned policy consumes, so a policy trained in the env runs unchanged on ROS.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from .config import CellConfig
from .geometry import polyline_arclength, polyline_point_at
from .perception import cable_crossing_in_fork


def obs_layout(n_forks: int, n_cable_points: int = 16) -> Dict[str, slice]:
    sizes = [
        ("tcp_pos", 3), ("tcp_yaw_sincos", 2), ("gripper", 1), ("wrench", 6),
        ("target_lead", 4), ("q", 6),
        ("cable", 3 * n_cable_points),
        ("connector_pos", 3), ("connector_axis", 3),
        ("forks", 4 * n_forks), ("holder", 4), ("anchor", 2),
        ("progress", n_forks + 1), ("time", 1),
    ]
    layout, i = {}, 0
    for name, n in sizes:
        layout[name] = slice(i, i + n)
        i += n
    return layout


def obs_dim(layout: Dict[str, slice]) -> int:
    return max(sl.stop for sl in layout.values())


def resample_polyline(points: np.ndarray, n: int) -> np.ndarray:
    s = polyline_arclength(points)
    return np.array([polyline_point_at(points, q)[0] for q in np.linspace(0.0, s[-1], n)])


def perceived_progress(obs: Dict[str, np.ndarray], cfg: CellConfig):
    """Fork routing flags and connector-seated flag from perception only."""
    bz = float(obs["board_z"][0])
    forks = [bool(cable_crossing_in_fork(obs["cable"], f, cfg.fork, bz)["inside"]) for f in obs["forks"]]
    seated = bool(np.linalg.norm(obs["connector_pos"][:2] - obs["holder_pos"][:2]) < 0.005
                  and abs(obs["connector_pos"][2] - obs["holder_pos"][2]) < 0.003)
    return forks, seated


def flatten_obs(obs: Dict[str, np.ndarray], cfg: CellConfig, layout: Dict[str, slice],
                n_cable_points: int = 16, max_episode_time: float = 150.0,
                elapsed: float = None) -> np.ndarray:
    """Observation dict -> float32 vector (forces in N/10, torques in N m, lengths in m)."""
    L = layout
    v = np.zeros(obs_dim(layout), dtype=np.float64)
    yaw = float(obs["tcp_yaw"][0])
    v[L["tcp_pos"]] = obs["tcp_pos"]
    v[L["tcp_yaw_sincos"]] = [np.sin(yaw), np.cos(yaw)]
    v[L["gripper"]] = obs["gripper"]
    v[L["wrench"]] = np.concatenate([obs["wrench"][:3] / 10.0, obs["wrench"][3:]])
    dyaw = np.arctan2(np.sin(obs["target_yaw"][0] - yaw), np.cos(obs["target_yaw"][0] - yaw))
    v[L["target_lead"]] = np.concatenate([obs["target_pos"] - obs["tcp_pos"], [dyaw]])
    v[L["q"]] = obs["q"]
    v[L["cable"]] = resample_polyline(obs["cable"], n_cable_points).reshape(-1)
    v[L["connector_pos"]] = obs["connector_pos"]
    v[L["connector_axis"]] = obs["connector_rot"][:, 0]
    v[L["forks"]] = np.concatenate([[f[0], f[1], np.sin(f[3]), np.cos(f[3])] for f in obs["forks"]])
    hy = float(obs["holder_yaw"][0])
    v[L["holder"]] = [obs["holder_pos"][0], obs["holder_pos"][1], np.sin(hy), np.cos(hy)]
    v[L["anchor"]] = obs["anchor_pos"][:2]
    forks, seated = perceived_progress(obs, cfg)
    v[L["progress"]] = [float(f) for f in forks] + [float(seated)]
    t = float(obs["time"][0]) if elapsed is None else float(elapsed)
    v[L["time"]] = [t / max_episode_time]
    return v.astype(np.float32)
