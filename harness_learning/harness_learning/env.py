"""Gymnasium environment: force-guided wire-harness routing with a UR5e-class arm.

Action (5,), all in [-1, 1], applied at 20 Hz on top of the 500 Hz Cartesian
compliance controller (see ``harness_core.actions``):
    [dx, dy, dz, dyaw, gripper]

Observation: flat float32 vector, layout in ``HarnessRoutingEnv.obs_layout``
(name -> slice). It contains proprioception, the payload-compensated wrist
wrench, the (noisy) perceived cable keypoints, connector and fixture poses and
perceived task progress. Forces are in N / 10, torques in N m, lengths in m.

Reward (weights configurable): +1 per fork the wire snaps into, -1 per fork it
leaves, +2 when the connector is seated, +5 on success, a small time penalty,
a penalty for large contact forces and -5 for a protective stop.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from harness_core.actions import ActionInterface, ActionSpec
from harness_core.cell import HarnessCell
from harness_core.config import CellConfig
from harness_core.observation import flatten_obs, obs_dim, obs_layout, resample_polyline  # noqa: F401

DEFAULT_REWARD = {
    "fork_routed": 1.0,
    "fork_lost": -1.0,
    "connector_seated": 2.0,
    "success": 5.0,
    "time": -0.002,
    "force": -0.02,
    "protective_stop": -5.0,
}


class HarnessRoutingEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, cfg: Optional[CellConfig] = None, config_path: Optional[str] = None,
                 randomize: bool = True, max_episode_time: float = 150.0,
                 n_cable_points: int = 16, reward_weights: Optional[Dict[str, float]] = None,
                 render_mode: Optional[str] = None, camera: str = "overview",
                 render_size: Tuple[int, int] = (480, 640)):
        super().__init__()
        if cfg is None:
            cfg = CellConfig.from_yaml(config_path) if config_path else CellConfig()
        self.cfg = cfg
        self.randomize = randomize
        self.max_episode_time = float(max_episode_time)
        self.n_cable_points = int(n_cable_points)
        self.reward_weights = dict(DEFAULT_REWARD, **(reward_weights or {}))
        self.render_mode = render_mode
        self.camera = camera
        self.render_size = render_size
        self.spec_actions = ActionSpec()
        self.n_forks = len(cfg.layout.fork_xy)

        self.obs_layout = self._make_layout()
        dim = obs_dim(self.obs_layout)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(dim,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32)

        self.cell: Optional[HarnessCell] = None
        self.iface: Optional[ActionInterface] = None
        self._renderer = None
        self.last_obs_dict: Dict[str, np.ndarray] = {}
        self.episode_seed: Optional[int] = None

    # ------------------------------------------------------------------ layout
    def _make_layout(self) -> Dict[str, slice]:
        return obs_layout(self.n_forks, self.n_cable_points)

    # --------------------------------------------------------------- gym API
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        self.episode_seed = int(seed) if seed is not None else int(self.np_random.integers(0, 2**31 - 1))
        if self.cell is None:
            self.cell = HarnessCell(self.cfg, seed=self.episode_seed, randomize=self.randomize)
        else:
            self.cell.reset(seed=self.episode_seed, randomize=self.randomize)
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        sim = self.cell.sim
        self.iface = ActionInterface(self.spec_actions, sim.instance.board_z)
        p, R = sim.tcp_pose()
        self.iface.reset(p, float(np.arctan2(R[1, 0], R[0, 0])), sim.gripper_opening())
        self._status = sim.task_status()
        self._observe()
        return self._flat(), self._info(0.0)

    def step(self, action: np.ndarray):
        assert self.cell is not None, "call reset() first"
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(5), -1.0, 1.0)
        sim = self.cell.sim
        p, R = sim.tcp_pose()
        self.iface.apply(a, p, float(np.arctan2(R[1, 0], R[0, 0])))
        self.cell.set_pose_target(self.iface.target_pos, yaw=self.iface.target_yaw)
        self.cell.set_gripper(self.iface.gripper)
        self.cell.step_time(self.cfg.sim.policy_dt)

        status = sim.task_status()
        w = self.reward_weights
        reward = w["time"]
        prev = self._status
        for was, now in zip(prev["forks_routed"], status["forks_routed"]):
            if now and not was:
                reward += w["fork_routed"]
            elif was and not now:
                reward += w["fork_lost"]
        if status["connector_seated"] and not prev["connector_seated"]:
            reward += w["connector_seated"]
        elif prev["connector_seated"] and not status["connector_seated"]:
            reward -= w["connector_seated"]
        force = float(np.linalg.norm(self.cell.wrench_world()[:3]))
        reward += w["force"] * float(np.clip((force - 25.0) / 25.0, 0.0, 1.0))
        stopped = bool(self.cell.ctrl.protective_stop)
        success = bool(status["success"])
        if success:
            reward += w["success"]
        if stopped:
            reward += w["protective_stop"]
        self._status = status
        self._observe()
        terminated = success or stopped
        truncated = (not terminated) and sim.time >= self.max_episode_time
        return self._flat(), float(reward), terminated, truncated, self._info(force)

    # ---------------------------------------------------------- observation
    def _observe(self) -> None:
        """One perception pass per step: the flat vector and the expert's dict view are
        built from the same noisy sample, so replays are bit-for-bit reproducible."""
        obs = self.cell.observe()
        obs["target_pos"] = self.iface.target_pos.copy()
        obs["target_yaw"] = np.array([self.iface.target_yaw])
        self.last_obs_dict = obs

    def _flat(self) -> np.ndarray:
        return flatten_obs(self.last_obs_dict, self.cfg, self.obs_layout, self.n_cable_points,
                           self.max_episode_time)

    def _info(self, force: float) -> Dict[str, Any]:
        st = self._status
        return {
            "episode_seed": self.episode_seed,
            "sim_time": self.cell.sim.time,
            "forks_routed": list(st["forks_routed"]),
            "n_routed": st["n_routed"],
            "connector_seated": st["connector_seated"],
            "is_success": st["success"],
            "protective_stop": bool(self.cell.ctrl.protective_stop),
            "contact_force": force,
        }

    # --------------------------------------------------------------- render
    def render(self):
        if self.render_mode != "rgb_array" or self.cell is None:
            return None
        import mujoco
        if self._renderer is None:
            h, w = self.render_size
            self._renderer = mujoco.Renderer(self.cell.sim.model, h, w)
        self._renderer.update_scene(self.cell.sim.data, camera=self.camera)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
