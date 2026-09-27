"""What a physics backend has to provide for the cell to run on it.

``HarnessSim`` (MuJoCo) is the reference implementation. A port to another engine -
Isaac Sim / PhysX is the one this repository has a benchmark suite for - only has to
satisfy this protocol for ``HarnessCell``, the controller, the expert, the Gymnasium
environment and the ROS node to work unchanged, because none of them touch the engine
directly.

The benchmark rigs (``benchmarks.py``, ``rigs_mujoco.py``, ``isaac/harness_isaac``) are
a *separate*, much smaller interface: they only build small scenes and return traces.
Port those first; they answer whether the full port is worth doing.
"""

from __future__ import annotations

from typing import Dict, Optional, Protocol, Tuple, runtime_checkable

import numpy as np


@runtime_checkable
class CellBackend(Protocol):
    """Engine-side state and commands for one harness cell."""

    # ------------------------------------------------------------- lifecycle
    def reset(self, seed: Optional[int] = None, randomize: Optional[bool] = None,
              instance=None, settle_time: float = 0.6) -> None: ...

    def step(self, n: int = 1) -> None:
        """Advance by n control periods of ``cfg.sim.control_dt``."""

    @property
    def time(self) -> float: ...

    # ----------------------------------------------------------- robot state
    @property
    def q(self) -> np.ndarray: ...

    @property
    def qd(self) -> np.ndarray: ...

    def tcp_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """TCP position and rotation matrix in world coordinates."""

    def gripper_opening(self) -> float: ...

    def ft_wrench(self, noise: bool = True) -> np.ndarray:
        """Payload-compensated wrist wrench in tool0 axes (6,)."""

    def zero_ft(self) -> None: ...

    # -------------------------------------------------------------- commands
    def set_joint_position(self, q: np.ndarray, qd: Optional[np.ndarray] = None) -> None: ...

    def set_joint_velocity(self, qd: np.ndarray) -> None: ...

    def set_gripper(self, opening: float) -> None: ...

    # ----------------------------------------------------------- cell state
    def cable_points(self) -> np.ndarray:
        """Wire centreline vertices (n + 1, 3) from the clamp to the connector."""

    def connector_pose(self) -> Tuple[np.ndarray, np.ndarray]: ...

    def holder_seat_pose(self) -> Tuple[np.ndarray, np.ndarray]: ...

    def fork_pose(self, i: int) -> Tuple[np.ndarray, np.ndarray]: ...

    def wire_in_fork(self, i: int) -> Dict[str, float]:
        """Where the wire crosses fork i's slot plane, in fork-local coordinates."""

    def connector_seated(self) -> Dict[str, float]: ...

    def task_status(self) -> Dict[str, object]: ...


REQUIRED_ATTRIBUTES = (
    "reset", "step", "time", "q", "qd", "tcp_pose", "gripper_opening", "ft_wrench",
    "zero_ft", "set_joint_position", "set_joint_velocity", "set_gripper", "cable_points",
    "connector_pose", "holder_seat_pose", "fork_pose", "wire_in_fork", "connector_seated",
    "task_status",
    # also used directly by HarnessCell
    "cfg", "instance", "kin", "rng", "n_forks", "data", "sid",
)


def missing_attributes(backend) -> Tuple[str, ...]:
    """Names a backend still has to provide (empty tuple means it is complete)."""
    return tuple(name for name in REQUIRED_ATTRIBUTES if not hasattr(backend, name))
