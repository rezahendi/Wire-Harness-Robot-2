"""Assemble the task-level observation dict from ROS topics.

The dict has the same keys as ``HarnessCell.observe()`` in simulation, so the
scripted expert and learned policies run unchanged on top of ROS (simulated or
real cell).
"""

from __future__ import annotations

import math
import threading
from typing import Dict, Optional

import numpy as np
from geometry_msgs.msg import PoseStamped, WrenchStamped
from harness_interfaces.msg import CableState
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from tf2_ros import Buffer, TransformException, TransformListener

from harness_core.config import CellConfig
from harness_core.geometry import quat_to_mat, tool_yaw


def _pose(msg: PoseStamped):
    p = msg.pose.position
    q = msg.pose.orientation
    return np.array([p.x, p.y, p.z]), quat_to_mat([q.w, q.x, q.y, q.z])


class RosObservation:
    """Caches the latest robot/perception messages and builds observation dicts."""

    def __init__(self, node: Node, cfg: CellConfig, wrench_filter_hz: float = 40.0):
        self.node = node
        self.cfg = cfg
        self.lock = threading.Lock()
        self.joint_names = list(cfg.robot.joint_names)
        self.q = None
        self.qd = None
        self.finger = None
        self.tcp = None
        self.wrench_s = None
        self.wrench_world = np.zeros(6)
        self._wrench_t = None
        self.cable = None
        self.connector = None
        self.holder = None
        self.protective_stop = False
        self.fc = wrench_filter_hz
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, node)
        node.create_subscription(JointState, "/joint_states", self._on_js, 20)
        node.create_subscription(PoseStamped, "/tcp_pose_broadcaster/pose", self._on_tcp, 20)
        node.create_subscription(WrenchStamped, "/force_torque_sensor_broadcaster/wrench", self._on_wrench, 20)
        node.create_subscription(CableState, "/perception/cable", self._on_cable, 5)
        node.create_subscription(PoseStamped, "/perception/connector_pose", self._on_conn, 5)
        node.create_subscription(PoseStamped, "/perception/holder_pose", self._on_holder, 5)
        node.create_subscription(Bool, "/harness_sim/protective_stop", self._on_stop, 5)

    # ------------------------------------------------------------ callbacks
    def _on_js(self, msg: JointState) -> None:
        idx = {n: i for i, n in enumerate(msg.name)}
        if not all(n in idx for n in self.joint_names):
            return
        with self.lock:
            self.q = np.array([msg.position[idx[n]] for n in self.joint_names])
            self.qd = np.array([msg.velocity[idx[n]] if msg.velocity else 0.0 for n in self.joint_names])
            if "finger_left_joint" in idx:
                self.finger = float(msg.position[idx["finger_left_joint"]])

    def _on_tcp(self, msg: PoseStamped) -> None:
        with self.lock:
            self.tcp = _pose(msg)

    def _on_wrench(self, msg: WrenchStamped) -> None:
        f, t = msg.wrench.force, msg.wrench.torque
        w_s = np.array([f.x, f.y, f.z, t.x, t.y, t.z])
        stamp = Time.from_msg(msg.header.stamp).nanoseconds * 1e-9
        with self.lock:
            if self.tcp is None:
                return
            R = self.tcp[1]
            f_w = R @ w_s[:3]
            r = -R[:, 2] * (self.cfg.robot.tcp_offset - self.cfg.robot.ft_thickness)
            t_w = R @ w_s[3:] + np.cross(r, f_w)
            w = np.concatenate([f_w, t_w])
            if self._wrench_t is None:
                self.wrench_world = w
            else:
                dt = max(stamp - self._wrench_t, 1e-4)
                a = 1.0 - math.exp(-2.0 * math.pi * self.fc * dt)
                self.wrench_world = self.wrench_world + a * (w - self.wrench_world)
            self._wrench_t = stamp

    def _on_cable(self, msg: CableState) -> None:
        with self.lock:
            self.cable = np.array([[p.x, p.y, p.z] for p in msg.points])

    def _on_conn(self, msg: PoseStamped) -> None:
        with self.lock:
            self.connector = _pose(msg)

    def _on_holder(self, msg: PoseStamped) -> None:
        with self.lock:
            self.holder = _pose(msg)

    def _on_stop(self, msg: Bool) -> None:
        with self.lock:
            self.protective_stop = bool(msg.data)

    # ---------------------------------------------------------------- query
    def missing(self) -> Optional[str]:
        with self.lock:
            for name in ("q", "tcp", "cable", "connector", "holder", "finger"):
                if getattr(self, name) is None:
                    return name
        return None

    def fixture_poses(self) -> Optional[Dict[str, np.ndarray]]:
        """Fork / anchor frames from TF (published by the simulator or a calibration)."""
        out = {}
        try:
            forks = []
            for i in range(len(self.cfg.layout.fork_xy)):
                tr = self.tf_buffer.lookup_transform("world", f"fork_{i}", Time(), timeout=Duration(seconds=0.5))
                p = tr.transform.translation
                q = tr.transform.rotation
                R = quat_to_mat([q.w, q.x, q.y, q.z])
                forks.append(np.array([p.x, p.y, p.z, tool_yaw(R)]))
            tr = self.tf_buffer.lookup_transform("world", "anchor", Time(), timeout=Duration(seconds=0.5))
            p = tr.transform.translation
            out["forks"] = np.array(forks)
            out["anchor_pos"] = np.array([p.x, p.y, p.z])
        except TransformException as exc:
            self.node.get_logger().warn(f"fixture TF not available yet: {exc}", throttle_duration_sec=2.0)
            return None
        return out

    def build(self, t: float, fixtures: Dict[str, np.ndarray], target_pos: np.ndarray,
              target_yaw: float) -> Dict[str, np.ndarray]:
        with self.lock:
            p, R = self.tcp
            cp, cR = self.connector
            hp, hR = self.holder
            return {
                "time": np.array([t]),
                "q": self.q.copy(),
                "qd": self.qd.copy(),
                "tcp_pos": p.copy(),
                "tcp_rot": R.copy(),
                "tcp_yaw": np.array([tool_yaw(R)]),
                "wrench": self.wrench_world.copy(),
                "gripper": np.array([2.0 * max(self.finger, 0.0)]),
                "cable": self.cable.copy(),
                "connector_pos": cp.copy(),
                "connector_rot": cR.copy(),
                "forks": fixtures["forks"].copy(),
                "holder_pos": hp.copy(),
                "holder_yaw": np.array([tool_yaw(hR)]),
                "anchor_pos": fixtures["anchor_pos"].copy(),
                "board_z": np.array([self.cfg.layout.board_thickness]),
                "protective_stop": np.array([float(self.protective_stop)]),
                "target_pos": np.asarray(target_pos, dtype=float).copy(),
                "target_yaw": np.array([float(target_yaw)]),
            }
