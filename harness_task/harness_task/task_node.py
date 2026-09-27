"""RouteHarness action server: runs the force-guided expert (or a learned policy)
through the robot's ROS interface.

The node closes the task-level loop at ``rate`` Hz (sim time): it assembles the
observation from ROS topics, asks the policy for an action in the shared 5-D
action space, integrates it into a Cartesian compliance target and publishes
that target plus the gripper command. The 500 Hz force/compliance loop runs
in the robot controller (here: the simulator node), as on a real UR5e.

Parameters
  config_file   cell YAML shared with the simulator (fixture geometry, gains)
  policy        "expert" or "package.module:object" for a learned policy. The
                object is called with the flat observation vector (same layout
                as the Gymnasium env) and must return a (5,) action; classes
                are instantiated with the CellConfig first.
  rate          policy rate [Hz] (default 20, like the env)
  autostart     start routing by itself after `autostart_delay` seconds
"""

from __future__ import annotations

import importlib
import math
import threading
import time
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from harness_interfaces.action import RouteHarness
from harness_interfaces.msg import TaskStatus
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger

from harness_core.actions import ActionInterface, ActionSpec
from harness_core.config import CellConfig
from harness_core.expert import HarnessExpert
from harness_core.geometry import mat_to_quat, tool_down_rotation
from harness_core.observation import flatten_obs, obs_layout, perceived_progress

from .ros_observation import RosObservation


class LearnedPolicy:
    """Adapter: flat-vector policy -> the expert-like step(obs_dict) interface."""

    def __init__(self, spec: str, cfg: CellConfig, max_episode_time: float = 150.0):
        module, _, attr = spec.partition(":")
        obj = getattr(importlib.import_module(module), attr or "policy")
        self.fn = obj(cfg) if isinstance(obj, type) else obj
        self.cfg = cfg
        self.layout = obs_layout(len(cfg.layout.fork_xy))
        self.max_episode_time = max_episode_time
        self.done = False
        self.failed = False
        self.phase = "policy"
        self.current_fork = 0
        self.t0 = None

    def reset(self):
        if hasattr(self.fn, "reset"):
            self.fn.reset()
        self.t0 = None

    def step(self, obs):
        t = float(obs["time"][0])
        self.t0 = t if self.t0 is None else self.t0
        vec = flatten_obs(obs, self.cfg, self.layout, 16, self.max_episode_time, elapsed=t - self.t0)
        act = self.fn.act(vec) if hasattr(self.fn, "act") else self.fn(vec)
        return np.clip(np.asarray(act, dtype=float).reshape(5), -1.0, 1.0)


class RoutingTaskNode(Node):
    def __init__(self):
        super().__init__("routing_task")
        p = self.declare_parameter
        p("config_file", "")
        p("policy", "expert")
        p("rate", 20.0)
        p("autostart", False)
        p("autostart_delay", 3.0)
        p("default_timeout", 300.0)
        p("verbose", True)
        gp = lambda n: self.get_parameter(n).value
        cfg_file = gp("config_file")
        self.cfg = CellConfig.from_yaml(cfg_file) if cfg_file else CellConfig()
        self.policy_spec = str(gp("policy"))
        self.rate = float(gp("rate"))
        self.verbose = bool(gp("verbose"))
        self.spec = ActionSpec()
        self.obs = RosObservation(self, self.cfg)
        self.pub_target = self.create_publisher(PoseStamped, "/cartesian_compliance_controller/target_frame", 10)
        self.pub_grip = self.create_publisher(Float64MultiArray, "/gripper_controller/commands", 10)
        self.truth: Optional[TaskStatus] = None
        self.create_subscription(TaskStatus, "/sim/task_status", self._on_truth, 5)
        self.zero_ft = self.create_client(Trigger, "/io_and_status_controller/zero_ftsensor")
        self._busy = threading.Lock()
        self.server = ActionServer(
            self, RouteHarness, "route_harness", execute_callback=self._execute,
            goal_callback=self._on_goal, cancel_callback=lambda _: CancelResponse.ACCEPT,
            callback_group=ReentrantCallbackGroup())
        if bool(gp("autostart")):
            self._auto_timer = self.create_timer(float(gp("autostart_delay")), self._autostart,
                                                 callback_group=MutuallyExclusiveCallbackGroup())
        self.get_logger().info(f"routing task ready (policy: {self.policy_spec}); "
                               f"send a goal to /route_harness")

    def _on_truth(self, msg: TaskStatus) -> None:
        self.truth = msg

    def now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _sleep_sim(self, seconds: float) -> None:
        """Sleep in ROS (sim) time; returns early if the clock stalls."""
        t_end = self.now() + seconds
        wall_end = time.monotonic() + 10.0 * seconds + 1.0
        while rclpy.ok() and self.now() < t_end and time.monotonic() < wall_end:
            time.sleep(0.002)

    # ----------------------------------------------------------------- goals
    def _on_goal(self, goal) -> GoalResponse:
        if self._busy.locked():
            self.get_logger().warn("already routing, rejecting new goal")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _autostart(self) -> None:
        self._auto_timer.cancel()
        threading.Thread(target=self._run_standalone, daemon=True).start()

    def _run_standalone(self) -> None:
        goal = RouteHarness.Goal()
        result = self._route(goal, handle=None)
        self.get_logger().info(f"autostart finished: success={result.success} ({result.message})")

    def _execute(self, handle):
        return self._route(handle.request, handle)

    def _make_policy(self, skip_connector: bool):
        if self.policy_spec == "expert":
            return HarnessExpert(self.cfg, self.spec, skip_connector=skip_connector, verbose=False)
        pol = LearnedPolicy(self.policy_spec, self.cfg)
        pol.reset()
        return pol

    def _route(self, goal, handle) -> RouteHarness.Result:
        result = RouteHarness.Result()
        if not self._busy.acquire(blocking=False):
            result.message = "busy"
            if handle is not None:
                handle.abort()
            return result
        try:
            return self._route_locked(goal, handle, result)
        finally:
            self._busy.release()

    def _route_locked(self, goal, handle, result):
        log = self.get_logger()
        # wait for robot + perception data
        t_wait = time.monotonic()
        while rclpy.ok() and self.obs.missing() is not None:
            if time.monotonic() - t_wait > 20.0:
                result.message = f"no data on {self.obs.missing()}"
                log.error(result.message)
                if handle is not None:
                    handle.abort()
                return result
            time.sleep(0.05)
        if self.zero_ft.wait_for_service(timeout_sec=1.0):
            self.zero_ft.call_async(Trigger.Request())
        # (the pause also lets the fixture frames of a just-rebuilt cell arrive)
        self._sleep_sim(0.3)
        fixtures = None
        while rclpy.ok() and fixtures is None and time.monotonic() - t_wait < 30.0:
            fixtures = self.obs.fixture_poses()
        if fixtures is None:
            result.message = "fixture frames (fork_i, anchor) missing in TF"
            if handle is not None:
                handle.abort()
            return result

        policy = self._make_policy(bool(goal.skip_connector))
        timeout = goal.timeout if goal.timeout > 0.0 else float(self.get_parameter("default_timeout").value)
        t0 = self.now()
        o = self.obs.build(t0, fixtures, np.zeros(3), 0.0)
        iface = ActionInterface(self.spec, self.cfg.layout.board_thickness)
        iface.reset(o["tcp_pos"], float(o["tcp_yaw"][0]), float(o["gripper"][0]))
        period = 1.0 / self.rate
        next_t = self.now()
        last_fb = -1.0
        last_log = 0
        log.info("routing started")
        while rclpy.ok():
            if handle is not None and handle.is_cancel_requested:
                handle.canceled()
                result.message = "canceled"
                return result
            now = self.now()
            if now - t0 > timeout:
                result.message = f"timeout after {timeout:.0f} s"
                break
            o = self.obs.build(now, fixtures, iface.target_pos, iface.target_yaw)
            if o["protective_stop"][0] > 0.5:
                result.message = "protective stop"
                break
            a = policy.step(o)
            iface.apply(a, o["tcp_pos"], float(o["tcp_yaw"][0]))
            self._publish_command(iface)
            if self.verbose and hasattr(policy, "log") and len(policy.log) > last_log:
                for _, msg in policy.log[last_log:]:
                    log.info(f"[expert] {msg}")
                last_log = len(policy.log)
            if handle is not None and now - last_fb > 0.5:
                last_fb = now
                fb = RouteHarness.Feedback()
                fb.phase = str(policy.phase)
                fb.current_fork = int(getattr(policy, "current_fork", 0))
                f = o["wrench"][:3]
                fb.contact_force = float(np.linalg.norm(f))
                fb.wire_tension = float(math.hypot(f[0], f[1]))
                fb.elapsed = float(now - t0)
                handle.publish_feedback(fb)
            if policy.done:
                break
            next_t += period
            while rclpy.ok() and self.now() < next_t:
                time.sleep(0.001)
        # final state from perception, averaged over half a second because the pose
        # estimates are noisy (and ground truth if the simulator publishes it)
        samples = [self.obs.build(self.now(), fixtures, iface.target_pos, iface.target_yaw)]
        t_end = self.now() + 0.5
        while rclpy.ok() and self.now() < t_end:
            self._sleep_sim(0.05)
            samples.append(self.obs.build(self.now(), fixtures, iface.target_pos, iface.target_yaw))
        o = samples[-1]
        for key in ("connector_pos", "holder_pos", "cable"):
            if all(s[key].shape == o[key].shape for s in samples):
                o[key] = np.mean([s[key] for s in samples], axis=0)
        forks, seated = perceived_progress(o, self.cfg)
        result.forks_routed = [bool(v) for v in forks]
        result.connector_seated = bool(seated)
        result.duration = float(self.now() - t0)
        want_conn = not bool(goal.skip_connector)
        result.success = bool(all(forks) and (seated or not want_conn) and not getattr(policy, "failed", False))
        if not result.message:
            result.message = getattr(policy, "fail_reason", "") or ("done" if result.success else "incomplete")
        if self.truth is not None:
            result.message += (f" | ground truth: forks {list(self.truth.forks_routed)}, "
                               f"connector seated {self.truth.connector_seated}")
        log.info(f"routing finished: success={result.success}, {result.message}, {result.duration:.1f} s")
        if handle is not None:
            if result.success:
                handle.succeed()
            else:
                handle.abort()
        return result

    def _publish_command(self, iface: ActionInterface) -> None:
        m = PoseStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = "world"
        tp = iface.target_pos
        m.pose.position.x, m.pose.position.y, m.pose.position.z = float(tp[0]), float(tp[1]), float(tp[2])
        w, x, y, z = mat_to_quat(tool_down_rotation(iface.target_yaw))
        m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w = x, y, z, w
        self.pub_target.publish(m)
        self.pub_grip.publish(Float64MultiArray(data=[float(iface.gripper)]))


def main(args=None):
    rclpy.init(args=args)
    node = RoutingTaskNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.try_shutdown()


if __name__ == "__main__":
    main()
