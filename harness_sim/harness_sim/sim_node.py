"""MuJoCo simulation of the wire-harness cell behind a UR-driver-like ROS 2 interface.

The node plays the role of the robot + its controller box + the cell: it steps
the physics in (scaled) real time and exposes the topics a real UR5e cell
running ``ur_robot_driver`` would, so task code can move to hardware with
minimal changes.

Published
  /clock                                       rosgraph_msgs/Clock (sim time)
  /joint_states                                sensor_msgs/JointState (arm + fingers)
  /force_torque_sensor_broadcaster/wrench      geometry_msgs/WrenchStamped (frame ft_frame,
                                               payload compensated, noisy)
  /tcp_pose_broadcaster/pose                   geometry_msgs/PoseStamped (TCP in world)
  /perception/cable                            harness_interfaces/CableState (noisy)
  /perception/connector_pose                   geometry_msgs/PoseStamped (noisy)
  /perception/holder_pose                      geometry_msgs/PoseStamped (seat, noisy)
  /harness/markers                             visualization_msgs/MarkerArray (cell for RViz)
  /sim/task_status                             harness_interfaces/TaskStatus (ground truth)
  /harness_sim/protective_stop                 std_msgs/Bool
  TF static: world -> fork_<i>, holder, anchor

Command interfaces (the most recent command source wins)
  /cartesian_compliance_controller/target_frame   geometry_msgs/PoseStamped (TCP target, world)
  /cartesian_compliance_controller/target_wrench  geometry_msgs/WrenchStamped (world axes)
  /forward_velocity_controller/commands           std_msgs/Float64MultiArray (6 joint velocities)
  /scaled_joint_trajectory_controller/follow_joint_trajectory   control_msgs FollowJointTrajectory
  /gripper_controller/commands                    std_msgs/Float64MultiArray ([opening m])
  /gripper_controller/gripper_cmd                 control_msgs GripperCommand (position = opening m)

Services
  /sim/reset                                    harness_interfaces/ResetCell
  /io_and_status_controller/zero_ftsensor       std_srvs/Trigger
  /dashboard_client/unlock_protective_stop      std_srvs/Trigger
"""

from __future__ import annotations

import math
import threading
import time
from typing import List, Optional

import numpy as np
import rclpy
from builtin_interfaces.msg import Time as TimeMsg
from control_msgs.action import FollowJointTrajectory, GripperCommand
from geometry_msgs.msg import Point, PoseStamped, TransformStamped, WrenchStamped
from harness_interfaces.msg import CableState, TaskStatus
from harness_interfaces.srv import ResetCell
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64MultiArray
from std_srvs.srv import Trigger
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import Marker, MarkerArray

from harness_core.cell import HarnessCell
from harness_core.config import CellConfig
from harness_core.geometry import mat_to_quat, quat_to_mat

import mujoco


def _time_msg(t: float) -> TimeMsg:
    sec = int(math.floor(t))
    return TimeMsg(sec=sec, nanosec=int((t - sec) * 1e9))


def _quat_xyzw(R: np.ndarray):
    w, x, y, z = mat_to_quat(R)
    return x, y, z, w


class MujocoSimNode(Node):
    def __init__(self):
        super().__init__("mujoco_sim")
        p = self.declare_parameter
        p("config_file", "")
        p("seed", 0)
        p("randomize", True)
        p("real_time_factor", 1.0)
        p("state_rate", 125.0)
        p("perception_rate", 20.0)
        p("marker_rate", 15.0)
        p("status_rate", 5.0)
        p("command_timeout", 0.2)
        p("viewer", False)
        p("perception_noise", True)
        gp = lambda n: self.get_parameter(n).value

        cfg_file = gp("config_file")
        self.cfg = CellConfig.from_yaml(cfg_file) if cfg_file else CellConfig()
        self.rtf = max(0.01, float(gp("real_time_factor")))
        self.cmd_timeout = float(gp("command_timeout"))
        self.noisy = bool(gp("perception_noise"))
        self.lock = threading.RLock()
        seed = int(gp("seed"))
        self.cell = HarnessCell(self.cfg, seed=seed, randomize=bool(gp("randomize")))
        self.episode_seed = seed
        self.time_offset = 0.0
        self.vel_stamp = -1.0
        self.gripper_force = self.cfg.robot.grip_force
        self.get_logger().info(f"cell ready: seed {seed}, wire {self.cell.sim.instance.wire_length:.2f} m, "
                               f"{self.cell.sim.n_forks} forks")

        cb = ReentrantCallbackGroup()
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.pub_clock = self.create_publisher(Clock, "/clock", 10)
        self.pub_js = self.create_publisher(JointState, "/joint_states", 10)
        self.pub_wrench = self.create_publisher(WrenchStamped, "/force_torque_sensor_broadcaster/wrench", 10)
        self.pub_tcp = self.create_publisher(PoseStamped, "/tcp_pose_broadcaster/pose", 10)
        self.pub_cable = self.create_publisher(CableState, "/perception/cable", 10)
        self.pub_conn = self.create_publisher(PoseStamped, "/perception/connector_pose", 10)
        self.pub_holder = self.create_publisher(PoseStamped, "/perception/holder_pose", 10)
        self.pub_markers = self.create_publisher(MarkerArray, "/harness/markers", 10)
        self.pub_status = self.create_publisher(TaskStatus, "/sim/task_status", 10)
        self.pub_stop = self.create_publisher(Bool, "/harness_sim/protective_stop", latched)
        # Fixture frames go to /tf_static like a StaticTransformBroadcaster would publish
        # them, but the whole set is replaced on every reset (the rclpy broadcaster only
        # ever adds frames it has not seen, so a rebuilt layout would keep old poses).
        self.pub_tf_static = self.create_publisher(TFMessage, "/tf_static", latched)

        self.create_subscription(PoseStamped, "/cartesian_compliance_controller/target_frame",
                                 self._on_target_frame, 10, callback_group=cb)
        self.create_subscription(WrenchStamped, "/cartesian_compliance_controller/target_wrench",
                                 self._on_target_wrench, 10, callback_group=cb)
        self.create_subscription(Float64MultiArray, "/forward_velocity_controller/commands",
                                 self._on_velocity, 10, callback_group=cb)
        self.create_subscription(Float64MultiArray, "/gripper_controller/commands",
                                 self._on_gripper_topic, 10, callback_group=cb)

        self.create_service(ResetCell, "/sim/reset", self._on_reset, callback_group=cb)
        self.create_service(Trigger, "/io_and_status_controller/zero_ftsensor", self._on_zero_ft,
                            callback_group=cb)
        self.create_service(Trigger, "/dashboard_client/unlock_protective_stop", self._on_unlock,
                            callback_group=cb)

        self.traj_server = ActionServer(
            self, FollowJointTrajectory, "/scaled_joint_trajectory_controller/follow_joint_trajectory",
            execute_callback=self._exec_trajectory, goal_callback=self._goal_trajectory,
            cancel_callback=lambda _: CancelResponse.ACCEPT, callback_group=cb)
        self.grip_server = ActionServer(
            self, GripperCommand, "/gripper_controller/gripper_cmd",
            execute_callback=self._exec_gripper, cancel_callback=lambda _: CancelResponse.ACCEPT,
            callback_group=cb)

        self._viewer_enabled = bool(gp("viewer"))
        self._viewer = None
        self._static_markers: List[Marker] = []
        self._after_reset()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    # ---------------------------------------------------------------- helpers
    def now_sim(self) -> float:
        return self.time_offset + self.cell.sim.time

    def _stamp(self):
        return _time_msg(self.now_sim())

    def _after_reset(self) -> None:
        self._publish_static_tf()
        self._static_markers = self._build_static_markers()
        self.pub_stop.publish(Bool(data=False))
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
            self._viewer = None
        if self._viewer_enabled:
            self._open_viewer()

    def _open_viewer(self) -> None:
        try:
            import mujoco.viewer
            self._viewer = mujoco.viewer.launch_passive(self.cell.sim.model, self.cell.sim.data)
        except Exception as exc:  # no display (e.g. headless WSL): keep running without it
            self.get_logger().warn(f"MuJoCo viewer unavailable: {exc}")
            self._viewer_enabled = False
            self._viewer = None

    # ------------------------------------------------------------- callbacks
    def _on_target_frame(self, msg: PoseStamped) -> None:
        q = msg.pose.orientation
        R = quat_to_mat([q.w, q.x, q.y, q.z])
        pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        with self.lock:
            wrench = self.cell.target.wrench if self.cell.mode == "compliance" else None
            self.cell.set_pose_target(pos, rotation=R, wrench=wrench)

    def _on_target_wrench(self, msg: WrenchStamped) -> None:
        w = np.array([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z,
                      msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z])
        with self.lock:
            t = self.cell.target
            self.cell.set_pose_target(t.position, rotation=t.rotation, wrench=w)

    def _on_velocity(self, msg: Float64MultiArray) -> None:
        if len(msg.data) != 6:
            self.get_logger().warn("forward_velocity_controller/commands needs 6 values", throttle_duration_sec=2.0)
            return
        with self.lock:
            self.cell.set_joint_velocity_command(np.array(msg.data, dtype=float))
            self.vel_stamp = time.monotonic()

    def _on_gripper_topic(self, msg: Float64MultiArray) -> None:
        if len(msg.data) >= 1:
            with self.lock:
                self.cell.set_gripper(float(msg.data[0]))

    def _on_reset(self, req: ResetCell.Request, res: ResetCell.Response):
        seed = int(req.seed) if req.seed >= 0 else int(np.random.randint(0, 2**31 - 1))
        with self.lock:
            self.time_offset = self.now_sim() + 0.1
            self.cell.reset(seed=seed, randomize=bool(req.randomize))
            self.episode_seed = seed
            self._after_reset()
        res.success = True
        res.seed = seed
        res.message = f"cell rebuilt (seed {seed}, randomize={req.randomize})"
        self.get_logger().info(res.message)
        return res

    def _on_zero_ft(self, req, res):
        with self.lock:
            self.cell.sim.zero_ft()
        res.success = True
        res.message = "F/T sensor zeroed"
        return res

    def _on_unlock(self, req, res):
        with self.lock:
            self.cell.ctrl.clear_protective_stop()
            self.cell.hold_current_pose()
        self.pub_stop.publish(Bool(data=False))
        res.success = True
        res.message = "protective stop cleared"
        return res

    # --------------------------------------------------------- joint trajectory
    def _goal_trajectory(self, goal: FollowJointTrajectory.Goal):
        names = list(goal.trajectory.joint_names)
        if sorted(names) != sorted(self.cfg.robot.joint_names) or not goal.trajectory.points:
            self.get_logger().warn(f"rejecting trajectory with joints {names}")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _exec_trajectory(self, handle):
        traj = handle.request.trajectory
        order = [list(traj.joint_names).index(n) for n in self.cfg.robot.joint_names]
        times = [pt.time_from_start.sec + 1e-9 * pt.time_from_start.nanosec for pt in traj.points]
        pos = [[pt.positions[i] for i in order] for pt in traj.points]
        vel = None
        if all(len(pt.velocities) == len(order) for pt in traj.points):
            vel = [[pt.velocities[i] for i in order] for pt in traj.points]
        with self.lock:
            self.cell.follow_joint_trajectory(np.array(times), np.array(pos), None if vel is None else np.array(vel))
            t_end = self.cell.sim.time + times[-1]
        result = FollowJointTrajectory.Result()
        while rclpy.ok():
            if handle.is_cancel_requested:
                with self.lock:
                    self.cell.hold_current_pose()
                handle.canceled()
                result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
                return result
            with self.lock:
                done = self.cell.mode != "trajectory"
                q = self.cell.sim.q
                late = self.cell.sim.time > t_end + 2.0
            fb = FollowJointTrajectory.Feedback()
            fb.joint_names = list(self.cfg.robot.joint_names)
            fb.actual.positions = [float(v) for v in q]
            handle.publish_feedback(fb)
            if done or late:
                break
            time.sleep(0.02)
        handle.succeed()
        result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
        return result

    # ----------------------------------------------------------------- gripper
    def _exec_gripper(self, handle):
        cmd = handle.request.command
        opening = float(cmd.position)
        effort = float(cmd.max_effort) if cmd.max_effort > 0.0 else self.cfg.robot.grip_force
        with self.lock:
            self.cell.sim.model.actuator_forcerange[self.cell.sim.act_gripper] = [-effort, effort]
            self.cell.set_gripper(opening)
        last, still = None, 0
        t0 = time.monotonic()
        while rclpy.ok() and time.monotonic() - t0 < 5.0 / self.rtf:
            if handle.is_cancel_requested:
                handle.canceled()
                return GripperCommand.Result()
            time.sleep(0.05)
            with self.lock:
                cur = self.cell.sim.gripper_opening()
            fb = GripperCommand.Feedback()
            fb.position = cur
            handle.publish_feedback(fb)
            if last is not None and abs(cur - last) < 2e-4:
                still += 1
            else:
                still = 0
            last = cur
            if still >= 3:
                break
        with self.lock:
            cur = self.cell.sim.gripper_opening()
            force = float(abs(self.cell.sim.data.actuator_force[self.cell.sim.act_gripper]))
        res = GripperCommand.Result()
        res.position = cur
        res.effort = force
        res.reached_goal = abs(cur - opening) < 0.002
        res.stalled = not res.reached_goal
        handle.succeed()
        return res

    # --------------------------------------------------------------- main loop
    def _run(self) -> None:
        dt = self.cfg.sim.control_dt
        gp = lambda n: float(self.get_parameter(n).value)
        every = lambda rate: max(1, int(round(1.0 / (rate * dt))))
        n_state = every(gp("state_rate"))
        n_perc = every(gp("perception_rate"))
        n_mark = every(gp("marker_rate"))
        n_status = every(gp("status_rate"))
        n_view = every(30.0)
        k = 0
        wall0 = time.monotonic()
        sim0 = self.now_sim()
        last_report = wall0
        stopped_prev = False
        while self._running and rclpy.ok():
            with self.lock:
                cell = self.cell
                if cell.mode == "velocity" and time.monotonic() - self.vel_stamp > self.cmd_timeout:
                    cell.set_joint_velocity_command(np.zeros(6))
                cell.step()
                k += 1
                try:
                    if k % n_state == 0:
                        self._publish_state()
                    if k % n_perc == 0:
                        self._publish_perception()
                    if k % n_mark == 0:
                        self._publish_markers(k % (n_mark * 15) == 0)
                    if k % n_status == 0:
                        self._publish_status()
                    stopped = bool(cell.ctrl.protective_stop)
                    if stopped != stopped_prev:
                        self.pub_stop.publish(Bool(data=stopped))
                        if stopped:
                            self.get_logger().error(cell.ctrl.stop_reason)
                        stopped_prev = stopped
                    if self._viewer is not None and k % n_view == 0:
                        if self._viewer.is_running():
                            self._viewer.sync()
                        else:
                            self._viewer = None
                except Exception as exc:  # never let a publishing hiccup kill the sim thread
                    self.get_logger().warn(f"publish error: {exc}", throttle_duration_sec=5.0)
                sim_elapsed = self.now_sim() - sim0
            # pace to real time * rtf
            ahead = sim_elapsed / self.rtf - (time.monotonic() - wall0)
            if ahead > 0.002:
                time.sleep(ahead)
            elif ahead < -0.5:
                wall0 = time.monotonic() - sim_elapsed / self.rtf   # do not try to catch up
            if time.monotonic() - last_report > 30.0:
                last_report = time.monotonic()
                self.get_logger().info(f"sim time {self.now_sim():.1f} s, mode {self.cell.mode}")

    # ------------------------------------------------------------- publishing
    def _publish_state(self) -> None:
        sim = self.cell.sim
        stamp = self._stamp()
        self.pub_clock.publish(Clock(clock=stamp))
        js = JointState()
        js.header.stamp = stamp
        js.name = list(self.cfg.robot.joint_names) + ["finger_left_joint", "finger_right_joint"]
        fq = sim.finger_positions()
        js.position = [float(v) for v in sim.q] + [float(max(fq[0], 0.0)), float(max(fq[1], 0.0))]
        js.velocity = [float(v) for v in sim.qd] + [0.0, 0.0]
        js.effort = [float(v) for v in sim.joint_effort] + [float(sim.data.actuator_force[sim.act_gripper])] * 2
        self.pub_js.publish(js)

        w = sim.ft_wrench(noise=True)
        wm = WrenchStamped()
        wm.header.stamp = stamp
        wm.header.frame_id = "ft_frame"
        wm.wrench.force.x, wm.wrench.force.y, wm.wrench.force.z = (float(v) for v in w[:3])
        wm.wrench.torque.x, wm.wrench.torque.y, wm.wrench.torque.z = (float(v) for v in w[3:])
        self.pub_wrench.publish(wm)

        pos, R = sim.tcp_pose()
        self.pub_tcp.publish(self._pose_msg(pos, R, stamp))

    def _pose_msg(self, pos, R, stamp) -> PoseStamped:
        m = PoseStamped()
        m.header.stamp = stamp
        m.header.frame_id = "world"
        m.pose.position.x, m.pose.position.y, m.pose.position.z = (float(v) for v in pos)
        x, y, z, w = _quat_xyzw(R)
        m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w = x, y, z, w
        return m

    def _publish_perception(self) -> None:
        obs = self.cell.observe(noisy=self.noisy)
        stamp = self._stamp()
        cs = CableState()
        cs.header.stamp = stamp
        cs.header.frame_id = "world"
        cs.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in obs["cable"]]
        cs.radius = float(self.cell.sim.instance.wire.radius)
        cs.length = float(self.cell.sim.instance.wire_length)
        self.pub_cable.publish(cs)
        self.pub_conn.publish(self._pose_msg(obs["connector_pos"], obs["connector_rot"], stamp))
        hy = float(obs["holder_yaw"][0])
        Rh = np.array([[math.cos(hy), -math.sin(hy), 0.0], [math.sin(hy), math.cos(hy), 0.0], [0.0, 0.0, 1.0]])
        self.pub_holder.publish(self._pose_msg(obs["holder_pos"], Rh, stamp))

    def _publish_status(self) -> None:
        st = self.cell.sim.task_status()
        m = TaskStatus()
        m.header.stamp = self._stamp()
        m.forks_routed = [bool(v) for v in st["forks_routed"]]
        m.connector_seated = bool(st["connector_seated"])
        m.success = bool(st["success"])
        m.max_contact_force = float(st["max_contact_force"])
        m.episode_seed = int(self.episode_seed)
        self.pub_status.publish(m)

    def _publish_static_tf(self) -> None:
        sim = self.cell.sim
        stamp = self._stamp()
        tfs = []

        def tf(name, pos, R):
            t = TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = "world"
            t.child_frame_id = name
            t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = (float(v) for v in pos)
            x, y, z, w = _quat_xyzw(R)
            t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = x, y, z, w
            return t
        for i in range(sim.n_forks):
            tfs.append(tf(f"fork_{i}", *sim.fork_pose(i)))
        tfs.append(tf("holder", *sim.holder_seat_pose()))
        a = sim.instance.anchor
        Ra = np.array([[math.cos(a.yaw), -math.sin(a.yaw), 0.0], [math.sin(a.yaw), math.cos(a.yaw), 0.0], [0, 0, 1.0]])
        tfs.append(tf("anchor", sim.data.site_xpos[sim.sid["anchor"]], Ra))
        self.pub_tf_static.publish(TFMessage(transforms=tfs))

    # ----------------------------------------------------------------- markers
    def _geom_marker(self, gid: int, mid: int, ns: str) -> Optional[Marker]:
        m_, d_ = self.cell.sim.model, self.cell.sim.data
        gtype = m_.geom_type[gid]
        size = m_.geom_size[gid]
        mk = Marker()
        mk.header.frame_id = "world"
        mk.ns = ns
        mk.id = mid
        mk.action = Marker.ADD
        pos = d_.geom_xpos[gid]
        R = d_.geom_xmat[gid].reshape(3, 3)
        mk.pose.position.x, mk.pose.position.y, mk.pose.position.z = (float(v) for v in pos)
        x, y, z, w = _quat_xyzw(R)
        mk.pose.orientation.x, mk.pose.orientation.y, mk.pose.orientation.z, mk.pose.orientation.w = x, y, z, w
        if gtype == mujoco.mjtGeom.mjGEOM_BOX:
            mk.type = Marker.CUBE
            mk.scale.x, mk.scale.y, mk.scale.z = (float(2 * v) for v in size)
        elif gtype in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
            mk.type = Marker.CYLINDER
            extra = size[0] if gtype == mujoco.mjtGeom.mjGEOM_CAPSULE else 0.0
            mk.scale.x = mk.scale.y = float(2 * size[0])
            mk.scale.z = float(2 * (size[1] + extra))
        else:
            return None
        matid = m_.geom_matid[gid]
        rgba = m_.mat_rgba[matid] if matid >= 0 else m_.geom_rgba[gid]
        gname = mujoco.mj_id2name(m_, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if gname == "board":        # textured in MuJoCo: use a plain wood colour in RViz
            rgba = (0.80, 0.69, 0.52, 1.0)
        elif gname == "table":
            rgba = (0.45, 0.47, 0.50, 1.0)
        mk.color.r, mk.color.g, mk.color.b, mk.color.a = (float(v) for v in rgba)
        if mk.color.a <= 0.0:
            mk.color.a = 1.0
        return mk

    def _fixture_geoms(self, dynamic: bool) -> List[int]:
        m_ = self.cell.sim.model
        out = []
        for g in range(m_.ngeom):
            name = mujoco.mj_id2name(m_, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            body = mujoco.mj_id2name(m_, mujoco.mjtObj.mjOBJ_BODY, m_.geom_bodyid[g]) or ""
            if name in ("floor",) or body.startswith("wire_"):
                continue
            is_jaw = "_prong_" in body
            is_fixture = (body.startswith("fork") or body in ("holder", "clamp")
                          or name in ("board", "table") or name.startswith("route_"))
            if dynamic and (is_jaw or body == "connector"):
                out.append(g)
            elif not dynamic and is_fixture and not is_jaw:
                out.append(g)
        return out

    def _build_static_markers(self) -> List[Marker]:
        marks = []
        for i, g in enumerate(self._fixture_geoms(dynamic=False)):
            mk = self._geom_marker(g, i, "fixtures")
            if mk is not None:
                marks.append(mk)
        return marks

    def _publish_markers(self, include_static: bool) -> None:
        stamp = self._stamp()
        arr = MarkerArray()
        if include_static or not hasattr(self, "_static_sent") or not self._static_sent:
            for mk in self._static_markers:
                mk.header.stamp = stamp
                arr.markers.append(mk)
            self._static_sent = True
        for i, g in enumerate(self._fixture_geoms(dynamic=True)):
            mk = self._geom_marker(g, i, "moving")
            if mk is not None:
                mk.header.stamp = stamp
                arr.markers.append(mk)
        wire = Marker()
        wire.header.frame_id = "world"
        wire.header.stamp = stamp
        wire.ns = "wire"
        wire.id = 0
        wire.type = Marker.LINE_STRIP
        wire.action = Marker.ADD
        wire.pose.orientation.w = 1.0
        wire.scale.x = float(2 * self.cell.sim.instance.wire.radius)
        wire.color.r, wire.color.g, wire.color.b, wire.color.a = 0.96, 0.55, 0.10, 1.0
        wire.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in self.cell.sim.cable_points()]
        arr.markers.append(wire)
        self.pub_markers.publish(arr)

    def destroy_node(self):
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MujocoSimNode()
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
