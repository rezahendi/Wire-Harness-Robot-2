"""MuJoCo simulation of the cell: the "hardware" layer.

``HarnessSim`` owns the MuJoCo model/data and exposes what a real UR5e cell
would expose: joint states, a joint-velocity/position servo interface, the
wrist F/T sensor (payload compensated, with noise and bias), a gripper, plus
privileged ground-truth queries (cable shape, fork and connector state) that a
perception system would provide on the real cell.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import mujoco
import numpy as np

from .config import CellConfig
from .geometry import mat_to_quat, polyline_arclength, rot_z
from .layout import CellInstance, sample_instance
from .scene import build_mjcf
from .ur_kinematics import URKinematics

GRAVITY = 9.81


class HarnessSim:
    def __init__(self, cfg: Optional[CellConfig] = None, seed: Optional[int] = None,
                 randomize: Optional[bool] = None, instance: Optional[CellInstance] = None):
        self.cfg = cfg.copy() if cfg is not None else CellConfig()
        self.kin = URKinematics(self.cfg.robot)
        self.rng = np.random.default_rng(seed)
        self.model: mujoco.MjModel = None
        self.data: mujoco.MjData = None
        self.reset(seed=seed, randomize=randomize, instance=instance)

    # ================================================================== reset
    def reset(self, seed: Optional[int] = None, randomize: Optional[bool] = None,
              instance: Optional[CellInstance] = None, settle_time: float = 0.6) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.instance = instance if instance is not None else sample_instance(
            self.cfg, self.rng, randomize=randomize, seed=seed)
        self.xml = build_mjcf(self.cfg, self.instance)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self._cache_ids()
        self.n_substeps = max(1, int(round(self.cfg.sim.control_dt / self.cfg.sim.timestep)))

        # Robot at home, gripper open, wire in its initial layout.
        q_home = np.array(self.cfg.robot.home_q, dtype=float)
        self.data.qpos[self.arm_qadr] = q_home
        self.q_ref = q_home.copy()
        self.qd_ref = np.zeros(6)
        self.gripper_ctrl = self.cfg.robot.finger_stroke
        self.gripper_target = self.gripper_ctrl
        self.data.qpos[self.finger_qadr] = self.cfg.robot.finger_stroke
        self._apply_initial_wire()
        self._write_ctrl()
        mujoco.mj_forward(self.model, self.data)

        # F/T sensor: per-episode bias (removed by zero_ft()) and payload model
        noise = self.cfg.noise
        self.ft_bias = np.concatenate([
            self.rng.normal(0.0, noise.ft_bias_force, 3),
            self.rng.normal(0.0, noise.ft_bias_torque, 3)])
        self.ft_offset = np.zeros(6)
        self.max_contact_force = 0.0
        self.max_contact_pair = None
        self._payload_mass = float(self.model.body_subtreemass[self.bid["gripper"]])
        com_w = self.data.subtree_com[self.bid["gripper"]]
        R_s = self.data.site_xmat[self.sid["ft_site"]].reshape(3, 3)
        self._payload_com = R_s.T @ (com_w - self.data.site_xpos[self.sid["ft_site"]])

        # Let the wire settle on the board.
        for _ in range(int(settle_time / self.cfg.sim.control_dt)):
            self.step()
        self.data.time = 0.0
        self.max_contact_force = 0.0
        self.max_contact_pair = None

    def _cache_ids(self) -> None:
        m = self.model
        body = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        site = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n)
        joint = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        self.arm_jid = [joint(n) for n in self.cfg.robot.joint_names]
        self.arm_qadr = np.array([m.jnt_qposadr[j] for j in self.arm_jid])
        self.arm_dadr = np.array([m.jnt_dofadr[j] for j in self.arm_jid])
        self.finger_jid = [joint("finger_left_joint"), joint("finger_right_joint")]
        self.finger_qadr = np.array([m.jnt_qposadr[j] for j in self.finger_jid])
        self.bid = {n: body(n) for n in ("gripper", "ft_sensor", "connector", "clamp", "holder",
                                         "wrist_3_link", "finger_left", "finger_right")}
        self.sid = {n: site(n) for n in ("tcp", "tool0", "ft_site", "anchor", "holder_seat", "connector")}
        n = self.instance.n_segments
        self.wire_bid = np.array([body(f"wire_{i}") for i in range(n)])
        self.wire_jqadr = np.array([m.jnt_qposadr[joint(f"wire_j{i}")] for i in range(1, n)])
        self.n_forks = len(self.instance.forks)
        self.fork_bid = [body(f"fork{i}") for i in range(self.n_forks)]
        self.fork_hinge_qadr = [
            (m.jnt_qposadr[joint(f"fork{i}_hinge_l")], m.jnt_qposadr[joint(f"fork{i}_hinge_r")])
            for i in range(self.n_forks)]
        self.arm_body_ids = [body(n) for n in ("shoulder_link", "upper_arm_link", "forearm_link",
                                               "wrist_1_link", "wrist_2_link", "wrist_3_link")]
        self.gripper_geom_ids = set(
            g for g in range(m.ngeom)
            if m.geom_bodyid[g] in (self.bid["gripper"], self.bid["finger_left"],
                                    self.bid["finger_right"], self.bid["ft_sensor"]))
        self.wire_geom_ids = set(g for g in range(m.ngeom) if m.geom_bodyid[g] in set(self.wire_bid))
        self.grasp_geom_ids = self.wire_geom_ids | {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "connector")}
        self.fork_geom_ids = [set(g for g in range(m.ngeom)
                                  if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
                                  and mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g).startswith(f"fork{i}_"))
                              for i in range(self.n_forks)]
        self.act_arm = np.arange(6)
        self.act_gripper = 6
        self.latch_eq = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, "connector_latch")
        self.latched = False

    def _apply_initial_wire(self) -> None:
        """Set the ball joints so the (straight-at-rest) wire follows the initial curve."""
        verts = self.instance.initial_wire
        dirs = np.diff(verts, axis=0)
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        up = np.array([0.0, 0.0, 1.0])
        frames = []
        for dvec in dirs:
            z = up - (up @ dvec) * dvec
            z /= np.linalg.norm(z)
            y = np.cross(z, dvec)
            frames.append(np.column_stack([dvec, y, z]))
        # root frame is fixed by the clamp (level, along the anchor yaw)
        R_prev = rot_z(self.instance.anchor.yaw)
        for i in range(1, len(frames)):
            R_rel = R_prev.T @ frames[i]
            self.data.qpos[self.wire_jqadr[i - 1]:self.wire_jqadr[i - 1] + 4] = mat_to_quat(R_rel)
            R_prev = frames[i]

    # =============================================================== commands
    def set_joint_velocity(self, qd: np.ndarray) -> None:
        """Joint velocity command (held until changed), like forward_velocity_controller."""
        lim = np.asarray(self.cfg.robot.joint_vel_limits)
        self.qd_ref = np.clip(np.asarray(qd, dtype=float), -lim, lim)

    def set_joint_position(self, q: np.ndarray, qd: Optional[np.ndarray] = None) -> None:
        """Direct joint position reference (trajectory following)."""
        self.q_ref = np.asarray(q, dtype=float).copy()
        self.qd_ref = np.zeros(6) if qd is None else np.asarray(qd, dtype=float).copy()

    def set_gripper(self, opening: float) -> None:
        """Target opening between the pads [m]. 0 (or less) closes with full force.
        The fingers move towards the target at the gripper's rated speed."""
        half = 0.5 * float(opening)
        self.gripper_target = float(np.clip(half, -0.006, self.cfg.robot.finger_stroke))
        if half <= 0.0:
            self.gripper_target = -0.006

    def zero_ft(self) -> None:
        """Re-zero the F/T sensor (like /io_and_status_controller/zero_ftsensor)."""
        self.ft_offset = self._ft_compensated_noiseless()

    # ================================================================== step
    def _write_ctrl(self) -> None:
        kp = np.asarray(self.cfg.robot.servo_kp)
        kv = np.asarray(self.cfg.robot.servo_kv)
        self.data.ctrl[self.act_arm] = self.q_ref + (kv / kp) * self.qd_ref
        self.data.ctrl[self.act_gripper] = self.gripper_ctrl

    def step(self, n: int = 1) -> None:
        """Advance by n control periods (control_dt each)."""
        dt = self.cfg.sim.control_dt
        lower = -np.asarray(self.cfg.robot.joint_limits)
        upper = np.asarray(self.cfg.robot.joint_limits)
        max_step = self.cfg.robot.finger_speed * dt
        for _ in range(n):
            self.q_ref = np.clip(self.q_ref + self.qd_ref * dt, lower, upper)
            self.gripper_ctrl += float(np.clip(self.gripper_target - self.gripper_ctrl, -max_step, max_step))
            self._write_ctrl()
            mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)
            self._track_contacts()
            if not self.latched and self.latch_eq >= 0:
                self._update_latch()

    def _update_latch(self) -> None:
        """Snap latch of the holder: a connector pressed fully into the pocket clicks in
        and is held there (a weld between connector and holder engages at its pose)."""
        s = self.connector_seated()
        h = self.cfg.holder
        if not (s["dz"] < h.latch_depth and abs(s["dx"]) < h.latch_offset
                and abs(s["dy"]) < h.latch_offset and s["tilt"] < 0.05):
            return
        m, d = self.model, self.data
        b1, b2 = self.bid["connector"], self.bid["holder"]
        R2 = d.xmat[b2].reshape(3, 3)
        eq = m.eq_data[self.latch_eq]
        eq[0:3] = R2.T @ (d.xpos[b1] - d.xpos[b2])      # anchor: connector centre, holder frame
        eq[3:6] = 0.0                                    # the same point in the connector frame
        q1_inv = np.zeros(4)
        mujoco.mju_negQuat(q1_inv, d.xquat[b1])
        mujoco.mju_mulQuat(eq[6:10], q1_inv, d.xquat[b2])   # relative orientation
        eq[10] = 1.0                                     # torque scale
        d.eq_active[self.latch_eq] = 1
        self.latched = True

    def _track_contacts(self) -> None:
        """Track the largest normal force between the gripper and anything but the
        grasped wire/connector (squeezing those is expected)."""
        d = self.data
        if d.ncon == 0:
            return
        f6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom1, c.geom2
            if g1 in self.gripper_geom_ids or g2 in self.gripper_geom_ids:
                other = g2 if g1 in self.gripper_geom_ids else g1
                if other in self.grasp_geom_ids:
                    continue
                mujoco.mj_contactForce(self.model, d, i, f6)
                if abs(f6[0]) > self.max_contact_force:
                    self.max_contact_force = abs(f6[0])
                    self.max_contact_pair = (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g1),
                                             mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g2),
                                             float(d.time))

    # ================================================================= state
    @property
    def time(self) -> float:
        return float(self.data.time)

    @property
    def q(self) -> np.ndarray:
        return self.data.qpos[self.arm_qadr].copy()

    @property
    def qd(self) -> np.ndarray:
        return self.data.qvel[self.arm_dadr].copy()

    @property
    def joint_effort(self) -> np.ndarray:
        return self.data.actuator_force[self.act_arm].copy()

    def gripper_opening(self) -> float:
        return float(2.0 * self.data.qpos[self.finger_qadr[0]])

    def finger_positions(self) -> np.ndarray:
        return self.data.qpos[self.finger_qadr].copy()

    def tcp_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        s = self.sid["tcp"]
        return self.data.site_xpos[s].copy(), self.data.site_xmat[s].reshape(3, 3).copy()

    def tool0_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        s = self.sid["tool0"]
        return self.data.site_xpos[s].copy(), self.data.site_xmat[s].reshape(3, 3).copy()

    def tcp_velocity(self) -> np.ndarray:
        v = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_SITE,
                                 self.sid["tcp"], v, 0)
        return np.concatenate([v[3:], v[:3]])  # [linear; angular] in world frame

    # ------------------------------------------------------------- F/T sensor
    def _ft_raw(self) -> np.ndarray:
        """Sensor reading in the ft_site frame: wrench exerted on the gripper by the arm side."""
        f = self.data.sensordata[0:3].copy()
        t = self.data.sensordata[3:6].copy()
        return np.concatenate([f, t])

    def _ft_compensated_noiseless(self) -> np.ndarray:
        """External wrench applied BY the environment ON the tool, expressed in the sensor
        frame (axes = tool0, reference point = sensor tool face), payload removed."""
        raw = self._ft_raw()
        R_s = self.data.site_xmat[self.sid["ft_site"]].reshape(3, 3)
        g_s = R_s.T @ np.array([0.0, 0.0, -GRAVITY * self._payload_mass])
        # MuJoCo reports the wrench the parent (sensor) applies to the child (gripper
        # subtree). In static equilibrium: raw + payload weight + external = 0.
        c = self._payload_com
        f_ext = -raw[:3] - g_s
        t_ext = -raw[3:] - np.array([c[1] * g_s[2] - c[2] * g_s[1], c[2] * g_s[0] - c[0] * g_s[2],
                                     c[0] * g_s[1] - c[1] * g_s[0]])
        return np.concatenate([f_ext, t_ext])

    def ft_wrench(self, noise: bool = True) -> np.ndarray:
        """What the robot driver would publish: payload compensated wrench in tool0 axes."""
        w = self._ft_compensated_noiseless() + self.ft_bias - self.ft_offset
        if noise:
            n = self.cfg.noise
            w = w + np.concatenate([self.rng.normal(0.0, n.ft_force_std, 3),
                                    self.rng.normal(0.0, n.ft_torque_std, 3)])
        return w

    def ft_frame_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        s = self.sid["ft_site"]
        return self.data.site_xpos[s].copy(), self.data.site_xmat[s].reshape(3, 3).copy()

    # ------------------------------------------------------------ cell state
    def cable_points(self) -> np.ndarray:
        """Wire centreline vertices (n_segments + 1, 3), from the clamp to the connector."""
        pts = self.data.xpos[self.wire_bid].copy()
        last = self.wire_bid[-1]
        end = self.data.xpos[last] + self.data.xmat[last].reshape(3, 3)[:, 0] * self.instance.wire.segment_length
        return np.vstack([pts, end])

    def cable_arclength(self) -> np.ndarray:
        return polyline_arclength(self.cable_points())

    def connector_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        b = self.bid["connector"]
        return self.data.xpos[b].copy(), self.data.xmat[b].reshape(3, 3).copy()

    def holder_seat_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        s = self.sid["holder_seat"]
        return self.data.site_xpos[s].copy(), self.data.site_xmat[s].reshape(3, 3).copy()

    def fork_pose(self, i: int) -> Tuple[np.ndarray, np.ndarray]:
        b = self.fork_bid[i]
        return self.data.xpos[b].copy(), self.data.xmat[b].reshape(3, 3).copy()

    def fork_prong_angles(self, i: int) -> Tuple[float, float]:
        a, b = self.fork_hinge_qadr[i]
        return float(self.data.qpos[a]), float(self.data.qpos[b])

    def wire_in_fork(self, i: int) -> Dict[str, float]:
        """Where the wire crosses fork i's slot plane (fork-local coordinates)."""
        f = self.cfg.fork
        p_f, R_f = self.fork_pose(i)
        pts = (self.cable_points() - p_f) @ R_f        # into fork frame (x along the slot)
        s_all = polyline_arclength(pts)
        best = None
        for k in range(len(pts) - 1):
            a, b = pts[k], pts[k + 1]
            if (a[0] - 0.0) * (b[0] - 0.0) <= 0.0 and abs(b[0] - a[0]) > 1e-9:
                t = -a[0] / (b[0] - a[0])
                p = a + t * (b - a)
                cand = (abs(p[1]) + max(0.0, p[2] - (f.post_height + f.prong_height)), p,
                        s_all[k] + t * np.linalg.norm(b - a))
                if best is None or cand[0] < best[0]:
                    best = cand
        if best is None:
            return {"routed": 0.0, "y": np.inf, "z": np.inf, "s": np.nan}
        _, p, s = best
        lip_bottom = f.post_height + f.prong_height - 2.0 * f.lip_radius
        inside = (abs(p[1]) < f.slot_width / 2 + 0.002
                  and f.post_height - 0.002 < p[2] < lip_bottom + 0.001)
        return {"routed": float(inside), "y": float(p[1]), "z": float(p[2]), "s": float(s)}

    def connector_seated(self) -> Dict[str, float]:
        """Connector in the pocket: centre within tolerance and its mating axis (x) along
        the holder axis. Roll about its own axis is free (square cross-section)."""
        p_c, R_c = self.connector_pose()
        p_h, R_h = self.holder_seat_pose()
        e = R_h.T @ (p_c - p_h)
        yaw_err = float(np.arctan2(R_h[:, 1] @ R_c[:, 0], R_h[:, 0] @ R_c[:, 0]))
        tilt = float(np.arccos(np.clip(R_c[:, 0] @ R_h[:, 0], -1.0, 1.0)))
        seated = (abs(e[0]) < 0.004 and abs(e[1]) < 0.0025 and abs(e[2]) < 0.0025
                  and abs(yaw_err) < 0.15 and tilt < 0.15)
        return {"seated": float(seated), "dx": float(e[0]), "dy": float(e[1]), "dz": float(e[2]),
                "yaw_err": yaw_err, "tilt": tilt}

    def gripper_touching(self, geom_set) -> bool:
        d = self.data
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom1, c.geom2
            if (g1 in self.gripper_geom_ids and g2 in geom_set) or (g2 in self.gripper_geom_ids and g1 in geom_set):
                return True
        return False

    def arm_min_height(self) -> float:
        """Lowest point of the arm link origins (safety check; links are visual only)."""
        return float(np.min(self.data.xpos[self.arm_body_ids][:, 2]))

    def task_status(self) -> Dict[str, object]:
        forks = [self.wire_in_fork(i) for i in range(self.n_forks)]
        seat = self.connector_seated()
        routed = [bool(f["routed"]) for f in forks]
        return {
            "forks_routed": routed,
            "n_routed": int(sum(routed)),
            "connector_seated": bool(seat["seated"]),
            "connector_latched": bool(self.latched),
            "seat": seat,
            "success": bool(all(routed) and seat["seated"]),
            "max_contact_force": float(self.max_contact_force),
        }
