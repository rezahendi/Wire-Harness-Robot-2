"""MJCF generator for the wire-harness cell.

The model is generated from a ``CellConfig`` and a sampled ``CellInstance`` so the
layout, wire properties and fixtures can be randomised per episode.

Collision groups (contype / conaffinity bit masks):
    1  static environment (table, board, fork posts, clamp, holder)
    2  wire and connector (no wire-wire self collision)
    4  gripper (F/T sensor, housing, fingers)
    8  fork prongs (spring-loaded, must not touch their own post)
Arm links are visual only; the task never brings them close to the board and the
safety checks in ``HarnessSim`` flag it if they do.
"""

from __future__ import annotations

from typing import List

import numpy as np

from .config import CellConfig
from .layout import CellInstance, Pose2

# contype, conaffinity
ENV = (1, 2 | 4)
WIRE = (2, 1 | 4 | 8)
GRIPPER = (4, 1 | 2 | 8)
PRONG = (8, 2 | 4)

SOLREF = "0.004 1"
SOLIMP = "0.95 0.99 0.0005"

# Names used by the simulator wrapper
ARM_BODIES = ("shoulder_link", "upper_arm_link", "forearm_link",
              "wrist_1_link", "wrist_2_link", "wrist_3_link")
FINGER_JOINTS = ("finger_left_joint", "finger_right_joint")


def _f(x: float) -> str:
    return f"{x:.6g}"


def _v(*xs) -> str:
    return " ".join(_f(float(x)) for x in xs)


def _quat_yaw(yaw: float) -> str:
    return _v(np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2))


def _col(group) -> str:
    return f'contype="{group[0]}" conaffinity="{group[1]}"'


def _xyaxes_lookat(pos, target, up=(0.0, 0.0, 1.0)) -> str:
    pos, target, up = (np.asarray(v, dtype=float) for v in (pos, target, up))
    fwd = target - pos
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    if np.linalg.norm(right) < 1e-6:
        right = np.cross(fwd, np.array([1.0, 0.0, 0.0]))
    right /= np.linalg.norm(right)
    cam_up = np.cross(right, fwd)
    return _v(*right, *cam_up)


def build_mjcf(cfg: CellConfig, inst: CellInstance) -> str:
    parts: List[str] = []
    w = inst.wire
    fr = inst.friction_scale
    sim = cfg.sim

    parts.append(f"""<mujoco model="wire_harness_cell">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{_f(sim.timestep)}" integrator="implicitfast" cone="elliptic" impratio="10"/>
  <size memory="64M"/>
  <extension>
    <plugin plugin="mujoco.elasticity.cable">
      <instance name="wire">
        <config key="twist" value="{_f(w.twist_modulus)}"/>
        <config key="bend" value="{_f(w.bend_modulus)}"/>
        <config key="vmax" value="0"/>
      </instance>
    </plugin>
  </extension>
  <visual>
    <global offwidth="1280" offheight="960" elevation="-25" azimuth="160"/>
    <quality shadowsize="4096" offsamples="4"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.5 0.5 0.5" specular="0.1 0.1 0.1"/>
    <map znear="0.01"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.93 0.95 0.98" rgb2="0.62 0.68 0.76" width="512" height="512"/>
    <texture name="floor_tex" type="2d" builtin="checker" rgb1="0.42 0.43 0.45" rgb2="0.38 0.39 0.41" width="512" height="512"/>
    <texture name="board_tex" type="2d" builtin="flat" rgb1="0.80 0.69 0.52" rgb2="0.74 0.62 0.45" mark="random" random="0.08" markrgb="0.70 0.58 0.42" width="512" height="512"/>
    <material name="floor" texture="floor_tex" texrepeat="8 8" reflectance="0.05"/>
    <material name="table" rgba="0.55 0.57 0.60 1"/>
    <material name="board" texture="board_tex" texrepeat="3 3" rgba="1 1 1 1"/>
    <material name="ur_light" rgba="0.80 0.82 0.84 1" specular="0.3" shininess="0.4"/>
    <material name="ur_cap" rgba="0.36 0.52 0.68 1" specular="0.3"/>
    <material name="ur_dark" rgba="0.18 0.19 0.21 1"/>
    <material name="ft" rgba="0.55 0.56 0.58 1" specular="0.5"/>
    <material name="gripper" rgba="0.14 0.15 0.17 1"/>
    <material name="pad" rgba="0.85 0.35 0.20 1"/>
    <material name="wire" rgba="0.96 0.55 0.10 1" specular="0.4"/>
    <material name="connector" rgba="0.12 0.12 0.13 1" specular="0.3"/>
    <material name="fork" rgba="0.16 0.36 0.78 1"/>
    <material name="lip" rgba="0.98 0.82 0.18 1"/>
    <material name="holder" rgba="0.22 0.55 0.35 1"/>
    <material name="clamp" rgba="0.30 0.30 0.32 1"/>
    <material name="route" rgba="1 1 1 0.55"/>
  </asset>
  <default>
    <geom solref="{SOLREF}" solimp="{SOLIMP}"/>
    <default class="visual">
      <geom contype="0" conaffinity="0" group="2"/>
    </default>
    <default class="env">
      <geom {_col(ENV)} friction="{_f(0.35 * fr)} 0.005 0.0001" group="0"/>
    </default>
    <default class="arm_joint">
      <joint type="hinge" axis="0 0 1" damping="1.0"/>
    </default>
  </default>
  <worldbody>
    <light name="key" pos="0.9 -0.8 1.6" dir="-0.4 0.45 -0.8" diffuse="0.55 0.55 0.55" castshadow="true"/>
    <light name="fill" pos="-0.3 0.7 1.4" dir="0.5 -0.5 -0.7" diffuse="0.25 0.25 0.25" castshadow="false"/>
    <geom name="floor" type="plane" size="3 3 0.1" pos="0 0 -0.75" material="floor" {_col(ENV)}/>
    <geom name="table" class="env" type="box" size="0.75 0.7 0.375" pos="0.35 0 -0.375" material="table"/>
""")
    parts.append(_cameras(cfg))
    parts.append(_board(cfg, inst))
    parts.append(_robot(cfg))
    for i, fork in enumerate(inst.forks):
        parts.append(_fork(cfg, inst, i, fork))
    parts.append(_holder(cfg, inst))
    parts.append(_clamp_and_wire(cfg, inst))
    parts.append("  </worldbody>\n")
    parts.append(_equality_actuators_sensors(cfg))
    parts.append("</mujoco>\n")
    return "".join(parts)


# ---------------------------------------------------------------------------
def _cameras(cfg: CellConfig) -> str:
    cx, cy = cfg.layout.board_center
    tgt = (cx, cy, 0.05)
    ov = (cx + 0.85, cy - 0.75, 0.62)
    side = (cx + 0.05, cy - 0.95, 0.45)
    return f"""    <camera name="overview" pos="{_v(*ov)}" xyaxes="{_xyaxes_lookat(ov, tgt)}" fovy="45"/>
    <camera name="side" pos="{_v(*side)}" xyaxes="{_xyaxes_lookat(side, (cx, cy, 0.06))}" fovy="45"/>
    <camera name="top" pos="{_v(cx, cy, 1.05)}" xyaxes="0 -1 0 1 0 0" fovy="45"/>
"""


def _board(cfg: CellConfig, inst: CellInstance) -> str:
    lay = cfg.layout
    cx, cy = lay.board_center
    sx, sy = lay.board_size
    t = lay.board_thickness
    out = [f'    <geom name="board" class="env" type="box" size="{_v(sx / 2, sy / 2, t / 2)}" '
           f'pos="{_v(cx, cy, t / 2)}" material="board"/>\n']
    # printed route (visual only), like the drawing glued on a real formboard
    pts = inst.route_points(cfg.connector.length)
    for i, (a, b) in enumerate(zip(pts[:-1], pts[1:])):
        mid = 0.5 * (a + b)
        d = b - a
        L = float(np.linalg.norm(d))
        yaw = float(np.arctan2(d[1], d[0]))
        out.append(f'    <geom name="route_{i}" class="visual" type="box" size="{_v(L / 2, 0.0025, 0.0003)}" '
                   f'pos="{_v(mid[0], mid[1], t + 0.0003)}" quat="{_quat_yaw(yaw)}" material="route"/>\n')
    return "".join(out)


def _robot(cfg: CellConfig) -> str:
    r = cfg.robot
    d, a = r.dh_d, r.dh_a
    lim = r.joint_limits
    arm = f"""    <body name="base_link" pos="0 0 0">
      <geom class="visual" type="cylinder" size="0.076 0.043" pos="0 0 0.043" material="ur_dark"/>
      <geom class="visual" type="cylinder" size="0.078 0.004" pos="0 0 0.088" material="ur_cap"/>
      <site name="base_link" pos="0 0 0" size="0.005" group="5"/>
      <body name="shoulder_link" quat="0 0 0 1" gravcomp="1">
        <inertial pos="0 0 0.12" mass="3.761" diaginertia="0.0103 0.0103 0.0067"/>
        <joint name="{r.joint_names[0]}" class="arm_joint" range="{_v(-lim[0], lim[0])}" armature="{_f(r.armature[0])}"/>
        <geom class="visual" type="cylinder" size="0.062 0.036" pos="0 0 0.128" material="ur_light"/>
        <body name="upper_arm_link" pos="{_v(a[0], 0, d[0])}" quat="0.707107 0.707107 0 0" gravcomp="1">
          <inertial pos="-0.2125 0 0.138" mass="8.058" diaginertia="0.0151 0.1339 0.1339"/>
          <joint name="{r.joint_names[1]}" class="arm_joint" range="{_v(-lim[1], lim[1])}" armature="{_f(r.armature[1])}"/>
          <geom class="visual" type="cylinder" size="0.062 0.105" pos="0 0 0.07" material="ur_light"/>
          <geom class="visual" type="cylinder" size="0.063 0.006" pos="0 0 -0.036" material="ur_cap"/>
          <geom class="visual" type="capsule" fromto="-0.03 0 0.138 -0.395 0 0.138" size="0.046" material="ur_light"/>
          <geom class="visual" type="cylinder" size="0.052 0.105" pos="{_v(a[1], 0, 0.074)}" material="ur_light"/>
          <geom class="visual" type="cylinder" size="0.053 0.006" pos="{_v(a[1], 0, 0.179)}" material="ur_cap"/>
          <body name="forearm_link" pos="{_v(a[1], 0, d[1])}" gravcomp="1">
            <inertial pos="-0.1961 0 0.007" mass="2.846" diaginertia="0.0041 0.0311 0.0311"/>
            <joint name="{r.joint_names[2]}" class="arm_joint" range="{_v(-lim[2], lim[2])}" armature="{_f(r.armature[2])}"/>
            <geom class="visual" type="capsule" fromto="-0.03 0 0.007 -0.36 0 0.007" size="0.038" material="ur_light"/>
            <geom class="visual" type="cylinder" size="0.042 0.06" pos="{_v(a[2], 0, 0.02)}" material="ur_light"/>
            <body name="wrist_1_link" pos="{_v(a[2], 0, d[2])}" gravcomp="1">
              <inertial pos="0 0 0.12" mass="1.37" diaginertia="0.0026 0.0026 0.0022"/>
              <joint name="{r.joint_names[3]}" class="arm_joint" range="{_v(-lim[3], lim[3])}" armature="{_f(r.armature[3])}"/>
              <geom class="visual" type="cylinder" size="0.042 0.05" pos="0 0 0.105" material="ur_light"/>
              <geom class="visual" type="cylinder" size="0.043 0.005" pos="0 0 0.16" material="ur_cap"/>
              <body name="wrist_2_link" pos="{_v(a[3], 0, d[3])}" quat="0.707107 0.707107 0 0" gravcomp="1">
                <inertial pos="0 0 0.08" mass="1.3" diaginertia="0.0026 0.0026 0.0022"/>
                <joint name="{r.joint_names[4]}" class="arm_joint" range="{_v(-lim[4], lim[4])}" armature="{_f(r.armature[4])}"/>
                <geom class="visual" type="cylinder" size="0.042 0.048" pos="0 0 0.078" material="ur_light"/>
                <geom class="visual" type="cylinder" size="0.043 0.005" pos="0 0 0.03" material="ur_cap"/>
                <body name="wrist_3_link" pos="{_v(a[4], 0, d[4])}" quat="0.707107 -0.707107 0 0" gravcomp="1">
                  <inertial pos="0 0 0.07" mass="0.365" diaginertia="0.0002 0.0002 0.0003"/>
                  <joint name="{r.joint_names[5]}" class="arm_joint" range="{_v(-lim[5], lim[5])}" armature="{_f(r.armature[5])}"/>
                  <geom class="visual" type="cylinder" size="0.042 0.035" pos="0 0 0.045" material="ur_light"/>
                  <geom class="visual" type="cylinder" size="0.032 0.005" pos="{_v(0, 0, d[5] - 0.005)}" material="ur_dark"/>
                  <site name="tool0" pos="{_v(0, 0, d[5])}" size="0.004" group="5"/>
{_tool(cfg, indent="                  ")}
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
"""
    return arm


def _tool(cfg: CellConfig, indent: str) -> str:
    """F/T sensor + parallel gripper mounted on tool0 (z = approach direction)."""
    r = cfg.robot
    d6 = r.dh_d[5]
    ft = r.ft_thickness
    tcp = r.tcp_offset
    house_len = 0.075
    # Gripper frame origin = sensor tool face; TCP is `tcp - ft` further along z.
    tcp_g = tcp - ft
    finger_base = house_len                   # fingers hang below the housing
    pad_h, pad_w, pad_t = 0.012, 0.010, 0.003  # pad half-sizes: along z, along x, thickness
    pad_center_z = tcp_g + 0.006 - pad_h      # pads end 6 mm beyond the TCP (towards the object)
    # geometry is expressed in the finger frame (origin at y = 0 when closed)
    finger_len = pad_center_z - finger_base
    fr = cfg.robot.pad_friction
    col = _col(GRIPPER)
    return f"""{indent}<body name="ft_sensor" pos="{_v(0, 0, d6)}" gravcomp="1">
{indent}  <inertial pos="0 0 {_f(ft / 2)}" mass="0.3" diaginertia="0.0002 0.0002 0.0003"/>
{indent}  <geom name="ft_body" type="cylinder" size="0.0375 {_f(ft / 2)}" pos="0 0 {_f(ft / 2)}" material="ft" {col}/>
{indent}  <geom class="visual" type="cylinder" size="0.0385 0.004" pos="0 0 {_f(ft * 0.3)}" material="ur_cap"/>
{indent}  <body name="gripper" pos="0 0 {_f(ft)}" gravcomp="1">
{indent}    <inertial pos="0 0 0.035" mass="{_f(r.gripper_mass - 0.1)}" diaginertia="0.0012 0.0010 0.0006"/>
{indent}    <site name="ft_site" pos="0 0 0" size="0.006" group="5"/>
{indent}    <site name="tcp" pos="0 0 {_f(tcp_g)}" size="0.004" rgba="1 0 0 1" group="5"/>
{indent}    <geom name="gripper_housing" type="box" size="0.032 0.047 {_f(house_len / 2)}" pos="0 0 {_f(house_len / 2)}" material="gripper" {col}/>
{indent}    <geom class="visual" type="box" size="0.033 0.020 0.004" pos="0 0 0.012" material="ur_cap"/>
{indent}    <camera name="wrist" mode="fixed" pos="-0.045 0 0.03" xyaxes="{_xyaxes_lookat((-0.045, 0, 0.03), (0.0, 0.0, tcp_g + 0.01), up=(1.0, 0.0, 0.0))}" fovy="75"/>
{indent}    <body name="finger_left" pos="0 0 {_f(finger_base)}" gravcomp="1">
{indent}      <inertial pos="0 0.008 {_f(finger_len / 2)}" mass="0.05" diaginertia="2e-5 2e-5 5e-6"/>
{indent}      <joint name="finger_left_joint" type="slide" axis="0 1 0" range="0 {_f(r.finger_stroke)}" damping="2" armature="0.02"/>
{indent}      <geom name="finger_left_bar" type="box" size="{_f(pad_w)} 0.005 {_f(finger_len / 2)}" pos="0 {_f(pad_t * 2 + 0.005)} {_f(finger_len / 2)}" material="gripper" {col}/>
{indent}      <geom name="pad_left" type="box" size="{_f(pad_w)} {_f(pad_t)} {_f(pad_h)}" pos="0 {_f(pad_t)} {_f(finger_len)}" material="pad" {col} condim="4" friction="{_f(fr)} 0.02 0.0001"/>
{indent}    </body>
{indent}    <body name="finger_right" pos="0 0 {_f(finger_base)}" gravcomp="1">
{indent}      <inertial pos="0 -0.008 {_f(finger_len / 2)}" mass="0.05" diaginertia="2e-5 2e-5 5e-6"/>
{indent}      <joint name="finger_right_joint" type="slide" axis="0 -1 0" range="0 {_f(r.finger_stroke)}" damping="2" armature="0.02"/>
{indent}      <geom name="finger_right_bar" type="box" size="{_f(pad_w)} 0.005 {_f(finger_len / 2)}" pos="0 {_f(-(pad_t * 2 + 0.005))} {_f(finger_len / 2)}" material="gripper" {col}/>
{indent}      <geom name="pad_right" type="box" size="{_f(pad_w)} {_f(pad_t)} {_f(pad_h)}" pos="0 {_f(-pad_t)} {_f(finger_len)}" material="pad" {col} condim="4" friction="{_f(fr)} 0.02 0.0001"/>
{indent}    </body>
{indent}  </body>
{indent}</body>"""


def _fork(cfg: CellConfig, inst: CellInstance, i: int, pose: Pose2) -> str:
    f = cfg.fork
    zb = inst.board_z
    hd = f.depth / 2
    yl = f.slot_width / 2 + f.prong_thickness / 2
    lip_y = f.lip_gap / 2 + f.lip_radius
    r_l = f.lip_radius
    h = f.prong_height
    col = _col(PRONG)
    fric = f'friction="{_f(0.2)} 0.005 0.0001"'

    def prong(side: str, sgn: int) -> str:
        # sgn = +1 for the +y jaw. Jaws are spring-loaded slides that open outwards:
        # the rounded lip tops wedge them open when the wire is pushed down, while the
        # flat barb underneath gives no opening force when the wire is pulled up.
        barb_half_y = (yl - f.lip_gap / 2) / 2
        barb_y = sgn * (f.lip_gap / 2 - yl) / 2
        return f"""      <body name="fork{i}_prong_{side}" pos="{_v(0, sgn * yl, f.post_height)}">
        <inertial pos="0 0 {_f(h / 2)}" mass="0.002" diaginertia="1.4e-7 1.3e-7 3e-8"/>
        <joint name="fork{i}_hinge_{side}" type="slide" axis="0 {sgn} 0" range="0 {_f(f.jaw_travel)}" stiffness="{_f(f.spring_stiffness)}" springref="{_f(-f.spring_preload)}" damping="{_f(f.damping)}" armature="{_f(f.armature)}"/>
        <geom name="fork{i}_prong_{side}" type="box" size="{_v(hd, f.prong_thickness / 2, h / 2)}" pos="0 0 {_f(h / 2)}" material="fork" {col} {fric}/>
        <geom name="fork{i}_lip_{side}" type="capsule" size="{_v(r_l, hd)}" pos="{_v(0, sgn * (lip_y - yl), h - r_l)}" quat="0.707107 0 0.707107 0" material="lip" {col} {fric}/>
        <geom name="fork{i}_barb_{side}" type="box" size="{_v(hd, barb_half_y, r_l / 2)}" pos="{_v(0, barb_y, h - 1.5 * r_l)}" material="lip" {col} {fric}/>
      </body>
"""
    return f"""    <body name="fork{i}" pos="{_v(pose.x, pose.y, zb)}" quat="{_quat_yaw(pose.yaw)}">
      <site name="fork{i}_slot" pos="{_v(0, 0, f.post_height)}" size="0.003" group="5"/>
      <geom name="fork{i}_post" class="env" type="box" size="{_v(hd, f.slot_width / 2 + f.prong_thickness, f.post_height / 2)}" pos="0 0 {_f(f.post_height / 2)}" material="fork"/>
      <geom name="fork{i}_foot" class="env" type="box" size="{_v(hd + 0.006, f.slot_width / 2 + f.prong_thickness + 0.006, 0.002)}" pos="0 0 0.002" material="fork"/>
{prong("l", +1)}{prong("r", -1)}    </body>
"""


def _holder(cfg: CellConfig, inst: CellInstance) -> str:
    h = cfg.holder
    c = cfg.connector
    zb = inst.board_z
    Lp = c.length + 2 * h.clearance
    Wp = c.width + 2 * h.clearance
    t = h.wall_thickness
    fh = h.floor_height
    wh = h.end_wall_height
    rh = h.rail_height
    ch = h.chamfer
    ws = h.wire_slot_width
    pose = inst.holder
    geoms = []
    geoms.append(f'<geom name="holder_base" class="env" type="box" size="{_v(Lp / 2 + t, Wp / 2 + t, fh / 2)}" pos="{_v(0, 0, fh / 2)}" material="holder"/>')
    # front wall with rounded lead-in on the inner top edge
    wall_z = fh + (wh - ch) / 2
    geoms.append(f'<geom name="holder_front" class="env" type="box" size="{_v(t / 2, Wp / 2 + t, (wh - ch) / 2)}" pos="{_v(Lp / 2 + t / 2, 0, wall_z)}" material="holder"/>')
    geoms.append(f'<geom name="holder_front_lead" class="env" type="capsule" size="{_v(ch, Wp / 2 + t - ch)}" pos="{_v(Lp / 2 + ch, 0, fh + wh - ch)}" quat="0.707107 0.707107 0 0" material="holder"/>')
    # back wall split by the wire slot
    half_y = (Wp / 2 + t - ws / 2) / 2
    for sgn, nm in ((1, "l"), (-1, "r")):
        yc = sgn * (ws / 2 + half_y)
        geoms.append(f'<geom name="holder_back_{nm}" class="env" type="box" size="{_v(t / 2, half_y, (wh - ch) / 2)}" pos="{_v(-(Lp / 2 + t / 2), yc, wall_z)}" material="holder"/>')
        geoms.append(f'<geom name="holder_back_lead_{nm}" class="env" type="capsule" size="{_v(ch, max(half_y - ch, 1e-4))}" pos="{_v(-(Lp / 2 + ch), yc, fh + wh - ch)}" quat="0.707107 0.707107 0 0" material="holder"/>')
        # low side rails with a rounded lead-in along their inner top edge
        geoms.append(f'<geom name="holder_rail_{nm}" class="env" type="box" size="{_v(Lp / 2, t / 2, rh / 2)}" pos="{_v(0, sgn * (Wp / 2 + t / 2), fh + rh / 2)}" material="holder"/>')
        rr = 0.5 * rh
        geoms.append(f'<geom name="holder_rail_lead_{nm}" class="env" type="capsule" size="{_v(rr, Lp / 2 - rr)}" pos="{_v(0, sgn * (Wp / 2 + rr), fh + rh)}" quat="0.707107 0 0.707107 0" material="holder"/>')
    body = "\n      ".join(geoms)
    seat_z = fh + c.height / 2
    return f"""    <body name="holder" pos="{_v(pose.x, pose.y, zb)}" quat="{_quat_yaw(pose.yaw)}">
      <site name="holder_seat" pos="{_v(0, 0, seat_z)}" size="0.003" group="5"/>
      {body}
    </body>
"""


def _clamp_and_wire(cfg: CellConfig, inst: CellInstance) -> str:
    w = inst.wire
    L = w.segment_length
    r = w.radius
    n = inst.n_segments
    zc = cfg.wire.clamp_height
    a = inst.anchor
    fr = inst.friction_scale
    c = cfg.connector
    seg_mass = np.pi * r * r * L * w.density
    col = _col(WIRE)
    # Chain is written straight (this is the stress-free shape seen by the cable
    # plugin); the curved initial layout is applied through qpos at reset.
    chain_open = []
    for i in range(n):
        pos = "0 0 0" if i == 0 else f"{_f(L)} 0 0"
        joint = "" if i == 0 else (f'<joint name="wire_j{i}" type="ball" damping="{_f(w.joint_damping)}" '
                                   f'armature="{_f(w.joint_armature)}"/>')
        chain_open.append(
            f'<body name="wire_{i}" pos="{pos}">{joint}'
            f'<geom name="wire_g{i}" type="capsule" size="{_v(r, L / 2)}" pos="{_f(L / 2)} 0 0" '
            f'quat="0.707107 0 -0.707107 0" mass="{_f(seg_mass)}" {col} condim="3" '
            f'friction="{_f(w.friction * fr)} 0.005 0.0001" material="wire"/>'
            f'<plugin instance="wire"/>')
    conn = (f'<body name="connector" pos="{_f(L + c.length / 2)} 0 0">'
            f'<inertial pos="0 0 0" mass="{_f(c.mass)}" diaginertia="4e-7 5e-7 5e-7"/>'
            f'<geom name="connector" type="box" size="{_v(c.length / 2, c.width / 2, c.height / 2)}" '
            f'{col} condim="4" friction="0.6 0.01 0.0001" material="connector"/>'
            f'<geom class="visual" type="box" size="{_v(0.001, c.width / 2 - 0.002, c.height / 2 - 0.003)}" '
            f'pos="{_f(c.length / 2)} 0 0" rgba="0.8 0.7 0.2 1"/>'
            f'<site name="connector" pos="0 0 0" size="0.003" group="5"/></body>')
    chain = "".join(chain_open) + conn + "</body>" * n
    return f"""    <body name="clamp" pos="{_v(a.x, a.y, inst.board_z)}" quat="{_quat_yaw(a.yaw)}">
      <site name="anchor" pos="{_v(L, 0, zc)}" size="0.003" group="5"/>
      <geom name="clamp_base" class="env" type="box" size="{_v(0.018, 0.018, (zc - r - 0.0005) / 2)}" pos="{_v(0.004, 0, (zc - r - 0.0005) / 2)}" material="clamp"/>
      <geom name="clamp_top" class="visual" type="box" size="{_v(0.018, 0.018, 0.005)}" pos="{_v(0.004, 0, zc + r + 0.005)}" material="clamp"/>
      <geom class="visual" type="cylinder" size="0.0035 0.009" pos="{_v(-0.006, 0, zc + r + 0.012)}" rgba="0.6 0.6 0.62 1"/>
      <body name="wire_root" pos="{_v(0, 0, zc)}">
        {chain}
      </body>
    </body>
"""


def _equality_actuators_sensors(cfg: CellConfig) -> str:
    r = cfg.robot
    acts = []
    for i, name in enumerate(r.joint_names):
        kp, kv = r.servo_kp[i], r.servo_kv[i]
        lim = r.joint_limits[i]
        acts.append(f'    <general name="{name}_servo" joint="{name}" gaintype="fixed" biastype="affine" '
                    f'gainprm="{_f(kp)}" biasprm="0 {_f(-kp)} {_f(-kv)}" ctrlrange="{_v(-lim - 0.5, lim + 0.5)}" '
                    f'forcerange="{_v(-r.joint_torque_limits[i], r.joint_torque_limits[i])}"/>')
    acts.append(f'    <general name="gripper" joint="finger_left_joint" gaintype="fixed" biastype="affine" '
                f'gainprm="5000" biasprm="0 -5000 -40" ctrlrange="{_v(-0.006, r.finger_stroke)}" '
                f'forcerange="{_v(-r.grip_force, r.grip_force)}"/>')
    latch = ('\n    <weld name="connector_latch" body1="connector" body2="holder" active="false" '
             'solref="0.01 1"/>' if cfg.holder.latch else "")
    return f"""  <equality>
    <joint name="finger_coupling" joint1="finger_right_joint" joint2="finger_left_joint" polycoef="0 1 0 0 0" solref="0.005 1"/>{latch}
  </equality>
  <actuator>
{chr(10).join(acts)}
  </actuator>
  <sensor>
    <force name="ft_force" site="ft_site"/>
    <torque name="ft_torque" site="ft_site"/>
  </sensor>
"""
