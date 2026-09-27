"""Configuration of the wire-harness cell.

Everything that shapes the cell (geometry, physics, controller gains, noise and
domain randomisation) lives in one nested dataclass, ``CellConfig``. It can be
loaded from / saved to YAML so the ROS nodes, the Gymnasium environment and the
demo recorder all build exactly the same cell.

Units: metres, radians, kilograms, newtons, seconds.
World frame: origin at the robot base, x towards the formboard, z up.
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import yaml


@dataclass
class WireParams:
    """Deformable wire bundle simulated with MuJoCo's cable (Cosserat rod) plugin."""

    radius: float = 0.003            # 6 mm bundle (several FLRY wires taped together)
    segment_length: float = 0.03
    slack: float = 0.075             # extra length on top of the routed path length
    density: float = 2500.0          # kg/m^3  -> ~70 g/m
    bend_modulus: float = 2.0e6      # Young's modulus [Pa] seen by the cable plugin
    twist_modulus: float = 1.3e6     # shear modulus [Pa]
    joint_damping: float = 5e-4
    joint_armature: float = 1e-5     # numerical stabiliser for the tiny segment inertia
    friction: float = 0.5
    clamp_height: float = 0.033      # height of the wire axis in the anchor clamp above the board


@dataclass
class ForkParams:
    """Snap-in harness fork: a post carrying two spring-loaded jaws with barbed lips."""

    post_height: float = 0.030       # slot bottom above the board surface
    prong_height: float = 0.028
    slot_width: float = 0.013        # free width between the prongs below the lips
    prong_thickness: float = 0.004
    depth: float = 0.005             # extent along the wire direction (thin clip)
    lip_radius: float = 0.003
    lip_gap: float = 0.0025          # free gap between the lips (< wire diameter -> snap fit)
    spring_stiffness: float = 150.0  # N/m per spring-loaded jaw
    spring_preload: float = 0.002    # m of spring preload against the closed stop
    jaw_travel: float = 0.0065       # m each jaw can open (lets a wire in at up to ~35 deg)
    damping: float = 0.5
    armature: float = 0.01           # kg (numerical stabiliser for the light jaws)


@dataclass
class ConnectorParams:
    length: float = 0.024            # along the wire axis
    width: float = 0.016
    height: float = 0.016
    mass: float = 0.008


@dataclass
class HolderParams:
    """Connector holder (pocket) with low side rails so the fingers can reach in."""

    clearance: float = 0.0008        # per side
    floor_height: float = 0.0015     # pocket floor above the board
    end_wall_height: float = 0.012   # above the floor
    rail_height: float = 0.003       # above the floor
    wall_thickness: float = 0.006
    wire_slot_width: float = 0.010
    chamfer: float = 0.0035          # size of the 45 deg lead-in on the end walls
    # Snap latch: once the connector is pressed fully into the pocket it clicks in and is
    # held (a weld constraint engages), like the latch of a real formboard holder.
    latch: bool = True
    latch_depth: float = 0.0008      # engages when the connector is within this of the floor
    latch_offset: float = 0.0015     # ... and centred in the pocket to within this


@dataclass
class LayoutParams:
    """Nominal formboard layout (before randomisation). Poses are (x, y, yaw)."""

    board_center: Tuple[float, float] = (0.48, 0.0)
    board_size: Tuple[float, float] = (0.44, 1.00)
    board_thickness: float = 0.02
    anchor_xy: Tuple[float, float] = (0.34, -0.24)
    fork_xy: List[Tuple[float, float]] = field(
        default_factory=lambda: [(0.48, -0.16), (0.58, -0.02), (0.54, 0.13)])
    holder_xy: Tuple[float, float] = (0.44, 0.22)
    # Initial wire: leaves the clamp towards fork 1, turns left by `turn` over an
    # arc of radius `arc_radius`, then runs straight (lying on the board/table).
    initial_turn: float = 1.15
    initial_arc_radius: float = 0.09
    initial_straight_bend: float = 0.0   # curvature of the rest of the wire [1/m]


@dataclass
class RobotParams:
    """UR5e-class arm (UR DH parameters) with wrist F/T sensor and parallel gripper."""

    dh_d: Tuple[float, ...] = (0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996)
    dh_a: Tuple[float, ...] = (0.0, -0.425, -0.3922, 0.0, 0.0, 0.0)
    dh_alpha: Tuple[float, ...] = (1.5707963267948966, 0.0, 0.0,
                                   1.5707963267948966, -1.5707963267948966, 0.0)
    joint_names: Tuple[str, ...] = ("shoulder_pan_joint", "shoulder_lift_joint",
                                    "elbow_joint", "wrist_1_joint", "wrist_2_joint",
                                    "wrist_3_joint")
    joint_limits: Tuple[float, ...] = (6.283, 6.283, 3.1416, 6.283, 6.283, 6.283)
    joint_vel_limits: Tuple[float, ...] = (3.14, 3.14, 3.14, 3.14, 3.14, 3.14)
    joint_torque_limits: Tuple[float, ...] = (150.0, 150.0, 150.0, 28.0, 28.0, 28.0)
    servo_kp: Tuple[float, ...] = (20000.0, 20000.0, 12000.0, 4000.0, 4000.0, 3000.0)
    servo_kv: Tuple[float, ...] = (600.0, 600.0, 300.0, 60.0, 60.0, 40.0)
    armature: Tuple[float, ...] = (0.3, 0.3, 0.15, 0.04, 0.04, 0.03)
    # Home: tool pointing straight down above the middle of the board.
    home_q: Tuple[float, ...] = (-0.3007, -1.6761, 1.9752, -1.8698, -1.5708, -1.8715)
    ft_thickness: float = 0.0375     # flange -> sensor tool side
    gripper_mass: float = 0.65       # housing + fingers (payload the controller compensates)
    tcp_offset: float = 0.17         # tool0 -> TCP (between the finger pads)
    finger_stroke: float = 0.03      # per finger -> max opening 60 mm
    finger_speed: float = 0.075      # m/s per finger (150 mm/s closing speed)
    grip_force: float = 30.0
    pad_friction: float = 1.2


@dataclass
class ControllerParams:
    """Cartesian compliance (admittance) controller, runs at `SimParams.control_dt`."""

    kp_lin: float = 6.0              # 1/s   position error -> velocity
    kp_rot: float = 5.0
    kf_lin: float = 0.002            # (m/s)/N  force error -> velocity (compliance)
    kf_rot: float = 0.08             # (rad/s)/(N m)
    max_lin_vel: float = 0.25
    max_rot_vel: float = 1.2
    max_lin_acc: float = 2.0
    max_rot_acc: float = 8.0
    force_deadband: float = 0.4
    torque_deadband: float = 0.03
    wrench_filter_hz: float = 40.0
    damping_lambda: float = 0.02     # damped least squares
    protective_stop_force: float = 120.0
    protective_stop_torque: float = 10.0


@dataclass
class SimParams:
    timestep: float = 0.001          # physics
    control_dt: float = 0.002        # compliance controller (500 Hz, like the UR5e)
    policy_dt: float = 0.05          # policy / expert (20 Hz)
    episode_time: float = 90.0


@dataclass
class NoiseParams:
    ft_force_std: float = 0.05
    ft_torque_std: float = 0.004
    ft_bias_force: float = 0.3       # random constant bias per episode (re-zero removes it)
    ft_bias_torque: float = 0.02
    cable_point_std: float = 0.0008  # "perception" noise on cable keypoints
    holder_pose_std: float = 0.0015  # perception noise on the holder position
    holder_yaw_std: float = 0.01


@dataclass
class RandomizationParams:
    enabled: bool = True
    anchor_pos: float = 0.008
    fork_pos: float = 0.012
    fork_yaw: float = 0.10
    holder_pos: float = 0.012
    holder_yaw: float = 0.10
    initial_turn: Tuple[float, float] = (0.95, 1.35)
    initial_arc_radius: Tuple[float, float] = (0.07, 0.12)
    initial_straight_bend: Tuple[float, float] = (-0.4, 0.5)
    bend_scale: Tuple[float, float] = (0.6, 1.3)
    friction_scale: Tuple[float, float] = (0.8, 1.25)
    slack: Tuple[float, float] = (0.06, 0.09)


@dataclass
class CellConfig:
    wire: WireParams = field(default_factory=WireParams)
    fork: ForkParams = field(default_factory=ForkParams)
    connector: ConnectorParams = field(default_factory=ConnectorParams)
    holder: HolderParams = field(default_factory=HolderParams)
    layout: LayoutParams = field(default_factory=LayoutParams)
    robot: RobotParams = field(default_factory=RobotParams)
    controller: ControllerParams = field(default_factory=ControllerParams)
    sim: SimParams = field(default_factory=SimParams)
    noise: NoiseParams = field(default_factory=NoiseParams)
    randomization: RandomizationParams = field(default_factory=RandomizationParams)

    # ------------------------------------------------------------------ helpers
    def to_dict(self) -> Dict[str, Any]:
        return _to_plain(dataclasses.asdict(self))

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "CellConfig":
        cfg = cls()
        if data:
            _update_dataclass(cfg, data)
        return cfg

    @classmethod
    def from_yaml(cls, path: str) -> "CellConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        # Allow ROS-style files with a top-level "cell:" key.
        if "cell" in data and isinstance(data["cell"], dict):
            data = data["cell"]
        return cls.from_dict(data)

    def to_yaml(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump({"cell": self.to_dict()}, f, sort_keys=False)

    def copy(self) -> "CellConfig":
        return copy.deepcopy(self)


def _to_plain(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    return obj


def _update_dataclass(obj: Any, data: Dict[str, Any]) -> None:
    for key, value in data.items():
        if not hasattr(obj, key):
            raise KeyError(f"Unknown config key '{key}' for {type(obj).__name__}")
        current = getattr(obj, key)
        if dataclasses.is_dataclass(current) and isinstance(value, dict):
            _update_dataclass(current, value)
        elif isinstance(current, tuple) and isinstance(value, (list, tuple)):
            if current and isinstance(current[0], tuple):
                setattr(obj, key, tuple(tuple(v) for v in value))
            else:
                setattr(obj, key, tuple(value))
        elif isinstance(current, list) and isinstance(value, (list, tuple)):
            if current and isinstance(current[0], tuple):
                setattr(obj, key, [tuple(v) for v in value])
            else:
                setattr(obj, key, list(value))
        else:
            setattr(obj, key, type(current)(value) if current is not None and not isinstance(current, bool) else value)
