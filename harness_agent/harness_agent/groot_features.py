"""What GR00T sees and does in the harness cell: shared by the demo recorder and the policy.

The policy gets the same information as the force-guided expert (no simulator ground truth):

    video  scene   a fixed camera over the formboard (256 x 256)
           wrist   the wrist camera next to the fingers (256 x 256)
    state  tcp      TCP position (m) and tool yaw as sin/cos                    5
           command  commanded target minus TCP (m), and its yaw lead (rad)      4
           gripper  opening between the pads (m)                                1
           wrench   filtered force/torque on the tool, world frame (N, Nm)      6
           goal     the fixture this skill works on and where the wire is      7
                    fixed before it: route_fork -> fork x, y, sin/cos yaw +
                    fixation point; insert_connector -> holder x, y, sin/cos
                    yaw + connector position
           cable    8 perceived wire keypoints (noisy perception)              24
           route    the gripper in the target fork's route frame (the line from 8
                    the fixation point through the fork): along, lateral, height
                    above the prongs, each coarse (/10 cm) and fine (tanh of
                    /1 cm), and the gripper yaw against the route as sin/cos
           wire     the nearest perceived wire point in the gripper frame,     8
                    coarse and fine, and the wire direction against the gripper
                    yaw (sin/cos of twice the angle: the fingers are symmetric)
           slot     where the wire crosses the target slot's plane: found,     3
                    lateral offset (tanh of /5 mm), depth below the prong tops
                    (tanh of /1 cm)
           pick     the planned grasp point on the wire (the expert's plan:    8
                    enough wire from the fixation to carry it over the fork) in
                    the gripper frame, coarse and fine, and the wire direction
                    there against the gripper yaw (sin/cos of twice the angle)
           seat     the wire's pull on the tool along the route (/10 N, tanh   5
                    of /5 N), the tool height against the planned seating height
                    (tanh of /1 cm), and how far past the fork the tool is against
                    the planned holding distance (/10 cm, tanh of /1 cm)
    action motion   TCP step dx dy dz (x 1 cm) and yaw step (x 0.15 rad)        4
           gripper  -1 open ... +1 closed                                       1
    language        "route the wire into fork F2" / "insert the connector into its holder"

Actions are the 20 Hz, 5-D actions of the cell's action interface, exactly what the expert
produces; the 500 Hz admittance controller underneath is unchanged.

The route, wire and slot keys (added after the first two models, which used the first 47
values only) give the policy the millimetre offsets it needs for grasping and seating
directly, instead of leaving it to subtract table coordinates. They come from the same
perception the expert uses. The pick and seat keys (layout v4, 79 values) add the expert's
plan, made once at the start of the skill from the same perception (harness_core.expert.
plan_route): where on the wire to grasp, and where to hold it while seating. A policy
trained without them grasped wherever the wire lay on average and pressed too close to the
fork. Recordings store every value of their layout; the runner sends a served model exactly
the keys it was trained with.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

BASE_LAYOUT: List[Tuple[str, int]] = [("tcp", 5), ("command", 4), ("gripper", 1), ("wrench", 6),
                                      ("goal", 7), ("cable", 24)]
GEOMETRY_LAYOUT: List[Tuple[str, int]] = [("route", 8), ("wire", 8), ("slot", 3)]
PLAN_LAYOUT: List[Tuple[str, int]] = [("pick", 8), ("seat", 5)]
V3_LAYOUT: List[Tuple[str, int]] = BASE_LAYOUT + GEOMETRY_LAYOUT
STATE_LAYOUT: List[Tuple[str, int]] = V3_LAYOUT + PLAN_LAYOUT           # what recordings store (v4)
ACTION_LAYOUT: List[Tuple[str, int]] = [("motion", 4), ("gripper", 1)]
BASE_DIM = sum(n for _, n in BASE_LAYOUT)
V3_DIM = sum(n for _, n in V3_LAYOUT)
STATE_DIM = sum(n for _, n in STATE_LAYOUT)
ACTION_DIM = sum(n for _, n in ACTION_LAYOUT)
STATE_LAYOUTS = {BASE_DIM: BASE_LAYOUT, V3_DIM: V3_LAYOUT, STATE_DIM: STATE_LAYOUT}   # by width, older sets too
CABLE_POINTS = 8
IMAGE_SIZE = 256
VIDEO_KEYS = ("scene", "wrist")
LANGUAGE_KEY = "annotation.human.task_description"
FPS = 20

BASE_NAMES = (["tcp.x", "tcp.y", "tcp.z", "tcp.sin_yaw", "tcp.cos_yaw",
               "command.dx", "command.dy", "command.dz", "command.dyaw", "gripper.opening",
               "wrench.fx", "wrench.fy", "wrench.fz", "wrench.tx", "wrench.ty", "wrench.tz",
               "goal.x", "goal.y", "goal.sin_yaw", "goal.cos_yaw", "goal.fix_x", "goal.fix_y", "goal.fix_z"]
              + [f"cable.{k}.{a}" for k in range(CABLE_POINTS) for a in "xyz"])
V3_NAMES = BASE_NAMES + [
    "route.along", "route.lateral", "route.height", "route.along_fine", "route.lateral_fine",
    "route.height_fine", "route.sin_dyaw", "route.cos_dyaw",
    "wire.x", "wire.y", "wire.z", "wire.x_fine", "wire.y_fine", "wire.z_fine", "wire.sin2_dyaw", "wire.cos2_dyaw",
    "slot.found", "slot.lateral_fine", "slot.depth_fine"]
STATE_NAMES = V3_NAMES + [
    "pick.x", "pick.y", "pick.z", "pick.x_fine", "pick.y_fine", "pick.z_fine", "pick.sin2_dyaw", "pick.cos2_dyaw",
    "seat.pull", "seat.pull_fine", "seat.height_fine", "seat.beyond", "seat.beyond_fine"]
STATE_NAMES_BY_WIDTH = {BASE_DIM: BASE_NAMES, V3_DIM: V3_NAMES, STATE_DIM: STATE_NAMES}
ACTION_NAMES = ["motion.dx", "motion.dy", "motion.dz", "motion.dyaw", "gripper"]


def instruction(skill: str, target: str = "") -> str:
    if skill == "route_fork":
        return f"route the wire into fork {target}"
    if skill == "insert_connector":
        return "insert the connector into its holder"
    if skill == "relocate_connector":
        return "move the connector to a clear spot"
    raise ValueError(f"no instruction for skill {skill!r}")


def layout_slices(layout: Sequence[Tuple[str, int]]) -> Dict[str, Tuple[int, int]]:
    out, k = {}, 0
    for name, n in layout:
        out[name] = (k, k + n)
        k += n
    return out


def _wrap(a: float) -> float:
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


def goal_vector(obs: Dict[str, np.ndarray], skill: str, fork_index: int, cfg) -> np.ndarray:
    """The fixture the skill works on (CAD pose) and the point the wire is fixed at before it."""
    if skill == "route_fork":
        fx, fy, fz, fyaw = (float(v) for v in obs["forks"][fork_index])
        if fork_index == 0:
            fix = np.asarray(obs["anchor_pos"], dtype=float)
        else:
            px, py, pz, _ = (float(v) for v in obs["forks"][fork_index - 1])
            fix = np.array([px, py, pz + cfg.fork.post_height + cfg.wire.radius])
        return np.array([fx, fy, np.sin(fyaw), np.cos(fyaw), fix[0], fix[1], fix[2]])
    hp = np.asarray(obs["holder_pos"], dtype=float)
    hyaw = float(obs["holder_yaw"][0])
    cp = np.asarray(obs["connector_pos"], dtype=float)
    return np.array([hp[0], hp[1], np.sin(hyaw), np.cos(hyaw), cp[0], cp[1], cp[2]])


def layout_for_width(width: int) -> List[Tuple[str, int]]:
    if width not in STATE_LAYOUTS:
        raise ValueError(f"no state layout with {width} values (known: {sorted(STATE_LAYOUTS)})")
    return STATE_LAYOUTS[width]


def _fine(x: float, scale: float) -> float:
    return float(np.tanh(x / scale))


def geometry_features(obs: Dict[str, np.ndarray], fork_index: int, cfg) -> Dict[str, np.ndarray]:
    """The gripper relative to what route_fork works on: the target fork's route line, the
    nearest bit of wire, and the wire's crossing of the slot (see the module docstring)."""
    from harness_core.geometry import closest_point_on_polyline
    from harness_core.perception import cable_crossing_in_fork

    tcp = np.asarray(obs["tcp_pos"], dtype=float)
    yaw = float(obs["tcp_yaw"][0])
    fork = np.asarray(obs["forks"][fork_index], dtype=float)
    bz = float(obs["board_z"][0])
    fix = goal_vector(obs, "route_fork", fork_index, cfg)[4:7]
    u = fork[:2] - fix[:2]
    length = float(np.linalg.norm(u))
    u = u / length if length > 1e-6 else np.array([np.cos(fork[3]), np.sin(fork[3])])
    n = np.array([-u[1], u[0]])
    d = tcp[:2] - fork[:2]
    along, lateral = float(d @ u), float(d @ n)
    top = cfg.fork.post_height + cfg.fork.prong_height
    height = float(tcp[2] - (bz + top))
    dyaw = yaw - float(np.arctan2(u[1], u[0]))
    route = [along / 0.1, lateral / 0.1, height / 0.1, _fine(along, 0.01), _fine(lateral, 0.01),
             _fine(height, 0.01), np.sin(dyaw), np.cos(dyaw)]

    cable = np.asarray(obs["cable"], dtype=float)
    p, _, _, k = closest_point_on_polyline(cable, tcp)
    off = np.asarray(p) - tcp
    c, sn = np.cos(yaw), np.sin(yaw)
    xg, yg, zg = c * off[0] + sn * off[1], -sn * off[0] + c * off[1], off[2]
    seg = cable[min(k + 1, len(cable) - 1)] - cable[k]
    twice = 2.0 * (float(np.arctan2(seg[1], seg[0])) - yaw)
    wire = [xg / 0.1, yg / 0.1, zg / 0.1, _fine(xg, 0.01), _fine(yg, 0.01), _fine(zg, 0.01),
            np.sin(twice), np.cos(twice)]

    chk = cable_crossing_in_fork(cable, fork, cfg.fork, bz)
    if np.isfinite(chk["y"]):
        slot = [1.0, _fine(chk["y"], 0.005), _fine(chk["z"] - top, 0.01)]
    else:
        slot = [0.0, 0.0, 0.0]
    return {"route": np.asarray(route, np.float32), "wire": np.asarray(wire, np.float32),
            "slot": np.asarray(slot, np.float32)}


def plan_features(obs: Dict[str, np.ndarray], fork_index: int, cfg,
                  plan: Optional[Dict[str, float]]) -> Dict[str, np.ndarray]:
    """The gripper against route_fork's plan (harness_core.expert.plan_route): the grasp point
    on the wire, the wire's pull along the route, and the seating height and holding distance."""
    from harness_core.geometry import polyline_arclength, polyline_point_at

    if plan is None:
        return {k: np.zeros(n, np.float32) for k, n in PLAN_LAYOUT}
    tcp = np.asarray(obs["tcp_pos"], dtype=float)
    yaw = float(obs["tcp_yaw"][0])
    c, sn = np.cos(yaw), np.sin(yaw)
    cable = np.asarray(obs["cable"], dtype=float)
    s = float(np.clip(plan["s_pick"], 0.05, polyline_arclength(cable)[-1] - 0.05))
    p_w, t_w = polyline_point_at(cable, s)
    off = np.asarray(p_w, dtype=float) - tcp
    xg, yg, zg = c * off[0] + sn * off[1], -sn * off[0] + c * off[1], off[2]
    twice = 2.0 * (float(np.arctan2(t_w[1], t_w[0])) - yaw)
    pick = [xg / 0.1, yg / 0.1, zg / 0.1, _fine(xg, 0.01), _fine(yg, 0.01), _fine(zg, 0.01),
            np.sin(twice), np.cos(twice)]

    fork = np.asarray(obs["forks"][fork_index], dtype=float)
    fix = goal_vector(obs, "route_fork", fork_index, cfg)[4:7]
    u = fork[:2] - fix[:2]
    length = float(np.linalg.norm(u))
    u = u / length if length > 1e-6 else np.array([np.cos(fork[3]), np.sin(fork[3])])
    f = np.asarray(obs["wrench"], dtype=float)[:3]
    pull = float(-(f[0] * u[0] + f[1] * u[1]))            # the expert's tension: the wire pulling back along -u
    beyond = float((tcp[:2] - fork[:2]) @ u) - float(plan["beyond"])
    seat = [pull / 10.0, _fine(pull, 5.0), _fine(float(tcp[2]) - float(plan["press_z"]), 0.01),
            beyond / 0.1, _fine(beyond, 0.01)]
    return {"pick": np.asarray(pick, np.float32), "seat": np.asarray(seat, np.float32)}


def state_parts(obs: Dict[str, np.ndarray], skill: str, fork_index: int, cfg,
                plan: Optional[Dict[str, float]] = None) -> Dict[str, np.ndarray]:
    """Every state key, by name: the base 47 values, the gripper-relative geometry and the
    gripper against route_fork's plan (zeros for skills other than route_fork, and for the
    plan keys when there is no plan)."""
    parts = split_state(state_vector(obs, goal_vector(obs, skill, fork_index, cfg)), BASE_LAYOUT)
    if skill == "route_fork":
        parts.update(geometry_features(obs, fork_index, cfg))
        parts.update(plan_features(obs, fork_index, cfg, plan))
    else:
        parts.update({k: np.zeros(n, np.float32) for k, n in GEOMETRY_LAYOUT + PLAN_LAYOUT})
    return parts


def join_state(parts: Dict[str, np.ndarray], layout: Sequence[Tuple[str, int]] = STATE_LAYOUT) -> np.ndarray:
    v = np.concatenate([np.asarray(parts[k], dtype=np.float32).reshape(n) for k, n in layout])
    return v.astype(np.float32)


def state_vector(obs: Dict[str, np.ndarray], goal: np.ndarray) -> np.ndarray:
    tcp = np.asarray(obs["tcp_pos"], dtype=float)
    yaw = float(obs["tcp_yaw"][0])
    tgt = np.asarray(obs["target_pos"], dtype=float)
    tyaw = float(obs["target_yaw"][0])
    cable = np.asarray(obs["cable"], dtype=float)
    idx = np.round(np.linspace(0, len(cable) - 1, CABLE_POINTS)).astype(int)
    parts = [tcp, [np.sin(yaw), np.cos(yaw)], tgt - tcp, [_wrap(tyaw - yaw)], [float(obs["gripper"][0])],
             np.asarray(obs["wrench"], dtype=float)[:6], goal, cable[idx].reshape(-1)]
    v = np.concatenate([np.asarray(p, dtype=float).reshape(-1) for p in parts]).astype(np.float32)
    assert v.shape == (BASE_DIM,), v.shape
    return v


def split_state(v: np.ndarray, layout: Optional[Sequence[Tuple[str, int]]] = None) -> Dict[str, np.ndarray]:
    """Named parts of a state vector; the layout follows from its width unless given."""
    layout = layout or layout_for_width(int(np.shape(v)[-1]))
    return {k: v[..., a:b] for k, (a, b) in layout_slices(layout).items()}


def join_action(parts: Dict[str, np.ndarray]) -> np.ndarray:
    """(horizon, 5) actions from the policy's per-key output."""
    return np.concatenate([np.asarray(parts[k], dtype=np.float32).reshape(-1, n) for k, n in ACTION_LAYOUT],
                          axis=-1)


class Cameras:
    """Renders the two policy views. The scene view is a free camera, so the MuJoCo model
    (and with it the physics) is unchanged."""

    SCENE = dict(lookat=(0.50, 0.03, 0.02), distance=0.78, azimuth=180.0, elevation=-58.0)

    def __init__(self, model, size: int = IMAGE_SIZE):
        import mujoco
        self._mj = mujoco
        self.size = size
        self.renderer = mujoco.Renderer(model, size, size)
        self.scene_cam = mujoco.MjvCamera()
        self.scene_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.scene_cam.lookat[:] = self.SCENE["lookat"]
        self.scene_cam.distance = self.SCENE["distance"]
        self.scene_cam.azimuth = self.SCENE["azimuth"]
        self.scene_cam.elevation = self.SCENE["elevation"]
        self.wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist")
        if self.wrist_id < 0:
            raise ValueError("the model has no 'wrist' camera")

    def render(self, data) -> Dict[str, np.ndarray]:
        r = self.renderer
        r.update_scene(data, camera=self.scene_cam)
        scene = r.render().copy()
        r.update_scene(data, camera=self.wrist_id)
        wrist = r.render().copy()
        return {"scene": scene, "wrist": wrist}

    def close(self) -> None:
        try:
            self.renderer.close()
        except Exception:
            pass


def observation_for_policy(images: Dict[str, np.ndarray], state, text: str,
                           state_keys: Optional[Sequence[str]] = None) -> Dict[str, object]:
    """One GR00T observation (batch 1, one time step), in Gr00tPolicy's nested format.

    ``state`` is a vector (47 or 66 values) or a dict of named parts; ``state_keys`` picks the
    keys the served model was trained with (default: all of them)."""
    parts = split_state(np.asarray(state, dtype=np.float32)) if not isinstance(state, dict) else state
    keys = list(state_keys) if state_keys else list(parts)
    return {
        "video": {k: images[k][None, None].astype(np.uint8) for k in VIDEO_KEYS},
        "state": {k: np.asarray(parts[k], dtype=np.float32)[None, None] for k in keys},
        "language": {LANGUAGE_KEY: [[text]]},
    }
