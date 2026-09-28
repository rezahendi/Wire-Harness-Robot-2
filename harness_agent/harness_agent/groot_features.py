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
    action motion   TCP step dx dy dz (x 1 cm) and yaw step (x 0.15 rad)        4
           gripper  -1 open ... +1 closed                                       1
    language        "route the wire into fork F2" / "insert the connector into its holder"

Actions are the 20 Hz, 5-D actions of the cell's action interface, exactly what the expert
produces; the 500 Hz admittance controller underneath is unchanged.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

STATE_LAYOUT: List[Tuple[str, int]] = [("tcp", 5), ("command", 4), ("gripper", 1), ("wrench", 6),
                                       ("goal", 7), ("cable", 24)]
ACTION_LAYOUT: List[Tuple[str, int]] = [("motion", 4), ("gripper", 1)]
STATE_DIM = sum(n for _, n in STATE_LAYOUT)
ACTION_DIM = sum(n for _, n in ACTION_LAYOUT)
CABLE_POINTS = 8
IMAGE_SIZE = 256
VIDEO_KEYS = ("scene", "wrist")
LANGUAGE_KEY = "annotation.human.task_description"
FPS = 20

STATE_NAMES = (["tcp.x", "tcp.y", "tcp.z", "tcp.sin_yaw", "tcp.cos_yaw",
                "command.dx", "command.dy", "command.dz", "command.dyaw", "gripper.opening",
                "wrench.fx", "wrench.fy", "wrench.fz", "wrench.tx", "wrench.ty", "wrench.tz",
                "goal.x", "goal.y", "goal.sin_yaw", "goal.cos_yaw", "goal.fix_x", "goal.fix_y", "goal.fix_z"]
               + [f"cable.{k}.{a}" for k in range(CABLE_POINTS) for a in "xyz"])
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
    assert v.shape == (STATE_DIM,), v.shape
    return v


def split_state(v: np.ndarray) -> Dict[str, np.ndarray]:
    return {k: v[..., a:b] for k, (a, b) in layout_slices(STATE_LAYOUT).items()}


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


def observation_for_policy(images: Dict[str, np.ndarray], state: np.ndarray, text: str) -> Dict[str, object]:
    """One GR00T observation (batch 1, one time step), in Gr00tPolicy's nested format."""
    parts = split_state(state.astype(np.float32))
    return {
        "video": {k: images[k][None, None].astype(np.uint8) for k in VIDEO_KEYS},
        "state": {k: v[None, None].astype(np.float32) for k, v in parts.items()},
        "language": {LANGUAGE_KEY: [[text]]},
    }
