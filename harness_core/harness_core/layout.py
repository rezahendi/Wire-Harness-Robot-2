"""Sampling of a concrete cell instance (board layout + wire) from a CellConfig."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .config import CellConfig, WireParams


@dataclass
class Pose2:
    x: float
    y: float
    yaw: float

    @property
    def xy(self) -> np.ndarray:
        return np.array([self.x, self.y])

    @property
    def dir(self) -> np.ndarray:
        return np.array([np.cos(self.yaw), np.sin(self.yaw)])

    def as_list(self) -> List[float]:
        return [float(self.x), float(self.y), float(self.yaw)]


@dataclass
class CellInstance:
    """Everything needed to build one episode's MuJoCo model."""

    board_z: float                      # z of the board top surface
    anchor: Pose2                       # clamp exit point, wire direction
    forks: List[Pose2]                  # slot centre, slot direction
    holder: Pose2                       # pocket centre, +x points away from the last fork
    wire: WireParams                    # resolved wire parameters (after randomisation)
    n_segments: int
    friction_scale: float
    initial_wire: np.ndarray            # (n_segments + 1, 3) vertex positions
    seed: Optional[int] = None

    @property
    def wire_length(self) -> float:
        return self.n_segments * self.wire.segment_length

    def route_points(self, connector_length: float) -> np.ndarray:
        """Ideal routed polyline (xy): anchor, forks, wire exit of the seated connector."""
        pts = [self.anchor.xy] + [f.xy for f in self.forks]
        pts.append(self.holder.xy - 0.5 * connector_length * self.holder.dir)
        return np.array(pts)

    def to_dict(self) -> dict:
        return {
            "board_z": self.board_z,
            "anchor": self.anchor.as_list(),
            "forks": [f.as_list() for f in self.forks],
            "holder": self.holder.as_list(),
            "wire": dataclasses.asdict(self.wire),
            "n_segments": self.n_segments,
            "friction_scale": self.friction_scale,
            "seed": self.seed,
        }


def _angle_of(v: np.ndarray) -> float:
    return float(np.arctan2(v[1], v[0]))


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def sample_instance(cfg: CellConfig, rng: Optional[np.random.Generator] = None,
                    randomize: Optional[bool] = None, seed: Optional[int] = None) -> CellInstance:
    """Sample (or build the nominal) cell instance."""
    if rng is None:
        rng = np.random.default_rng(seed)
    rnd = cfg.randomization
    randomize = rnd.enabled if randomize is None else randomize
    lay = cfg.layout

    def jitter(scale: float) -> float:
        return float(rng.uniform(-scale, scale)) if randomize else 0.0

    def uniform(rng_pair, nominal):
        return float(rng.uniform(*rng_pair)) if randomize else float(nominal)

    board_z = float(lay.board_thickness)
    anchor_xy = np.array(lay.anchor_xy) + np.array([jitter(rnd.anchor_pos), jitter(rnd.anchor_pos)])
    fork_xy = [np.array(p) + np.array([jitter(rnd.fork_pos), jitter(rnd.fork_pos)]) for p in lay.fork_xy]
    holder_xy = np.array(lay.holder_xy) + np.array([jitter(rnd.holder_pos), jitter(rnd.holder_pos)])

    pts = [anchor_xy] + fork_xy + [holder_xy]
    anchor_yaw = _angle_of(pts[1] - pts[0])
    forks = []
    for i, p in enumerate(fork_xy):
        d_in = _unit(pts[i + 1] - pts[i])
        d_out = _unit(pts[i + 2] - pts[i + 1])
        yaw = _angle_of(_unit(d_in + d_out)) + jitter(rnd.fork_yaw)
        forks.append(Pose2(float(p[0]), float(p[1]), yaw))
    holder_yaw = _angle_of(holder_xy - fork_xy[-1]) + jitter(rnd.holder_yaw)
    anchor = Pose2(float(anchor_xy[0]), float(anchor_xy[1]), anchor_yaw)
    holder = Pose2(float(holder_xy[0]), float(holder_xy[1]), holder_yaw)

    wire = dataclasses.replace(cfg.wire)
    wire.slack = uniform(rnd.slack, cfg.wire.slack)
    bend_scale = uniform(rnd.bend_scale, 1.0)
    wire.bend_modulus = cfg.wire.bend_modulus * bend_scale
    wire.twist_modulus = cfg.wire.twist_modulus * bend_scale
    friction_scale = uniform(rnd.friction_scale, 1.0)

    route = np.array([anchor.xy] + [f.xy for f in forks]
                     + [holder.xy - 0.5 * cfg.connector.length * holder.dir])
    path_len = float(np.sum(np.linalg.norm(np.diff(route, axis=0), axis=1)))
    n_seg = int(np.ceil((path_len + wire.slack) / wire.segment_length))

    inst = CellInstance(board_z=board_z, anchor=anchor, forks=forks, holder=holder, wire=wire,
                        n_segments=n_seg, friction_scale=friction_scale,
                        initial_wire=np.zeros((n_seg + 1, 3)), seed=seed)

    # Initial wire shape: rejection-sample a curve that stays clear of the fixtures,
    # falling back to a deterministic search around the nominal shape.
    candidates = []
    if randomize:
        for _ in range(60):
            candidates.append((float(rng.uniform(*rnd.initial_turn)),
                               float(rng.uniform(*rnd.initial_arc_radius)),
                               float(rng.uniform(*rnd.initial_straight_bend))))
    grid = [(lay.initial_turn + dt, lay.initial_arc_radius + dr, lay.initial_straight_bend + db)
            for dt in (0.0, -0.1, 0.1, -0.2, 0.2) for dr in (0.0, 0.02, -0.02) for db in (0.0, 0.3, -0.3, 0.6)]
    candidates += sorted(grid, key=lambda c: abs(c[0] - lay.initial_turn) + 3 * abs(c[1] - lay.initial_arc_radius)
                         + 0.2 * abs(c[2] - lay.initial_straight_bend))
    verts = None
    for turn, radius, bend in candidates:
        verts = initial_wire_curve(cfg, inst, turn, radius, bend)
        if _wire_is_clear(cfg, inst, verts):
            break
    inst.initial_wire = verts
    return inst


def support_height(cfg: CellConfig, xy: np.ndarray) -> float:
    """Height of the surface under a point: board top or table (z = 0)."""
    cx, cy = cfg.layout.board_center
    sx, sy = cfg.layout.board_size
    inside = abs(xy[0] - cx) <= sx / 2 and abs(xy[1] - cy) <= sy / 2
    return float(cfg.layout.board_thickness) if inside else 0.0


def initial_wire_curve(cfg: CellConfig, inst: CellInstance, turn: float, radius: float,
                       straight_bend: float) -> np.ndarray:
    """Planar curve leaving the clamp along the anchor direction, arcing left then running on."""
    n = inst.n_segments
    L = inst.wire.segment_length
    r = inst.wire.radius
    theta = inst.anchor.yaw
    p = inst.anchor.xy.copy()
    arc_len = abs(turn) * radius
    xy = [p.copy()]
    headings = []
    for i in range(n):
        s_mid = (i + 0.5) * L
        # first segment stays straight inside the clamp
        if i == 0:
            kappa = 0.0
        elif s_mid < L + arc_len:
            kappa = np.sign(turn) / radius
        else:
            kappa = straight_bend
        theta = theta + kappa * L
        headings.append(theta)
        p = p + L * np.array([np.cos(theta), np.sin(theta)])
        xy.append(p.copy())
    xy = np.array(xy)
    z_clamp = inst.board_z + cfg.wire.clamp_height
    verts = np.zeros((n + 1, 3))
    verts[:, :2] = xy
    for i in range(n + 1):
        s = i * L
        z_rest = support_height(cfg, xy[i]) + r
        w = np.clip(1.0 - (s - L) / 0.06, 0.0, 1.0) if s > L else 1.0
        w = w * w * (3 - 2 * w)
        verts[i, 2] = z_rest + (z_clamp - z_rest) * w
    return verts


def _wire_is_clear(cfg: CellConfig, inst: CellInstance, verts: np.ndarray) -> bool:
    # dense samples along the wire plus the connector
    pts = []
    for a, b in zip(verts[:-1], verts[1:]):
        for t in np.linspace(0.0, 1.0, 4, endpoint=False):
            pts.append(a + t * (b - a))
    d_end = _unit(verts[-1, :2] - verts[-2, :2])
    for t in np.linspace(0.0, cfg.connector.length, 4):
        pts.append(np.array([*(verts[-1, :2] + t * d_end), verts[-1, 2]]))
    pts = np.array(pts)[3:]  # ignore the part inside the clamp
    xy = pts[:, :2]
    for f in inst.forks:
        if np.min(np.linalg.norm(xy - f.xy, axis=1)) < 0.045:
            return False
    if np.min(np.linalg.norm(xy - inst.holder.xy, axis=1)) < 0.06:
        return False
    # stay on the board (a wire end hanging over the board edge snags when dragged),
    # inside the robot's reachable area and away from the robot base
    cx, cy = cfg.layout.board_center
    sx, sy = cfg.layout.board_size
    margin = 0.025
    if (np.any(np.abs(xy[:, 0] - cx) > sx / 2 - margin)
            or np.any(np.abs(xy[:, 1] - cy) > sy / 2 - margin)):
        return False
    rad = np.linalg.norm(xy, axis=1)
    if rad.max() > 0.74 or rad.min() < 0.26:
        return False
    # no self-intersection (coarse)
    v2 = verts[:, :2]
    for i in range(len(v2)):
        dist = np.linalg.norm(v2[i + 3:] - v2[i], axis=1)
        if dist.size and dist.min() < 2.5 * cfg.wire.radius + 0.004:
            return False
    return True
