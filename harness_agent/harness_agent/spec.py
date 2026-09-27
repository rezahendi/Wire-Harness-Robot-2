"""Harness specification: what a harness drawing says, in a form the cell can build.

A formboard drawing gives fixture positions in board millimetres, the route of each wire
through the fixtures and the part numbers involved. ``HarnessSpec`` holds exactly that,
``validate()`` checks whether the cell can build it (and says why not in words an agent
can act on), and ``to_cell_config()`` turns it into the cell's layout.

Board coordinates (as on the drawing): ``u`` runs along the long edge of the board,
``v`` along the short edge, both in mm from the lower-left corner, with the robot below
the lower edge. In robot coordinates that is

    x = board_x_min + v / 1000,   y = board_y_min + u / 1000

The cell currently builds one wire from a clamp, through any number of forks, into one
connector holder; the format already allows several wires so specs do not have to change
when the cell grows.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from harness_core.config import CellConfig

STIFFNESS_CLASSES = {"soft": 0.6, "nominal": 1.0, "stiff": 1.4}


@dataclass
class Fixture:
    id: str
    at_mm: Tuple[float, float]           # (u, v) on the board
    kind: str = "fork"                   # fork | clamp | holder
    part_number: str = ""


@dataclass
class Connector:
    id: str
    holder_at_mm: Tuple[float, float]
    part_number: str = ""
    description: str = ""


@dataclass
class Wire:
    id: str
    route: List[str]                     # fork ids, in order
    start: str                           # clamp id
    end: str                             # connector id
    part_number: str = ""
    diameter_mm: float = 6.0
    stiffness: str = "nominal"           # soft | nominal | stiff
    slack_mm: float = 75.0


@dataclass
class Issue:
    severity: str                        # error | warning
    code: str
    message: str

    def as_dict(self) -> Dict[str, str]:
        return {"severity": self.severity, "code": self.code, "message": self.message}


@dataclass
class HarnessSpec:
    name: str
    revision: str = "A"
    board_size_mm: Tuple[float, float] = (1000.0, 440.0)
    clamps: List[Fixture] = field(default_factory=list)
    forks: List[Fixture] = field(default_factory=list)
    connectors: List[Connector] = field(default_factory=list)
    wires: List[Wire] = field(default_factory=list)
    notes: str = ""

    # ----------------------------------------------------------------- io
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HarnessSpec":
        h = d.get("harness", {})
        board = d.get("board", {})
        clamps = [Fixture(c["id"], tuple(c["at_mm"]), "clamp", c.get("part_number", ""))
                  for c in d.get("clamps", [])]
        if "clamp" in d:                                   # single-clamp shorthand
            c = d["clamp"]
            clamps.append(Fixture(c["id"], tuple(c["at_mm"]), "clamp", c.get("part_number", "")))
        forks = [Fixture(f["id"], tuple(f["at_mm"]), "fork", f.get("part_number", ""))
                 for f in d.get("forks", [])]
        conns = [Connector(c["id"], tuple(c["holder_at_mm"]), c.get("part_number", ""),
                           c.get("description", "")) for c in d.get("connectors", [])]
        wires = [Wire(w["id"], list(w["route"]), w["from"], w["to"], w.get("part_number", ""),
                      float(w.get("diameter_mm", 6.0)), str(w.get("stiffness", "nominal")),
                      float(w.get("slack_mm", 75.0))) for w in d.get("wires", [])]
        return cls(name=str(h.get("name", "unnamed harness")), revision=str(h.get("revision", "A")),
                   board_size_mm=tuple(board.get("size_mm", (1000.0, 440.0))),
                   clamps=clamps, forks=forks, connectors=conns, wires=wires,
                   notes=str(h.get("notes", "")))

    @classmethod
    def from_yaml(cls, path: str) -> "HarnessSpec":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f) or {})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "harness": {"name": self.name, "revision": self.revision, "notes": self.notes},
            "board": {"size_mm": list(self.board_size_mm)},
            "clamps": [{"id": c.id, "at_mm": list(c.at_mm), "part_number": c.part_number}
                       for c in self.clamps],
            "forks": [{"id": f.id, "at_mm": list(f.at_mm)} for f in self.forks],
            "connectors": [{"id": c.id, "holder_at_mm": list(c.holder_at_mm),
                            "part_number": c.part_number, "description": c.description}
                           for c in self.connectors],
            "wires": [{"id": w.id, "from": w.start, "route": list(w.route), "to": w.end,
                       "part_number": w.part_number, "diameter_mm": w.diameter_mm,
                       "stiffness": w.stiffness, "slack_mm": w.slack_mm} for w in self.wires],
        }

    def to_yaml(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    # ------------------------------------------------------------ lookups
    def fork(self, fid: str) -> Optional[Fixture]:
        return next((f for f in self.forks if f.id == fid), None)

    def clamp(self, cid: str) -> Optional[Fixture]:
        return next((c for c in self.clamps if c.id == cid), None)

    def connector(self, cid: str) -> Optional[Connector]:
        return next((c for c in self.connectors if c.id == cid), None)

    def summary(self) -> Dict[str, Any]:
        """Compact description for an agent prompt."""
        w = self.wires[0] if self.wires else None
        return {
            "harness": f"{self.name} rev {self.revision}",
            "wires": [{"id": x.id, "part_number": x.part_number, "diameter_mm": x.diameter_mm,
                       "stiffness": x.stiffness, "route": [x.start] + x.route + [x.end]}
                      for x in self.wires],
            "forks": {f.id: list(f.at_mm) for f in self.forks},
            "connector": ({"id": w.end, "part_number": self.connector(w.end).part_number}
                          if w and self.connector(w.end) else None),
        }


# ------------------------------------------------------------ coordinates
def board_frame(cfg: CellConfig) -> Tuple[float, float]:
    """Robot-frame (x_min, y_min) of the board corner that is (0, 0) on the drawing."""
    cx, cy = cfg.layout.board_center
    sx, sy = cfg.layout.board_size
    return cx - sx / 2.0, cy - sy / 2.0


def board_to_robot(uv_mm, cfg: CellConfig) -> Tuple[float, float]:
    x0, y0 = board_frame(cfg)
    u, v = uv_mm
    return x0 + float(v) / 1000.0, y0 + float(u) / 1000.0


def robot_to_board(xy, cfg: CellConfig) -> Tuple[float, float]:
    x0, y0 = board_frame(cfg)
    x, y = xy
    return (float(y) - y0) * 1000.0, (float(x) - x0) * 1000.0


# ------------------------------------------------------------- validation
LIMITS = {
    "edge_margin_mm": 30.0,
    "min_spacing_mm": 60.0,
    "min_segment_mm": 60.0,
    "max_turn_deg": 110.0,
    "reach_min_m": 0.30,
    "reach_max_m": 0.80,
    "min_slack_mm": 40.0,
    "max_forks": 6,
}


def _route_xy(spec: HarnessSpec, wire: Wire, cfg: CellConfig) -> List[Tuple[str, np.ndarray]]:
    pts = []
    c = spec.clamp(wire.start)
    if c:
        pts.append((c.id, np.array(board_to_robot(c.at_mm, cfg))))
    for fid in wire.route:
        f = spec.fork(fid)
        if f:
            pts.append((f.id, np.array(board_to_robot(f.at_mm, cfg))))
    k = spec.connector(wire.end)
    if k:
        pts.append((k.id, np.array(board_to_robot(k.holder_at_mm, cfg))))
    return pts


def validate(spec: HarnessSpec, cfg: Optional[CellConfig] = None) -> List[Issue]:
    """Can this cell build this harness? Errors block the build, warnings do not."""
    cfg = cfg or CellConfig()
    issues: List[Issue] = []
    err = lambda code, msg: issues.append(Issue("error", code, msg))
    warn = lambda code, msg: issues.append(Issue("warning", code, msg))
    L = LIMITS

    if len(spec.wires) != 1:
        err("wire_count", f"the cell routes exactly one wire per build; the spec has {len(spec.wires)}")
    if not spec.wires:
        return issues
    w = spec.wires[0]
    if spec.clamp(w.start) is None:
        err("unknown_clamp", f"wire {w.id} starts at '{w.start}', which is not a clamp in the spec")
    if spec.connector(w.end) is None:
        err("unknown_connector", f"wire {w.id} ends at '{w.end}', which is not a connector in the spec")
    for fid in w.route:
        if spec.fork(fid) is None:
            err("unknown_fork", f"wire {w.id} is routed through '{fid}', which is not a fork in the spec")
    if len(w.route) == 0:
        err("empty_route", f"wire {w.id} has no forks in its route")
    if len(w.route) > L["max_forks"]:
        err("too_many_forks", f"{len(w.route)} forks in one route; the cell supports up to {L['max_forks']}")
    if len(set(w.route)) != len(w.route):
        err("repeated_fork", f"wire {w.id} passes the same fork twice")
    if w.stiffness not in STIFFNESS_CLASSES:
        err("stiffness", f"unknown stiffness class '{w.stiffness}' (use {', '.join(STIFFNESS_CLASSES)})")
    if not 5.0 <= w.diameter_mm <= 7.0:
        err("diameter", f"bundle diameter {w.diameter_mm} mm does not fit the forks on this board, "
                        f"which are sized for 5-7 mm bundles")
    if w.slack_mm < L["min_slack_mm"]:
        err("slack", f"slack {w.slack_mm:.0f} mm is too little to lift the wire over the forks "
                     f"(need at least {L['min_slack_mm']:.0f} mm)")
    structural = {"wire_count", "unknown_clamp", "unknown_connector", "unknown_fork", "empty_route"}
    if any(i.code in structural for i in issues):
        return issues                         # geometry checks need a complete route

    su, sv = spec.board_size_mm
    for name, (u, v) in ([(c.id, c.at_mm) for c in spec.clamps] + [(f.id, f.at_mm) for f in spec.forks]
                         + [(k.id, k.holder_at_mm) for k in spec.connectors]):
        m = L["edge_margin_mm"]
        if not (m <= u <= su - m and m <= v <= sv - m):
            err("off_board", f"{name} at ({u:.0f}, {v:.0f}) mm is outside the board or within "
                             f"{m:.0f} mm of its edge")

    pts = _route_xy(spec, w, cfg)
    for name, p in pts:
        r = float(np.linalg.norm(p))
        if not L["reach_min_m"] <= r <= L["reach_max_m"]:
            err("reach", f"{name} is {1000 * r:.0f} mm from the robot base; the arm works between "
                         f"{1000 * L['reach_min_m']:.0f} and {1000 * L['reach_max_m']:.0f} mm")
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = 1000.0 * float(np.linalg.norm(pts[i][1] - pts[j][1]))
            if d < L["min_spacing_mm"]:
                err("spacing", f"{pts[i][0]} and {pts[j][0]} are {d:.0f} mm apart; fixtures need "
                               f"{L['min_spacing_mm']:.0f} mm so the gripper fits between them")
    for i in range(1, len(pts)):
        seg = 1000.0 * float(np.linalg.norm(pts[i][1] - pts[i - 1][1]))
        if seg < L["min_segment_mm"]:
            err("short_segment", f"segment {pts[i - 1][0]} -> {pts[i][0]} is only {seg:.0f} mm")
    for i in range(1, len(pts) - 1):
        a = pts[i][1] - pts[i - 1][1]
        b = pts[i + 1][1] - pts[i][1]
        cosang = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        turn = math.degrees(math.acos(max(-1.0, min(1.0, cosang))))
        if turn > L["max_turn_deg"]:
            err("sharp_turn", f"the route turns {turn:.0f} deg at {pts[i][0]}; the forks hold at most "
                              f"{L['max_turn_deg']:.0f} deg")
        elif turn > 80.0:
            warn("tight_turn", f"the route turns {turn:.0f} deg at {pts[i][0]}; expect extra seating attempts")
    # crossing segments (the wire would have to pass over itself)
    for i in range(1, len(pts)):
        for j in range(i + 2, len(pts)):
            if _segments_cross(pts[i - 1][1], pts[i][1], pts[j - 1][1], pts[j][1]):
                err("self_crossing", f"segment {pts[i - 1][0]}->{pts[i][0]} crosses "
                                     f"{pts[j - 1][0]}->{pts[j][0]}")
    # fixtures that sit on the route without being part of it
    route_ids = {n for n, _ in pts}
    for f in spec.forks:
        if f.id in route_ids:
            continue
        p = np.array(board_to_robot(f.at_mm, cfg))
        for i in range(1, len(pts)):
            if _point_segment_distance(p, pts[i - 1][1], pts[i][1]) < 0.035:
                warn("fixture_in_path", f"unused fork {f.id} sits within 35 mm of segment "
                                        f"{pts[i - 1][0]}->{pts[i][0]}; the wire may snag on it")
    return issues


def _segments_cross(a, b, c, d) -> bool:
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
    return (o1 * o2 < 0) and (o3 * o4 < 0)


def _point_segment_distance(p, a, b) -> float:
    ab = b - a
    t = float(np.clip((p - a) @ ab / max(ab @ ab, 1e-12), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def has_errors(issues: List[Issue]) -> bool:
    return any(i.severity == "error" for i in issues)


# ------------------------------------------------------------ cell config
def to_cell_config(spec: HarnessSpec, base: Optional[CellConfig] = None) -> CellConfig:
    """Cell configuration that builds this harness (the first wire)."""
    cfg = copy.deepcopy(base) if base is not None else CellConfig()
    w = spec.wires[0]
    clamp = spec.clamp(w.start)
    conn = spec.connector(w.end)
    cfg.layout.anchor_xy = board_to_robot(clamp.at_mm, cfg)
    cfg.layout.fork_xy = [board_to_robot(spec.fork(fid).at_mm, cfg) for fid in w.route]
    cfg.layout.holder_xy = board_to_robot(conn.holder_at_mm, cfg)
    cfg.wire.radius = w.diameter_mm / 2000.0
    cfg.wire.slack = w.slack_mm / 1000.0
    scale = STIFFNESS_CLASSES.get(w.stiffness, 1.0)
    cfg.wire.bend_modulus *= scale
    cfg.wire.twist_modulus *= scale
    return cfg
