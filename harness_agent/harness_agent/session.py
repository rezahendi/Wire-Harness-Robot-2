"""A build session: the simulated cell for one harness spec, driven skill by skill.

The force-guided expert already knows how to route a wire into a fork and how to seat a
connector; what it does not do is decide. ``CellSession`` exposes each of its skills as a
separate, bounded call that returns a structured result, so a planner (the Nemotron agent,
the scripted planner, or a person at a terminal) makes the decisions:

    session = CellSession(spec, seed=3)
    session.route_fork("F1")         # -> SkillResult(ok, messages, forces, perceived state)
    session.status()                 # what perception says now
    session.insert_connector()

What the planner sees is *perception only* (noisy cable keypoints, connector pose, the
wrist wrench): the same information a real cell has. Ground truth from the simulator is
recorded next to it for scoring, and never shown to the planner.

With a ``VisualInspector`` attached, ``inspect`` also photographs each fixture with the
inspection camera and asks a vision-language model about it: a second, independent check
whose verdicts are logged against ground truth as well.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Generator, List, Optional

import numpy as np

from harness_core.config import CellConfig
from harness_core.expert import HarnessExpert
from harness_core.geometry import polyline_arclength, wrap_angle
from harness_core.perception import arclength_near, cable_crossing_in_fork

from .spec import HarnessSpec, has_errors, robot_to_board, to_cell_config, validate


def run_until_phase(gen: Generator, expert, phase: str) -> Generator:
    """Run a skill generator until the expert enters ``phase``, then abandon it."""
    try:
        action = next(gen)
        while expert.phase != phase:
            obs = yield action
            action = gen.send(obs)
    except StopIteration:
        return False
    gen.close()
    return True


@dataclass
class SkillResult:
    skill: str
    args: Dict[str, Any]
    ok: bool
    outcome: str                         # short machine-readable outcome
    messages: List[str]                  # what the skill reported while running
    sim_time_start: float
    sim_time_end: float
    max_force: float
    perceived: Dict[str, Any] = field(default_factory=dict)   # shown to the planner
    truth: Dict[str, Any] = field(default_factory=dict)       # hidden, for scoring
    wall_time: float = 0.0

    def for_planner(self) -> Dict[str, Any]:
        """What the planner is told: no ground truth."""
        return {"skill": self.skill, "args": self.args, "ok": self.ok, "outcome": self.outcome,
                "duration_s": round(self.sim_time_end - self.sim_time_start, 1),
                "max_contact_force_N": round(self.max_force, 1),
                "messages": self.messages[-8:], "state": self.perceived}

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CellSession:
    """One harness build in the simulated cell."""

    def __init__(self, spec: HarnessSpec, seed: int = 0, randomize: bool = False,
                 base_cfg: Optional[CellConfig] = None, max_sim_time: float = 900.0,
                 render: bool = False, camera: str = "overview", frame_every: float = 0.5,
                 on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
                 inspector: Optional[Any] = None, inspection_dir: Optional[str] = None):
        from harness_learning.env import HarnessRoutingEnv   # imported late: pulls in MuJoCo

        self.spec = spec
        self.issues = validate(spec, base_cfg)
        self.feasible = not has_errors(self.issues)
        self.cfg = to_cell_config(spec, base_cfg) if self.feasible else None
        self.route = list(spec.wires[0].route) if spec.wires else []
        self.connector_id = spec.wires[0].end if spec.wires else ""
        self.seed = seed
        self.randomize = randomize
        self.max_sim_time = max_sim_time
        self.frame_every = frame_every
        self.frames: List[np.ndarray] = []
        self.events: List[Dict[str, Any]] = []
        self.on_event = on_event
        self.finished = False
        self.env = None
        self.expert = None
        self.faults: set = set()              # injected by a benchmark scenario, e.g. "slip_on_insert"
        self.inspector = inspector           # VisualInspector (or None: perception only)
        self.inspection_dir = inspection_dir
        self.visual_checks: List[Dict[str, Any]] = []
        self._camera = None
        self._looks: Dict[str, int] = {}     # inspections per target (each one uses a new view)
        self._last_status: Optional[Dict[str, Any]] = None
        if self.feasible:
            self.env = HarnessRoutingEnv(cfg=self.cfg, randomize=randomize,
                                         max_episode_time=max_sim_time + 60.0,
                                         render_mode="rgb_array" if render else None, camera=camera)
            self.env.reset(seed=seed)
            self.expert = HarnessExpert(self.cfg, self.env.spec_actions)
            self.expert._last_obs = self.env.last_obs_dict
            self._last_frame_t = -1e9
            self._grab_frame(force=True)

    # ------------------------------------------------------------- helpers
    @property
    def obs(self) -> Dict[str, np.ndarray]:
        return self.env.last_obs_dict

    @property
    def sim_time(self) -> float:
        return float(self.obs["time"][0]) if self.env is not None else 0.0

    def _grab_frame(self, force: bool = False) -> None:
        if self.env is None or self.env.render_mode != "rgb_array":
            return
        if force or self.sim_time - self._last_frame_t >= self.frame_every:
            self.frames.append(self.env.render())
            self._last_frame_t = self.sim_time

    def _emit(self, event: Dict[str, Any]) -> None:
        self.events.append(event)
        if self.on_event is not None:
            self.on_event(event)

    def _fork_index(self, fork_id: str) -> Optional[int]:
        return self.route.index(fork_id) if fork_id in self.route else None

    def _drive(self, gen: Generator, budget: float) -> Dict[str, Any]:
        """Step the cell with the actions a skill generator yields until it returns."""
        ex, env = self.expert, self.env
        ex._last_obs = self.obs
        t0 = self.sim_time
        log0 = len(ex.log)
        max_f = 0.0
        value, reason = None, "done"
        try:
            action = next(gen)
            while True:
                _, _, _, _, info = env.step(action)
                ex._last_obs = self.obs
                max_f = max(max_f, float(info["contact_force"]))
                self._grab_frame()
                if info["protective_stop"]:
                    reason = "protective_stop"
                    gen.close()
                    break
                if self.sim_time - t0 > budget:
                    reason = "timeout"
                    gen.close()
                    break
                if self.sim_time > self.max_sim_time:
                    reason = "session_time_limit"
                    gen.close()
                    break
                action = gen.send(self.obs)
        except StopIteration as stop:
            value = stop.value
        return {"value": value, "reason": reason, "max_force": max_f, "t0": t0,
                "messages": [m for _, m in ex.log[log0:]]}

    def _hold(self, seconds: float, collect: Optional[List[Dict[str, np.ndarray]]] = None) -> None:
        steps = max(1, int(round(seconds / self.cfg.sim.policy_dt)))
        for _ in range(steps):
            self.env.step(self.expert._hold())
            self.expert._last_obs = self.obs
            self._grab_frame()
            if collect is not None:
                collect.append(self.obs)

    def settled_obs(self, seconds: float = 0.4) -> Dict[str, np.ndarray]:
        """Hold still and average the noisy perception outputs over the samples.

        One sample of the holder position is off by 1.5 mm (1 sigma) per axis, which is
        enough to call a seated connector unseated; eight samples bring that to 0.5 mm.
        """
        samples: List[Dict[str, np.ndarray]] = []
        self._hold(seconds, collect=samples)
        out = dict(samples[-1])
        for key in ("cable", "connector_pos", "holder_pos"):
            out[key] = np.mean([np.asarray(o[key], dtype=float) for o in samples], axis=0)
        yaws = np.array([float(o["holder_yaw"][0]) for o in samples])
        out["holder_yaw"] = np.array([math.atan2(np.sin(yaws).mean(), np.cos(yaws).mean())])
        return out

    def _result(self, skill: str, args: Dict[str, Any], ok: bool, outcome: str,
                run: Dict[str, Any], wall: float) -> SkillResult:
        perceived = self.perceive(self.settled_obs(0.3))
        res = SkillResult(skill=skill, args=args, ok=ok, outcome=outcome,
                          messages=run.get("messages", []), sim_time_start=run.get("t0", self.sim_time),
                          sim_time_end=self.sim_time, max_force=run.get("max_force", 0.0),
                          perceived=perceived, truth=self.truth(), wall_time=wall)
        self._emit({"type": "skill", **res.as_dict()})
        return res

    def _refuse(self, skill: str, args: Dict[str, Any], outcome: str, message: str) -> SkillResult:
        t = self.sim_time
        res = SkillResult(skill=skill, args=args, ok=False, outcome=outcome, messages=[message],
                          sim_time_start=t, sim_time_end=t, max_force=0.0,
                          perceived=self.perceive() if self.env is not None else {},
                          truth=self.truth() if self.env is not None else {})
        self._emit({"type": "refused", **res.as_dict()})
        return res

    # ----------------------------------------------------------- perception
    def perceive(self, obs: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Any]:
        """The cell's state as perception sees it (this is all a planner gets).

        ``obs`` defaults to the latest single observation; pass ``settled_obs()`` for the
        averaged one.
        """
        if self.env is None:
            return {}
        obs = self.obs if obs is None else obs
        cfg = self.cfg
        bz = float(obs["board_z"][0])
        cable = obs["cable"]
        forks = {}
        for fid, fpose in zip(self.route, obs["forks"]):
            chk = cable_crossing_in_fork(cable, fpose, cfg.fork, bz)
            seated = bool(chk["inside"])
            forks[fid] = {"wire_in_slot": seated,
                          "wire_offset_mm": None if not np.isfinite(chk["y"]) else round(1000 * chk["y"], 1),
                          "wire_height_mm": None if not np.isfinite(chk["z"]) else round(1000 * chk["z"], 1)}
        cp, cR = obs["connector_pos"], obs["connector_rot"]
        hp = obs["holder_pos"]
        yaw_h = float(obs["holder_yaw"][0])
        xh = np.array([math.cos(yaw_h), math.sin(yaw_h)])
        d = cp[:2] - hp[:2]
        standing = abs(float(cR[2, 0])) > 0.5
        yaw_c = math.atan2(cR[1, 0], cR[0, 0])
        fixtures = [(fid, np.asarray(f[:2])) for fid, f in zip(self.route, obs["forks"])]
        nearest = min(((fid, float(np.linalg.norm(cp[:2] - p))) for fid, p in fixtures),
                      key=lambda x: x[1]) if fixtures else ("", float("inf"))
        # tilt: how far the connector's long axis leaves the board plane (roll about it is free).
        # A connector can sit in the pocket with one end up on a rail; its centre is then
        # still within tolerance, so the pose check has to look at tilt and yaw too.
        tilt = math.degrees(math.asin(min(1.0, abs(float(cR[2, 0])))))
        yaw_err = math.degrees(wrap_angle(yaw_c - yaw_h))
        in_pocket = (abs(float(d @ xh)) < 0.004 and abs(float(d @ np.array([-xh[1], xh[0]]))) < 0.003
                     and abs(float(cp[2] - hp[2])) < 0.003 and not standing
                     and tilt < 7.0 and abs(yaw_err) < 8.5)
        s_total = float(polyline_arclength(cable)[-1])
        last_routed = None
        for fid in self.route:
            if forks[fid]["wire_in_slot"]:
                last_routed = fid
        free_len = None
        if last_routed is not None:
            f = obs["forks"][self.route.index(last_routed)]
            s_last = arclength_near(cable, np.array([f[0], f[1], f[2] + cfg.fork.post_height]))
            free_len = round(1000 * (s_total - s_last))
        tcp = obs["tcp_pos"]
        return {
            "sim_time_s": round(self.sim_time, 1),
            "forks": forks,
            "forks_in_slot": [fid for fid in self.route if forks[fid]["wire_in_slot"]],
            "next_fork": next((fid for fid in self.route if not forks[fid]["wire_in_slot"]), None),
            "connector": {
                "id": self.connector_id,
                "in_holder": bool(in_pocket),
                "offset_from_holder_mm": [round(1000 * float(d @ xh), 1),
                                          round(1000 * float(d @ np.array([-xh[1], xh[0]])), 1),
                                          round(1000 * float(cp[2] - hp[2]), 1)],
                "standing_on_end": bool(standing),
                "tilt_deg": round(tilt, 1),
                "yaw_error_deg": round(yaw_err, 1),
                "at_board_mm": [round(v) for v in robot_to_board(cp[:2], cfg)],
                "nearest_fork": nearest[0],
                "nearest_fork_distance_mm": round(1000 * nearest[1]),
            },
            "free_wire_after_last_fork_mm": free_len,
            "gripper_opening_mm": round(1000 * float(obs["gripper"][0]), 1),
            "tcp_height_above_board_mm": round(1000 * float(tcp[2] - bz)),
            "protective_stop": bool(obs["protective_stop"][0] > 0.5),
        }

    def truth(self) -> Dict[str, Any]:
        """Simulator ground truth, for scoring only."""
        if self.env is None:
            return {}
        st = self.env.cell.sim.task_status()
        return {"forks_routed": dict(zip(self.route, [bool(v) for v in st["forks_routed"]])),
                "connector_seated": bool(st["connector_seated"]),
                "connector_latched": bool(st.get("connector_latched", False)),
                "success": bool(st["success"]),
                "max_contact_force": round(float(st["max_contact_force"]), 1)}

    # --------------------------------------------------------------- skills
    def status(self) -> Dict[str, Any]:
        """Settle briefly and report perception averaged over a few samples."""
        if self.env is None:
            return {"feasible": False, "issues": [i.as_dict() for i in self.issues]}
        state = self.perceive(self.settled_obs(0.4))
        self._emit({"type": "status", "sim_time": self.sim_time, "perceived": state, "truth": self.truth()})
        return state

    def route_fork(self, fork_id: str, attempt: int = 0, pick_offset_mm: Optional[float] = None) -> SkillResult:
        args = {"fork_id": fork_id, "attempt": attempt, "pick_offset_mm": pick_offset_mm}
        if self.env is None:
            return self._refuse("route_fork", args, "infeasible_spec", "the spec failed validation")
        i = self._fork_index(fork_id)
        if i is None:
            return self._refuse("route_fork", args, "unknown_fork",
                                f"{fork_id} is not on this wire's route {self.route}")
        state = self.perceive()
        missing = [f for f in self.route[:i] if not state["forks"][f]["wire_in_slot"]]
        if missing:
            return self._refuse("route_fork", args, "previous_fork_not_seated",
                                f"route {fork_id} only after {', '.join(missing)}: the wire is anchored "
                                f"at the previous fork, so those must hold it first")
        t_wall = time.perf_counter()
        self.expert.current_fork = i
        offset = None if pick_offset_mm is None else float(pick_offset_mm) / 1000.0
        run = self._drive(self.expert._route_fork(i, int(attempt), offset), budget=60.0)
        ok = bool(run["value"]) and run["reason"] == "done"
        outcome = "routed" if ok else _classify_fork_failure(run)
        # leave the cell tidy for the next decision: gripper open, tool up
        if not ok and run["reason"] == "done":
            self._drive(self._clear_board(), budget=6.0)
        return self._result("route_fork", args, ok, outcome, run, time.perf_counter() - t_wall)

    def _clear_board(self) -> Generator:
        """Open the gripper and lift the tool clear of the fixtures."""
        ex = self.expert
        yield from ex._set_grip(-1.0, 0.3)
        up = ex._obs["tcp_pos"].copy()
        up[2] = max(float(up[2]), ex._board_z() + 0.10)
        yield from ex._goto(up, None, tol=0.01, timeout=4.0)
        return True

    def insert_connector(self) -> SkillResult:
        args: Dict[str, Any] = {}
        if self.env is None:
            return self._refuse("insert_connector", args, "infeasible_spec", "the spec failed validation")
        state = self.perceive()
        missing = [f for f in self.route if not state["forks"][f]["wire_in_slot"]]
        if missing:
            return self._refuse("insert_connector", args, "forks_not_routed",
                                f"seat the connector after the wire is in every fork; missing: {', '.join(missing)}")
        c = state["connector"]
        if c["standing_on_end"]:
            return self._refuse("insert_connector", args, "connector_standing",
                                "the connector stands on its end and cannot be grasped; tip it over first")
        if not self.expert._connector_ok_to_grasp():
            return self._refuse("insert_connector", args, "connector_blocked",
                                f"the connector lies {c['nearest_fork_distance_mm']} mm from {c['nearest_fork']}; "
                                f"the fingers would hit the fork, relocate it first")
        t_wall = time.perf_counter()
        self.expert.current_fork = -1
        if "slip_on_insert" in self.faults:              # benchmark fault: lose the connector on the way
            self.faults.discard("slip_on_insert")
            run = self._drive(run_until_phase(self.expert._insert_connector(auto_recover=False), self.expert,
                                              "connector_descend"), budget=60.0)
            self._drive(self._clear_board(), budget=6.0)
            run["messages"].append("the connector slipped out of the fingers above the holder")
            self._emit({"type": "disturbance", "label": "connector slipped out of the fingers",
                        "sim_time": self.sim_time, "truth": self.truth()})
            return self._result("insert_connector", args, False, "grasp_slipped", run,
                                time.perf_counter() - t_wall)
        run = self._drive(self.expert._insert_connector(auto_recover=False), budget=60.0)
        ok = bool(run["value"]) and run["reason"] == "done"
        outcome = "seated" if ok else _classify_connector_failure(run)
        return self._result("insert_connector", args, ok, outcome, run, time.perf_counter() - t_wall)

    def tip_connector(self) -> SkillResult:
        args: Dict[str, Any] = {}
        if self.env is None:
            return self._refuse("tip_connector", args, "infeasible_spec", "the spec failed validation")
        t_wall = time.perf_counter()
        run = self._drive(self.expert._tip_connector(), budget=30.0)
        ok = run["reason"] == "done" and not self.perceive()["connector"]["standing_on_end"]
        return self._result("tip_connector", args, ok, "lying_flat" if ok else "still_standing",
                            run, time.perf_counter() - t_wall)

    def relocate_connector(self) -> SkillResult:
        args: Dict[str, Any] = {}
        if self.env is None:
            return self._refuse("relocate_connector", args, "infeasible_spec", "the spec failed validation")
        t_wall = time.perf_counter()
        run = self._drive(self.expert._relocate_connector(), budget=40.0)
        ok = run["reason"] == "done" and bool(self.expert._connector_ok_to_grasp())
        return self._result("relocate_connector", args, ok, "graspable" if ok else "still_blocked",
                            run, time.perf_counter() - t_wall)

    def retreat(self) -> SkillResult:
        args: Dict[str, Any] = {}
        if self.env is None:
            return self._refuse("retreat", args, "infeasible_spec", "the spec failed validation")
        t_wall = time.perf_counter()
        run = self._drive(self.expert._retreat(), budget=15.0)
        return self._result("retreat", args, run["reason"] == "done", "home", run, time.perf_counter() - t_wall)

    def inspect(self, target: str = "all") -> Dict[str, Any]:
        """Look at the board and report on a target: a fork id, the connector, or ``all``.

        Perception (cable keypoints, connector pose, averaged over a short settle) always
        answers. With a visual inspector attached, the camera also photographs each target
        and the vision model gives its own verdict; ``disagreements`` lists the targets
        where the two differ. Asking again about a target takes new camera views.
        """
        if self.env is None:
            return {"feasible": False}
        state = self.status()
        self._last_status = state
        out: Dict[str, Any] = {"target": target, "sim_time_s": state["sim_time_s"], "method": "perception",
                               "arm_clear": state["tcp_height_above_board_mm"] >= 150}
        if target in ("all", "board"):
            out["forks"] = state["forks"]
            out["connector"] = state["connector"]
            targets = list(self.route) + [self.connector_id]
        elif target in (self.connector_id, "connector"):
            out["connector"] = state["connector"]
            targets = [self.connector_id]
        elif target in state["forks"]:
            out["fork"] = {target: state["forks"][target]}
            targets = [target]
        else:
            out["error"] = f"unknown target {target!r}; use a fork id, {self.connector_id!r}, or 'all'"
            targets = []
        if self.inspector is not None and targets:
            out["method"] = "perception+vision"
            out["vision"], disagree = {}, []
            for t, verdict in zip(targets, self.look_all(targets)):
                out["vision"][t] = verdict.for_planner()
                perceived = (state["connector"]["in_holder"] if t == self.connector_id
                             else state["forks"][t]["wire_in_slot"])
                if verdict.seated is not None and verdict.seated != perceived:
                    disagree.append(t)
            out["disagreements"] = disagree
        self._last_status = None
        self._emit({"type": "inspect", "sim_time": self.sim_time, "result": out, "truth": self.truth()})
        return out

    # -------------------------------------------------------------- vision
    def camera(self):
        if self._camera is None:
            from .vision import InspectionCamera
            self._camera = InspectionCamera(self.env.cell.sim)
        return self._camera

    def photograph(self, target: str, view: Optional[int] = None, style: Optional[str] = None):
        """Composite image of a fork or the connector holder: (kind, image, view index).

        ``style`` picks the camera views (see vision.VIEWS); default: the inspector's."""
        from .vision import DEFAULT_STYLE, compose
        style = style or getattr(self.inspector, "style", None) or DEFAULT_STYLE
        if view is None:
            view = self._looks.get(target, 0)
            self._looks[target] = view + 1
        obs = self.obs
        if target == self.connector_id or target == "connector":
            pose = (*[float(v) for v in obs["holder_pos"]], float(obs["holder_yaw"][0]))
            views = self.camera().views("connector", pose, self.cfg, view, style)
            return "connector", compose(views, f"Holder for connector {self.connector_id}"), view
        i = self.route.index(target)
        views = self.camera().views("fork", obs["forks"][i], self.cfg, view, style)
        return "fork", compose(views, f"Fork {target}"), view

    def look(self, target: str):
        """Photograph a target and ask the visual inspector; logs the verdict with ground truth."""
        return self.look_all([target])[0]

    def look_all(self, targets: List[str]):
        """Photograph the targets (renderer: this thread), ask about them in parallel (the
        model calls wait on the network), and log every verdict with ground truth."""
        from concurrent.futures import ThreadPoolExecutor
        shots = [(t, *self.photograph(t)) for t in targets]
        with ThreadPoolExecutor(max_workers=max(1, min(4, len(shots)))) as pool:
            verdicts = list(pool.map(lambda sh: self.inspector.ask(sh[1], sh[0], sh[2], view=sh[3]), shots))
        truth = self.truth()
        state = self._last_status or self.perceive()
        for (target, kind, image, view), verdict in zip(shots, verdicts):
            if self.inspection_dir:
                os.makedirs(self.inspection_dir, exist_ok=True)
                name = f"{len(self.visual_checks) + 1:02d}_{target}_view{view}.jpg"
                image.save(os.path.join(self.inspection_dir, name), quality=90)
                verdict.image = name
            record = {"sim_time": round(self.sim_time, 2), "target": target, "kind": kind,
                      "verdict": verdict.as_dict(),
                      "perceived": (state["connector"]["in_holder"] if kind == "connector"
                                    else state["forks"][target]["wire_in_slot"]),
                      "truth": (truth["connector_seated"] if kind == "connector"
                                else truth["forks_routed"][target])}
            self.visual_checks.append(record)
            self._emit({"type": "visual_check", **record})
        return verdicts

    def summary(self) -> Dict[str, Any]:
        return {"spec": self.spec.name, "route": self.route, "connector": self.connector_id,
                "seed": self.seed, "randomize": self.randomize, "feasible": self.feasible,
                "issues": [i.as_dict() for i in self.issues],
                "sim_time_s": round(self.sim_time, 1),
                "truth": self.truth() if self.env is not None else {}}

    def close(self) -> None:
        if self._camera is not None:
            self._camera.close()
            self._camera = None
        if self.env is not None:
            self.env.close()


def _classify_fork_failure(run: Dict[str, Any]) -> str:
    if run["reason"] != "done":
        return run["reason"]
    text = " ".join(run["messages"]).lower()
    for key, label in (("carry-over aborted", "carry_over_force_limit"),
                       ("grasp failed", "grasp_failed"),
                       ("no touch-down", "no_touch_down"),
                       ("no reachable pick", "no_pick_point"),
                       ("not retained", "wire_not_retained")):
        if key in text:
            return label
    return "not_routed"


def _classify_connector_failure(run: Dict[str, Any]) -> str:
    if run["reason"] != "done":
        return run["reason"]
    text = " ".join(run["messages"]).lower()
    for key, label in (("not lying flat", "connector_standing"),
                       ("approach failed", "approach_failed"),
                       ("descent blocked", "grasp_blocked_by_fixture"),
                       ("grasp failed", "grasp_failed"),
                       ("did not reach seat height", "not_seated_after_search"),
                       ("seated=false", "not_seated")):
        if key in text:
            return label
    return "not_seated"
