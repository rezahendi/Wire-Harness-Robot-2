"""Builds for the web app: one at a time in a worker thread, every step streamed as events.

A build is the same thing ``run_build`` does (a planner calling the robot's skills in the
simulated cell), recorded as it happens:

    scene        once: the cell's geometry (boxes, capsules, cylinders) for the 3D view
    frame        20 Hz: every moving body's pose, the tool, the controller's target, the
                 filtered force, the gripper, the skill's phase, the expert's new log lines
    plan         Nemotron's reply (its reasoning and the tool call it chose)
    call_start   a tool call begins (name, arguments)
    call_end     it returns (outcome, who executed it: expert or GR00T, force, messages)
    disturbance  a scripted disturbance (wire pulled out of a fork, connector slipping)
    visual_check the camera check's verdict (vision model on Token Factory)
    done         the result: ground truth, the planner's own verdict, its report, files

Everything is kept in memory for clients that join late and written to the build's folder
(events.json for the app's replay, trace.json, report.md and scene3d.json as run_build and
replay3d write them).
"""

from __future__ import annotations

import base64
import json
import os
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ..replay3d import RecordingSession, _short, scene_document

SCENE_KEYS = ("bodies", "moving", "static", "geoms", "forks", "holder", "connector_body", "board_z", "camera",
              "wire_bodies")


@dataclass
class BuildOptions:
    spec: str                                   # spec file
    planner: str = "scripted"                   # scripted | nemotron
    seed: int = 0
    randomize: bool = True
    scenario: str = "nominal"                   # nominal | popped_wire | slip_on_insert | both
    vision: bool = False                        # camera check by a Token Factory vision model
    groot: Optional[str] = None                 # GR00T policy server HOST:PORT; "" = 127.0.0.1:5556; None = off
    groot_attempts: str = "0"                   # GR00T takes the first attempt; a retry goes to the expert
    groot_ensemble: Optional[float] = 0.1       # a chunk every 4 steps, overlapping chunks averaged (None: off)
    groot_seat_assist: Optional[float] = 1.5    # force-controlled seating after this long stuck (None: off)
    model: Optional[str] = None                 # Token Factory planner model (default: picked automatically)
    max_turns: int = 40
    pace: float = 1.0                           # 1: no faster than real time; 0: as fast as it runs


class LiveSession(RecordingSession):
    """A RecordingSession that hands the scene and every recorded frame to ``sink``."""

    def __init__(self, *args: Any, sink: Optional[Callable[[Dict[str, Any]], None]] = None, pace: float = 1.0,
                 **kwargs: Any):
        self._sink = sink
        self._pace = float(pace)
        self._header_sent = False
        self._log_sent = 0
        self._clock: Optional[tuple] = None
        super().__init__(*args, **kwargs)

    def scene_header(self) -> Dict[str, Any]:
        from ..replay3d import PHASES
        return {"type": "scene", **{k: self.rec[k] for k in SCENE_KEYS}, "route": list(self.route),
                "connector": self.connector_id, "pose_scale": 20000.0, "quat_scale": 32767.0,
                "phase_text": dict(PHASES), "controller": {k: getattr(self.cfg.controller, k) for k in
                                                           ("max_lin_vel", "protective_stop_force")}}

    def _record(self) -> None:
        n = len(self.rec["t"])
        super()._record()
        if self._sink is None or len(self.rec["t"]) == n:
            return
        if not self._header_sent:
            self._header_sent = True
            self._sink(self.scene_header())
        self._sink(self._frame_event(len(self.rec["t"]) - 1))
        self._keep_pace()

    def _frame_event(self, i: int) -> Dict[str, Any]:
        r = self.rec
        names = sorted(self._phase_ids, key=self._phase_ids.get)
        log = self.expert.log[self._log_sent:] if self.expert is not None else []
        self._log_sent += len(log)
        return {"type": "frame", "i": i, "t": r["t"][i],
                "pose": base64.b64encode(np.ascontiguousarray(r["poses"][i], dtype="<i2").tobytes()).decode("ascii"),
                "tcp": r["tcp"][i], "target": r["target"][i], "target_yaw": r["target_yaw"][i],
                "wrench": r["wrench"][i][:3], "grip": r["gripper"][i], "phase": names[r["phase"][i]],
                "call": r["call"][i], "truth": r["truth"][i], "mode": 1 if r["mode"][i] == "disturbance" else 0,
                "pokes": r["pokes"].get(str(i)), "log": [[round(float(t), 2), m] for t, m in log]}

    def _keep_pace(self) -> None:
        """No faster than ``pace`` x real time (a live build is easier to follow at robot speed)."""
        if self._pace <= 0:
            return
        now, t = time.perf_counter(), float(self.rec["t"][-1])
        if self._clock is None:
            self._clock = (now, t)
            return
        wait = self._clock[0] + (t - self._clock[1]) / self._pace - now
        if wait > 0:
            time.sleep(min(wait, 0.5))
        elif wait < -1.0:                         # the simulator is behind: do not catch up later
            self._clock = (now, t)


def _slim_result(result: Dict[str, Any]) -> Dict[str, Any]:
    out = _short(result)
    for k in ("executed_by", "duration_s", "max_contact_force_N", "skill"):
        if k in result:
            out[k] = result[k]
    if "messages" in result:
        out["messages"] = list(result["messages"])[-4:]
    return out


class Build:
    """One build: its options, its events so far, and whoever is listening."""

    def __init__(self, options: BuildOptions, root: str):
        self.id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
        self.options = options
        self.dir = os.path.join(root, self.id)
        self.status = "queued"
        self.created = time.time()
        self.events: List[Dict[str, Any]] = []
        self.summary: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._listeners: List[Callable[[Dict[str, Any]], None]] = []

    def emit(self, event: Dict[str, Any]) -> None:
        with self._lock:
            event = {**event, "seq": len(self.events)}
            self.events.append(event)
            listeners = list(self._listeners)
        for fn in listeners:
            try:
                fn(event)
            except Exception:
                pass

    def subscribe(self, fn: Callable[[Dict[str, Any]], None]) -> List[Dict[str, Any]]:
        """Start listening; returns the events so far (send those first)."""
        with self._lock:
            self._listeners.append(fn)
            return list(self.events)

    def unsubscribe(self, fn: Callable[[Dict[str, Any]], None]) -> None:
        with self._lock:
            if fn in self._listeners:
                self._listeners.remove(fn)

    def info(self) -> Dict[str, Any]:
        return {"id": self.id, "status": self.status, "created": self.created, "options": asdict(self.options),
                "summary": self.summary, "events": len(self.events)}


def run(build: Build) -> None:
    """Run one build (in the worker thread) and write its files."""
    from ..agent import NemotronPlanner, ScriptedPlanner
    from ..disturbances import make_scenario
    from ..run_build import _jsonable, write_report
    from ..spec import HarnessSpec

    o = build.options
    os.makedirs(build.dir, exist_ok=True)
    build.status = "running"
    session = runner = None
    calls: Dict[int, Dict[str, Any]] = {}
    reasoning = {"last": ""}
    t_wall = time.perf_counter()
    try:
        spec = HarnessSpec.from_yaml(o.spec)
        client = inspector = None
        if o.planner == "nemotron" or o.vision:
            from ..llm import TokenFactoryClient
            client = TokenFactoryClient()
        if o.vision:
            from ..vision import default_inspector
            inspector = default_inspector(client)
        session = LiveSession(spec, seed=o.seed, randomize=o.randomize, inspector=inspector,
                              inspection_dir=os.path.join(build.dir, "inspection") if inspector else None,
                              sink=build.emit, pace=o.pace)
        if not session.feasible:
            build.emit({"type": "infeasible", "issues": [{"severity": i.severity, "message": i.message}
                                                         for i in session.issues]})

        def on_live(ev: Dict[str, Any]) -> None:
            kind = ev.get("type")
            if kind == "call_start":
                session.call_index = int(ev["index"])
                calls[ev["index"]] = {"name": ev["name"], "arguments": ev.get("arguments") or {},
                                      "t_start": ev.get("sim_time", 0.0), "reasoning": reasoning["last"],
                                      "disturbances": []}
                reasoning["last"] = ""
                build.emit(ev)
            elif kind == "call_end":
                res = _slim_result(ev.get("result") or {})
                c = calls.setdefault(ev["index"], {"name": ev["name"], "arguments": {}, "t_start": 0.0,
                                                   "reasoning": "", "disturbances": []})
                c.update({"t_end": ev.get("sim_time", 0.0), "result": res})
                build.emit({"type": "call_end", "index": ev["index"], "name": ev["name"],
                            "sim_time": ev.get("sim_time"), "result": res})
            elif kind == "plan":
                reasoning["last"] = (ev.get("reasoning") or ev.get("content") or "").strip()
                build.emit(ev)
            elif kind == "disturbance":
                d = {"type": "disturbance", "label": ev.get("label"), "sim_time": ev.get("sim_time")}
                if calls:
                    calls[max(calls)]["disturbances"].append({"label": d["label"], "t": d["sim_time"]})
                build.emit(d)
            elif kind == "visual_check":
                v = ev.get("verdict") or {}
                build.emit({"type": "visual_check", "target": ev.get("target"), "truth": ev.get("truth"),
                            "perceived": ev.get("perceived"), "sim_time": ev.get("sim_time"),
                            "verdict": {k: v.get(k) for k in ("seated", "confidence", "model", "evidence",
                                                              "image", "error")}})

        session.on_live = on_live
        if o.groot is not None and session.feasible:
            from ..groot_skill import connect_runner
            runner = connect_runner(o.groot or None, o.groot_attempts, seat_assist=o.groot_seat_assist,
                                    ensemble_decay=o.groot_ensemble,
                                    execute_horizon=4 if o.groot_ensemble is not None else 8)
            session.skill_runners["route_fork"] = runner
            session.expert_after_lost_wire = True       # recoveries after a lost wire go to the expert
        scenario = make_scenario(o.scenario, session.route) if session.feasible else None
        for trig in (scenario.triggers if scenario is not None else []):    # mark the disturbance frames
            act = trig.action

            def wrapped(s, _act=act):
                s.mode = "disturbance"
                try:
                    return _act(s)
                finally:
                    s.mode = "skill"
            trig.action = wrapped
        quiet = (lambda *_: None)
        if o.planner == "nemotron":
            planner = NemotronPlanner(session, client=client, model=o.model, log=quiet)
            build.emit({"type": "planner", "planner": "nemotron", "model": planner.model})
        else:
            planner = ScriptedPlanner(session, log=quiet)
            build.emit({"type": "planner", "planner": "scripted", "model": ""})
        result = planner.run(o.max_turns, scenario=scenario).as_dict()
        if session.feasible:
            session.call_index = len(result.get("tool_calls") or [])
            session._hold(1.0)                    # a second of stillness at the end of the replay

        result["events"] = session.events
        result["session"] = session.summary()
        result["visual_checks"] = session.visual_checks
        if client is not None and not result.get("usage"):
            result["usage"] = client.usage.as_dict()
        if runner is not None:
            from ..groot_skill import route_stats
            result["routing"] = route_stats(result.get("tool_calls"))
        with open(os.path.join(build.dir, "trace.json"), "w", encoding="utf-8") as f:
            json.dump(_jsonable(result), f, indent=1)
        write_report(os.path.join(build.dir, "report.md"), spec, result, session)
        if session.feasible:
            ordered = [calls[k] for k in sorted(calls)]
            doc = scene_document(session, ordered, result.get("disturbances") or [], {
                "spec": spec.name, "revision": spec.revision, "planner": result.get("planner"),
                "model": result.get("model"), "seed": o.seed, "scenario": o.scenario,
                "success": result.get("success"), "claimed_success": result.get("claimed_success"),
                "replay_matched": True, "record_wall_s": round(time.perf_counter() - t_wall, 1)})
            with open(os.path.join(build.dir, "scene3d.json"), "w", encoding="utf-8") as f:
                json.dump(_jsonable(doc), f, separators=(",", ":"))
        u = result.get("usage") or {}
        build.summary = {"spec": spec.name, "planner": result.get("planner"), "model": result.get("model"),
                         "success": bool(result.get("success")), "claimed_success": result.get("claimed_success"),
                         "report": result.get("report") or "", "robot_time_s": result.get("sim_time"),
                         "wall_time_s": round(time.perf_counter() - t_wall, 1),
                         "tool_calls": len(result.get("tool_calls") or []), "error": result.get("error", ""),
                         "routing": result.get("routing"), "scenario": o.scenario, "seed": o.seed,
                         "tokens": int(u.get("prompt_tokens", 0)) + int(u.get("completion_tokens", 0)),
                         "truth": _jsonable(result.get("truth") or {})}
        build.status = "done"
        build.emit({"type": "done", **build.summary, "files": sorted(os.listdir(build.dir))})
    except BaseException as exc:                   # report it to the page instead of dying quietly
        build.status = "failed"
        build.summary = {"error": f"{type(exc).__name__}: {exc}"}
        build.emit({"type": "failed", "error": build.summary["error"], "traceback": traceback.format_exc()[-3000:]})
    finally:
        if runner is not None:
            runner.close()
            runner.client.close()
        if session is not None:
            session.close()
        try:
            _write_json(os.path.join(build.dir, "events.json"), build.events, separators=(",", ":"))
            _write_json(os.path.join(build.dir, "build.json"), build.info(), indent=1)
        except OSError:
            pass


def _write_json(path: str, data: Any, **kwargs: Any) -> None:
    """Write a file whole or not at all (a reader never sees half of it)."""
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, default=str, **kwargs)
    os.replace(tmp, path)


class BuildManager:
    """Keeps the builds of this server run and runs them one at a time."""

    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.builds: Dict[str, Build] = {}
        self._queue: List[Build] = []
        self._cv = threading.Condition()
        self._worker = threading.Thread(target=self._loop, name="builds", daemon=True)
        self._worker.start()

    def start(self, options: BuildOptions) -> Build:
        build = Build(options, self.root)
        with self._cv:
            self.builds[build.id] = build
            self._queue.append(build)
            self._cv.notify()
        return build

    def busy(self) -> bool:
        return any(b.status in ("queued", "running") for b in self.builds.values())

    def _loop(self) -> None:
        while True:
            with self._cv:
                while not self._queue:
                    self._cv.wait()
                build = self._queue.pop(0)
            run(build)

    def past(self) -> List[Dict[str, Any]]:
        """Builds of earlier server runs, from their folders (newest first)."""
        out = []
        for name in sorted(os.listdir(self.root), reverse=True):
            path = os.path.join(self.root, name, "build.json")
            if name in self.builds or not os.path.exists(path):
                continue
            try:
                with open(path, encoding="utf-8") as f:
                    out.append(json.load(f))
            except (OSError, ValueError):
                continue
        return out
