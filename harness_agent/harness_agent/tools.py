"""The robot's skills as tools a language model can call (OpenAI function-calling schema).

Each tool maps onto one ``CellSession`` call. Results are compact JSON so a build of
twenty-odd calls stays well inside a small context, and they contain perception only.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from .session import CellSession


def tool_schemas(route: List[str], connector_id: str) -> List[Dict[str, Any]]:
    def fn(name: str, description: str, properties: Optional[Dict[str, Any]] = None,
           required: Optional[List[str]] = None) -> Dict[str, Any]:
        return {"type": "function", "function": {
            "name": name, "description": description,
            "parameters": {"type": "object", "properties": properties or {},
                           "required": required or []}}}

    return [
        fn("get_status",
           "Settle for a moment and report what the cameras and sensors see: which forks hold "
           "the wire, where the connector is relative to its holder, free wire length."),
        fn("route_fork",
           "Pick the wire, carry it over the fork and snap it into the fork's barbed jaws, with "
           "force feedback. Forks must be routed in route order.",
           {"fork_id": {"type": "string", "enum": list(route), "description": "fork to route"},
            "attempt": {"type": "integer", "minimum": 0, "maximum": 3,
                        "description": "failed tries of this fork so far: 0 for the first try (and when "
                                       "re-routing a fork that lost the wire later); each increment picks "
                                       "the wire 20 mm further along"},
            "pick_offset_mm": {"type": "number", "minimum": -20, "maximum": 60,
                               "description": "optional explicit shift of the pick point along the wire"}},
           ["fork_id"]),
        fn("tip_connector",
           "The connector stands on its end: grasp the wire close to it and pull it over so it "
           "lies flat."),
        fn("relocate_connector",
           "The connector lies where the fingers cannot get around it (next to a fork, or on top of "
           "its holder's rails or walls): lift it by the wire and lay it down flat on a free spot "
           "near the holder."),
        fn("insert_connector",
           f"Grasp connector {connector_id}, approach its holder from behind, find the pocket with a "
           "force-controlled spiral search and press it home. Only when every fork holds the wire."),
        fn("inspect",
           "Check a target: a fork id, the connector id, or 'all'. Returns what perception measures "
           "and, when the camera check is on, a vision verdict per fixture (seated, confidence, "
           "evidence) plus the fixtures where the two disagree. Asking again takes new camera angles.",
           {"target": {"type": "string", "description": f"one of {route + [connector_id, 'all']}"}},
           ["target"]),
        fn("retreat", "Move the arm up and back to its home pose, clear of the board."),
        fn("finish",
           "End the build and report. Call it once, when the harness is complete or when you "
           "have decided it cannot be completed.",
           {"success": {"type": "boolean", "description": "true only if every fork holds the wire "
                                                          "and the connector is seated"},
            "report": {"type": "string", "description": "what was done, what failed and why, "
                                                        "attempts per step"}},
           ["success", "report"]),
    ]


PHYSICAL_TOOLS = ("route_fork", "tip_connector", "relocate_connector", "insert_connector", "retreat")


class ToolBox:
    """Dispatches tool calls to a CellSession, with budgets and argument checks.

    It also holds the model to evidence: ``finish(success=true)`` is refused unless an
    inspection of everything, made after the last physical skill with the arm parked
    clear of the board, shows every fixture seated. A planner cannot talk its way to a
    success verdict.
    """

    def __init__(self, session: CellSession, max_fork_calls: int = 12, max_connector_calls: int = 8,
                 scenario: Optional[Any] = None):
        self.session = session
        self.scenario = scenario                 # disturbances.Scenario for the recovery benchmark
        self.disturbances: List[Dict[str, Any]] = []
        if scenario is not None:
            session.faults |= set(scenario.faults)
        self.max_fork_calls = max_fork_calls
        self.max_connector_calls = max_connector_calls
        self.fork_calls = 0
        self.connector_calls = 0
        self.finished: Optional[Dict[str, Any]] = None
        self.calls: List[Dict[str, Any]] = []
        self.last_inspection: Optional[Dict[str, Any]] = None   # inspect('all') since the last skill
        self.refused_verdicts = 0
        self._handlers: Dict[str, Callable[..., Any]] = {
            "get_status": self._status,
            "route_fork": self._route_fork,
            "tip_connector": self._tip,
            "relocate_connector": self._relocate,
            "insert_connector": self._insert,
            "inspect": self._inspect,
            "retreat": self._retreat,
            "finish": self._finish,
        }

    def call(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        handler = self._handlers.get(name)
        if handler is None:
            out = {"error": f"unknown tool {name!r}", "available": sorted(self._handlers)}
        elif self.finished is not None and name != "finish":
            out = {"error": "the build is already finished"}
        else:
            try:
                out = handler(**(arguments or {}))
            except TypeError as exc:
                out = {"error": f"bad arguments for {name}: {exc}"}
            if name in PHYSICAL_TOOLS:
                self.last_inspection = None           # the cell changed: earlier evidence is stale
            self._fire_triggers(name, arguments or {}, out)
        self.calls.append({"name": name, "arguments": arguments, "result": out,
                           "sim_time": round(self.session.sim_time, 2)})
        return out

    def _fire_triggers(self, name: str, arguments: Dict[str, Any], out: Dict[str, Any]) -> None:
        """Scenario disturbances happen between tool calls; the planner meets their effect
        in the next state it is shown, not in this result."""
        if self.scenario is None or self.session.env is None:
            return
        for trig in self.scenario.triggers:
            if trig.matches(name, arguments, out):
                trig.fired = True
                effect = trig.action(self.session)
                event = {"label": trig.label, "after": name, "arguments": arguments,
                         "sim_time": round(self.session.sim_time, 2), "effect": effect}
                self.disturbances.append(event)
                self.session._emit({"type": "disturbance", **event, "truth": self.session.truth()})

    # ------------------------------------------------------------ handlers
    def _status(self) -> Dict[str, Any]:
        return {"state": self.session.status()}

    def _route_fork(self, fork_id: str, attempt: int = 0, pick_offset_mm: Optional[float] = None,
                    **_: Any) -> Dict[str, Any]:
        if self.fork_calls >= self.max_fork_calls:
            return {"error": f"fork budget used up ({self.max_fork_calls} route_fork calls); finish the build"}
        self.fork_calls += 1
        return self.session.route_fork(str(fork_id), int(attempt or 0), pick_offset_mm).for_planner()

    def _connector_budget(self) -> Optional[Dict[str, Any]]:
        if self.connector_calls >= self.max_connector_calls:
            return {"error": f"connector budget used up ({self.max_connector_calls} calls); finish the build"}
        self.connector_calls += 1
        return None

    def _tip(self, **_: Any) -> Dict[str, Any]:
        return self._connector_budget() or self.session.tip_connector().for_planner()

    def _relocate(self, **_: Any) -> Dict[str, Any]:
        return self._connector_budget() or self.session.relocate_connector().for_planner()

    def _insert(self, **_: Any) -> Dict[str, Any]:
        return self._connector_budget() or self.session.insert_connector().for_planner()

    def _inspect(self, target: str = "all", **_: Any) -> Dict[str, Any]:
        out = self.session.inspect(str(target))
        if str(target) in ("all", "board") and "error" not in out and out.get("forks") is not None:
            self.last_inspection = out
        return out

    def _retreat(self, **_: Any) -> Dict[str, Any]:
        return self.session.retreat().for_planner()

    def _finish(self, success: bool = False, report: str = "", **_: Any) -> Dict[str, Any]:
        if isinstance(success, str):
            success = success.strip().lower() in ("true", "yes", "1")
        if success:
            problem = self._success_evidence_problem()
            if problem:
                self.refused_verdicts += 1
                return {"error": problem, "refused": "finish"}
        self.finished = {"success": bool(success), "report": str(report)}
        self.session.finished = True
        return {"ok": True, "recorded": self.finished}

    def _success_evidence_problem(self) -> str:
        if not self.session.feasible:
            return "the spec failed validation, so nothing was built: finish with success=false"
        insp = self.last_inspection
        if insp is None:
            return ("a success verdict needs evidence: call inspect with target 'all' after the last "
                    "skill, then finish")
        if insp.get("arm_clear") is False:
            return ("the arm was over the board during the inspection, so the check is not valid: "
                    "retreat, inspect with target 'all' again, then finish")
        bad = [f for f, v in (insp.get("forks") or {}).items() if not v.get("wire_in_slot")]
        if not (insp.get("connector") or {}).get("in_holder"):
            bad.append(self.session.connector_id)
        if bad:
            return (f"the last inspection shows {', '.join(bad)} not seated: fix that or finish with "
                    "success=false")
        return ""
