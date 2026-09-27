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
                        "description": "0 for the first try; each increment picks the wire 20 mm further along"},
            "pick_offset_mm": {"type": "number", "minimum": -20, "maximum": 60,
                               "description": "optional explicit shift of the pick point along the wire"}},
           ["fork_id"]),
        fn("tip_connector",
           "The connector stands on its end: grasp the wire close to it and pull it over so it "
           "lies flat."),
        fn("relocate_connector",
           "The connector lies where the fingers cannot reach it (next to a fork): drag it to a "
           "free spot near the holder."),
        fn("insert_connector",
           f"Grasp connector {connector_id}, approach its holder from behind, find the pocket with a "
           "force-controlled spiral search and press it home. Only when every fork holds the wire."),
        fn("inspect",
           "Look at the board and check a target: a fork id, the connector id, or 'all'.",
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


class ToolBox:
    """Dispatches tool calls to a CellSession, with budgets and argument checks."""

    def __init__(self, session: CellSession, max_fork_calls: int = 12, max_connector_calls: int = 8):
        self.session = session
        self.max_fork_calls = max_fork_calls
        self.max_connector_calls = max_connector_calls
        self.fork_calls = 0
        self.connector_calls = 0
        self.finished: Optional[Dict[str, Any]] = None
        self.calls: List[Dict[str, Any]] = []
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
        self.calls.append({"name": name, "arguments": arguments, "result": out,
                           "sim_time": round(self.session.sim_time, 2)})
        return out

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
        return self.session.inspect(str(target))

    def _retreat(self, **_: Any) -> Dict[str, Any]:
        return self.session.retreat().for_planner()

    def _finish(self, success: bool = False, report: str = "", **_: Any) -> Dict[str, Any]:
        self.finished = {"success": bool(success), "report": str(report)}
        self.session.finished = True
        return {"ok": True, "recorded": self.finished}
