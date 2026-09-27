"""The build agent: a Nemotron model supervising the cell through tool calls.

The split of work is deliberate. The force-guided skills handle everything where physics
matters (touch-down, taut carry, snap-in, spiral search), because a language model at a
few calls per second cannot close a force loop. The model handles what a script handles
badly: reading the job, checking every step against what perception reports, going back
when an earlier fork lost the wire, choosing a recovery, and writing an honest report.

Planners that can drive a build:

    NemotronPlanner   Nemotron on Nebius Token Factory (tool calling)
    ScriptedPlanner   the same decisions hard-coded, as a baseline and a test double

Both go through the same ToolBox, so their builds are directly comparable.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .llm import TokenFactoryClient, assistant_message_for_history
from .session import CellSession
from .tools import ToolBox, tool_schemas

SYSTEM_PROMPT = """You are the cell supervisor of a robotic wire-harness assembly cell.

The cell has a UR5e-class arm with a wrist force/torque sensor and a parallel gripper, a
formboard with snap-in forks, a clamp that holds one end of the wire, and a holder for the
connector at the other end. The robot has reliable force-controlled skills. You decide
which skill runs next, check its result, and recover from failures. You never control
motion directly.

How a build works:
1. The wire runs from the clamp through the forks in route order; then its connector is
   seated in the holder.
2. Route forks strictly in order. A fork can only be routed when every earlier fork holds
   the wire, because the wire is anchored at the previous fork.
3. Every skill result contains the new state of the cell; read it before the next step
   (get_status is only needed for a fresh look). If a fork that held the wire before has
   lost it, route that fork again before going on.
4. Seat the connector only when every fork holds the wire. If it stands on its end, tip it
   over. If it lies against a fork, relocate it. Then insert it.
5. Finish in three calls: retreat (the camera needs the arm out of the way), inspect with
   target 'all', then finish with an honest verdict based on that inspection. The cell
   refuses finish(success=true) unless an inspection of everything, made after the last
   skill with the arm retreated, shows every fixture seated.

Inspection:
- Perception (the tracked wire and connector pose) is the primary check and decides the
  verdict. The camera check, when it is on, is a second opinion from a vision model.
- If the two disagree about a fixture, inspect that fixture once more: the camera takes
  new angles. If they still disagree, keep the verdict from perception and list the
  fixture in the report as flagged for a manual visual check.

Recovery:
- carry_over_force_limit: the wire went taut while carried. Retry with attempt + 1 (more
  wire between the anchor and the gripper).
- wire_not_retained or not_routed: retry with attempt + 1.
- grasp_failed or no_touch_down: retry with attempt + 1.
- After three failed attempts on the same fork, stop and report.
- not_seated or not_seated_after_search: check the connector state and insert again. Stop
  after four insertion attempts.
- insert_connector ends with grasp_slipped, grasp_failed or grasp_blocked_by_fixture: the
  connector is lying on or against its holder, where the fingers cannot get around it.
  Call relocate_connector (it lifts it off and lays it down clear), then insert again. Do
  this after every such failure, not only the first. If relocate_connector reports
  still_on_holder, call it again (it often frees the connector on a later try); after three
  relocations in a row that leave it on the holder, stop and report.
- protective_stop: stop at once and report; do not retry.
- A tool that refuses (outcome such as previous_fork_not_seated or connector_standing)
  tells you what to do first. Do that.

Be decisive: one tool call per turn, no commentary between calls. The final report states
what was built, how many attempts each step needed, what failed and why."""


@dataclass
class BuildResult:
    planner: str
    claimed_success: Optional[bool]
    report: str
    truth: Dict[str, Any]
    sim_time: float
    wall_time: float
    turns: int
    tool_calls: List[Dict[str, Any]]
    messages: List[Dict[str, Any]] = field(default_factory=list)
    usage: Dict[str, Any] = field(default_factory=dict)
    model: str = ""
    error: str = ""
    refused_verdicts: int = 0            # finish(success=true) calls refused for lack of evidence
    scenario: str = "nominal"
    disturbances: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return bool(self.truth.get("success"))

    def as_dict(self) -> Dict[str, Any]:
        return {"planner": self.planner, "model": self.model, "claimed_success": self.claimed_success,
                "success": self.success, "report": self.report, "truth": self.truth,
                "sim_time": round(self.sim_time, 1), "wall_time": round(self.wall_time, 1),
                "turns": self.turns, "tool_calls": self.tool_calls, "messages": self.messages,
                "usage": self.usage, "error": self.error, "refused_verdicts": self.refused_verdicts,
                "scenario": self.scenario, "disturbances": self.disturbances}


def opening_message(session: CellSession) -> str:
    summary = session.spec.summary()
    issues = [i.as_dict() for i in session.issues]
    parts = ["Build this harness.",
             "Spec: " + json.dumps(summary),
             "Validation: " + (json.dumps(issues) if issues else "no issues")]
    if session.feasible:
        parts.append("Current state: " + json.dumps(session.status()))
    else:
        parts.append("The spec failed validation, so the cell will refuse every skill.")
    return "\n".join(parts)


# --------------------------------------------------------------- policies
# insert_connector outcomes that mean the fingers could not get around the connector
GRASP_TROUBLE = ("grasp_slipped", "grasp_failed", "grasp_blocked_by_fixture")


class ScriptedPolicy:
    """The hard-coded version of the decisions in SYSTEM_PROMPT."""

    def __init__(self, route: List[str], connector_id: str):
        self.route = route
        self.connector_id = connector_id
        self.attempts: Dict[str, int] = {}      # route_fork calls per fork (for the report)
        self.fails: Dict[str, int] = {}         # failed route_fork calls per fork
        self.stuck = 0                          # relocations in a row that left the connector on the holder
        self.inserts = 0
        self.stage = "start"
        self.inspection: Optional[Dict[str, Any]] = None
        self.flagged: List[str] = []

    def next(self, last: Optional[Dict[str, Any]], feasible: bool) -> Tuple[str, Dict[str, Any]]:
        if not feasible:
            return "finish", {"success": False, "report": "The spec failed validation; nothing was built."}
        if self.stage == "rechecking":
            self.flagged = list((last or {}).get("disagreements") or [])
            ok = self._inspection_ok(self.inspection)
            return "finish", {"success": ok, "report": self._report(ok)}
        if self.stage == "finishing":
            self.inspection = last
            ok = self._inspection_ok(last)
            disagree = list((last or {}).get("disagreements") or [])
            if disagree:
                self.stage = "rechecking"
                target = disagree[0] if len(disagree) == 1 else "all"
                return "inspect", {"target": target}
            return "finish", {"success": ok, "report": self._report(ok)}
        if self.stage == "retreated":
            self.stage = "finishing"
            return "inspect", {"target": "all"}
        state = (last or {}).get("state") or {}
        if last and last.get("outcome") == "protective_stop":
            return "finish", {"success": False, "report": "Protective stop; build aborted."}
        if last and last.get("skill") == "route_fork" and not last.get("ok") and last.get("args"):
            failed = last["args"].get("fork_id")
            if failed and last.get("outcome") not in ("previous_fork_not_seated", "unknown_fork"):
                self.fails[failed] = self.fails.get(failed, 0) + 1
        forks = state.get("forks") or {}
        todo = next((f for f in self.route if not forks.get(f, {}).get("wire_in_slot")), None)
        if todo is not None:
            n = self.fails.get(todo, 0)
            if n >= 3:
                return "finish", {"success": False, "report": f"{todo} failed three times; stopped."}
            self.attempts[todo] = self.attempts.get(todo, 0) + 1
            return "route_fork", {"fork_id": todo, "attempt": n}
        conn = state.get("connector") or {}
        if conn.get("in_holder"):
            self.stage = "retreated"
            return "retreat", {}
        if self.inserts >= 4:
            return "finish", {"success": False, "report": "Connector not seated after four attempts."}
        if last and last.get("outcome") == "connector_standing" or conn.get("standing_on_end"):
            return "tip_connector", {}
        if last and last.get("outcome") == "connector_blocked":
            return "relocate_connector", {}
        if last and last.get("skill") == "insert_connector" and last.get("outcome") in GRASP_TROUBLE:
            return "relocate_connector", {}
        if last and last.get("outcome") == "still_on_holder":
            self.stuck += 1
            if self.stuck >= 3:
                return "finish", {"success": False, "report": "The connector is stuck on its holder: three "
                                                               "relocations could not lift it clear."}
            return "relocate_connector", {}
        if last and last.get("skill") == "relocate_connector":
            self.stuck = 0
        self.inserts += 1
        return "insert_connector", {}

    @staticmethod
    def _inspection_ok(last: Optional[Dict[str, Any]]) -> bool:
        if not last:
            return False
        forks = last.get("forks") or {}
        conn = last.get("connector") or {}
        return bool(forks) and all(v.get("wire_in_slot") for v in forks.values()) and bool(conn.get("in_holder"))

    def _report(self, ok: bool) -> str:
        att = ", ".join(f"{f}: {n}" for f, n in self.attempts.items())
        text = (f"{'Complete' if ok else 'Incomplete'}. Fork attempts {att}; "
                f"connector insertions {self.inserts}.")
        if self.flagged:
            text += f" Flagged for a manual visual check: {', '.join(self.flagged)}."
        return text


class ScriptedPlanner:
    name = "scripted"

    def __init__(self, session: CellSession, log: Callable[[str], None] = print):
        self.session = session
        self.log = log

    def run(self, max_turns: int = 60, scenario: Optional[Any] = None) -> BuildResult:
        t0 = time.perf_counter()
        box = ToolBox(self.session, scenario=scenario)
        policy = ScriptedPolicy(self.session.route, self.session.connector_id)
        last: Optional[Dict[str, Any]] = None
        if self.session.feasible:
            last = box.call("get_status", {})
        turns = 0
        while box.finished is None and turns < max_turns:
            name, args = policy.next(last, self.session.feasible)
            last = box.call(name, args)
            self.log(f"[scripted] {name}({_fmt_args(args)}) -> {_brief(last)}")
            turns += 1
        return _result(self.name, "", box, self.session, t0, turns, [], {})


class ScriptedLLM:
    """Stands in for Token Factory in tests: answers every chat call with the next tool
    call of the scripted policy, through exactly the code path the real model uses."""

    def __init__(self, route: List[str], connector_id: str):
        self.policy = ScriptedPolicy(route, connector_id)
        self.usage = type("U", (), {"as_dict": staticmethod(lambda: {"calls": 0})})()
        self.n = 0

    def chat(self, model: str, messages: List[Dict[str, Any]], tools=None, **_: Any) -> Dict[str, Any]:
        last = None
        for m in reversed(messages):
            if m.get("role") == "tool":
                last = json.loads(m["content"])
                break
        feasible = "failed validation" not in messages[1]["content"]
        if last is None and feasible:
            # the opening message carries the current state
            text = messages[1]["content"]
            state = json.loads(text.split("Current state: ", 1)[1]) if "Current state: " in text else {}
            last = {"state": state}
        name, args = self.policy.next(last, feasible)
        self.n += 1
        return {"role": "assistant", "content": "", "reasoning": "",
                "tool_calls": [{"id": f"call_{self.n}", "name": name, "arguments": args}]}


class NemotronPlanner:
    """Nemotron on Nebius Token Factory drives the build through tool calls."""

    name = "nemotron"

    def __init__(self, session: CellSession, client: Optional[Any] = None, model: Optional[str] = None,
                 temperature: float = 0.2, max_tokens: int = 2048,
                 log: Callable[[str], None] = print, extra: Optional[Dict[str, Any]] = None):
        self.session = session
        self.client = client or TokenFactoryClient()
        self.model = model or self.client.planner_model()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.log = log
        self.extra = extra or {}

    def run(self, max_turns: int = 40, scenario: Optional[Any] = None) -> BuildResult:
        t0 = time.perf_counter()
        box = ToolBox(self.session, scenario=scenario)
        tools = tool_schemas(self.session.route, self.session.connector_id)
        messages: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT},
                                          {"role": "user", "content": opening_message(self.session)}]
        transcript: List[Dict[str, Any]] = []
        turns, nudges, error = 0, 0, ""
        while box.finished is None and turns < max_turns:
            turns += 1
            try:
                msg = self.client.chat(self.model, messages, tools=tools, temperature=self.temperature,
                                       max_tokens=self.max_tokens, **self.extra)
            except Exception as exc:                            # network, quota, bad request
                error = f"{type(exc).__name__}: {exc}"
                self.log(f"[nemotron] model call failed: {error}")
                break
            messages.append(assistant_message_for_history(msg))
            transcript.append({"turn": turns, "role": "assistant", "content": msg.get("content", ""),
                               "reasoning": (msg.get("reasoning") or "")[:2000],
                               "tool_calls": msg.get("tool_calls", [])})
            if not msg.get("tool_calls"):
                nudges += 1
                if nudges > 3:
                    error = "the model stopped calling tools"
                    break
                messages.append({"role": "user", "content": "Continue with a tool call. Call finish when the "
                                                             "build is complete or cannot be completed."})
                continue
            for call in msg["tool_calls"][:1]:                  # one skill at a time
                result = box.call(call["name"], call.get("arguments") or {})
                self.log(f"[nemotron] {call['name']}({_fmt_args(call.get('arguments'))}) -> {_brief(result)}")
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)})
                transcript.append({"turn": turns, "role": "tool", "name": call["name"],
                                   "arguments": call.get("arguments"), "result": result})
            for call in msg["tool_calls"][1:]:                  # answer extra calls without running them
                messages.append({"role": "tool", "tool_call_id": call["id"],
                                 "content": json.dumps({"error": "one tool call per turn; call it again"})})
        usage = self.client.usage.as_dict() if hasattr(self.client, "usage") else {}
        res = _result(self.name, self.model, box, self.session, t0, turns, transcript, usage)
        res.error = error
        return res


# --------------------------------------------------------------- helpers
def _result(name: str, model: str, box: ToolBox, session: CellSession, t0: float, turns: int,
            transcript: List[Dict[str, Any]], usage: Dict[str, Any]) -> BuildResult:
    fin = box.finished or {}
    return BuildResult(planner=name, claimed_success=fin.get("success"), report=fin.get("report", ""),
                       truth=session.truth() if session.env is not None else {"success": False},
                       sim_time=session.sim_time, wall_time=time.perf_counter() - t0, turns=turns,
                       tool_calls=box.calls, messages=transcript, usage=usage, model=model,
                       refused_verdicts=box.refused_verdicts,
                       scenario=box.scenario.name if box.scenario is not None else "nominal",
                       disturbances=box.disturbances)


def _fmt_args(args: Optional[Dict[str, Any]]) -> str:
    if not args:
        return ""
    return ", ".join(f"{k}={v!r}" for k, v in args.items() if v is not None and k != "report")


def _brief(result: Dict[str, Any]) -> str:
    if "error" in result:
        return f"error: {result['error']}"
    if "outcome" in result:
        state = result.get("state") or {}
        return f"{result['outcome']} (forks in slot: {state.get('forks_in_slot')})"
    if "recorded" in result:
        return f"finished success={result['recorded']['success']}"
    if "state" in result:
        s = result["state"]
        return f"status: forks {s.get('forks_in_slot')}, connector in holder={s.get('connector', {}).get('in_holder')}"
    if "forks" in result or "connector" in result:
        return "inspected"
    return str(result)[:120]
