import json
import os

import pytest

from harness_agent.agent import NemotronPlanner, ScriptedLLM
from harness_agent.llm import normalise_message
from harness_agent.session import CellSession
from harness_agent.spec import HarnessSpec
from harness_agent.tools import tool_schemas

SPECS = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "specs")
SLOW = os.environ.get("HARNESS_SLOW_TESTS") == "1"


def spec(name):
    return HarnessSpec.from_yaml(os.path.join(SPECS, name + ".yaml"))


def test_tool_schemas_are_valid_function_definitions():
    tools = tool_schemas(["F1", "F2"], "X1")
    names = [t["function"]["name"] for t in tools]
    assert names == ["get_status", "route_fork", "tip_connector", "relocate_connector",
                     "insert_connector", "inspect", "retreat", "finish"]
    for t in tools:
        json.dumps(t)
        assert t["type"] == "function"
        assert t["function"]["parameters"]["type"] == "object"


def test_tool_calls_are_recovered_from_structured_and_text_replies():
    structured = {"content": None, "tool_calls": [{"id": "a", "function": {
        "name": "route_fork", "arguments": "{\"fork_id\": \"F1\"}"}}]}
    msg = normalise_message(structured)
    assert msg["tool_calls"][0]["arguments"] == {"fork_id": "F1"}
    text = {"content": "<think>plan</think><tool_call>{\"name\": \"get_status\", \"arguments\": {}}</tool_call>"}
    msg = normalise_message(text)
    assert msg["tool_calls"][0]["name"] == "get_status"
    assert "plan" not in msg["content"]


def test_infeasible_spec_is_refused_without_moving_the_robot():
    session = CellSession(spec("demo_infeasible"))
    assert not session.feasible
    res = NemotronPlanner(session, client=ScriptedLLM(session.route, session.connector_id), model="stub").run()
    assert res.claimed_success is False
    assert [c["name"] for c in res.tool_calls] == ["finish"]


def test_skills_refuse_out_of_order_requests():
    session = CellSession(spec("demo_3fork"), seed=0)
    res = session.route_fork("F3")
    assert not res.ok and res.outcome == "previous_fork_not_seated"
    assert session.insert_connector().outcome == "forks_not_routed"
    assert "truth" not in res.for_planner()          # the planner never sees ground truth


@pytest.mark.skipif(not SLOW, reason="full build (~1 min); set HARNESS_SLOW_TESTS=1")
def test_agent_loop_builds_the_nominal_harness():
    session = CellSession(spec("demo_3fork"), seed=0)
    res = NemotronPlanner(session, client=ScriptedLLM(session.route, session.connector_id), model="stub").run()
    assert res.success and res.claimed_success
    assert [c["name"] for c in res.tool_calls[-2:]] == ["inspect", "finish"]
    assert res.refused_verdicts == 0
