"""Recovery benchmark: scenarios, triggers, the summary table, and (slow) a disturbed build."""

import os

import pytest

from harness_agent.agent import ScriptedPolicy
from harness_agent.benchmark import summarize
from harness_agent.disturbances import SCENARIOS, make_scenario

SPECS = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "specs")
SLOW = os.environ.get("HARNESS_SLOW_TESTS") == "1"


def test_scenarios_fire_once_after_the_right_call():
    route = ["F1", "F2", "F3"]
    assert make_scenario("nominal", route).triggers == []
    both = make_scenario("both", route)
    assert both.faults == ["slip_on_insert"] and [t.arg for t in both.triggers] == ["F2"]
    pop = both.triggers[0]
    assert not pop.matches("route_fork", {"fork_id": "F1"}, {"ok": True})
    assert not pop.matches("route_fork", {"fork_id": "F2"}, {"ok": False})   # only after a success
    assert pop.matches("route_fork", {"fork_id": "F2"}, {"ok": True})
    pop.fired = True
    assert not pop.matches("route_fork", {"fork_id": "F2"}, {"ok": True})
    assert set(SCENARIOS) == {"nominal", "popped_wire", "slip_on_insert", "both"}
    with pytest.raises(ValueError):
        make_scenario("earthquake", route)


def test_scripted_policy_counts_failed_attempts_only():
    policy = ScriptedPolicy(["F1", "F2"], "X1")
    state = {"forks": {"F1": {"wire_in_slot": False}, "F2": {"wire_in_slot": False}}}
    assert policy.next({"state": state}, True) == ("route_fork", {"fork_id": "F1", "attempt": 0})
    routed = {"forks": {"F1": {"wire_in_slot": True}, "F2": {"wire_in_slot": False}}}
    ok = {"skill": "route_fork", "args": {"fork_id": "F1"}, "ok": True, "outcome": "routed", "state": routed}
    assert policy.next(ok, True) == ("route_fork", {"fork_id": "F2", "attempt": 0})
    # F1 lost the wire again later (a snag): back to F1 with a fresh first attempt
    lost = {"skill": "route_fork", "args": {"fork_id": "F2"}, "ok": False,
            "outcome": "previous_fork_not_seated", "state": state}
    assert policy.next(lost, True) == ("route_fork", {"fork_id": "F1", "attempt": 0})
    failed = {"skill": "route_fork", "args": {"fork_id": "F1"}, "ok": False, "outcome": "grasp_failed",
              "state": state}
    assert policy.next(failed, True) == ("route_fork", {"fork_id": "F1", "attempt": 1})


def test_scripted_policy_relocates_a_connector_it_cannot_grasp():
    policy = ScriptedPolicy(["F1"], "X1")
    done = {"forks": {"F1": {"wire_in_slot": True}}, "connector": {"in_holder": False, "standing_on_end": False}}
    assert policy.next({"state": done}, True) == ("insert_connector", {})
    for outcome in ("grasp_slipped", "grasp_failed", "grasp_blocked_by_fixture"):
        failed = {"skill": "insert_connector", "ok": False, "outcome": outcome, "state": done}
        assert policy.next(failed, True) == ("relocate_connector", {})
    moved = {"skill": "relocate_connector", "ok": True, "outcome": "graspable", "state": done}
    assert policy.next(moved, True) == ("insert_connector", {})


def test_relocation_must_leave_the_connector_flat_and_clear_of_the_holder():
    from harness_agent.session import connector_lies_clear
    board = {"standing_on_end": False, "tilt_deg": 3.0, "offset_from_holder_mm": [30.0, 60.0, -1.5]}
    assert connector_lies_clear(board)
    on_rail = dict(board, offset_from_holder_mm=[5.7, 15.1, 8.8], tilt_deg=14.5)   # seen after a failed lift
    assert not connector_lies_clear(on_rail)
    assert not connector_lies_clear(dict(board, tilt_deg=20.0))
    assert not connector_lies_clear(dict(board, offset_from_holder_mm=[5.0, 10.0, -1.5]))
    assert not connector_lies_clear(dict(board, holder_in_finger_path=True))    # fingers would hit the holder


def test_scripted_policy_retries_a_stuck_relocation_three_times():
    policy = ScriptedPolicy(["F1"], "X1")
    done = {"forks": {"F1": {"wire_in_slot": True}}, "connector": {"in_holder": False, "standing_on_end": False}}
    stuck = {"skill": "relocate_connector", "ok": False, "outcome": "still_on_holder", "state": done}
    assert policy.next(stuck, True) == ("relocate_connector", {})
    assert policy.next(stuck, True) == ("relocate_connector", {})
    name, args = policy.next(stuck, True)
    assert name == "finish" and args["success"] is False


def test_summary_table_counts_successes_and_honest_verdicts():
    rows = [{"planner": "scripted", "scenario": "nominal", "seed": s, "success": s != 1, "honest": True,
             "tool_calls": 8, "robot_s": 70.0, "tokens": 0} for s in range(3)]
    rows.append({"planner": "nemotron", "scenario": "both", "seed": 0, "success": False, "honest": False,
                 "tool_calls": 12, "robot_s": 90.0, "tokens": 40000})
    text = summarize(rows)
    assert "| scripted | nominal | 3 | 2/3 | 3/3 | 8.0 | 70 s | - |" in text
    assert "| nemotron | both | 1 | 0/1 | 0/1 | 12.0 | 90 s | 40,000 |" in text


@pytest.mark.skipif(not SLOW, reason="two disturbed builds (~3 min); set HARNESS_SLOW_TESTS=1")
def test_scripted_planner_recovers_from_a_popped_wire_and_a_slipped_connector():
    from harness_agent.agent import ScriptedPlanner
    from harness_agent.session import CellSession
    from harness_agent.spec import HarnessSpec

    session = CellSession(HarnessSpec.from_yaml(os.path.join(SPECS, "demo_3fork.yaml")), seed=0, randomize=True)
    res = ScriptedPlanner(session, log=lambda *_: None).run(scenario=make_scenario("both", session.route))
    assert [d["label"] for d in res.disturbances] == ["wire pulled out of F2"]
    assert any(c["result"].get("outcome") == "grasp_slipped" for c in res.tool_calls)
    assert res.success and res.claimed_success


def test_annotated_video_follows_the_tool_calls(tmp_path):
    import json

    import imageio
    import numpy as np

    from harness_agent.annotate import _reasoning_per_call, annotate

    trace = {"planner": "nemotron", "model": "nvidia/nemotron-3-super-120b-a12b", "scenario": "popped_wire",
             "session": {"spec": "test board"}, "success": True, "claimed_success": True,
             "tool_calls": [{"name": "route_fork", "arguments": {"fork_id": "F1"}, "sim_time": 1.0,
                             "result": {"ok": True, "outcome": "routed"}},
                            {"name": "route_fork", "arguments": {"fork_id": "F2"}, "sim_time": 2.0,
                             "result": {"ok": False, "outcome": "previous_fork_not_seated"}},
                            {"name": "finish", "arguments": {"success": True}, "sim_time": 2.0,
                             "result": {"ok": True, "recorded": {"success": True}}}],
             "messages": [{"role": "assistant", "reasoning": "route F1 first"}, {"role": "tool"},
                          {"role": "assistant", "content": "then F2"}, {"role": "tool"},
                          {"role": "assistant", "reasoning": "done"}, {"role": "tool"}],
             "disturbances": [{"label": "wire pulled out of F1", "sim_time": 1.2}]}
    with open(tmp_path / "trace.json", "w") as f:
        json.dump(trace, f)
    assert _reasoning_per_call(trace) == ["route F1 first", "then F2", "done"]
    frames = [np.full((48, 64, 3), 40 * k, np.uint8) for k in range(5)]
    path = annotate(str(tmp_path), frames=frames, frame_times=[0.0, 0.5, 1.0, 1.5, 2.0], fps=4)
    reader = imageio.get_reader(path)
    n = reader.count_frames()
    shape = reader.get_data(0).shape
    reader.close()
    assert n == 5 + 3 * 4 and shape == (1080, 1920, 3)       # frames + a three-second hold on the verdict
