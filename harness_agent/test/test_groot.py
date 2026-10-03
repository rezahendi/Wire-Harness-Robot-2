"""GR00T integration: data layout, dataset files, and the policy-server protocol (no GPU needed)."""

import json
import os
import threading

import numpy as np
import pytest

from harness_agent import groot_features as gf

SLOW = os.environ.get("HARNESS_SLOW_TESTS") == "1"


def fake_obs(yaw=0.3, target_yaw=0.5):
    return {"tcp_pos": np.array([0.4, 0.1, 0.2]), "tcp_yaw": np.array([yaw]),
            "target_pos": np.array([0.41, 0.1, 0.19]), "target_yaw": np.array([target_yaw]),
            "gripper": np.array([0.034]), "wrench": np.arange(6, dtype=float),
            "cable": np.linspace([0, 0, 0], [1, 1, 1], 25),
            "forks": np.array([[0.5, -0.1, 0.02, 0.8], [0.6, 0.0, 0.02, 1.4]]),
            "anchor_pos": np.array([0.3, -0.3, 0.03]), "holder_pos": np.array([0.45, 0.2, 0.03]),
            "holder_yaw": np.array([2.4]), "connector_pos": np.array([0.7, 0.1, 0.03])}


class _Cfg:
    class fork:
        post_height = 0.015
    class wire:
        radius = 0.003


def test_state_vector_layout_matches_the_names_and_slices():
    assert gf.STATE_DIM == len(gf.STATE_NAMES) == 47 and gf.ACTION_DIM == len(gf.ACTION_NAMES) == 5
    obs = fake_obs(yaw=3.1, target_yaw=-3.1)                      # the yaw lead wraps around pi
    goal = gf.goal_vector(obs, "route_fork", 1, _Cfg)
    v = gf.state_vector(obs, goal)
    parts = gf.split_state(v)
    assert np.allclose(parts["tcp"][:3], obs["tcp_pos"])
    assert np.allclose(parts["command"][:3], [0.01, 0.0, -0.01], atol=1e-6)
    assert abs(parts["command"][3] - (2 * np.pi - 6.2)) < 1e-5
    assert np.allclose(parts["goal"][:2], [0.6, 0.0]) and np.allclose(parts["goal"][4:6], [0.5, -0.1])
    assert abs(parts["goal"][6] - (0.02 + 0.015 + 0.003)) < 1e-6    # fixed on top of the previous fork
    assert np.allclose(parts["cable"][:3], 0) and np.allclose(parts["cable"][-3:], 1)
    first = gf.goal_vector(obs, "route_fork", 0, _Cfg)               # F1: fixed at the clamp
    assert np.allclose(first[4:], obs["anchor_pos"])


def test_modality_json_matches_the_groot_config_keys():
    from harness_agent.groot_data import info_json, modality_json
    m = modality_json()
    assert list(m["state"]) == ["tcp", "command", "gripper", "wrench", "goal", "cable"]
    assert m["state"]["cable"] == {"start": 23, "end": 47} and m["action"]["gripper"] == {"start": 4, "end": 5}
    assert m["annotation"] == {"human.task_description": {"original_key": "task_index"}}
    info = info_json(3, 900, 2, 256)
    assert info["features"]["observation.state"]["shape"] == [47]
    assert info["features"]["observation.images.wrist"]["shape"] == [256, 256, 3]
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    cfg = open(os.path.join(here, "groot", "harness_config.py")).read()
    for key in list(m["state"]) + ["motion", "scene", "wrist", "annotation.human.task_description"]:
        assert f'"{key}"' in cfg


def _episode(staging, seed, target, n=40, order=1):
    from harness_agent.groot_data import EpisodeBuffer, write_episode
    ep = EpisodeBuffer("route_fork", target, None)
    for t in range(n):
        for v in gf.VIDEO_KEYS:
            ep.frames[v].append(np.full((64, 64, 3), 10 * t % 255, np.uint8))
        ep.states.append(np.full(gf.STATE_DIM, t, np.float32))
        ep.actions.append(np.array([t / n, 0, 0, 0, 1 if t < n // 2 else -1], np.float32))
    write_episode(ep, os.path.join(str(staging), f"s{seed:06d}_{order:02d}"),
                  {"seed": seed, "order": order, "skill": "route_fork", "target": target, "attempt": 0,
                   "scenario": "nominal"})


def _tiny_dataset(tmp_path, n=40):
    from harness_agent.groot_data import merge
    staging = tmp_path / "staging"
    for k, target in enumerate(["F1", "F2"]):
        _episode(staging, 1000 + k, target, n)
    out = tmp_path / "set"
    merge(str(staging), str(out), size=64)
    return out


def test_recorded_set_has_the_lerobot_layout_groot_reads(tmp_path):
    from harness_agent.groot_data import check
    out = _tiny_dataset(tmp_path)
    res = check(str(out))
    assert res["problems"] == [] and res["episodes"] == 2 and res["frames"] == 80
    tasks = [json.loads(line) for line in open(out / "meta" / "tasks.jsonl")]
    assert [t["task"] for t in tasks] == ["route the wire into fork F1", "route the wire into fork F2"]
    extra = [json.loads(line) for line in open(out / "meta" / "harness_episodes.jsonl")]
    assert [(e["episode_index"], e["seed"], e["target"]) for e in extra] == [(0, 1000, "F1"), (1, 1001, "F2")]
    import pyarrow.parquet as pq
    t = pq.read_table(out / "data" / "chunk-000" / "episode_000001.parquet").to_pydict()
    assert t["episode_index"][0] == 1 and t["index"][0] == 40 and t["task_index"][0] == 1
    assert len(t["observation.state"][0]) == gf.STATE_DIM


def test_merge_combines_sets_once_and_writes_groot_stats(tmp_path):
    from harness_agent.groot_data import _fingerprint, check, merge
    first = _tiny_dataset(tmp_path)                       # seeds 1000 (F1) and 1001 (F2), packaged
    more = tmp_path / "more" / "staging"                  # a second recording: 1001 again, and 1002
    _episode(more, 1001, "F2")
    _episode(more, 1002, "F3", n=30)
    out = tmp_path / "all"
    res = merge([str(first), str(tmp_path / "more")], str(out))
    assert res["episodes"] == 3 and res["duplicates"] == 1 and res["frames"] == 110
    assert check(str(out))["problems"] == []
    assert not more.exists() and check(str(first))["episodes"] == 2       # staging packaged, the first set kept
    extra = [json.loads(line) for line in open(out / "meta" / "harness_episodes.jsonl")]
    assert [(e["seed"], e["target"], e["length"]) for e in extra] == [(1000, "F1", 40), (1001, "F2", 40),
                                                                       (1002, "F3", 30)]
    stats = json.load(open(out / "meta" / "stats.json"))
    info = json.load(open(out / "meta" / "info.json"))
    for col in ("action", "observation.state", "timestamp"):
        assert stats["__fingerprints__"][col] == _fingerprint(col, info["features"][col])
        assert set(stats[col]) == {"mean", "std", "min", "max", "q01", "q99"}
    assert len(stats["observation.state"]["mean"]) == gf.STATE_DIM and stats["action"]["max"][4] == 1.0
    with pytest.raises(SystemExit, match="already holds a dataset"):
        merge(str(first), str(out))


def test_recording_resumes_and_ctrl_c_packages_what_is_done(tmp_path, monkeypatch):
    from harness_agent import groot_data as gd
    calls = []

    def fake_record_seed(seed):
        calls.append(seed)
        if seed == 3:
            raise KeyboardInterrupt                       # Ctrl-c during the third build
        _episode(gd._REC["staging"], seed, "F1", n=20)
        return {"seed": seed, "kept": 1, "dropped": 0, "seconds": 0.1}

    monkeypatch.setattr(gd, "record_seed", fake_record_seed)
    out = tmp_path / "rec"
    staging = out / "staging"
    _episode(staging, 1, "F1", n=20)                     # seed 1 finished in an earlier, interrupted run
    os.makedirs(staging / gd.DONE_DIR)
    (staging / gd.DONE_DIR / "1.json").write_text("{}")
    res = gd.record([1, 2, 3, 4], str(out), workers=1, skills=["route_fork"], popped=0.0,
                    spec=gd._spec_path("demo_3fork.yaml"), size=64)
    assert calls == [2, 3] and res["stopped"] and res["episodes"] == 2
    assert gd.check(str(out))["problems"] == [] and not staging.exists()
    with pytest.raises(SystemExit, match="already holds a packaged dataset"):
        gd.record([5], str(out), workers=1, skills=["route_fork"], popped=0.0,
                  spec=gd._spec_path("demo_3fork.yaml"), size=64)


def test_eval_report_counts_crashes_and_server_time():
    from harness_agent.groot_eval import latency_line, summarize
    rows = [{"seed": 0, "fork": "F1", "ok": True, "outcome": "routed", "seconds": 14.0, "max_force_N": 9.0,
             "truth_routed": True, "chunks": 30, "inference_s": 7.5},
            {"seed": 0, "fork": "F2", "ok": False, "outcome": "timeout", "seconds": 40.0, "max_force_N": 4.0,
             "truth_routed": False, "chunks": 100, "inference_s": 30.0},
            {"seed": 1, "fork": "F2", "skipped": "expert failed on F1 (timeout)"},
            {"seed": 1, "fork": "F1", "error": "GrootError: no answer"}]
    text = summarize(rows, "t")
    assert "| all | 2 | 1/2 (50%) |" in text and "1 trials skipped" in text and "1 trials crashed" in text
    assert "Failures: timeout 1" in text
    line = latency_line(rows, workers=4)
    assert "130 action chunks, 288 ms per round trip" in line and "4 workers" in line
    assert latency_line([rows[2]], workers=1) == ""


def test_several_clients_share_one_policy_server(tmp_path):
    from harness_agent.groot_client import GrootClient
    from harness_agent.groot_replay_server import ReplayPolicy, serve
    policy = ReplayPolicy(str(_tiny_dataset(tmp_path)), execution_horizon=8)
    port = 5597
    th = threading.Thread(target=serve, args=(policy,), kwargs={"host": "127.0.0.1", "port": port}, daemon=True)
    th.start()
    imgs = {k: np.zeros((256, 256, 3), np.uint8) for k in gf.VIDEO_KEYS}
    obs = gf.observation_for_policy(imgs, np.zeros(gf.STATE_DIM, np.float32), "route the wire into fork F1")
    got, errors = [], []

    def worker():
        c = GrootClient("127.0.0.1", port, timeout_ms=5000)
        try:
            for _ in range(5):
                got.append(c.get_action(obs)["motion"].shape)
        except Exception as exc:          # pragma: no cover - reported below
            errors.append(exc)
        finally:
            c.close()

    workers = [threading.Thread(target=worker) for _ in range(3)]
    for w in workers:
        w.start()
    for w in workers:
        w.join(timeout=20)
    stop = GrootClient("127.0.0.1", port, timeout_ms=5000)
    stop.call("kill")
    stop.close()
    th.join(timeout=5)
    assert not errors and got == [(1, 16, 4)] * 15 and policy.observations == 15


def test_recovery_pushes_follow_the_experts_phase():
    from harness_agent.groot_data import RecoveryNoise
    phase = {"now": "pick_close"}
    noise = RecoveryNoise(1.0, np.random.default_rng(0), phase=lambda: phase["now"])
    a = np.array([0.2, -0.1, 0.0, 0.0, 1.0])
    assert np.array_equal(noise(a), a)                                   # closing the gripper: no push
    phase["now"] = "route_descend"
    pushed = np.array([noise(a) for _ in range(200)])
    assert np.all(pushed[:, 2] == 0.0) and np.all(pushed[:, 4] == 1.0)   # no vertical push, gripper untouched
    assert pushed[:, :2].std() > 0.02
    phase["now"] = "pick_approach"
    out = np.array([noise(a) for _ in range(2000)])
    assert noise.kicks > 5 and np.all(np.abs(out[:, :4]) <= 1.0)
    assert np.array_equal(RecoveryNoise(0.0, np.random.default_rng(0), phase=lambda: "pick_approach")(a), a)


def test_route_stats_count_who_routed_and_skip_refusals():
    from harness_agent.groot_skill import route_stats
    calls = [{"name": "get_status", "result": {"state": {}}},
             {"name": "route_fork", "result": {"skill": "route_fork", "ok": True, "outcome": "routed",
                                               "executed_by": "GR00T N1.7 (fine-tuned)"}},
             {"name": "route_fork", "result": {"skill": "route_fork", "ok": False, "outcome": "timeout",
                                               "executed_by": "GR00T N1.7 (fine-tuned)"}},
             {"name": "route_fork", "result": {"skill": "route_fork", "ok": False,
                                               "outcome": "previous_fork_not_seated"}},
             {"name": "route_fork", "result": {"skill": "route_fork", "ok": True, "outcome": "routed"}},
             {"name": "route_fork", "result": {"error": "fork budget used up"}}]
    assert route_stats(calls) == {"groot_routes": 2, "groot_ok": 1, "expert_routes": 1, "expert_ok": 1}


def test_connect_runner_needs_a_server():
    from harness_agent.groot_skill import connect_runner
    with pytest.raises(SystemExit, match="no GR00T policy server at 127.0.0.1:5595"):
        connect_runner("127.0.0.1:5595", timeout_ms=300)


@pytest.mark.skipif(not os.environ.get("GROOT_REPO"), reason="set GROOT_REPO to an Isaac-GR00T checkout")
def test_stats_are_the_ones_groot_would_compute(tmp_path):
    import sys
    sys.path.insert(0, os.environ["GROOT_REPO"])
    try:
        from gr00t.data import stats as gs
    except Exception as exc:                              # GR00T's own dependencies missing here
        pytest.skip(f"cannot import gr00t.data.stats: {exc}")
    out = _tiny_dataset(tmp_path)
    ours = json.load(open(out / "meta" / "stats.json"))
    assert gs.check_stats_validity(out, ["action", "observation.state", "timestamp"])   # GR00T reuses ours
    theirs = gs.calculate_dataset_statistics(list(out.glob(gs.LE_ROBOT_DATA_FILENAME)),
                                             ["action", "observation.state", "timestamp"])
    for col, values in theirs.items():
        for k, v in values.items():
            assert np.allclose(ours[col][k], v, atol=1e-6), (col, k)


def test_client_talks_to_a_replay_server_like_groot(tmp_path):
    from harness_agent.groot_client import GrootClient, GrootError
    from harness_agent.groot_replay_server import ReplayPolicy, serve
    out = _tiny_dataset(tmp_path)
    policy = ReplayPolicy(str(out), execution_horizon=8)
    port = 5599
    th = threading.Thread(target=serve, args=(policy,), kwargs={"host": "127.0.0.1", "port": port, "max_requests": 6},
                          daemon=True)
    th.start()
    client = GrootClient("127.0.0.1", port, timeout_ms=5000)
    try:
        assert client.ping()
        assert client.reset({"episode_index": 1})["episode_index"] == 1
        imgs = {k: np.zeros((256, 256, 3), np.uint8) for k in gf.VIDEO_KEYS}
        obs = gf.observation_for_policy(imgs, np.zeros(gf.STATE_DIM, np.float32), "route the wire into fork F2")
        a = client.get_action(obs)
        assert a["motion"].shape == (1, 16, 4) and a["gripper"].shape == (1, 16, 1)
        chunk = gf.join_action({k: v[0] for k, v in a.items()})
        assert chunk.shape == (16, 5) and np.isclose(chunk[3, 0], 3 / 40)
        b = client.get_action(obs)                                   # advanced by the execution horizon
        assert np.isclose(gf.join_action({k: v[0] for k, v in b.items()})[0, 0], 8 / 40)
        obs["state"]["cable"] = obs["state"]["cable"].astype(np.float64)   # wrong dtype: rejected, as GR00T does
        with pytest.raises(GrootError, match="bad observation"):
            client.get_action(obs)
        client.call("kill")
    finally:
        client.close()
        th.join(timeout=5)


@pytest.mark.skipif(not os.environ.get("GROOT_REPO"), reason="set GROOT_REPO to an Isaac-GR00T checkout")
def test_payloads_survive_groots_own_serializer():
    import importlib.util
    import sys
    import types
    repo = os.environ["GROOT_REPO"]
    sys.path.insert(0, repo)
    pkg = types.ModuleType("gr00t.policy")
    pkg.__path__ = [os.path.join(repo, "gr00t", "policy")]
    sys.modules.setdefault("gr00t.policy", pkg)
    for name in ("policy", "server_client"):
        spec = importlib.util.spec_from_file_location(f"gr00t.policy.{name}", os.path.join(repo, "gr00t", "policy", f"{name}.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"gr00t.policy.{name}"] = mod
        spec.loader.exec_module(mod)
    from harness_agent.groot_client import pack, unpack
    ser = sys.modules["gr00t.policy.server_client"].MsgSerializer
    obs = gf.observation_for_policy({k: np.ones((256, 256, 3), np.uint8) for k in gf.VIDEO_KEYS},
                                    np.arange(gf.STATE_DIM, dtype=np.float32), "route the wire into fork F1")
    got = ser.from_bytes(pack({"endpoint": "get_action", "data": {"observation": obs}}))["data"]["observation"]
    assert np.array_equal(got["state"]["goal"], obs["state"]["goal"])
    back = unpack(ser.to_bytes([{"motion": np.zeros((1, 16, 4), np.float32)}, {}]))
    assert back[0]["motion"].shape == (1, 16, 4)


@pytest.mark.skipif(not SLOW, reason="records a build and replays it (~5 min); set HARNESS_SLOW_TESTS=1")
def test_replayed_demo_routes_the_wire_through_the_policy_loop(tmp_path):
    """Record one scripted build, serve its F1 episode, and let the GR00T runner execute it."""
    from harness_agent.groot_client import GrootClient
    from harness_agent.groot_data import main as record_main
    from harness_agent.groot_eval import run_trial
    from harness_agent.groot_replay_server import ReplayPolicy, serve
    from harness_agent.groot_skill import GrootRunner
    from harness_agent.spec import HarnessSpec
    out = tmp_path / "set"
    assert record_main(["record", "--out", str(out), "--seeds", "1000", "--workers", "1", "--popped", "0"]) == 0
    th = threading.Thread(target=serve, args=(ReplayPolicy(str(out)),), kwargs={"host": "127.0.0.1", "port": 5598},
                          daemon=True)
    th.start()
    client = GrootClient("127.0.0.1", 5598, timeout_ms=10000)
    spec_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "specs")
    runner = GrootRunner(client, forks=["F1"], attempts=None)
    row = run_trial(HarnessSpec.from_yaml(os.path.join(spec_dir, "demo_3fork.yaml")), 1000, "F1", runner,
                    replay_episode=0)
    client.call("kill")
    assert row["ok"] and row["truth_routed"] and row["controller"].startswith("GR00T")
