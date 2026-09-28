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


def _tiny_dataset(tmp_path, n=40):
    from harness_agent.groot_data import EpisodeBuffer, merge, write_episode
    staging = tmp_path / "staging"
    for k, target in enumerate(["F1", "F2"]):
        ep = EpisodeBuffer("route_fork", target, None)
        for t in range(n):
            for v in gf.VIDEO_KEYS:
                ep.frames[v].append(np.full((64, 64, 3), 10 * t % 255, np.uint8))
            ep.states.append(np.full(gf.STATE_DIM, t, np.float32))
            ep.actions.append(np.array([t / n, 0, 0, 0, 1 if t < n // 2 else -1], np.float32))
        write_episode(ep, str(staging / f"s{1000 + k:06d}_01"),
                      {"seed": 1000 + k, "order": 1, "skill": "route_fork", "target": target, "attempt": 0})
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
