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
        prong_height = 0.028
        slot_width = 0.013
        lip_radius = 0.003

    class wire:
        radius = 0.003


def test_state_vector_layout_matches_the_names_and_slices():
    assert gf.BASE_DIM == len(gf.BASE_NAMES) == 47 and gf.ACTION_DIM == len(gf.ACTION_NAMES) == 5
    assert gf.V3_DIM == len(gf.V3_NAMES) == 66 and gf.STATE_LAYOUT[:6] == gf.BASE_LAYOUT
    assert gf.STATE_DIM == len(gf.STATE_NAMES) == 79 and gf.STATE_LAYOUT[:9] == gf.V3_LAYOUT
    assert gf.STATE_NAMES[:66] == gf.V3_NAMES and sorted(gf.STATE_LAYOUTS) == [47, 66, 79]
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


def test_geometry_is_measured_from_the_gripper():
    obs = fake_obs(yaw=0.0)
    obs["board_z"] = np.array([0.0])
    obs["forks"] = np.array([[0.5, 0.0, 0.0, 0.0], [0.6, 0.0, 0.0, 0.0]])      # slots open along x
    obs["anchor_pos"] = np.array([0.3, 0.0, 0.03])           # route to F1 runs along +x
    obs["tcp_pos"] = np.array([0.52, 0.004, 0.05])           # 2 cm beyond F1, 4 mm to its left
    xs = np.linspace(0.3, 0.7, 41)
    obs["cable"] = np.stack([xs, np.full_like(xs, 0.001), np.full_like(xs, 0.035)], axis=1)  # straight, in the slot
    parts = gf.state_parts(obs, "route_fork", 0, _Cfg)
    r, w, sl = parts["route"], parts["wire"], parts["slot"]
    assert np.allclose(r[:3], [0.2, 0.04, (0.05 - 0.043) / 0.1], atol=1e-5)
    assert np.isclose(r[3], np.tanh(2.0)) and np.isclose(r[4], np.tanh(0.4)) and np.isclose(r[7], 1.0)
    assert np.allclose(w[:3], [0.0, -0.03, -0.15], atol=1e-5)    # wire 3 mm to the right, 15 mm below
    assert np.isclose(w[7], 1.0, atol=1e-6)                       # wire along the gripper's x axis
    assert sl[0] == 1.0 and np.isclose(sl[1], np.tanh(0.001 / 0.005), atol=1e-4)
    assert sl[2] < 0                                               # below the prong tops: in the slot
    v = gf.join_state(parts)
    assert v.shape == (gf.STATE_DIM,) and np.allclose(gf.split_state(v)["route"], r)
    other = gf.state_parts(obs, "insert_connector", -1, _Cfg)
    assert not other["route"].any() and not other["slot"].any()
    base = gf.observation_for_policy({k: np.zeros((4, 4, 3), np.uint8) for k in gf.VIDEO_KEYS}, parts, "t",
                                     [k for k, _ in gf.BASE_LAYOUT])
    assert list(base["state"]) == [k for k, _ in gf.BASE_LAYOUT]   # what a model of the base layout gets


def test_plan_keys_measure_the_gripper_against_the_experts_plan():
    obs = fake_obs(yaw=0.0)
    obs["board_z"] = np.array([0.0])
    obs["forks"] = np.array([[0.5, 0.0, 0.0, 0.0], [0.6, 0.0, 0.0, 0.0]])
    obs["anchor_pos"] = np.array([0.3, 0.0, 0.03])            # route to F1 runs along +x
    xs = np.linspace(0.3, 0.7, 41)
    obs["cable"] = np.stack([xs, np.full_like(xs, 0.002), np.full_like(xs, 0.003)], axis=1)  # along x, s=0 at 0.3
    plan = {"s_pick": 0.25, "beyond": 0.07, "press_z": 0.014}          # grasp at x = 0.55
    obs["tcp_pos"] = np.array([0.55, 0.0, 0.03])
    obs["wrench"] = np.array([-4.0, 0.0, 0.0, 0.0, 0.0, 0.0])           # the wire pulls back towards the fixation
    parts = gf.state_parts(obs, "route_fork", 0, _Cfg, plan=plan)
    pick, seat = parts["pick"], parts["seat"]
    assert np.allclose(pick[:3], [0.0, 0.02, -0.27], atol=1e-4)       # 2 mm to the left, 27 mm below
    assert np.isclose(pick[4], np.tanh(0.2), atol=1e-4) and np.isclose(pick[7], 1.0)
    assert np.isclose(seat[0], 0.4) and np.isclose(seat[2], np.tanh(1.6), atol=1e-4)
    assert np.isclose(seat[3], (0.05 - 0.07) / 0.1, atol=1e-5)       # 5 cm past the fork, plan says 7
    v = gf.join_state(parts)
    assert v.shape == (79,) and np.allclose(gf.split_state(v)["seat"], seat)
    assert not gf.state_parts(obs, "route_fork", 0, _Cfg)["pick"].any()          # no plan: zeros
    assert gf.join_state(parts, gf.V3_LAYOUT).shape == (66,)


def test_the_expert_plans_route_fork_with_the_shared_plan():
    from harness_agent.session import CellSession
    from harness_agent.spec import HarnessSpec
    from harness_core.expert import plan_route
    spec_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "specs")
    s = CellSession(HarnessSpec.from_yaml(os.path.join(spec_dir, "demo_3fork.yaml")), seed=1003, randomize=True)
    try:
        obs = s.obs
        s.expert._last_obs = obs
        gen = s.expert._route_fork(0)
        next(gen)                                         # plans, then asks for its first action
        gen.close()
        mine = plan_route(obs, 0, s.cfg, s.expert.p)
        assert mine is not None and s.expert.route_plan == mine
        assert 0.0 < mine["beyond"] < 0.2 and mine["s_pick"] > 0.1
    finally:
        s.close()


def test_raw_frames_travel_with_the_episodes(tmp_path):
    from harness_agent.groot_data import EpisodeBuffer, config_for, merge, write_episode
    ep = EpisodeBuffer("route_fork", "F1", None, plan_fn=lambda: {"s_pick": 0.25, "beyond": 0.07, "press_z": 0.01})
    obs = dict(fake_obs(), time=np.array([0.5]), board_z=np.array([0.0]))
    for t in range(6):
        for v in gf.VIDEO_KEYS:
            ep.frames[v].append(np.full((64, 64, 3), 40 * t, np.uint8))
        ep.states.append(np.zeros(gf.STATE_DIM, np.float32))
        ep.actions.append(np.zeros(5, np.float32))
        ep.keep_raw(obs)
    write_episode(ep, str(tmp_path / "staging" / "s001000_01"),
                  {"seed": 1000, "order": 1, "skill": "route_fork", "target": "F1", "scenario": "nominal"})
    merge(str(tmp_path / "staging"), str(tmp_path / "set"), size=64)
    raw = np.load(tmp_path / "set" / "raw" / "chunk-000" / "episode_000000.npz")
    assert raw["cable"].shape == (6, 25, 3) and np.allclose(raw["plan"][0], [0.25, 0.07, 0.01])
    assert config_for(str(tmp_path / "set")).endswith(os.path.join("groot", "harness_config.py"))
    assert os.path.exists(config_for(str(tmp_path / "set")))


def test_modality_json_matches_the_groot_config_keys():
    from harness_agent.groot_data import info_json, modality_json
    m = modality_json()
    assert list(m["state"]) == ["tcp", "command", "gripper", "wrench", "goal", "cable", "route", "wire", "slot",
                                "pick", "seat"]
    assert m["state"]["cable"] == {"start": 23, "end": 47} and m["state"]["slot"] == {"start": 63, "end": 66}
    assert m["state"]["pick"] == {"start": 66, "end": 74} and m["state"]["seat"] == {"start": 74, "end": 79}
    assert list(modality_json(66)["state"])[-1] == "slot"
    assert m["action"]["gripper"] == {"start": 4, "end": 5}
    assert m["annotation"] == {"human.task_description": {"original_key": "task_index"}}
    assert list(modality_json(47)["state"]) == ["tcp", "command", "gripper", "wrench", "goal", "cable"]
    info = info_json(3, 900, 2, 256)
    assert info["features"]["observation.state"]["shape"] == [79]
    assert len(info["features"]["observation.state"]["names"]) == 79
    assert info_json(3, 900, 2, 256, state_dim=66)["features"]["observation.state"]["shape"] == [66]
    assert info_json(3, 900, 2, 256, state_dim=47)["features"]["observation.state"]["shape"] == [47]
    assert info["features"]["observation.images.wrist"]["shape"] == [256, 256, 3]
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    for name, dim in (("harness_config.py", 79), ("harness_config_v3.py", 66), ("harness_config_base.py", 47)):
        cfg = open(os.path.join(here, "groot", name)).read()
        keys = list(modality_json(dim)["state"])
        flat = " ".join(cfg.replace("'", '"').split()).replace("[ ", "[").replace(", ]", "]")
        assert f"modality_keys={json.dumps(keys)}" in flat, name
        for key in ["motion", "scene", "wrist", "annotation.human.task_description"]:
            assert f'"{key}"' in cfg


def _episode(staging, seed, target, n=40, order=1, width=gf.STATE_DIM):
    from harness_agent.groot_data import EpisodeBuffer, write_episode
    ep = EpisodeBuffer("route_fork", target, None)
    for t in range(n):
        for v in gf.VIDEO_KEYS:
            ep.frames[v].append(np.full((64, 64, 3), 10 * t % 255, np.uint8))
        ep.states.append(np.full(width, t, np.float32))
        ep.actions.append(np.array([t / n, 0, 0, 0, 1 if t < n // 2 else -1], np.float32))
    write_episode(ep, os.path.join(str(staging), f"s{seed:06d}_{order:02d}"),
                  {"seed": seed, "order": order, "skill": "route_fork", "target": target, "attempt": 0,
                   "scenario": "nominal"})


def _tiny_dataset(tmp_path, n=40, width=gf.STATE_DIM, name="set"):
    from harness_agent.groot_data import merge
    staging = tmp_path / f"{name}_staging"
    for k, target in enumerate(["F1", "F2"]):
        _episode(staging, 1000 + k, target, n, width=width)
    out = tmp_path / name
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
    assert [(e["seed"], e["target"], e["length"]) for e in extra] == [
        (1000, "F1", 40), (1001, "F2", 40), (1002, "F3", 30)]
    stats = json.load(open(out / "meta" / "stats.json"))
    info = json.load(open(out / "meta" / "info.json"))
    for col in ("action", "observation.state", "timestamp"):
        assert stats["__fingerprints__"][col] == _fingerprint(col, info["features"][col])
        assert set(stats[col]) == {"mean", "std", "min", "max", "q01", "q99"}
    assert len(stats["observation.state"]["mean"]) == gf.STATE_DIM and stats["action"]["max"][4] == 1.0
    with pytest.raises(SystemExit, match="already holds a dataset"):
        merge(str(first), str(out))


def test_sets_with_different_state_layouts_are_not_merged(tmp_path):
    from harness_agent.groot_data import check, merge
    old = _tiny_dataset(tmp_path, width=gf.BASE_DIM, name="old")             # the first two sets: 47 values
    new = _tiny_dataset(tmp_path, name="new")
    assert check(str(old))["problems"] == [] and check(str(new))["problems"] == []
    info = json.load(open(old / "meta" / "info.json"))
    assert info["features"]["observation.state"]["shape"] == [47]
    assert list(json.load(open(old / "meta" / "modality.json"))["state"])[-1] == "cable"
    with pytest.raises(SystemExit, match="different state layouts"):
        merge([str(old), str(new)], str(tmp_path / "mixed"))


def test_runner_sends_the_state_keys_the_served_model_knows(tmp_path):
    from harness_agent.groot_client import GrootClient, GrootError
    from harness_agent.groot_replay_server import ReplayPolicy, serve
    from harness_agent.groot_skill import GrootRunner
    old = _tiny_dataset(tmp_path, width=gf.BASE_DIM, name="old")
    port = 5596
    th = threading.Thread(target=serve, args=(ReplayPolicy(str(old)),), kwargs={"host": "127.0.0.1", "port": port},
                          daemon=True)
    th.start()
    client = GrootClient("127.0.0.1", port, timeout_ms=5000)
    try:
        runner = GrootRunner(client)
        base = [k for k, _ in gf.BASE_LAYOUT]
        assert runner.state_keys() == base
        imgs = {k: np.zeros((256, 256, 3), np.uint8) for k in gf.VIDEO_KEYS}
        parts = gf.split_state(np.zeros(gf.STATE_DIM, np.float32))
        assert client.get_action(gf.observation_for_policy(imgs, parts, "route the wire into fork F1", base))
        with pytest.raises(GrootError, match="not trained with"):                # all 9 keys: refused
            client.get_action(gf.observation_for_policy(imgs, parts, "route the wire into fork F1"))
        client.call("kill")
    finally:
        client.close()
        th.join(timeout=5)


def test_ensembling_weights_newer_chunks_more():
    from harness_agent.groot_skill import ensemble_action
    old = np.zeros((16, 5))
    new = np.ones((16, 5))
    assert np.allclose(ensemble_action([(0, old)], 3, 0.1), 0.0)
    mixed = ensemble_action([(0, old), (8, new)], 8, 0.1)                 # ages 8 and 0
    assert np.allclose(mixed, 1.0 / (1.0 + np.exp(-0.8)))
    assert np.allclose(ensemble_action([(0, old), (8, new)], 8, 0.0), 0.5)  # decay 0: plain average
    ramp = np.arange(16.0)[:, None] * np.ones((1, 5))
    assert np.allclose(ensemble_action([(4, ramp)], 10, 0.3), 6.0)        # a chunk's own step 6


def test_a_stalled_try_is_cleared_and_started_over(monkeypatch):
    from harness_agent.groot_skill import GrootRunner

    class FakeSession:
        cfg, route = _Cfg, ["F1"]

        def __init__(self):
            self.cleared = []
            self.obs = dict(fake_obs(yaw=0.0), board_z=np.array([0.0]), time=np.array([0.0]),
                            forks=np.array([[0.5, 0.0, 0.0, 0.0]]), gripper=np.array([0.06]),
                            tcp_pos=np.array([0.2, 0.2, 0.2]))           # on the wire (the 0..1 diagonal)

        def _clear_board(self):
            self.cleared.append(float(self.obs["time"][0]))
            yield np.zeros(5)
            return True

    def run(gripper_at, seconds, restarts):
        session = FakeSession()
        runner = GrootRunner(client=None, restarts=restarts)
        monkeypatch.setattr(runner, "_chunk", lambda *a: np.zeros((16, 5), np.float32))
        gen = runner.route_fork(session, 0)
        next(gen)
        for k in range(int(round(seconds / 0.05))):
            t = (k + 1) * 0.05
            session.obs = dict(session.obs, time=np.array([t]), gripper=np.array([gripper_at(t)]))
            gen.send(session.obs)
        gen.close()
        return runner, session

    runner, session = run(lambda t: 0.06, 30.0, restarts=1)       # never does anything: one restart, no more
    assert runner.last["tries"] == 2 and len(session.cleared) == 1
    assert runner.last["stalls"][0]["why"] == "no progress for 12 s" and abs(session.cleared[0] - 12.0) < 0.1
    runner, session = run(lambda t: 0.005 if t < 1.0 else 0.06, 10.0, restarts=2)   # grasps, then lets go
    assert runner.last["milestones"].get("in_hand") is not None
    assert runner.last["stalls"][0]["why"] == "let go outside the slot" and abs(session.cleared[0] - 3.0) < 0.1
    runner, session = run(lambda t: 0.06, 30.0, restarts=0)       # the default: no restarts
    assert runner.last["tries"] == 1 and not session.cleared


def _stuck_on_the_prongs(t, y=0.004):
    """A wire held over fork F1's slot, resting on the prongs (60 mm up, ``y`` off the slot), the
    tool low beyond the fork; fork at (0.5, 0, board 0), the wire along x."""
    cable = np.linspace([0.3, y, 0.06], [0.62, y, 0.06], 33)
    return dict(fake_obs(yaw=0.0), board_z=np.array([0.0]), time=np.array([t]), cable=cable,
                forks=np.array([[0.5, 0.0, 0.0, 0.0]]), gripper=np.array([0.006]),
                tcp_pos=np.array([0.58, y, 0.05]), target_pos=np.array([0.58, y, 0.05]))


class _SeatingExpert:
    """Stands in for the expert's seat_from_here and _route_fork: after ``steps`` steps the wire
    is in the slot, the gripper open and the tool up."""

    def __init__(self, session, steps=20, works=True):
        self.session, self.steps, self.works = session, steps, works
        self.p, self._grip, self.calls = None, -1.0, []
        self.route_plan = None

    def _route_fork(self, i):
        self.route_plan = {"s_pick": 0.31, "beyond": 0.075, "press_z": 0.014}     # plans first, as the expert
        self.calls.append(("route", i))
        for k in range(self.steps):
            yield np.array([0.1, 0.0, 0.0, 0.0, -1.0])
            if k == 2:
                self.session.holding = True                    # picks the wire up
            if k == self.steps // 2 and self.works:
                self.session.inside = True
        return self.works

    def _hold(self):
        return np.array([0.0, 0.0, 0.0, 0.0, self._grip])

    def seat_from_here(self, i):
        self.calls.append((i, self._grip))
        for k in range(self.steps):
            yield np.array([0.0, 0.1, -0.1, 0.0, self._grip])
            if k == self.steps // 2 and self.works:
                self.session.inside = True
        self._grip = -1.0
        return self.works


def test_the_seat_assist_takes_over_a_wire_stuck_on_the_prongs(monkeypatch):
    from harness_agent.groot_skill import ASSIST_LETGO, GrootRunner

    class FakeSession:
        cfg, route = _Cfg, ["F1"]

        def __init__(self):
            self.inside = False
            self.obs = _stuck_on_the_prongs(0.0)
            self.expert = _SeatingExpert(self)

        def _clear_board(self):
            yield np.zeros(5)
            return True

    def run(seconds, assist, grip=1.0, works=True, sink=None, restarts=0):
        session = FakeSession()
        session.expert.works = works
        runner = GrootRunner(client=None, seat_assist=assist, takeover_sink=sink, restarts=restarts)
        monkeypatch.setattr(runner, "_chunk", lambda *a: np.tile([0.0, 0.0, 0.0, 0.0, grip], (16, 1)))
        monkeypatch.setattr(runner, "cameras", lambda s: type("Cams", (), {"render": lambda self, d: {
            k: np.zeros((8, 8, 3), np.uint8) for k in gf.VIDEO_KEYS}})())
        session.env = type("Env", (), {"cell": type("Cell", (), {"sim": type("Sim", (), {"data": None})})})
        gen = runner.route_fork(session, 0)
        next(gen)
        done = None
        for k in range(int(round(seconds / 0.05))):
            t = (k + 1) * 0.05
            obs = _stuck_on_the_prongs(t)
            if session.inside:                                  # in the slot, released, tool up
                obs.update(cable=np.linspace([0.3, 0.0, 0.03], [0.62, 0.0, 0.03], 33), gripper=np.array([0.05]),
                           tcp_pos=np.array([0.58, 0.0, 0.15]))
            session.obs = obs
            try:
                gen.send(obs)
            except StopIteration as stop:
                done = stop.value
                break
        gen.close()
        return runner, session, done

    runner, session, done = run(10.0, assist=None)                 # off: the policy keeps pressing
    assert done is None and "assists" not in runner.last and not session.expert.calls
    episodes = []
    runner, session, done = run(10.0, assist=1.5, sink=lambda ep, info: episodes.append((ep, info)))
    assert done is True and session.expert.calls == [(0, 1.0)]     # the fingers stay closed on the wire
    a = runner.last["assists"]
    assert len(a) == 1 and a[0]["why"] == "stuck" and a[0]["ok"] and abs(a[0]["t"] - 1.5) < 0.11
    assert runner.last["milestones"]["inside"] > 1.5 and runner.last["furthest"] == "released"
    ep, info = episodes[0]
    assert info["ok"] and len(ep) == 20 + gf.FPS and len(ep.states[0]) == gf.STATE_DIM   # seating + a second's hold
    assert ep.target == "F1" and np.allclose(ep.actions[0], [0.0, 0.1, -0.1, 0.0, 1.0])
    runner, session, done = run(10.0, assist=5.0, grip=ASSIST_LETGO - 0.2)   # starts to let go: taken over at once
    assert done is True and runner.last["assists"][0]["why"] == "letting go" and runner.last["assists"][0]["t"] < 0.2
    runner, session, done = run(20.0, assist=1.5, works=False)     # the expert fails too: the policy carries on
    assert done is None and len(runner.last["assists"]) == 1 and not runner.last["assists"][0]["ok"]
    runner, session, done = run(20.0, assist=1.5, works=False, restarts=1)   # ... or, with restarts, starts over
    assert runner.last["stalls"][0]["why"] == "seat assist failed"


def test_route_takeovers_and_a_hovering_wire(monkeypatch):
    """With route_assist the expert redoes the route when the policy stalls before the slot; a wire
    hanging over the slot without coming down goes to the seat assist."""
    from harness_agent.groot_skill import ROUTE_NO_GRASP_S, GrootRunner

    def hovering(t, y=0.004):          # held and lined up over F1, the tool 10 cm up: never "low"
        return dict(_stuck_on_the_prongs(t, y), cable=np.linspace([0.3, y, 0.04], [0.62, y, 0.105], 33),
                    tcp_pos=np.array([0.58, y, 0.10]))

    def empty_handed(t):               # the gripper open above the board, nowhere near the wire
        return dict(_stuck_on_the_prongs(t), gripper=np.array([0.05]), tcp_pos=np.array([0.40, 0.10, 0.12]))

    def run(obs_at, seconds=20.0, **kw):
        session = type("S", (), {"cfg": _Cfg, "route": ["F1"], "inside": False})()
        session.obs = obs_at(0.0)
        session.expert = _SeatingExpert(session)
        session.env = type("Env", (), {"cell": type("Cell", (), {"sim": type("Sim", (), {"data": None})})})
        episodes = []
        runner = GrootRunner(client=None, takeover_sink=lambda ep, info: episodes.append((ep, info)), **kw)
        monkeypatch.setattr(runner, "_chunk", lambda *a: np.tile([0.0, 0.0, 0.0, 0.0, 1.0], (16, 1)))
        monkeypatch.setattr(runner, "cameras", lambda s: type("Cams", (), {"render": lambda self, d: {
            k: np.zeros((8, 8, 3), np.uint8) for k in gf.VIDEO_KEYS}})())
        gen = runner.route_fork(session, 0)
        next(gen)
        done = None
        for k in range(int(round(seconds / 0.05))):
            obs = obs_at((k + 1) * 0.05)
            if getattr(session, "holding", False):
                obs["gripper"] = np.array([0.006])
            if session.inside:
                obs.update(cable=np.linspace([0.3, 0.0, 0.03], [0.62, 0.0, 0.03], 33), gripper=np.array([0.05]),
                           tcp_pos=np.array([0.58, 0.0, 0.15]))
            session.obs = obs
            try:
                gen.send(obs)
            except StopIteration as stop:
                done = stop.value
                break
        gen.close()
        return runner, session, done, episodes

    runner, session, done, eps = run(empty_handed, route_assist=True)
    a = runner.last["assists"]
    assert done is True and a[0]["kind"] == "route" and a[0]["why"] == "no grasp" and a[0]["ok"]
    assert abs(a[0]["t"] - ROUTE_NO_GRASP_S) < 0.11 and session.expert.calls == [("route", 0)]
    ep, info = eps[0]
    assert info["kind"] == "route" and len(ep) == 20 + gf.FPS
    assert np.allclose(ep.raw["plan"][0], [0.31, 0.075, 0.014])    # the states use the expert's new plan
    runner, session, done, eps = run(empty_handed, seat_assist=1.5)  # route takeovers are off by default
    assert done is None and "assists" not in runner.last
    runner, session, done, eps = run(hovering, seat_assist=1.5)
    a = runner.last["assists"]
    assert done is True and (a[0]["kind"], a[0]["why"]) == ("seat", "hovering") and abs(a[0]["t"] - 6.0) < 0.11


def test_the_expert_seats_a_wire_handed_over_off_the_slot():
    """The expert's own route to F1, stopped as the descent starts; the tool is moved 1 cm sideways
    and down, so the wire comes down beside the fork; seat_from_here lifts it clear, lines it up
    again and seats it."""
    from harness_agent.session import CellSession
    from harness_agent.spec import HarnessSpec
    from harness_core.perception import cable_crossing_in_fork
    spec_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "specs")
    s = CellSession(HarnessSpec.from_yaml(os.path.join(spec_dir, "demo_3fork.yaml")), seed=1003, randomize=True)
    try:
        s.status()
        ex, cfg = s.expert, s.cfg
        bz = float(s.obs["board_z"][0])
        seen = {}

        def handover():
            gen = ex._route_fork(0)
            a = next(gen)
            while ex.phase != "route_descend":
                obs = yield a
                a = gen.send(obs)
            gen.close()
            _, _, _, n = ex._route_frame(0)
            goal = ex._obs["tcp_pos"].copy()
            goal[:2] += 0.010 * n
            goal[2] = bz + 0.03
            yaw = float(ex._obs["tcp_yaw"][0])
            for _ in range(80):
                if np.linalg.norm(ex._obs["target_pos"] - goal) < 0.001:
                    break
                yield ex._action_toward(ex._obs, goal, yaw, speed=0.08)
            seen["before"] = cable_crossing_in_fork(ex._obs["cable"], ex._obs["forks"][0], cfg.fork, bz)
            return (yield from ex.seat_from_here(0))

        run = s._drive(handover(), budget=60.0)
        chk = cable_crossing_in_fork(s.obs["cable"], s.obs["forks"][0], cfg.fork, bz)
        assert abs(seen["before"]["y"]) > cfg.fork.slot_width / 2 and not seen["before"]["inside"]
        assert run["value"] is True and chk["inside"]
        assert any("lifting it clear" in m for m in run["messages"])
    finally:
        s.close()


def test_the_full_system_retries_a_failed_groot_call_with_the_expert():
    """--fallback: a GR00T call that fails (here one that never moves, with a 1 s limit) is followed
    by the expert's retry on the cleared board, and the row and summary report the system."""
    from harness_agent.groot_eval import run_trial, summarize
    from harness_agent.spec import HarnessSpec

    class StuckRunner:
        name, max_seconds, restarts, seat_assist, route_assist = "GR00T (stuck)", 1.0, 0, None, False

        def __init__(self):
            self.last = {}
            self.client = type("Client", (), {"calls": 0, "seconds": 0.0})()

        def wants(self, skill, target="", attempt=0):
            return True

        def route_fork(self, session, i):
            self.last = {"fork": session.route[i], "calls": 0, "grasped": False, "furthest": None, "milestones": {}}
            while True:
                yield session.expert._hold()

        def close(self):
            pass

    spec_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "specs")
    row = run_trial(HarnessSpec.from_yaml(os.path.join(spec_dir, "demo_3fork.yaml")), 1003, "F1", StuckRunner(),
                    fallback=True)
    assert not row["ok"] and row["outcome"] == "timeout"
    assert row["fallback"]["ok"] and row["system_ok"] and row["system_truth_routed"]
    text = summarize([row], "t")
    assert "| all | 1 | 0/1 (0%)" in text and "Full system" in text and "1/1 wires routed" in text
    assert "the expert's retry 1 of the 1 it got" in text


def test_takeovers_are_written_as_episodes_merge_keeps_apart(tmp_path):
    from harness_agent.groot_data import EpisodeBuffer, episodes_in, merge
    from harness_agent.groot_eval import summarize, takeover_writer
    sink = takeover_writer(str(tmp_path / "eval" / "takeovers"), 8003, "F2")
    ep = EpisodeBuffer("route_fork", "F2", lambda o: np.zeros(gf.STATE_DIM, np.float32))
    obs = dict(fake_obs(), time=np.array([0.5]), board_z=np.array([0.0]))
    for _ in range(5):
        ep.add(obs, {k: np.zeros((64, 64, 3), np.uint8) for k in gf.VIDEO_KEYS}, np.ones(5))
    sink(ep, {"t": 12.4, "why": "stuck", "ok": True})
    kind, found, _ = episodes_in(str(tmp_path / "eval" / "takeovers"))
    assert kind == "staging" and len(found) == 1
    m = found[0]["meta"]
    assert (m["seed"], m["target"], m["scenario"], m["takeover_why"], m["length"]) == (8003, "F2", "takeover",
                                                                                       "stuck", 5)
    _episode(tmp_path / "rec" / "staging", 8003, "F2", n=5)        # a recorded demo on the same board
    res = merge([str(tmp_path / "rec"), str(tmp_path / "eval" / "takeovers")], str(tmp_path / "set"), size=64)
    assert res["episodes"] == 2 and res["duplicates"] == 0
    rows = [{"seed": 0, "fork": "F1", "ok": True, "outcome": "routed", "seconds": 14.0, "max_force_N": 9.0,
             "truth_routed": True, "assists": []},
            {"seed": 0, "fork": "F2", "ok": True, "outcome": "routed", "seconds": 19.0, "max_force_N": 9.0,
             "truth_routed": True, "assists": [{"t": 12.0, "why": "stuck", "ok": True}]},
            {"seed": 1, "fork": "F1", "ok": False, "outcome": "timeout", "seconds": 40.0, "max_force_N": 9.0,
             "truth_routed": False, "assists": [{"t": 13.0, "why": "letting go", "ok": False}]}]
    text = summarize(rows, "t")
    assert "2 trials handed the wire to the expert's seating" in text and "it seated 1 of them" in text
    assert "Routed without the expert's help: 1/3" in text and "Route takeovers" not in text
    rows[0]["assists"] = [{"t": 9.0, "why": "no grasp", "ok": True, "kind": "route"}]
    text = summarize(rows, "t")
    assert "Route takeovers (collecting data): 1 trials handed the wire to the expert to redo the route " \
           "(no grasp 1); it routed 1 of them." in text and "Routed without the expert's help: 0/3" in text


def test_bad_descents_stay_in_their_ranges():
    from harness_agent.groot_data import BAD_DESCENT, bad_descent
    rng = np.random.default_rng(0)
    bad = [bad_descent(rng) for _ in range(400)]
    lat = np.array([b["lateral"] for b in bad]) * 1000.0
    lo, hi = BAD_DESCENT["lateral_mm"]
    assert np.all((np.abs(lat) >= lo) & (np.abs(lat) <= hi)) and (lat > 0).mean() > 0.4 and (lat < 0).mean() > 0.4
    assert all(BAD_DESCENT["height_mm"][0] <= 1000 * b["height"] <= BAD_DESCENT["height_mm"][1] for b in bad)
    assert all(abs(b["yaw"]) <= np.radians(BAD_DESCENT["yaw_deg"]) + 1e-9 for b in bad)


@pytest.mark.skipif(not SLOW, reason="records a build with seat recoveries (~5 min); set HARNESS_SLOW_TESTS=1")
def test_seat_recoveries_are_recorded_from_the_hand_over(tmp_path):
    from harness_agent.groot_data import episodes_in, main as record_main
    out = tmp_path / "rec"
    assert record_main(["record", "--out", str(out), "--seeds", "1000", "--workers", "1", "--popped", "0",
                        "--size", "64", "--seat-recoveries", "1.0", "--hold-after", "1.0"]) == 0
    _, found, _ = episodes_in(str(out))
    kinds = [it["meta"].get("kind") for it in found]
    assert len(found) >= 2 and set(kinds) == {"seat_recovery"}
    assert all(it["meta"]["length"] < 250 for it in found)          # from the hand-over, not the whole route


def test_an_interrupted_evaluation_keeps_its_finished_trials(tmp_path, monkeypatch):
    from harness_agent import groot_eval as ge
    out = tmp_path / "eval"
    out.mkdir()
    old = [{"seed": 0, "fork": "F1", "ok": True, "outcome": "routed", "seconds": 14.0, "max_force_N": 9.0,
            "truth_routed": True},
           {"seed": 0, "fork": "F2", "error": "GrootError: no answer"}]            # crashed: run again
    (out / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in old))
    ran = []

    def fake_trial(task):
        ran.append(task[:2])
        return {"seed": task[0], "fork": task[1], "ok": False, "outcome": "timeout", "seconds": 40.0,
                "max_force_N": 5.0, "truth_routed": False}

    monkeypatch.setattr(ge, "_trial", fake_trial)
    assert ge.main(["--expert", "--forks", "F1,F2", "--seeds", "0-1", "--out", str(out)]) == 0
    assert ran == [(0, "F2"), (1, "F1"), (1, "F2")]
    rows = [json.loads(line) for line in open(out / "results.jsonl")]
    assert len(rows) == 4 and sum(r["ok"] for r in rows) == 1 and (out / "summary.md").exists()


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
def test_modality_keys_are_read_from_groots_reply():
    sc = _groot_server_client()
    MsgSerializer, ModalityConfig = sc.MsgSerializer, sc.ModalityConfig
    from harness_agent.groot_client import modality_keys_from, unpack
    keys = ["tcp", "command", "gripper", "wrench", "goal", "cable", "route", "wire", "slot"]
    reply = MsgSerializer.to_bytes({"state": ModalityConfig(delta_indices=[0], modality_keys=keys),
                                    "video": ModalityConfig(delta_indices=[0], modality_keys=["scene", "wrist"])})
    assert modality_keys_from(unpack(reply)) == {"state": keys, "video": ["scene", "wrist"]}


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


def _groot_server_client():
    """GR00T's server_client module from the checkout in GROOT_REPO, without importing torch."""
    import importlib.util
    import sys
    import types
    repo = os.environ["GROOT_REPO"]
    if repo not in sys.path:
        sys.path.insert(0, repo)
    if "gr00t.policy.server_client" not in sys.modules:
        pkg = types.ModuleType("gr00t.policy")
        pkg.__path__ = [os.path.join(repo, "gr00t", "policy")]
        sys.modules.setdefault("gr00t.policy", pkg)
        for name in ("policy", "server_client"):
            spec = importlib.util.spec_from_file_location(f"gr00t.policy.{name}",
                                                          os.path.join(repo, "gr00t", "policy", f"{name}.py"))
            mod = importlib.util.module_from_spec(spec)
            sys.modules[f"gr00t.policy.{name}"] = mod
            spec.loader.exec_module(mod)
    return sys.modules["gr00t.policy.server_client"]


@pytest.mark.skipif(not os.environ.get("GROOT_REPO"), reason="set GROOT_REPO to an Isaac-GR00T checkout")
def test_payloads_survive_groots_own_serializer():
    from harness_agent.groot_client import pack, unpack
    ser = _groot_server_client().MsgSerializer
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
