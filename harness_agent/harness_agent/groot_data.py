"""Record the force-guided expert's skills as a GR00T N1.7 fine-tuning set (LeRobot v2 format).

    python -m harness_agent.groot_data bench                       # how fast this machine records
    python -m harness_agent.groot_data record --out data/harness_route --builds 330 --workers 7
    python -m harness_agent.groot_data check data/harness_route --preview data/route_preview.png
    python -m harness_agent.groot_data merge data/harness_route data/harness_route2 --out data/route_all

Every build is a full scripted build of a randomised board (layout, wire stiffness, friction,
slack and initial wire shape all vary with the seed); a share of the builds has the wire
pulled out of F2 after it was routed, so re-routes from a disturbed wire are in the set too.
Each successful skill call becomes one episode:

    route_fork F1/F2/F3   ->  "route the wire into fork F2"
    insert_connector      ->  "insert the connector into its holder"   (with --skills)

with two camera views, the state vector and the 5-D actions defined in groot_features.py.
Failed calls are dropped. Seeds start at 1000 by default, so the benchmark seeds (0-99)
stay unseen for evaluation.

With --noise the executed motion is pushed around while the expert's clean action is
recorded, so the demos show how to get back on track (DART); --hold-after adds a short,
recorded stand-still after each successful call, so the policy learns to stop when done.

Recording can be stopped and resumed. Finished builds are marked in <out>/staging, so the
same command run again (after a preempted VM, say) skips them. Ctrl-c stops the recording
and packages the episodes finished so far. ``merge`` combines recorded sets into one.

Rendering needs an OpenGL backend: MUJOCO_GL=egl on a GPU machine, osmesa elsewhere.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from . import groot_features as gf

CHUNK_SIZE = 1000
DEFAULT_SPEC = "demo_3fork.yaml"
DONE_DIR = "_done"          # in staging: one marker per finished build, for resuming


def single_threaded_math() -> None:
    """One BLAS/OpenMP thread per worker process (inherited by workers started after this):
    the workers already fill the cores, and oversubscribed math libraries slow all of them."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")


def _spec_path(name: str) -> str:
    if os.path.exists(name):
        return name
    here = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    for d in (os.path.join(here, "specs"),):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    try:
        from ament_index_python.packages import get_package_share_directory
        p = os.path.join(get_package_share_directory("harness_agent"), "specs", name)
        if os.path.exists(p):
            return p
    except Exception:
        pass
    raise SystemExit(f"spec {name!r} not found")


# ---------------------------------------------------------------- recording
class EpisodeBuffer:
    def __init__(self, skill: str, target: str, state_fn):
        self.skill, self.target, self.state_fn = skill, target, state_fn       # state_fn(obs) -> vector
        self.text = gf.instruction(skill, target)
        self.frames: Dict[str, List[np.ndarray]] = {k: [] for k in gf.VIDEO_KEYS}
        self.states: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []

    def __len__(self) -> int:
        return len(self.actions)


class RecoveryNoise:
    """Pushes on the executed motion for DART-style demos. The recorder keeps the expert's
    clean action as the label while the robot executes the pushed one, and the closed-loop
    expert shows how to get back on track. Two kinds of push, both scaled by ``scale``:

    * a smooth jitter (an Ornstein-Uhlenbeck process per axis), and
    * now and then, while the gripper moves in free space, a kick of 2-4 steps that throws it
      1.5-3 cm off its path.

    Pushes depend on the expert's phase: none during the guarded touch-down on the wire (a
    push reads as a touch), while the gripper closes, seats the wire or lets go, and no
    vertical push while it lowers the wire."""

    SIGMA = np.array([0.20, 0.20, 0.10, 0.15])        # jitter std of dx, dy, dz, dyaw (action units)
    GAIN = {"pick_approach": (1.0, 1.0, 1.0, 1.0),     # free space above the wire
            "route_lift": (0.5, 0.5, 0.3, 0.5),        # carrying the taut wire: gentler
            "route_transit": (0.5, 0.5, 0.3, 0.5),
            "route_descend": (0.4, 0.4, 0.0, 0.4)}     # lowering the wire beyond the fork
    KICK_PHASES = ("pick_approach",)
    KICK_RATE = 0.5                                  # kicks per second of a kick phase, at scale 1

    def __init__(self, scale: float, rng: np.random.Generator, phase=lambda: "", dt: float = 1.0 / gf.FPS,
                 tau: float = 0.5):
        self.scale, self.rng, self.phase, self.dt = float(scale), rng, phase, dt
        self.a = math.exp(-dt / tau)
        self.x = np.zeros(4)
        self.kick_left, self.kick = 0, np.zeros(4)
        self.kicks = 0

    def reset(self) -> None:
        self.x[:] = 0.0
        self.kick_left = 0

    def __call__(self, action: np.ndarray) -> np.ndarray:
        ph = self.phase()
        gain = np.asarray(self.GAIN.get(ph, (0.0, 0.0, 0.0, 0.0)))
        self.x = self.a * self.x + math.sqrt(1.0 - self.a ** 2) * self.rng.standard_normal(4) * self.SIGMA
        out = np.asarray(action, dtype=float).copy()
        out[:4] += self.scale * gain * self.x
        if (self.kick_left == 0 and ph in self.KICK_PHASES
                and self.rng.random() < self.KICK_RATE * self.dt * self.scale):
            ang = self.rng.uniform(0.0, 2.0 * math.pi)
            self.kick = 0.8 * np.array([math.cos(ang), math.sin(ang), self.rng.uniform(-0.1, 0.3), 0.0])
            self.kick_left = int(self.rng.integers(2, 5))
            self.kicks += 1
        if self.kick_left > 0:
            if ph in self.KICK_PHASES:
                out[:4] = self.kick                  # the push replaces the expert's motion
            self.kick_left -= 1
        out[:4] = np.clip(out[:4], -1.0, 1.0)
        return out


def make_recording_session(spec, seed: int, randomize: bool, skills: Sequence[str], sink, size: int,
                           noise: float = 0.0, hold_after: float = 0.0):
    """A CellSession whose route_fork / insert_connector calls are recorded as episodes.

    ``sink(buffer, result, session)`` receives every finished episode (successful or not).
    ``noise`` > 0 pushes the executed motion while recording (the scale of the pushes);
    ``hold_after`` > 0 records that many seconds of standing still after a successful call."""
    from .session import CellSession

    class DemoSession(CellSession):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._ep: Optional[EpisodeBuffer] = None
            self._cams = gf.Cameras(self.env.cell.sim.model, size) if self.env is not None else None
            self.step_hook = self._on_step
            self._noise = (RecoveryNoise(noise, np.random.default_rng(seed + 104729),
                                         phase=lambda: getattr(self.expert, "phase", ""))
                           if noise > 0 else None)

        def _on_step(self, action: np.ndarray) -> None:
            ep = self._ep
            if ep is None:
                return
            obs = self.obs
            imgs = self._cams.render(self.env.cell.sim.data)
            for k in gf.VIDEO_KEYS:
                ep.frames[k].append(imgs[k])
            ep.states.append(ep.state_fn(obs))
            ep.actions.append(np.clip(np.asarray(action, dtype=np.float32).reshape(5), -1.0, 1.0))

        def _record(self, skill: str, target: str, state_fn, run):
            self._ep = EpisodeBuffer(skill, target, state_fn)
            if self._noise is not None:
                self._noise.reset()
                self.action_noise = self._noise
            try:
                res = run()
                self.action_noise = None
                if hold_after > 0 and getattr(res, "ok", False) and len(self._ep):
                    self._record_hold(hold_after)
            finally:
                self.action_noise = None
                ep, self._ep = self._ep, None
            if len(ep):
                sink(ep, res, self)
            return res

        def _record_hold(self, seconds: float) -> None:
            """Stand still with the gripper as it is, recorded: the policy learns to stop when done."""
            for _ in range(max(1, int(round(seconds * gf.FPS)))):
                a = self.expert._hold()
                self._on_step(a)
                self.env.step(a)
                self.expert._last_obs = self.obs
                self._grab_frame()

        def route_fork(self, fork_id: str, attempt: int = 0, pick_offset_mm: Optional[float] = None):
            i = self._fork_index(fork_id)
            if "route_fork" not in skills or i is None or self.env is None:
                return super().route_fork(fork_id, attempt, pick_offset_mm)
            cfg = self.cfg
            return self._record("route_fork", fork_id,
                                lambda obs: gf.join_state(gf.state_parts(obs, "route_fork", i, cfg)),
                                lambda: super(DemoSession, self).route_fork(fork_id, attempt, pick_offset_mm))

        def insert_connector(self):
            if "insert_connector" not in skills or self.env is None:
                return super().insert_connector()
            cfg = self.cfg
            return self._record("insert_connector", self.connector_id,
                                lambda obs: gf.join_state(gf.state_parts(obs, "insert_connector", -1, cfg)),
                                lambda: super(DemoSession, self).insert_connector())

        def close(self):
            if self._cams is not None:
                self._cams.close()
            super().close()

    return DemoSession(spec, seed=seed, randomize=randomize)


def write_episode(ep: EpisodeBuffer, folder: str, meta: Dict[str, Any], fps: int = gf.FPS) -> None:
    """One episode in staging form: data.parquet (indices filled in at merge), one mp4 per view."""
    import imageio
    import pyarrow as pa
    import pyarrow.parquet as pq

    os.makedirs(folder, exist_ok=True)
    n = len(ep)
    table = pa.table({
        "observation.state": pa.array([s.tolist() for s in ep.states], type=pa.list_(pa.float32())),
        "action": pa.array([a.tolist() for a in ep.actions], type=pa.list_(pa.float32())),
        "timestamp": pa.array(np.arange(n, dtype=np.float32) / fps, type=pa.float32()),
        "frame_index": pa.array(np.arange(n, dtype=np.int64)),
        "episode_index": pa.array(np.zeros(n, dtype=np.int64)),
        "index": pa.array(np.arange(n, dtype=np.int64)),
        "task_index": pa.array(np.zeros(n, dtype=np.int64)),
    })
    pq.write_table(table, os.path.join(folder, "data.parquet"))
    for k in gf.VIDEO_KEYS:
        w = imageio.get_writer(os.path.join(folder, f"{k}.mp4"), fps=fps, codec="libx264", quality=8,
                               pixelformat="yuv420p", macro_block_size=16,
                               ffmpeg_params=["-g", str(fps), "-bf", "0"])
        try:
            for f in ep.frames[k]:
                w.append_data(f)
        finally:
            w.close()
    with open(os.path.join(folder, "meta.json"), "w") as f:
        json.dump({**meta, "task": ep.text, "length": n}, f)


_REC: Dict[str, Any] = {}


def _init_recorder(job: Dict[str, Any], pool: bool = False) -> None:
    """Per worker process: the job settings and the parsed spec."""
    if pool:
        import signal
        signal.signal(signal.SIGINT, signal.SIG_IGN)     # Ctrl-c is the parent's business
    from .spec import HarnessSpec
    _REC.clear()
    _REC.update(job)
    _REC["spec_obj"] = HarnessSpec.from_yaml(job["spec"])


def done_seeds(staging: str) -> set:
    d = os.path.join(staging, DONE_DIR)
    if not os.path.isdir(d):
        return set()
    return {int(n[:-5]) for n in os.listdir(d) if n.endswith(".json") and n[:-5].isdigit()}


def scripted_build(session, scenario_name: str, routing_only: bool = False) -> None:
    """The scripted build the demos come from. With ``routing_only`` it stops where the
    connector stage would begin (a fifth to a third of a build's simulated time): nothing
    after that point is recorded, and every call before it is exactly the full build's."""
    from .agent import ScriptedPlanner, ScriptedPolicy
    from .disturbances import make_scenario
    from .tools import ToolBox

    scenario = make_scenario(scenario_name, session.route)
    if not routing_only:
        ScriptedPlanner(session, log=lambda *_: None).run(scenario=scenario)
        return
    box = ToolBox(session, scenario=scenario)                 # as ScriptedPlanner.run, up to the connector
    policy = ScriptedPolicy(session.route, session.connector_id)
    last = box.call("get_status", {})
    for _ in range(60):
        if box.finished is not None:
            break
        name, args = policy.next(last, session.feasible)
        if name != "route_fork":
            break
        last = box.call(name, args)


def record_seed(seed: int) -> Dict[str, Any]:
    """Worker: one scripted build; its successful skill calls go to staging as episodes."""
    staging, skills = _REC["staging"], tuple(_REC["skills"])
    t0 = time.perf_counter()
    for old in glob.glob(os.path.join(staging, f"s{seed:06d}_*")):     # left by an interrupted run
        shutil.rmtree(old, ignore_errors=True)
    rng = np.random.default_rng(seed + 7919)
    scenario_name = "popped_wire" if rng.random() < _REC["popped"] else "nominal"
    noise = round(float(np.random.default_rng(seed + 31337).uniform(0.0, 1.0)) * _REC.get("noise", 0.0), 3)
    hold = float(_REC.get("hold_after", 0.0))
    res: Dict[str, Any] = {"seed": seed, "scenario": scenario_name, "kept": 0, "dropped": 0}
    order = 0

    def sink(ep: EpisodeBuffer, out, session) -> None:
        nonlocal order
        order += 1
        if not getattr(out, "ok", False):
            res["dropped"] += 1
            return
        args = getattr(out, "args", {}) or {}
        write_episode(ep, os.path.join(staging, f"s{seed:06d}_{order:02d}"), {
            "seed": seed, "order": order, "skill": ep.skill, "target": ep.target,
            "attempt": args.get("attempt", 0), "scenario": scenario_name, "outcome": out.outcome,
            "max_force": round(float(out.max_force), 2),
            "t_start": round(float(out.sim_time_start), 2),
            **({"noise": noise} if noise else {}), **({"hold_s": hold} if hold else {})})
        res["kept"] += 1

    try:
        session = make_recording_session(_REC["spec_obj"], seed, True, skills, sink, _REC["size"],
                                         noise=noise, hold_after=hold)
        try:
            if session.feasible:
                scripted_build(session, scenario_name, routing_only=set(skills) <= {"route_fork"})
        finally:
            session.close()
    except Exception as exc:                             # a broken build must not end the recording
        res["error"] = f"{type(exc).__name__}: {exc}"
    res["seconds"] = round(time.perf_counter() - t0, 1)
    if "error" not in res:                               # a crashed build is tried again on resume
        os.makedirs(os.path.join(staging, DONE_DIR), exist_ok=True)
        with open(os.path.join(staging, DONE_DIR, f"{seed}.json"), "w") as f:
            json.dump(res, f)
    return res


def _duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds >= 3600:
        return f"{seconds // 3600} h {seconds % 3600 // 60:02d} min"
    return f"{seconds // 60} min" if seconds >= 120 else f"{seconds} s"


def record(seeds: Sequence[int], out: str, workers: int, skills: Sequence[str], popped: float, spec: str,
           size: int, noise: float = 0.0, hold_after: float = 0.0) -> Dict[str, Any]:
    """Record the builds that are not done yet, then package everything staged into ``out``."""
    if os.path.exists(os.path.join(out, "meta", "info.json")):
        raise SystemExit(f"{out} already holds a packaged dataset. Record into a new folder and combine the two:\n"
                         f"  python -m harness_agent.groot_data merge {out} NEW_FOLDER --out COMBINED_FOLDER")
    staging = os.path.join(out, "staging")
    os.makedirs(os.path.join(staging, DONE_DIR), exist_ok=True)
    done = done_seeds(staging)
    todo = [s for s in seeds if s not in done]
    if len(todo) < len(seeds):
        print(f"resuming: {len(seeds) - len(todo)} of {len(seeds)} builds were already recorded in {staging}")
    workers = max(1, min(workers, len(todo)))
    job = {"staging": staging, "skills": list(skills), "popped": popped, "spec": spec, "size": size,
           "noise": noise, "hold_after": hold_after}
    print(f"recording {len(todo)} builds on {workers} worker{'s' if workers > 1 else ''} "
          f"(skills: {', '.join(skills)}) -> {out}\nCtrl-c stops and packages the episodes recorded so far.",
          flush=True)
    t0 = time.perf_counter()
    tally = {"builds": 0, "kept": 0, "dropped": 0, "errors": 0}

    def progress(res: Dict[str, Any]) -> None:
        tally["builds"] += 1
        tally["kept"] += res["kept"]
        tally["dropped"] += res["dropped"]
        tally["errors"] += "error" in res
        n = tally["builds"]
        left = (time.perf_counter() - t0) / n * (len(todo) - n)
        note = f" ERROR {res['error']}" if "error" in res else ""
        print(f"[{n}/{len(todo)}] seed {res['seed']}: {res['kept']} episodes in {res['seconds']:.0f} s{note} | "
              f"{tally['kept']} episodes so far, about {_duration(left)} left", flush=True)

    stopped = False
    try:
        if workers == 1:
            _init_recorder(job)
            for seed in todo:
                progress(record_seed(seed))
        elif todo:
            import multiprocessing as mp
            single_threaded_math()
            with mp.get_context("spawn").Pool(workers, initializer=_init_recorder, initargs=(job, True)) as pool:
                for res in pool.imap_unordered(record_seed, todo):
                    progress(res)
    except KeyboardInterrupt:
        stopped = True
        print("\nstopped: packaging the episodes recorded so far (about a minute) ...", flush=True)
    summary = merge(staging, out, size)
    summary.update({"dropped": tally["dropped"], "errors": tally["errors"], "stopped": stopped,
                    "seconds": round(time.perf_counter() - t0)})
    return summary


# ------------------------------------------------------------------- merge
def modality_json(state_dim: int = gf.STATE_DIM) -> Dict[str, Any]:
    st = gf.layout_slices(gf.layout_for_width(state_dim))
    ac = gf.layout_slices(gf.ACTION_LAYOUT)
    return {
        "state": {k: {"start": a, "end": b} for k, (a, b) in st.items()},
        "action": {k: {"start": a, "end": b} for k, (a, b) in ac.items()},
        "video": {k: {"original_key": f"observation.images.{k}"} for k in gf.VIDEO_KEYS},
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }


def info_json(total_episodes: int, total_frames: int, total_tasks: int, size: int, fps: int = gf.FPS,
              state_dim: int = gf.STATE_DIM) -> Dict[str, Any]:
    def video(k):
        return {"dtype": "video", "shape": [size, size, 3], "names": ["height", "width", "channels"],
                "info": {"video.height": size, "video.width": size, "video.codec": "h264",
                         "video.pix_fmt": "yuv420p", "video.is_depth_map": False, "video.fps": fps,
                         "video.channels": 3, "has_audio": False}}
    scalar = lambda dt: {"dtype": dt, "shape": [1], "names": None}   # noqa: E731
    return {
        "codebase_version": "v2.1",
        "robot_type": "harness_cell_ur5e_ft_gripper",
        "total_episodes": total_episodes, "total_frames": total_frames, "total_tasks": total_tasks,
        "total_videos": total_episodes * len(gf.VIDEO_KEYS),
        "total_chunks": (total_episodes + CHUNK_SIZE - 1) // CHUNK_SIZE, "chunks_size": CHUNK_SIZE,
        "fps": fps, "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "action": {"dtype": "float32", "shape": [gf.ACTION_DIM], "names": gf.ACTION_NAMES},
            "observation.state": {"dtype": "float32", "shape": [state_dim],
                                  "names": gf.STATE_NAMES_BY_WIDTH[state_dim]},
            **{f"observation.images.{k}": video(k) for k in gf.VIDEO_KEYS},
            "timestamp": scalar("float32"), "frame_index": scalar("int64"), "episode_index": scalar("int64"),
            "index": scalar("int64"), "task_index": scalar("int64"),
        },
    }


def _fingerprint(feature: str, meta: Dict[str, Any]) -> str:
    """Same cache key as gr00t.data.stats, so GR00T reuses our statistics instead of recomputing."""
    import hashlib
    payload = json.dumps({"feature": feature, "dtype": meta.get("dtype"), "shape": meta.get("shape")},
                         sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_stats(dataset: str) -> Dict[str, Any]:
    """meta/stats.json as GR00T's generate_stats writes it (mean, std, min, max, q01, q99 per float column).
    Fine-tuning would compute it itself, but GR00T's replay server needs it to exist."""
    import glob as _glob

    import pyarrow.parquet as pq
    with open(os.path.join(dataset, "meta", "info.json")) as f:
        features = json.load(f)["features"]
    cols = [k for k, v in features.items() if "float" in v["dtype"]]
    data: Dict[str, List[np.ndarray]] = {c: [] for c in cols}
    for path in sorted(_glob.glob(os.path.join(dataset, "data", "*", "*.parquet"))):
        t = pq.read_table(path, columns=cols).to_pydict()
        for c in cols:
            data[c].append(np.asarray(t[c], dtype=np.float32).reshape(len(t[c]), -1))
    stats: Dict[str, Any] = {}
    for c in cols:
        a = np.vstack(data[c]) if data[c] else np.zeros((1, 1), np.float32)
        stats[c] = {"mean": a.mean(0).tolist(), "std": a.std(0).tolist(), "min": a.min(0).tolist(),
                    "max": a.max(0).tolist(), "q01": np.quantile(a, 0.01, axis=0).tolist(),
                    "q99": np.quantile(a, 0.99, axis=0).tolist()}
    stats["__fingerprints__"] = {c: _fingerprint(c, features[c]) for c in cols}
    with open(os.path.join(dataset, "meta", "stats.json"), "w") as f:
        json.dump(stats, f, indent=4)
    return stats


def episodes_in(source: str) -> Tuple[str, List[Dict[str, Any]], Optional[int]]:
    """The episodes of a recorded folder: a packaged dataset, a recording's staging folder, or the
    recording folder holding it. Returns (kind, [{meta, parquet, videos}], image size or None)."""
    if os.path.exists(os.path.join(source, "meta", "info.json")):
        meta_dir = os.path.join(source, "meta")
        with open(os.path.join(meta_dir, "info.json")) as f:
            info = json.load(f)
        with open(os.path.join(meta_dir, "episodes.jsonl")) as f:
            episodes = [json.loads(line) for line in f]
        extra: Dict[int, Dict[str, Any]] = {}
        if os.path.exists(os.path.join(meta_dir, "harness_episodes.jsonl")):
            with open(os.path.join(meta_dir, "harness_episodes.jsonl")) as f:
                for line in f:
                    e = json.loads(line)
                    extra[int(e["episode_index"])] = e
        found = []
        for e in episodes:
            i = int(e["episode_index"])
            chunk = i // info["chunks_size"]
            meta = {k: v for k, v in extra.get(i, {}).items() if k != "episode_index"}
            meta["task"] = e["tasks"][0]
            meta["length"] = e["length"]
            found.append({"meta": meta,
                          "parquet": os.path.join(source, info["data_path"].format(episode_chunk=chunk,
                                                                                    episode_index=i)),
                          "videos": {k: os.path.join(source, info["video_path"].format(
                              episode_chunk=chunk, video_key=f"observation.images.{k}", episode_index=i))
                              for k in gf.VIDEO_KEYS}})
        size = info["features"][f"observation.images.{gf.VIDEO_KEYS[0]}"]["shape"][0]
        return "dataset", found, int(size)
    staging = os.path.join(source, "staging") if os.path.isdir(os.path.join(source, "staging")) else source
    if not os.path.isdir(staging):
        raise SystemExit(f"{source}: no dataset and no recorded episodes here")
    found = []
    for d in sorted(os.listdir(staging)):
        folder = os.path.join(staging, d)
        if not os.path.exists(os.path.join(folder, "meta.json")):      # written last: unfinished otherwise
            continue
        with open(os.path.join(folder, "meta.json")) as f:
            meta = json.load(f)
        found.append({"meta": meta, "parquet": os.path.join(folder, "data.parquet"),
                      "videos": {k: os.path.join(folder, f"{k}.mp4") for k in gf.VIDEO_KEYS}})
    return "staging", found, None


def merge(sources: Union[str, Sequence[str]], out: str, size: Optional[int] = None,
          keep_staging: bool = False) -> Dict[str, Any]:
    """Package recorded episodes into one LeRobot v2 dataset with GR00T's meta files.

    Sources are packaged datasets and staging folders; files are copied, so an interrupted
    merge can simply be run again. Staging folders are removed once the dataset is complete,
    unless ``keep_staging``. Episodes recorded twice (same seed, call, skill, target, scenario
    and length) are kept once."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    sources = [sources] if isinstance(sources, str) else list(sources)
    if os.path.exists(os.path.join(out, "meta", "info.json")):
        raise SystemExit(f"{out} already holds a dataset; merge into a new folder")
    items: List[Dict[str, Any]] = []
    staged: List[str] = []
    seen: set = set()
    duplicates = 0
    widths: Dict[str, int] = {}                          # state width per source: sets must not mix layouts
    for src in sources:
        kind, found, src_size = episodes_in(src)
        if found:
            first = pq.read_table(found[0]["parquet"], columns=["observation.state"]).to_pydict()["observation.state"]
            widths[src] = len(first[0]) if first else gf.STATE_DIM
        if src_size is not None:
            if size is None:
                size = src_size
            elif src_size != size:
                raise SystemExit(f"{src} has {src_size} px images, the merged set {size} px")
        for it in found:
            m = it["meta"]
            key = (m.get("seed"), m.get("order"), m.get("skill"), m.get("target"), m.get("scenario"),
                   m.get("length"))
            if m.get("seed") is not None and key in seen:
                duplicates += 1
                continue
            seen.add(key)
            items.append(it)
        if kind == "staging" and not keep_staging:
            staged.append(os.path.join(src, "staging") if os.path.isdir(os.path.join(src, "staging")) else src)
    size = size or gf.IMAGE_SIZE
    state_dims = sorted(set(widths.values()))
    if len(state_dims) > 1:
        raise SystemExit(f"these sets have different state layouts ({', '.join(map(str, state_dims))} values: "
                         f"{widths}); record them again with the same code to combine them")
    state_dim = state_dims[0] if state_dims else gf.STATE_DIM
    tasks = sorted({it["meta"]["task"] for it in items})
    os.makedirs(os.path.join(out, "meta"), exist_ok=True)
    index = 0
    episodes, extra = [], []
    for ep_idx, it in enumerate(items):
        m = it["meta"]
        chunk = ep_idx // CHUNK_SIZE
        table = pq.read_table(it["parquet"])
        n = table.num_rows
        table = table.set_column(table.schema.get_field_index("episode_index"), "episode_index",
                                 pa.array(np.full(n, ep_idx, dtype=np.int64)))
        table = table.set_column(table.schema.get_field_index("index"), "index",
                                 pa.array(np.arange(index, index + n, dtype=np.int64)))
        table = table.set_column(table.schema.get_field_index("task_index"), "task_index",
                                 pa.array(np.full(n, tasks.index(m["task"]), dtype=np.int64)))
        ddir = os.path.join(out, "data", f"chunk-{chunk:03d}")
        os.makedirs(ddir, exist_ok=True)
        pq.write_table(table, os.path.join(ddir, f"episode_{ep_idx:06d}.parquet"))
        for k in gf.VIDEO_KEYS:
            vdir = os.path.join(out, "videos", f"chunk-{chunk:03d}", f"observation.images.{k}")
            os.makedirs(vdir, exist_ok=True)
            shutil.copy2(it["videos"][k], os.path.join(vdir, f"episode_{ep_idx:06d}.mp4"))
        episodes.append({"episode_index": ep_idx, "tasks": [m["task"]], "length": n})
        extra.append({"episode_index": ep_idx, **{k: v for k, v in m.items() if k not in ("task", "length")},
                      "length": n, "task": m["task"]})
        index += n
    meta = os.path.join(out, "meta")
    with open(os.path.join(meta, "episodes.jsonl"), "w") as f:
        f.writelines(json.dumps(e) + "\n" for e in episodes)
    with open(os.path.join(meta, "tasks.jsonl"), "w") as f:
        f.writelines(json.dumps({"task_index": i, "task": t}) + "\n" for i, t in enumerate(tasks))
    with open(os.path.join(meta, "modality.json"), "w") as f:
        json.dump(modality_json(state_dim), f, indent=2)
    with open(os.path.join(meta, "info.json"), "w") as f:
        json.dump(info_json(len(episodes), index, len(tasks), size, state_dim=state_dim), f, indent=2)
    with open(os.path.join(meta, "harness_episodes.jsonl"), "w") as f:
        f.writelines(json.dumps(e) + "\n" for e in extra)
    write_stats(out)
    for st in staged:
        shutil.rmtree(st, ignore_errors=True)
    return {"episodes": len(episodes), "frames": index, "tasks": tasks, "duplicates": duplicates}


# ------------------------------------------------------------------- check
def check(dataset: str, preview: Optional[str] = None) -> Dict[str, Any]:
    """Consistency checks a fine-tuning run would trip over, plus a summary."""
    import imageio
    import pyarrow.parquet as pq

    meta = os.path.join(dataset, "meta")
    with open(os.path.join(meta, "info.json")) as f:
        info = json.load(f)
    with open(os.path.join(meta, "episodes.jsonl")) as f:
        episodes = [json.loads(line) for line in f]
    problems: List[str] = []
    lengths, by_task = [], {}
    act_min, act_max = np.full(gf.ACTION_DIM, np.inf), np.full(gf.ACTION_DIM, -np.inf)
    for e in episodes:
        i, chunk = e["episode_index"], e["episode_index"] // info["chunks_size"]
        path = os.path.join(dataset, info["data_path"].format(episode_chunk=chunk, episode_index=i))
        t = pq.read_table(path).to_pydict()
        n = len(t["action"])
        if n != e["length"]:
            problems.append(f"episode {i}: parquet has {n} rows, episodes.jsonl says {e['length']}")
        width = info["features"]["observation.state"]["shape"][0]
        if any(len(s) != width for s in t["observation.state"][:3]):
            problems.append(f"episode {i}: state width is not {width}")
        a = np.asarray(t["action"], dtype=np.float32)
        act_min, act_max = np.minimum(act_min, a.min(0)), np.maximum(act_max, a.max(0))
        for k in gf.VIDEO_KEYS:
            vp = os.path.join(dataset, info["video_path"].format(episode_chunk=chunk, video_key=f"observation.images.{k}",
                                                                 episode_index=i))
            if not os.path.exists(vp):
                problems.append(f"episode {i}: missing {vp}")
                continue
            r = imageio.get_reader(vp)
            nf = r.count_frames()
            r.close()
            if nf != n:
                problems.append(f"episode {i}: {k} video has {nf} frames for {n} rows")
        lengths.append(n)
        by_task[e["tasks"][0]] = by_task.get(e["tasks"][0], 0) + 1
    if preview and episodes:
        _preview(dataset, info, episodes, preview)
    return {"episodes": len(episodes), "frames": int(sum(lengths)),
            "hours": round(sum(lengths) / info["fps"] / 3600, 2),
            "length_s": {"min": round(min(lengths) / info["fps"], 1), "median": round(float(np.median(lengths)) / info["fps"], 1),
                         "max": round(max(lengths) / info["fps"], 1)} if lengths else {},
            "per_task": by_task, "action_range": [act_min.round(2).tolist(), act_max.round(2).tolist()],
            "problems": problems}


def _preview(dataset: str, info: Dict[str, Any], episodes: List[Dict[str, Any]], out: str) -> None:
    """A contact sheet: first, middle and last frame of both views for a few episodes."""
    import imageio
    from PIL import Image, ImageDraw

    pick = episodes[:: max(1, len(episodes) // 4)][:4]
    s = 192
    sheet = Image.new("RGB", (6 * s, len(pick) * (s + 22)), (245, 245, 245))
    d = ImageDraw.Draw(sheet)
    for row, e in enumerate(pick):
        i, chunk = e["episode_index"], e["episode_index"] // info["chunks_size"]
        col = 0
        for k in gf.VIDEO_KEYS:
            vp = os.path.join(dataset, info["video_path"].format(episode_chunk=chunk, video_key=f"observation.images.{k}",
                                                                 episode_index=i))
            frames = [np.asarray(f) for f in imageio.get_reader(vp)]
            for j in (0, len(frames) // 2, len(frames) - 1):
                sheet.paste(Image.fromarray(frames[j]).resize((s, s)), (col * s, row * (s + 22) + 22))
                col += 1
        d.text((6, row * (s + 22) + 4), f"episode {i}: {e['tasks'][0]}  ({e['length']} steps)", fill=(20, 20, 20))
    sheet.save(out)


# ------------------------------------------------------------------- bench
def _gl_strings() -> Dict[str, str]:
    """Who renders: the GPU's OpenGL driver, or a software fallback (needs a current GL context)."""
    try:
        from OpenGL import GL
        return {k: GL.glGetString(e).decode() for k, e in
                (("vendor", GL.GL_VENDOR), ("renderer", GL.GL_RENDERER), ("version", GL.GL_VERSION))}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def bench(spec: str = DEFAULT_SPEC, seed: int = 1000, size: int = gf.IMAGE_SIZE, frames: int = 40) -> Dict[str, Any]:
    """How fast this machine records, per worker: the expert routes F1 without cameras (physics
    and control), then the two policy views are rendered ``frames`` times."""
    import mujoco

    from .session import CellSession
    from .spec import HarnessSpec

    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        cpus = os.cpu_count() or 1
    out: Dict[str, Any] = {"cpus": cpus, "MUJOCO_GL": os.environ.get("MUJOCO_GL", "(not set)"),
                           "mujoco": mujoco.__version__}
    session = CellSession(HarnessSpec.from_yaml(_spec_path(spec)), seed=seed, randomize=True)
    try:
        t = time.perf_counter()
        session.status()
        routed = session.route_fork(session.route[0]).ok
        wall = time.perf_counter() - t
        sim = session.sim_time
        cams = gf.Cameras(session.env.cell.sim.model, size)
        try:
            cams.render(session.env.cell.sim.data)                  # warm up; leaves the context current
            out["gl"] = _gl_strings()
            t = time.perf_counter()
            for _ in range(frames):
                cams.render(session.env.cell.sim.data)
            render_ms = 1000 * (time.perf_counter() - t) / frames
        finally:
            cams.close()
    finally:
        session.close()
    step_ms = 1000 * wall / (sim * gf.FPS)                          # physics + control per 20 Hz step
    out["physics"] = {"robot_s": round(sim, 1), "wall_s": round(wall, 1), "x_realtime": round(sim / wall, 2),
                      "ms_per_step": round(step_ms, 1), "routed": bool(routed)}
    out["render_ms_per_step"] = round(render_ms, 1)                  # both views
    out["recording_x_realtime"] = round(1000 / gf.FPS / (step_ms + render_ms), 2)
    who = (out["gl"].get("renderer", "") + " " + out["gl"].get("vendor", "")).lower()
    out["software_rendering"] = any(w in who for w in ("llvmpipe", "softpipe", "swrast", "software"))
    return out


def _bench_report(b: Dict[str, Any]) -> str:
    gl = b["gl"].get("renderer") or b["gl"].get("error", "?")
    lines = [f"physics + control  {b['physics']['ms_per_step']:.1f} ms per 20 Hz step "
             f"({b['physics']['x_realtime']:.1f}x real time; F1 {'routed' if b['physics']['routed'] else 'NOT routed'})",
             f"rendering          {b['render_ms_per_step']:.1f} ms per step for both views ({gl})",
             f"recording          {b['recording_x_realtime']:.2f}x real time per worker, so a 15 s routing "
             f"episode takes about {15 / max(b['recording_x_realtime'], 1e-3):.0f} s; {b['cpus']} CPUs here"]
    if b["software_rendering"]:
        lines.append("warning: OpenGL runs in software on the CPU. On a GPU machine set MUJOCO_GL=egl and check "
                     "that the NVIDIA EGL library is installed (libnvidia-gl / libEGL_nvidia).")
    return "\n".join(lines)


# --------------------------------------------------------------------- CLI
def _parse_seeds(text: str) -> List[int]:
    out: List[int] = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record", help="run scripted builds and record skill episodes (resumes where it stopped)")
    r.add_argument("--out", required=True)
    r.add_argument("--builds", type=int, default=100)
    r.add_argument("--seed-start", type=int, default=1000)
    r.add_argument("--seeds", help="explicit seeds, e.g. 1000-1099 (overrides --builds/--seed-start)")
    r.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    r.add_argument("--skills", default="route_fork", help="comma list: route_fork,insert_connector")
    r.add_argument("--popped", type=float, default=0.25, help="share of builds where F2 loses the wire")
    r.add_argument("--spec", default=DEFAULT_SPEC)
    r.add_argument("--size", type=int, default=gf.IMAGE_SIZE)
    r.add_argument("--noise", type=float, default=0.0,
                   help="recovery demos: push the executed motion while the expert's clean action is "
                        "recorded; each build gets a random scale up to this (try 1.0)")
    r.add_argument("--hold-after", type=float, default=0.0,
                   help="record this many seconds of standing still after each successful call (try 1.0)")
    c = sub.add_parser("check", help="validate a recorded set and print a summary")
    c.add_argument("dataset")
    c.add_argument("--preview", help="write a contact sheet PNG here")
    m = sub.add_parser("merge", help="package recorded folders (datasets, or a stopped recording) into one dataset")
    m.add_argument("sources", nargs="+")
    m.add_argument("--out", required=True)
    m.add_argument("--size", type=int, help="image size (default: from the sources)")
    m.add_argument("--keep-staging", action="store_true", help="keep the staged episodes after packaging")
    st = sub.add_parser("stats", help="(re)write meta/stats.json the way GR00T computes it")
    st.add_argument("dataset")
    b = sub.add_parser("bench", help="how fast this machine records (renderer, render time, physics speed)")
    b.add_argument("--spec", default=DEFAULT_SPEC)
    b.add_argument("--seed", type=int, default=1000)
    b.add_argument("--size", type=int, default=gf.IMAGE_SIZE)
    b.add_argument("--json", action="store_true", help="print the raw numbers")
    args = ap.parse_args(argv)

    if args.cmd == "check":
        res = check(args.dataset, args.preview)
        print(json.dumps(res, indent=2))
        return 1 if res["problems"] else 0
    if args.cmd == "merge":
        res = merge(args.sources, args.out, args.size, keep_staging=args.keep_staging)
        note = f", {res['duplicates']} duplicates left out" if res["duplicates"] else ""
        print(f"{res['episodes']} episodes ({res['frames']} frames){note} -> {args.out}; tasks: {res['tasks']}")
        return 0
    if args.cmd == "stats":
        write_stats(args.dataset)
        print(f"wrote {os.path.join(args.dataset, 'meta', 'stats.json')}")
        return 0
    if args.cmd == "bench":
        res = bench(args.spec, args.seed, args.size)
        print(json.dumps(res, indent=2) if args.json else _bench_report(res))
        return 0

    seeds = _parse_seeds(args.seeds) if args.seeds else list(range(args.seed_start, args.seed_start + args.builds))
    skills = [s.strip() for s in args.skills.split(",") if s.strip()]
    res = record(seeds, args.out, args.workers, skills, args.popped, _spec_path(args.spec), args.size,
                 noise=args.noise, hold_after=args.hold_after)
    print(f"{res['episodes']} episodes ({res['frames']} frames) in {args.out}, {res['dropped']} failed skill calls "
          f"dropped, {res['seconds']} s; tasks: {res['tasks']}")
    if res["errors"]:
        print(f"warning: {res['errors']} builds crashed; their errors are in the progress lines above")
    return 0


if __name__ == "__main__":
    sys.exit(main())
