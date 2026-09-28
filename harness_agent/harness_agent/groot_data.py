"""Record the force-guided expert's skills as a GR00T N1.7 fine-tuning set (LeRobot v2 format).

    python -m harness_agent.groot_data record --out data/harness_route --builds 400 --workers 12
    python -m harness_agent.groot_data check data/harness_route          # counts, lengths, preview

Every build is a full scripted build of a randomised board (layout, wire stiffness, friction,
slack and initial wire shape all vary with the seed); a share of the builds has the wire
pulled out of F2 after it was routed, so re-routes from a disturbed wire are in the set too.
Each successful skill call becomes one episode:

    route_fork F1/F2/F3   ->  "route the wire into fork F2"
    insert_connector      ->  "insert the connector into its holder"   (with --skills)

with two camera views, the state vector and the 5-D actions defined in groot_features.py.
Failed calls are dropped. Seeds start at 1000 by default, so the benchmark seeds (0-99)
stay unseen for evaluation.

Rendering needs an OpenGL backend: MUJOCO_GL=egl on a GPU machine, osmesa elsewhere.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from . import groot_features as gf

CHUNK_SIZE = 1000
DEFAULT_SPEC = "demo_3fork.yaml"


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
    def __init__(self, skill: str, target: str, goal_fn):
        self.skill, self.target, self.goal_fn = skill, target, goal_fn
        self.text = gf.instruction(skill, target)
        self.frames: Dict[str, List[np.ndarray]] = {k: [] for k in gf.VIDEO_KEYS}
        self.states: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []

    def __len__(self) -> int:
        return len(self.actions)


def make_recording_session(spec, seed: int, randomize: bool, skills: Sequence[str], sink, size: int):
    """A CellSession whose route_fork / insert_connector calls are recorded as episodes.

    ``sink(buffer, result, session)`` receives every finished episode (successful or not)."""
    from .session import CellSession

    class DemoSession(CellSession):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._ep: Optional[EpisodeBuffer] = None
            self._cams = gf.Cameras(self.env.cell.sim.model, size) if self.env is not None else None
            self.step_hook = self._on_step

        def _on_step(self, action: np.ndarray) -> None:
            ep = self._ep
            if ep is None:
                return
            obs = self.obs
            imgs = self._cams.render(self.env.cell.sim.data)
            for k in gf.VIDEO_KEYS:
                ep.frames[k].append(imgs[k])
            ep.states.append(gf.state_vector(obs, ep.goal_fn(obs)))
            ep.actions.append(np.clip(np.asarray(action, dtype=np.float32).reshape(5), -1.0, 1.0))

        def _record(self, skill: str, target: str, goal_fn, run):
            self._ep = EpisodeBuffer(skill, target, goal_fn)
            try:
                res = run()
            finally:
                ep, self._ep = self._ep, None
            if len(ep):
                sink(ep, res, self)
            return res

        def route_fork(self, fork_id: str, attempt: int = 0, pick_offset_mm: Optional[float] = None):
            i = self._fork_index(fork_id)
            if "route_fork" not in skills or i is None or self.env is None:
                return super().route_fork(fork_id, attempt, pick_offset_mm)
            cfg = self.cfg
            return self._record("route_fork", fork_id,
                                lambda obs: gf.goal_vector(obs, "route_fork", i, cfg),
                                lambda: super(DemoSession, self).route_fork(fork_id, attempt, pick_offset_mm))

        def insert_connector(self):
            if "insert_connector" not in skills or self.env is None:
                return super().insert_connector()
            cfg = self.cfg
            return self._record("insert_connector", self.connector_id,
                                lambda obs: gf.goal_vector(obs, "insert_connector", -1, cfg),
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


def record_builds(job: Dict[str, Any]) -> Dict[str, Any]:
    """Worker: run scripted builds for the given seeds, write successful episodes to staging."""
    from .agent import ScriptedPlanner
    from .disturbances import make_scenario
    from .spec import HarnessSpec

    spec = HarnessSpec.from_yaml(job["spec"])
    staging, skills = job["staging"], tuple(job["skills"])
    kept = dropped = 0
    t0 = time.perf_counter()
    for seed in job["seeds"]:
        rng = np.random.default_rng(seed + 7919)
        scenario_name = "popped_wire" if rng.random() < job["popped"] else "nominal"
        counter = {"k": 0}

        def sink(ep: EpisodeBuffer, res, session, seed=seed, scenario_name=scenario_name, counter=counter):
            nonlocal kept, dropped
            counter["k"] += 1
            if not getattr(res, "ok", False):
                dropped += 1
                return
            name = f"s{seed:06d}_{counter['k']:02d}"
            args = getattr(res, "args", {}) or {}
            write_episode(ep, os.path.join(staging, name), {
                "seed": seed, "order": counter["k"], "skill": ep.skill, "target": ep.target,
                "attempt": args.get("attempt", 0), "scenario": scenario_name, "outcome": res.outcome,
                "max_force": round(float(res.max_force), 2),
                "t_start": round(float(res.sim_time_start), 2)})
            kept += 1

        session = make_recording_session(spec, seed, True, skills, sink, job["size"])
        try:
            if not session.feasible:
                continue
            ScriptedPlanner(session, log=lambda *_: None).run(scenario=make_scenario(scenario_name, session.route))
        finally:
            session.close()
    return {"kept": kept, "dropped": dropped, "seeds": len(job["seeds"]),
            "wall_s": round(time.perf_counter() - t0, 1)}


# ------------------------------------------------------------------- merge
def modality_json() -> Dict[str, Any]:
    st = gf.layout_slices(gf.STATE_LAYOUT)
    ac = gf.layout_slices(gf.ACTION_LAYOUT)
    return {
        "state": {k: {"start": a, "end": b} for k, (a, b) in st.items()},
        "action": {k: {"start": a, "end": b} for k, (a, b) in ac.items()},
        "video": {k: {"original_key": f"observation.images.{k}"} for k in gf.VIDEO_KEYS},
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }


def info_json(total_episodes: int, total_frames: int, total_tasks: int, size: int, fps: int = gf.FPS) -> Dict[str, Any]:
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
            "observation.state": {"dtype": "float32", "shape": [gf.STATE_DIM], "names": gf.STATE_NAMES},
            **{f"observation.images.{k}": video(k) for k in gf.VIDEO_KEYS},
            "timestamp": scalar("float32"), "frame_index": scalar("int64"), "episode_index": scalar("int64"),
            "index": scalar("int64"), "task_index": scalar("int64"),
        },
    }


def merge(staging: str, out: str, size: int, move: bool = True) -> Dict[str, Any]:
    """Turn staged episodes into one LeRobot v2 dataset with GR00T's meta files."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    names = sorted(d for d in os.listdir(staging) if os.path.exists(os.path.join(staging, d, "meta.json")))
    metas = []
    for d in names:
        with open(os.path.join(staging, d, "meta.json")) as f:
            metas.append(json.load(f))
    tasks: List[str] = []
    for m in metas:
        if m["task"] not in tasks:
            tasks.append(m["task"])
    tasks.sort()
    os.makedirs(os.path.join(out, "meta"), exist_ok=True)
    index = 0
    episodes, extra = [], []
    for ep_idx, (d, m) in enumerate(zip(names, metas)):
        chunk = ep_idx // CHUNK_SIZE
        src = os.path.join(staging, d)
        table = pq.read_table(os.path.join(src, "data.parquet"))
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
            (shutil.move if move else shutil.copy2)(os.path.join(src, f"{k}.mp4"),
                                                     os.path.join(vdir, f"episode_{ep_idx:06d}.mp4"))
        episodes.append({"episode_index": ep_idx, "tasks": [m["task"]], "length": n})
        extra.append({"episode_index": ep_idx, **{k: v for k, v in m.items() if k != "task"}, "task": m["task"]})
        index += n
    meta = os.path.join(out, "meta")
    with open(os.path.join(meta, "episodes.jsonl"), "w") as f:
        f.writelines(json.dumps(e) + "\n" for e in episodes)
    with open(os.path.join(meta, "tasks.jsonl"), "w") as f:
        f.writelines(json.dumps({"task_index": i, "task": t}) + "\n" for i, t in enumerate(tasks))
    with open(os.path.join(meta, "modality.json"), "w") as f:
        json.dump(modality_json(), f, indent=2)
    with open(os.path.join(meta, "info.json"), "w") as f:
        json.dump(info_json(len(episodes), index, len(tasks), size), f, indent=2)
    with open(os.path.join(meta, "harness_episodes.jsonl"), "w") as f:
        f.writelines(json.dumps(e) + "\n" for e in extra)
    if move:
        shutil.rmtree(staging, ignore_errors=True)
    return {"episodes": len(episodes), "frames": index, "tasks": tasks}


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
        if any(len(s) != gf.STATE_DIM for s in t["observation.state"][:3]):
            problems.append(f"episode {i}: state width is not {gf.STATE_DIM}")
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
    r = sub.add_parser("record", help="run scripted builds and record skill episodes")
    r.add_argument("--out", required=True)
    r.add_argument("--builds", type=int, default=100)
    r.add_argument("--seed-start", type=int, default=1000)
    r.add_argument("--seeds", help="explicit seeds, e.g. 1000-1099 (overrides --builds/--seed-start)")
    r.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    r.add_argument("--skills", default="route_fork", help="comma list: route_fork,insert_connector")
    r.add_argument("--popped", type=float, default=0.25, help="share of builds where F2 loses the wire")
    r.add_argument("--spec", default=DEFAULT_SPEC)
    r.add_argument("--size", type=int, default=gf.IMAGE_SIZE)
    c = sub.add_parser("check", help="validate a recorded set and print a summary")
    c.add_argument("dataset")
    c.add_argument("--preview", help="write a contact sheet PNG here")
    args = ap.parse_args(argv)

    if args.cmd == "check":
        res = check(args.dataset, args.preview)
        print(json.dumps(res, indent=2))
        return 1 if res["problems"] else 0

    seeds = _parse_seeds(args.seeds) if args.seeds else list(range(args.seed_start, args.seed_start + args.builds))
    staging = os.path.join(args.out, "staging")
    os.makedirs(staging, exist_ok=True)
    skills = [s.strip() for s in args.skills.split(",") if s.strip()]
    workers = max(1, min(args.workers, len(seeds)))
    jobs = [{"seeds": seeds[k::workers], "staging": staging, "skills": skills, "popped": args.popped,
             "spec": _spec_path(args.spec), "size": args.size} for k in range(workers)]
    t0 = time.perf_counter()
    print(f"recording {len(seeds)} builds on {workers} workers (skills: {', '.join(skills)}) -> {args.out}")
    if workers == 1:
        results = [record_builds(jobs[0])]
    else:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(workers) as pool:
            results = []
            for res in pool.imap_unordered(record_builds, jobs):
                results.append(res)
                print(f"  worker done: {res}", flush=True)
    summary = merge(staging, args.out, args.size)
    kept = sum(r["kept"] for r in results)
    dropped = sum(r["dropped"] for r in results)
    print(f"{summary['episodes']} episodes ({summary['frames']} frames), {dropped} failed skill calls dropped, "
          f"{time.perf_counter() - t0:.0f} s; tasks: {summary['tasks']}")
    if kept != summary["episodes"]:
        print(f"warning: workers reported {kept} episodes but {summary['episodes']} were merged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
