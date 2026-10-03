"""Closed-loop evaluation of a fine-tuned GR00T routing policy in the simulated cell.

    # on the GPU machine, with GR00T's policy server running on port 5556:
    python -m harness_agent.groot_eval --forks F1,F2,F3 --seeds 0-19 --workers 4 --out eval/route_v1
    python -m harness_agent.groot_eval --forks F1 --seeds 0-9 --expert --out eval/expert   # same trials, expert

Each trial resets a randomised board with the given seed (seeds 0-99 are never used for
training data), lets the expert route the forks before the target fork, then hands the
target fork to the policy. Success means the policy grasped the wire, left it in the slot,
released and lifted clear within the time limit; the simulator's ground truth is logged next
to it. Writes results.jsonl and summary.md (success rate with a 95% Wilson interval per fork).

Trials run in --workers processes that share the one policy server. The server answers one
request at a time, so with several workers the round trip per action chunk includes waiting
in line; the simulator, not the GPU, is usually the bottleneck.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .groot_client import DEFAULT_PORT
from .groot_data import _parse_seeds, _spec_path, single_threaded_math


def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def run_trial(spec, seed: int, fork: str, runner, video_dir: Optional[str] = None,
              replay_episode: Optional[int] = None) -> Dict[str, Any]:
    from .session import CellSession

    session = CellSession(spec, seed=seed, randomize=True, render=video_dir is not None, frame_every=0.1)
    try:
        session.status()          # settle and look first, as a build does (the recorded demos start this way)
        route = session.route
        k = route.index(fork)
        for before in route[:k]:                         # the expert sets up the earlier forks
            r = session.route_fork(before)
            if not r.ok:
                return {"seed": seed, "fork": fork, "skipped": f"expert failed on {before} ({r.outcome})"}
        t0 = session.sim_time
        calls0, secs0 = 0, 0.0
        if runner is not None:
            if replay_episode is not None:
                runner.client.reset({"episode_index": int(replay_episode)})
            session.skill_runners["route_fork"] = runner
            calls0, secs0 = runner.client.calls, runner.client.seconds
        res = session.route_fork(fork)
        truth = session.truth()
        out = {"seed": seed, "fork": fork, "ok": bool(res.ok), "outcome": res.outcome,
               "controller": res.controller, "seconds": round(session.sim_time - t0, 1),
               "max_force_N": round(float(res.max_force), 1),
               "truth_routed": bool(truth["forks_routed"].get(fork)),
               "perceived_in_slot": bool(res.perceived["forks"][fork]["wire_in_slot"]),
               "chunks": (runner.client.calls - calls0) if runner is not None else None,
               "inference_s": round(runner.client.seconds - secs0, 2) if runner is not None else None,
               "messages": res.messages[:3]}
        if video_dir is not None and session.frames:
            import imageio
            os.makedirs(video_dir, exist_ok=True)
            path = os.path.join(video_dir, f"{fork}_seed{seed:03d}_{'ok' if res.ok else res.outcome}.mp4")
            imageio.mimwrite(path, session.frames, fps=10, macro_block_size=8)
            out["video"] = os.path.relpath(path, os.path.dirname(video_dir))
        return out
    finally:
        if runner is not None:
            runner.close()                               # this trial's camera renderers
        session.close()


# ------------------------------------------------------------- trial workers
_WORKER: Dict[str, Any] = {}


class _ServerGone(RuntimeError):
    pass


def _init_worker(opts: Dict[str, Any]) -> None:
    """Per process: the spec, and a client + runner of its own (the server takes several clients)."""
    import signal
    if opts.get("pool"):
        signal.signal(signal.SIGINT, signal.SIG_IGN)     # Ctrl-c is the parent's business
    from .spec import HarnessSpec
    _WORKER.clear()
    _WORKER["opts"] = opts
    _WORKER["spec"] = HarnessSpec.from_yaml(opts["spec"])
    _WORKER["runner"] = None
    if not opts["expert"]:
        from .groot_client import GrootClient
        from .groot_skill import GrootRunner
        client = GrootClient(opts["host"], opts["port"], timeout_ms=opts["timeout_ms"])
        _WORKER["runner"] = GrootRunner(client, forks=opts["forks"], attempts=None,
                                        execute_horizon=opts["execute_horizon"], max_seconds=opts["max_seconds"])


def _close_worker() -> None:
    runner = _WORKER.get("runner")
    if runner is not None:
        runner.close()
        runner.client.close()
    _WORKER.clear()


def _trial(task: Tuple[int, str, Optional[int]]) -> Dict[str, Any]:
    seed, fork, episode = task
    opts = _WORKER["opts"]
    t0 = time.perf_counter()
    try:
        row = run_trial(_WORKER["spec"], seed, fork, _WORKER["runner"], opts["video_dir"], replay_episode=episode)
    except Exception as exc:                             # one broken trial must not end the evaluation
        row = {"seed": seed, "fork": fork, "error": f"{type(exc).__name__}: {exc}"}
    row["wall_s"] = round(time.perf_counter() - t0, 1)
    return row


# ------------------------------------------------------------------ report
def summarize(rows: List[Dict[str, Any]], title: str) -> str:
    lines = [f"# {title}", "", "| fork | trials | success | 95% interval | truth routed | median time | median max force |",
             "|---|---|---|---|---|---|---|"]
    forks = sorted({r["fork"] for r in rows if "ok" in r})
    for f in forks + ["all"]:
        rs = [r for r in rows if "ok" in r and (f == "all" or r["fork"] == f)]
        if not rs:
            continue
        n, k = len(rs), sum(r["ok"] for r in rs)
        lo, hi = wilson(k, n)
        tr = sum(r["truth_routed"] for r in rs)
        oks = [r["seconds"] for r in rs if r["ok"]]
        lines.append(f"| {f} | {n} | {k}/{n} ({100 * k / n:.0f}%) | {100 * lo:.0f}-{100 * hi:.0f}% | {tr}/{n} | "
                     f"{(np.median(oks) if oks else float('nan')):.1f} s | "
                     f"{np.median([r['max_force_N'] for r in rs]):.1f} N |")
    skipped = [r for r in rows if "skipped" in r]
    if skipped:
        lines += ["", f"{len(skipped)} trials skipped (the expert failed before the target fork)."]
    errors = [r for r in rows if "error" in r]
    if errors:
        lines += ["", f"{len(errors)} trials crashed, first: seed {errors[0]['seed']} {errors[0]['fork']}: "
                      f"{errors[0]['error']}"]
    fails: Dict[str, int] = {}
    for r in rows:
        if "ok" in r and not r["ok"]:
            fails[r["outcome"]] = fails.get(r["outcome"], 0) + 1
    if fails:
        lines += ["", "Failures: " + ", ".join(f"{k} {v}" for k, v in sorted(fails.items(), key=lambda x: -x[1]))]
    return "\n".join(lines) + "\n"


def latency_line(rows: List[Dict[str, Any]], workers: int) -> str:
    timed = [r for r in rows if r.get("chunks")]
    if not timed:
        return ""
    chunks = sum(r["chunks"] for r in timed)
    secs = sum(r["inference_s"] for r in timed)
    per_trial = [1000 * r["inference_s"] / r["chunks"] for r in timed]
    share = f" ({workers} workers sharing the server, waiting in line included)" if workers > 1 else ""
    return (f"Policy server: {chunks} action chunks, {1000 * secs / chunks:.0f} ms per round trip on average, "
            f"slowest trial {max(per_trial):.0f} ms{share}.\n")


def _state(row: Dict[str, Any]) -> str:
    if "error" in row:
        return f"ERROR {row['error']}"
    if "skipped" in row:
        return row["skipped"]
    return f"{'OK ' if row['ok'] else 'FAIL'} {row['outcome']:<14} {row['seconds']:5.1f} s"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--forks", default="F1,F2,F3")
    ap.add_argument("--seeds", default="0-19")
    ap.add_argument("--spec", default="demo_3fork.yaml")
    ap.add_argument("--workers", type=int, default=1,
                    help="trials run in parallel, all asking the same policy server (try 4 on an 8-vCPU VM)")
    ap.add_argument("--execute-horizon", type=int, default=8)
    ap.add_argument("--max-seconds", type=float, default=40.0)
    ap.add_argument("--timeout", type=float, default=120.0, help="seconds to wait for one answer from the server")
    ap.add_argument("--expert", action="store_true", help="run the expert on the same trials (baseline)")
    ap.add_argument("--video", action="store_true", help="save an overview video of every trial")
    ap.add_argument("--replay-dataset", help="the server replays this recorded set (GR00T's --dataset-path mode or "
                                             "harness_agent.groot_replay_server): each trial selects the episode "
                                             "recorded with the same seed and fork, to test the loop end to end")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    forks = [f.strip() for f in args.forks.split(",") if f.strip()]
    seeds = _parse_seeds(args.seeds)
    os.makedirs(args.out, exist_ok=True)
    if not args.expert:
        from .groot_client import GrootClient
        probe = GrootClient(args.host, args.port, timeout_ms=10000)
        alive = probe.ping()
        probe.close()
        if not alive:
            raise SystemExit(f"no GR00T policy server at {args.host}:{args.port} "
                             f"(start gr00t/eval/run_gr00t_server.py with --port {args.port})")
    episode_of: Dict[tuple, int] = {}
    if args.replay_dataset:
        with open(os.path.join(args.replay_dataset, "meta", "harness_episodes.jsonl")) as f:
            for line in f:
                e = json.loads(line)
                if e.get("skill") == "route_fork" and int(e.get("attempt", 0)) == 0:
                    episode_of.setdefault((int(e["seed"]), e["target"]), int(e["episode_index"]))
    tasks: List[Tuple[int, str, Optional[int]]] = []
    for seed in seeds:
        for fork in forks:
            if args.replay_dataset and (seed, fork) not in episode_of:
                print(f"seed {seed:3d} {fork}: no recorded episode to replay, skipped")
                continue
            tasks.append((seed, fork, episode_of.get((seed, fork)) if args.replay_dataset else None))
    workers = max(1, min(args.workers, len(tasks)))
    if args.replay_dataset and workers > 1:
        print("replay mode runs one trial at a time (the server replays one episode at a time)")
        workers = 1
    opts = {"spec": _spec_path(args.spec), "expert": args.expert, "host": args.host, "port": args.port,
            "timeout_ms": int(1000 * args.timeout), "forks": forks, "execute_horizon": args.execute_horizon,
            "max_seconds": args.max_seconds, "video_dir": os.path.join(args.out, "videos") if args.video else None,
            "pool": workers > 1}
    who = "expert" if args.expert else "GR00T"
    print(f"{len(tasks)} trials with {who} on {workers} worker{'s' if workers > 1 else ''} -> {args.out}", flush=True)

    rows: List[Dict[str, Any]] = []
    t0 = time.perf_counter()
    results_path = os.path.join(args.out, "results.jsonl")
    lost = {"in_a_row": 0}
    with open(results_path, "w") as f:
        def emit(row: Dict[str, Any]) -> None:
            rows.append(row)
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"[{len(rows)}/{len(tasks)}] seed {row['seed']:3d} {row['fork']}: {_state(row)}", flush=True)
            lost["in_a_row"] = lost["in_a_row"] + 1 if "GrootError" in row.get("error", "") else 0
            if lost["in_a_row"] >= 3:
                raise _ServerGone(f"the policy server stopped answering ({row['error']})")

        try:
            if workers == 1:
                _init_worker(opts)
                try:
                    for task in tasks:
                        emit(_trial(task))
                finally:
                    _close_worker()
            else:
                import multiprocessing as mp
                single_threaded_math()
                with mp.get_context("spawn").Pool(workers, initializer=_init_worker, initargs=(opts,)) as pool:
                    for row in pool.imap_unordered(_trial, tasks):
                        emit(row)
        except KeyboardInterrupt:
            print(f"\nstopped after {len(rows)} of {len(tasks)} trials; the summary covers those")
        except _ServerGone as exc:
            print(f"\n{exc}; stopped after {len(rows)} of {len(tasks)} trials")

    order = {f: i for i, f in enumerate(forks)}
    rows.sort(key=lambda r: (r["seed"], order.get(r["fork"], 99)))
    with open(results_path, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    text = summarize(rows, f"Routing with {who}: {len(seeds)} seeds x {', '.join(forks)}")
    lat = latency_line(rows, workers)
    if lat:
        text += "\n" + lat
    text += f"\nWall time {time.perf_counter() - t0:.0f} s for {len(rows)} trials on {workers} worker(s).\n"
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
