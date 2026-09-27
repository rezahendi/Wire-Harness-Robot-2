"""Build one harness in the simulated cell and write a trace, a report and (optionally) a video.

    python -m harness_agent.run_build --spec specs/demo_3fork.yaml --planner nemotron
    python -m harness_agent.run_build --spec specs/demo_4fork.yaml --planner scripted --video
    python -m harness_agent.run_build --spec specs/demo_3fork.yaml --planner expert --seed 7 --randomize

Planners:
    nemotron   Nemotron on Nebius Token Factory decides every step (needs NEBIUS_API_KEY)
    scripted   the same decisions hard-coded, through the same tools
    expert     the original monolithic expert with its built-in recoveries (baseline)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, Optional

import numpy as np


def _jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def run_expert(session) -> Dict[str, Any]:
    """The monolithic expert on the same cell (no tools, no planner)."""
    from harness_core.expert import HarnessExpert
    env = session.env
    expert = HarnessExpert(session.cfg, env.spec_actions)
    t0 = time.perf_counter()
    while not expert.done and session.sim_time < session.max_sim_time:
        env.step(expert.step(env.last_obs_dict))
        session._grab_frame()
    return {"planner": "expert", "success": bool(session.truth()["success"]),
            "claimed_success": not expert.failed, "report": expert.fail_reason or "done",
            "truth": session.truth(), "sim_time": round(session.sim_time, 1),
            "wall_time": round(time.perf_counter() - t0, 1),
            "tool_calls": [], "messages": [{"role": "expert_log", "content": m} for _, m in expert.log],
            "usage": {}, "turns": 0}


def write_report(path: str, spec, result: Dict[str, Any], session) -> None:
    lines = [f"# Build report: {spec.name} rev {spec.revision}", "",
             f"* planner: **{result['planner']}** {result.get('model') or ''}",
             f"* seed {session.seed}, randomised layout: {session.randomize}",
             f"* result (simulator ground truth): **{'SUCCESS' if result['success'] else 'FAILED'}**",
             f"* planner's own verdict: {result.get('claimed_success')}",
             f"* robot time {result['sim_time']} s, wall time {result['wall_time']} s", ""]
    if session.issues:
        lines += ["## Validation", ""] + [f"* {i.severity}: {i.message}" for i in session.issues] + [""]
    if result.get("tool_calls"):
        lines += ["## Steps", "", "| # | tool | arguments | outcome | robot time |", "|---|---|---|---|---|"]
        for k, c in enumerate(result["tool_calls"], 1):
            res = c.get("result", {})
            outcome = res.get("outcome") or ("error: " + res["error"] if "error" in res else
                                             ("done" if "recorded" in res else "ok"))
            args = {a: v for a, v in (c.get("arguments") or {}).items() if a != "report" and v is not None}
            lines.append(f"| {k} | `{c['name']}` | {json.dumps(args) if args else ''} | {outcome} | "
                         f"{c.get('sim_time', '')} s |")
        lines.append("")
    lines += ["## Planner report", "", result.get("report") or "(none)", ""]
    if result.get("usage"):
        u = result["usage"]
        lines += ["## Model usage", "", f"{u.get('calls', 0)} calls, {u.get('prompt_tokens', 0)} prompt + "
                  f"{u.get('completion_tokens', 0)} completion tokens, {u.get('seconds', 0)} s", ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True, help="harness spec YAML")
    ap.add_argument("--planner", default="scripted", choices=("nemotron", "scripted", "expert"))
    ap.add_argument("--model", default=None, help="Token Factory model id (default: picked automatically)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--randomize", action="store_true", help="perturb the layout and wire around the spec")
    ap.add_argument("--max-turns", type=int, default=40)
    ap.add_argument("--video", action="store_true", help="also write video.mp4 (slower)")
    ap.add_argument("--out", default=None, help="output directory (default: runs/<spec>_<planner>_<seed>)")
    args = ap.parse_args(argv)
    if args.video:
        from harness_core.render_util import choose_gl_backend
        choose_gl_backend()

    from .agent import NemotronPlanner, ScriptedPlanner
    from .drawing import render_drawing
    from .session import CellSession
    from .spec import HarnessSpec

    spec = HarnessSpec.from_yaml(args.spec)
    name = os.path.splitext(os.path.basename(args.spec))[0]
    out = args.out or os.path.join("runs", f"{name}_{args.planner}_{args.seed}")
    os.makedirs(out, exist_ok=True)
    render_drawing(spec, os.path.join(out, "drawing.png"))
    session = CellSession(spec, seed=args.seed, randomize=args.randomize, render=args.video,
                          frame_every=0.25)
    print(f"{spec.name} rev {spec.revision}: route {' > '.join(session.route)} > {session.connector_id}, "
          f"planner {args.planner}, seed {args.seed}")
    for i in session.issues:
        print(f"  {i.severity}: {i.message}")

    if args.planner == "expert":
        if not session.feasible:
            print("spec failed validation; the expert cannot run")
            return 1
        result = run_expert(session)
    elif args.planner == "scripted":
        result = ScriptedPlanner(session).run(args.max_turns).as_dict()
    else:
        planner = NemotronPlanner(session, model=args.model)
        print(f"  model: {planner.model}")
        result = planner.run(args.max_turns).as_dict()

    result["events"] = session.events
    result["session"] = session.summary()
    with open(os.path.join(out, "trace.json"), "w", encoding="utf-8") as f:
        json.dump(_jsonable(result), f, indent=1)
    write_report(os.path.join(out, "report.md"), spec, result, session)
    if args.video and session.frames:
        import imageio
        imageio.mimsave(os.path.join(out, "video.mp4"), session.frames, fps=16, macro_block_size=1)
    print(f"\n{'SUCCESS' if result['success'] else 'FAILED'} in {result['sim_time']} s robot time "
          f"({result['wall_time']} s wall) -> {out}")
    if result.get("error"):
        print(f"planner error: {result['error']}")
    session.close()
    return 0 if result["success"] else 2


if __name__ == "__main__":
    sys.exit(main())
