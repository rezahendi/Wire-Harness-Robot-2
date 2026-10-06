"""Build one harness in the simulated cell and write a trace, a report and (optionally) a video.

    python -m harness_agent.run_build --spec specs/demo_3fork.yaml --planner nemotron
    python -m harness_agent.run_build --spec specs/demo_4fork.yaml --planner scripted --video
    python -m harness_agent.run_build --spec specs/demo_3fork.yaml --planner expert --seed 7 --randomize
    python -m harness_agent.run_build --spec specs/demo_3fork.yaml --planner nemotron --vision
    python -m harness_agent.run_build --spec specs/demo_3fork.yaml --planner nemotron --groot   # GR00T routes

Planners:
    nemotron   Nemotron on Nebius Token Factory decides every step (needs NEBIUS_API_KEY)
    scripted   the same decisions hard-coded, through the same tools
    expert     the original monolithic expert with its built-in recoveries (baseline)

--vision adds the camera check: every inspection photographs the fixtures and a vision
model on Token Factory gives its verdict next to perception's (images in inspection/).

--groot hands route_fork to the fine-tuned GR00T N1.7 policy served at HOST:PORT (default
127.0.0.1:5556): the planner's tool call becomes GR00T's instruction, and a failed attempt
is retried by the force-guided expert (--groot-attempts all lets GR00T take retries too).
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
    from .groot_skill import REFUSED
    lines = [f"# Build report: {spec.name} rev {spec.revision}", "",
             f"* planner: **{result['planner']}** {result.get('model') or ''}",
             f"* seed {session.seed}, randomised layout: {session.randomize}",
             f"* result (simulator ground truth): **{'SUCCESS' if result['success'] else 'FAILED'}**",
             f"* planner's own verdict: {result.get('claimed_success')}",
             f"* robot time {result['sim_time']} s, wall time {result['wall_time']} s", ""]
    if session.issues:
        lines += ["## Validation", ""] + [f"* {i.severity}: {i.message}" for i in session.issues] + [""]
    if result.get("tool_calls"):
        lines += ["## Steps", "", "| # | tool | arguments | outcome | executed by | robot time |",
                  "|---|---|---|---|---|---|"]
        for k, c in enumerate(result["tool_calls"], 1):
            res = c.get("result", {})
            outcome = res.get("outcome") or ("error: " + res["error"] if "error" in res else
                                             ("done" if "recorded" in res else "ok"))
            args = {a: v for a, v in (c.get("arguments") or {}).items() if a != "report" and v is not None}
            ran = res.get("skill") and "duration_s" in res and res.get("outcome") not in REFUSED
            who = res.get("executed_by") or ("expert" if ran else "")
            lines.append(f"| {k} | `{c['name']}` | {json.dumps(args) if args else ''} | {outcome} | {who} | "
                         f"{c.get('sim_time', '')} s |")
        lines.append("")
    lines += ["## Planner report", "", result.get("report") or "(none)", ""]
    checks = getattr(session, "visual_checks", None) or []
    if checks:
        agree = sum(c["verdict"]["seated"] == c["truth"] for c in checks)
        by_model: Dict[str, int] = {}
        for c in checks:
            by_model[c["verdict"]["model"]] = by_model.get(c["verdict"]["model"], 0) + 1
        answered = ", ".join(f"`{m}` {n}" for m, n in by_model.items())
        lines += ["## Visual inspection", "",
                  f"{len(checks)} camera checks (answered by {answered}); the vision verdict "
                  f"matched ground truth in {agree}, perception in "
                  f"{sum(c['perceived'] == c['truth'] for c in checks)}.", "",
                  "| # | target | vision (confidence) | perception | truth | evidence |", "|---|---|---|---|---|---|"]

        def word(v):
            return "no answer" if v is None else ("seated" if v else "NOT seated")

        for k, c in enumerate(checks, 1):
            v = c["verdict"]
            img = f"[{c['target']}](inspection/{v['image']})" if v.get("image") else c["target"]
            lines.append(f"| {k} | {img} | {word(v['seated'])} ({v['confidence']:.2f}, "
                         f"{v['model'].split('/')[-1]}) | {word(c['perceived'])} | "
                         f"{word(c['truth'])} | {(v.get('evidence') or v.get('error') or '').replace('|', '/')} |")
        lines.append("")
        last = {}
        for c in checks:
            last[c["target"]] = c
        shots = [c["verdict"]["image"] for c in last.values() if c["verdict"].get("image")]
        lines += [f"![{os.path.splitext(n)[0]}](inspection/{n})" for n in shots] + [""]
    if result.get("usage"):
        u = result["usage"]
        lines += ["## Model usage", "", f"{u.get('calls', 0)} calls, {u.get('prompt_tokens', 0)} prompt + "
                  f"{u.get('completion_tokens', 0)} completion tokens, {u.get('seconds', 0)} s", ""]
        for model, m in (u.get("per_model") or {}).items():
            lines.append(f"* `{model}`: {m['calls']} calls, {m['prompt_tokens']} + {m['completion_tokens']} "
                         f"tokens, {m['seconds']:.1f} s")
        lines.append("")
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
    ap.add_argument("--scenario", default="nominal",
                    help="inject disturbances: nominal, popped_wire, slip_on_insert, both (see disturbances.py)")
    ap.add_argument("--video", action="store_true", help="also write video.mp4 (slower)")
    ap.add_argument("--vision", action="store_true",
                    help="camera check: a Token Factory vision model inspects each fixture too")
    ap.add_argument("--vision-model", default=None,
                    help="vision model id (default: the measured-best one your key can use, Kimi K3)")
    ap.add_argument("--vision-fallback", default=None,
                    help="model that answers when the first gives no verdict (default: MiniCPM-V; 'none')")
    ap.add_argument("--vision-style", default="v3", choices=("v1", "v2", "v3"), help="camera views and question style")
    ap.add_argument("--vision-refs", default="packaged",
                    help="labelled example images: 'packaged' (default), a refs/ folder, or 'none'")
    ap.add_argument("--groot", nargs="?", const="", default=None, metavar="HOST:PORT",
                    help="route forks with the GR00T policy server there (default 127.0.0.1:5556)")
    ap.add_argument("--groot-attempts", default="0",
                    help="route_fork attempts GR00T takes: 0 (retries go to the expert), 0,1 or all")
    ap.add_argument("--groot-restarts", type=int, default=0,
                    help="within one attempt, GR00T starts over after a stalled try up to N times (try 2)")
    ap.add_argument("--groot-seconds", type=float, default=None,
                    help="time limit of one GR00T attempt (default 40 s, 60 s with restarts)")
    ap.add_argument("--groot-ensemble", type=float, default=None, metavar="DECAY",
                    help="ask GR00T for a chunk every 4 steps and average the overlapping chunks (try 0.1)")
    ap.add_argument("--groot-seat-assist", type=float, default=None, metavar="SECONDS",
                    help="hybrid: the expert's seating takes over a wire GR00T holds stuck over the slot this long "
                         "(try 1.5)")
    ap.add_argument("--out", default=None, help="output directory (default: runs/<spec>_<planner>_<seed>)")
    args = ap.parse_args(argv)
    if args.video or args.vision:
        from harness_core.render_util import choose_gl_backend
        choose_gl_backend()

    from .agent import NemotronPlanner, ScriptedPlanner
    from .drawing import render_drawing
    from .session import CellSession
    from .spec import HarnessSpec

    spec = HarnessSpec.from_yaml(args.spec)
    name = os.path.splitext(os.path.basename(args.spec))[0]
    label = args.planner + ("_groot" if args.groot is not None else "")
    out = args.out or os.path.join("runs", f"{name}_{label}_{args.seed}")
    os.makedirs(out, exist_ok=True)
    render_drawing(spec, os.path.join(out, "drawing.png"))
    client, inspector = None, None
    if args.planner == "nemotron" or args.vision:
        from .llm import TokenFactoryClient
        client = TokenFactoryClient()
    if args.vision:
        from .vision import default_inspector
        inspector = default_inspector(client, model=args.vision_model, fallback_model=args.vision_fallback,
                                      style=args.vision_style,
                                      refs=None if args.vision_refs == "none" else args.vision_refs)
    session = CellSession(spec, seed=args.seed, randomize=args.randomize, render=args.video,
                          frame_every=0.25, inspector=inspector,
                          inspection_dir=os.path.join(out, "inspection") if inspector else None,
                          render_size=(960, 1280) if args.video else (480, 640))
    print(f"{spec.name} rev {spec.revision}: route {' > '.join(session.route)} > {session.connector_id}, "
          f"planner {args.planner}, seed {args.seed}")
    for i in session.issues:
        print(f"  {i.severity}: {i.message}")
    if inspector is not None:
        print(f"  camera check: {inspector.describe()}")
    runner = None
    if args.groot is not None and session.feasible:
        from .groot_skill import connect_runner
        runner = connect_runner(args.groot or None, args.groot_attempts, restarts=args.groot_restarts,
                                max_seconds=args.groot_seconds or (60.0 if args.groot_restarts else 40.0),
                                seat_assist=args.groot_seat_assist, ensemble_decay=args.groot_ensemble,
                                execute_horizon=4 if args.groot_ensemble is not None else 8)
        session.skill_runners["route_fork"] = runner
        print(f"  route_fork: {runner.name} on attempts {args.groot_attempts}, the expert otherwise")

    from .disturbances import make_scenario
    scenario = make_scenario(args.scenario, session.route) if session.feasible else None
    if scenario is not None and scenario.name != "nominal":
        print(f"  scenario: {scenario.name} ({scenario.description})")
    if args.planner == "expert":
        if not session.feasible:
            print("spec failed validation; the expert cannot run")
            return 1
        result = run_expert(session)
        if inspector is not None:                        # the expert does not inspect: do it after
            session.retreat()
            result["final_inspection"] = session.inspect("all")
    elif args.planner == "scripted":
        result = ScriptedPlanner(session).run(args.max_turns, scenario=scenario).as_dict()
    else:
        planner = NemotronPlanner(session, client=client, model=args.model)
        print(f"  model: {planner.model}")
        result = planner.run(args.max_turns, scenario=scenario).as_dict()

    result["events"] = session.events
    result["session"] = session.summary()
    result["visual_checks"] = session.visual_checks
    result["frame_times"] = session.frame_times
    if runner is not None:
        from .groot_skill import route_stats
        result["routing"] = route_stats(result.get("tool_calls"))
        r = result["routing"]
        print(f"  routing: GR00T {r['groot_ok']}/{r['groot_routes']} routed, expert {r['expert_ok']}/"
              f"{r['expert_routes']}; {runner.client.calls} action chunks")
    if client is not None and not result.get("usage"):
        result["usage"] = client.usage.as_dict()
    with open(os.path.join(out, "trace.json"), "w", encoding="utf-8") as f:
        json.dump(_jsonable(result), f, indent=1)
    write_report(os.path.join(out, "report.md"), spec, result, session)
    if args.video and session.frames:
        import imageio

        from .annotate import annotate
        imageio.mimsave(os.path.join(out, "video.mp4"), session.frames, fps=16, macro_block_size=1)
        annotate(out, frames=session.frames, frame_times=session.frame_times, title=spec.name)
        print(f"video: {os.path.join(out, 'video.mp4')} and video_annotated.mp4")
    print(f"\n{'SUCCESS' if result['success'] else 'FAILED'} in {result['sim_time']} s robot time "
          f"({result['wall_time']} s wall) -> {out}")
    if result.get("error"):
        print(f"planner error: {result['error']}")
    if runner is not None:
        runner.close()
        runner.client.close()
    session.close()
    return 0 if result["success"] else 2


if __name__ == "__main__":
    sys.exit(main())
