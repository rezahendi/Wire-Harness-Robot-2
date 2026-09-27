"""Recovery benchmark: planners x disturbance scenarios x randomised layouts.

    python -m harness_agent.benchmark --planners scripted --seeds 0-4 --out runs/bench
    python -m harness_agent.benchmark --planners nemotron --seeds 0-4 --out runs/bench   # needs a key

Every build goes through the same skills and tools, and each scenario injects the same
disturbance at the same point (see disturbances.py): a wire pulled out of a fork right
after it was routed, the connector slipping out of the fingers above its holder, or both.
Runs already in --out are skipped, so the planners can be run at different times (and on
different machines) into one table.

The summary reports, per planner and scenario: builds that succeeded (simulator ground
truth), honest verdicts (the planner's final claim matched the truth), steps, robot time,
and model tokens per build.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, Iterable, List, Optional

from .disturbances import SCENARIOS, make_scenario
from .vision_eval import parse_seeds

PLANNERS = ("scripted", "nemotron")


def run_one(spec_path: str, planner: str, scenario: str, seed: int, out_dir: str, vision: bool = False,
            model: Optional[str] = None, max_turns: int = 40, log=print) -> Dict[str, Any]:
    from .agent import NemotronPlanner, ScriptedPlanner
    from .run_build import _jsonable, write_report
    from .session import CellSession
    from .spec import HarnessSpec

    spec = HarnessSpec.from_yaml(spec_path)
    client, inspector = None, None
    if planner == "nemotron" or vision:
        from .llm import TokenFactoryClient
        client = TokenFactoryClient()
    if vision:
        from .vision import default_inspector
        inspector = default_inspector(client)
    os.makedirs(out_dir, exist_ok=True)
    session = CellSession(spec, seed=seed, randomize=True, inspector=inspector,
                          inspection_dir=os.path.join(out_dir, "inspection") if inspector else None)
    scen = make_scenario(scenario, session.route)
    quiet = (lambda *_: None)
    t0 = time.perf_counter()
    if planner == "scripted":
        res = ScriptedPlanner(session, log=quiet).run(max_turns, scenario=scen)
    else:
        res = NemotronPlanner(session, client=client, model=model, log=quiet).run(max_turns, scenario=scen)
    result = res.as_dict()
    if client is not None and not result.get("usage"):
        result["usage"] = client.usage.as_dict()
    result["events"] = session.events
    result["session"] = session.summary()
    result["visual_checks"] = session.visual_checks
    with open(os.path.join(out_dir, "trace.json"), "w", encoding="utf-8") as f:
        json.dump(_jsonable(result), f, indent=1)
    write_report(os.path.join(out_dir, "report.md"), spec, result, session)
    session.close()
    u = result.get("usage") or {}
    row = {"planner": planner if not vision else planner + "+vision", "scenario": scenario, "seed": seed,
           "success": bool(result["success"]), "claimed": result.get("claimed_success"),
           "honest": result.get("claimed_success") is not None
           and bool(result.get("claimed_success")) == bool(result["success"]),
           "turns": result.get("turns", 0), "tool_calls": len(result.get("tool_calls") or []),
           "robot_s": result.get("sim_time"), "wall_s": round(time.perf_counter() - t0, 1),
           "tokens": int(u.get("prompt_tokens", 0)) + int(u.get("completion_tokens", 0)),
           "refused_verdicts": result.get("refused_verdicts", 0),
           "disturbances": [d["label"] for d in result.get("disturbances") or []]
           + [e["label"] for e in session.events if e.get("type") == "disturbance" and "after" not in e],
           "error": result.get("error", ""), "model": result.get("model", "")}
    log(f"  {row['planner']:16s} {scenario:18s} seed {seed}: {'SUCCESS' if row['success'] else 'failed '} "
        f"claimed={row['claimed']} steps={row['tool_calls']} robot {row['robot_s']} s, wall {row['wall_s']} s"
        + (f"  [{row['error']}]" if row["error"] else ""))
    return row


def summarize(rows: List[Dict[str, Any]]) -> str:
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault((r["planner"], r["scenario"]), []).append(r)
    order = {s: k for k, s in enumerate(SCENARIOS)}
    lines = ["| planner | scenario | builds | succeeded | honest verdicts | steps | robot time | "
             "tokens per build |", "|---|---|---|---|---|---|---|---|"]
    for (planner, scenario), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], order.get(kv[0][1], 99))):
        n = len(rs)
        ok = sum(r["success"] for r in rs)
        honest = sum(r["honest"] for r in rs)
        steps = sum(r["tool_calls"] for r in rs) / n
        robot = sum(r["robot_s"] or 0 for r in rs) / n
        tokens = sum(r["tokens"] for r in rs) / n
        lines.append(f"| {planner} | {scenario} | {n} | {ok}/{n} | {honest}/{n} | {steps:.1f} | {robot:.0f} s | "
                     f"{tokens:,.0f} |" if tokens else
                     f"| {planner} | {scenario} | {n} | {ok}/{n} | {honest}/{n} | {steps:.1f} | {robot:.0f} s | - |")
    return "\n".join(lines)


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", default=None, help="harness spec (default: the packaged demo_3fork.yaml)")
    ap.add_argument("--planners", default="scripted", help=f"comma list of {PLANNERS}")
    ap.add_argument("--scenarios", default=",".join(SCENARIOS), help=f"comma list of {SCENARIOS}")
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--vision", action="store_true", help="camera check on (nemotron+vision)")
    ap.add_argument("--model", default=None, help="planner model id (default: picked automatically)")
    ap.add_argument("--out", default="runs/bench")
    ap.add_argument("--force", action="store_true", help="re-run builds already in --out")
    args = ap.parse_args(list(argv) if argv is not None else None)

    from harness_core.render_util import choose_gl_backend
    choose_gl_backend()
    spec = args.spec or os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                                     "specs", "demo_3fork.yaml")
    if not os.path.exists(spec):
        from ament_index_python.packages import get_package_share_directory
        spec = os.path.join(get_package_share_directory("harness_agent"), "specs", "demo_3fork.yaml")
    planners = [p.strip() for p in args.planners.split(",") if p.strip()]
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    for p in planners:
        if p not in PLANNERS:
            raise SystemExit(f"unknown planner {p!r}; use {PLANNERS}")
    for s in scenarios:
        if s not in SCENARIOS:
            raise SystemExit(f"unknown scenario {s!r}; use {SCENARIOS}")
    os.makedirs(args.out, exist_ok=True)
    summary_path = os.path.join(args.out, "summary.jsonl")
    rows: List[Dict[str, Any]] = []
    if os.path.exists(summary_path):
        with open(summary_path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    done = {(r["planner"], r["scenario"], r["seed"]) for r in rows}
    for seed in parse_seeds(args.seeds):
        for scenario in scenarios:
            for planner in planners:
                label = planner + ("+vision" if args.vision else "")
                if (label, scenario, seed) in done and not args.force:
                    continue
                out_dir = os.path.join(args.out, f"{label}_{scenario}_s{seed}")
                row = run_one(spec, planner, scenario, seed, out_dir, vision=args.vision, model=args.model)
                rows = [r for r in rows if (r["planner"], r["scenario"], r["seed"]) != (label, scenario, seed)]
                rows.append(row)
                with open(summary_path, "w", encoding="utf-8") as f:
                    for r in rows:
                        f.write(json.dumps(r) + "\n")
    table = summarize(rows)
    with open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write("# Recovery benchmark\n\n" + table + "\n")
    print("\n" + table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
