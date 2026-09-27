"""Measure how well a vision model inspects the formboard, against simulator ground truth.

Two steps, so the slow part (simulation) runs once and every model sees the same images:

    # 1. Build harnesses and photograph fixtures along the way: seated, not yet routed,
    #    and deliberately botched (wire released over the fork, wire not pressed in,
    #    connector dropped on its holder). The simulator labels every image.
    python -m harness_agent.vision_eval make-set --out vision_set --seeds 0-11

    # 2. Ask one or more vision models about every image and score them.
    python -m harness_agent.vision_eval score --set vision_set --model google/gemma-3-27b-it

``score`` writes one results file per model and ``report.md`` with accuracy, defect
recall, false alarms, latency and a sheet of the images each model got wrong. The same
images are also scored with perception (the tracked cable keypoints), as the baseline.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Generator, Iterable, List, Optional, Sequence

import numpy as np

FAULTS = ("released_early", "not_pressed", "connector_dropped")


# ------------------------------------------------------------------ helpers
def parse_seeds(text: str) -> List[int]:
    """'0-11' or '0,3,5' or '0-3,10' -> list of seeds."""
    out: List[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def model_slug(model: str) -> str:
    return model.replace("/", "__").replace(":", "_")


def _until_phase(gen: Generator, expert, phase: str) -> Generator:
    """Run a skill generator until the expert enters ``phase``, then abandon it."""
    try:
        action = next(gen)
        while expert.phase != phase:
            obs = yield action
            action = gen.send(obs)
    except StopIteration:
        return False
    gen.close()
    return True


def botch_fork(session, fork_id: str, mode: str) -> None:
    """Reproduce a routing mistake: let go of the wire before it is pressed into the fork.

    released_early: open the gripper as soon as the wire has been carried over the fork.
    not_pressed:    lower the wire past the fork, then open before the snap-in push.
    """
    stop = {"released_early": "route_descend", "not_pressed": "route_seat"}[mode]
    ex = session.expert
    i = session.route.index(fork_id)
    ex.current_fork = i
    session._drive(_until_phase(ex._route_fork(i, 0, None), ex, stop), budget=60.0)
    session._drive(session._clear_board(), budget=6.0)


def botch_connector(session) -> None:
    """Carry the connector to its holder and let go of it above the pocket, unpressed."""
    ex = session.expert
    state = session.perceive()
    if state["connector"]["standing_on_end"]:
        session.tip_connector()
    if not ex._connector_ok_to_grasp():
        session.relocate_connector()
    ex.current_fork = -1
    session._drive(_until_phase(ex._insert_connector(auto_recover=False), ex, "connector_descend"), budget=60.0)
    session._drive(session._clear_board(), budget=6.0)


def route_with_retries(session, fork_id: str, attempts: int = 3) -> bool:
    for attempt in range(attempts):
        if session.route_fork(fork_id, attempt).ok:
            return True
    return False


def insert_with_recovery(session, attempts: int = 4) -> bool:
    for _ in range(attempts):
        if session.perceive()["connector"]["standing_on_end"]:
            session.tip_connector()
        res = session.insert_connector()
        if res.outcome == "connector_blocked":
            session.relocate_connector()
            continue
        if res.ok:
            return True
    return False


def _difficulty(kind: str, truth: bool, detail: Dict[str, Any]) -> str:
    if truth:
        return "seated"
    if kind == "fork":
        y, z = detail.get("wire_offset_mm"), detail.get("wire_height_mm")
        near = y is not None and z is not None and abs(y) < 25.0 and z < 90.0
    else:
        near = float(np.linalg.norm(detail.get("offset_from_holder_mm", [1e3])[:2])) < 30.0
    return "hard" if near else "easy"


# ----------------------------------------------------------------- make-set
def make_set(out: str, spec_path: str, seeds: Sequence[int], randomize: bool = True,
             faults: bool = True, log=print) -> List[Dict[str, Any]]:
    from harness_core.render_util import choose_gl_backend
    choose_gl_backend()
    from .session import CellSession
    from .spec import HarnessSpec

    spec = HarnessSpec.from_yaml(spec_path)
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    labels: List[Dict[str, Any]] = []
    for seed in seeds:
        t0 = time.perf_counter()
        session = CellSession(spec, seed=seed, randomize=randomize)
        if not session.feasible:
            raise SystemExit(f"{spec_path} fails validation")
        route, cid = session.route, session.connector_id
        rng = np.random.default_rng(10_000 + seed)
        n_before = len(labels)

        def snap(target: str, stage: str, fault: Optional[str] = None) -> None:
            if session.perceive()["tcp_height_above_board_mm"] < 150:
                session.retreat()               # park the arm out of the camera's view
            state = session.perceive(session.settled_obs(0.4))
            kind, image, view = session.photograph(target, view=int(rng.integers(0, 3)))
            truth = session.truth()
            if kind == "fork":
                t, p, detail = truth["forks_routed"][target], state["forks"][target]["wire_in_slot"], \
                    state["forks"][target]
            else:
                t, p, detail = truth["connector_seated"], state["connector"]["in_holder"], state["connector"]
            ident = f"s{seed:03d}_{len(labels) - n_before:02d}_{target}"
            path = os.path.join("images", ident + ".jpg")
            image.save(os.path.join(out, path), quality=90)
            labels.append({"id": ident, "image": path, "kind": kind, "target": target, "seed": seed,
                           "spec": os.path.basename(spec_path), "stage": stage, "fault": fault,
                           "view": view, "truth": bool(t), "perceived": bool(p),
                           "difficulty": _difficulty(kind, bool(t), detail), "detail": detail})

        snap(route[0], "start")
        fault_fork = route[seed % len(route)] if faults else None
        all_routed = True
        for k, fid in enumerate(route):
            if fid == fault_fork:
                mode = FAULTS[(seed // len(route)) % 2]
                botch_fork(session, fid, mode)
                snap(fid, "botched", mode)
            ok = route_with_retries(session, fid)
            snap(fid, "routed" if ok else "route_failed")
            if not ok:
                all_routed = False
                break
            if k + 1 < len(route):
                snap(route[k + 1], "next")
        if all_routed:
            if faults and seed % 2 == 0:
                botch_connector(session)
                snap(cid, "botched", "connector_dropped")
            ok = insert_with_recovery(session)
            snap(cid, "inserted" if ok else "insert_failed")
            session.retreat()
            for fid in route:
                snap(fid, "final")
        session.close()
        new = labels[n_before:]
        log(f"seed {seed}: {len(new)} images ({sum(not l['truth'] for l in new)} not seated), "
            f"success={session.truth().get('success')} in {time.perf_counter() - t0:.0f} s")
        with open(os.path.join(out, "labels.jsonl"), "w", encoding="utf-8") as f:
            for lab in labels:
                f.write(json.dumps(lab) + "\n")
    return labels


def read_set(set_dir: str) -> List[Dict[str, Any]]:
    with open(os.path.join(set_dir, "labels.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# -------------------------------------------------------------------- score
def metrics(rows: Sequence[Dict[str, Any]], pred_key: str = "pred") -> Dict[str, Any]:
    """Accuracy and defect-detection numbers. A defect is a fixture that is NOT seated;
    an unusable answer counts as wrong (and is reported separately)."""
    n = len(rows)
    if n == 0:
        return {"n": 0}
    pred = [r.get(pred_key) for r in rows]
    truth = [bool(r["truth"]) for r in rows]
    correct = sum(p is not None and p == t for p, t in zip(pred, truth))
    defects = [i for i, t in enumerate(truth) if not t]
    good = [i for i, t in enumerate(truth) if t]
    caught = sum(pred[i] is False for i in defects)
    false_alarms = sum(pred[i] is False for i in good)
    flagged = sum(p is False for p in pred)
    out = {"n": n, "accuracy": correct / n,
           "unusable": sum(p is None for p in pred) / n,
           "defects": len(defects), "defect_recall": caught / len(defects) if defects else None,
           "false_alarm_rate": false_alarms / len(good) if good else None,
           "precision": caught / flagged if flagged else None}
    secs = [r["seconds"] for r in rows if r.get("seconds") is not None]
    if secs:
        out["median_s"] = statistics.median(secs)
        out["p90_s"] = float(np.percentile(secs, 90))
    conf_right = [r["confidence"] for r, p, t in zip(rows, pred, truth) if p is not None and p == t
                  and r.get("confidence") is not None]
    conf_wrong = [r["confidence"] for r, p, t in zip(rows, pred, truth) if p is not None and p != t
                  and r.get("confidence") is not None]
    if conf_right:
        out["confidence_when_right"] = float(np.mean(conf_right))
    if conf_wrong:
        out["confidence_when_wrong"] = float(np.mean(conf_wrong))
    return out


def breakdown(rows: Sequence[Dict[str, Any]], pred_key: str = "pred") -> Dict[str, Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = {"all": list(rows)}
    for r in rows:
        groups.setdefault(f"kind={r['kind']}", []).append(r)
        groups.setdefault(f"difficulty={r['difficulty']}", []).append(r)
        if r.get("fault"):
            groups.setdefault(f"fault={r['fault']}", []).append(r)
    return {k: metrics(v, pred_key) for k, v in groups.items()}


def score_model(set_dir: str, model: str, client=None, workers: int = 4, limit: Optional[int] = None,
                log=print) -> List[Dict[str, Any]]:
    from PIL import Image

    from .vision import VisualInspector

    labels = read_set(set_dir)[:limit] if limit else read_set(set_dir)
    inspector = VisualInspector(client=client, model=model)

    def one(lab: Dict[str, Any]) -> Dict[str, Any]:
        with Image.open(os.path.join(set_dir, lab["image"])) as im:
            v = inspector.ask(lab["kind"], lab["target"], im.convert("RGB"), view=lab["view"])
        return {**lab, "model": model, "pred": v.seated, "confidence": v.confidence,
                "evidence": v.evidence, "seconds": round(v.seconds, 2), "error": v.error, "raw": v.raw}

    rows: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for k, row in enumerate(pool.map(one, labels), 1):
            rows.append(row)
            mark = "ok " if row["pred"] is not None and row["pred"] == row["truth"] else (
                "?? " if row["pred"] is None else "XX ")
            log(f"  [{k:3d}/{len(labels)}] {mark}{row['id']:24s} truth={'seated' if row['truth'] else 'NOT'}"
                f"  model={row['pred']} ({row['confidence']:.2f}, {row['seconds']:.1f} s)"
                + (f"  {row['error']}" if row["error"] else ""))
    with open(os.path.join(set_dir, f"results_{model_slug(model)}.jsonl"), "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return rows


def error_sheet(set_dir: str, rows: Sequence[Dict[str, Any]], path: str, max_items: int = 12) -> Optional[str]:
    """Tile the images a model got wrong, with what it said."""
    from PIL import Image, ImageDraw

    from .vision import _font

    wrong = [r for r in rows if r.get("pred") is None or r["pred"] != r["truth"]][:max_items]
    if not wrong:
        return None
    w = 640
    tiles = []
    for r in wrong:
        with Image.open(os.path.join(set_dir, r["image"])) as im:
            im = im.convert("RGB")
            im = im.resize((w, round(im.height * w / im.width)))
        cap = Image.new("RGB", (w, 44), (255, 255, 255))
        d = ImageDraw.Draw(cap)
        said = "no answer" if r["pred"] is None else ("seated" if r["pred"] else "not seated")
        d.text((6, 3), f"{r['id']}  truth: {'seated' if r['truth'] else 'NOT seated'}  model: {said} "
                       f"({r.get('confidence', 0):.2f})", fill=(160, 20, 20), font=_font(14))
        d.text((6, 23), (r.get("evidence") or r.get("error") or "")[:95], fill=(40, 40, 40), font=_font(13))
        tile = Image.new("RGB", (w, im.height + 44), (255, 255, 255))
        tile.paste(im, (0, 0))
        tile.paste(cap, (0, im.height))
        tiles.append(tile)
    cols = 2
    rows_n = (len(tiles) + cols - 1) // cols
    th = max(t.height for t in tiles)
    sheet = Image.new("RGB", (cols * w + (cols - 1) * 8, rows_n * th + (rows_n - 1) * 8), (230, 230, 230))
    for k, t in enumerate(tiles):
        sheet.paste(t, ((k % cols) * (w + 8), (k // cols) * (th + 8)))
    sheet.save(path, quality=85)
    return path


def _pct(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{100 * v:.0f}%"


def write_report(set_dir: str, results: Dict[str, List[Dict[str, Any]]], usage: Dict[str, Any]) -> str:
    labels = read_set(set_dir)
    n_def = sum(not l["truth"] for l in labels)
    lines = ["# Visual inspection accuracy", "",
             f"{len(labels)} images from {len({l['seed'] for l in labels})} simulated builds "
             f"({n_def} not seated, {len(labels) - n_def} seated; "
             f"{sum(bool(l.get('fault')) for l in labels)} from deliberately botched steps). "
             "Labels are simulator ground truth. A *defect* is a fixture that is not seated; "
             "*defect recall* is the share of defects the check catches, *false alarms* the "
             "share of good fixtures it rejects. An unusable answer counts as wrong.", "",
             "| check | accuracy | defect recall | false alarms | hard defects caught | unusable | "
             "median latency |", "|---|---|---|---|---|---|---|"]
    base = breakdown(labels, "perceived")
    lines.append(f"| perception (cable keypoints) | {_pct(base['all']['accuracy'])} | "
                 f"{_pct(base['all']['defect_recall'])} | {_pct(base['all']['false_alarm_rate'])} | "
                 f"{_pct(base.get('difficulty=hard', {}).get('defect_recall'))} | 0% | - |")
    for model, rows in results.items():
        b = breakdown(rows)
        a = b["all"]
        lines.append(f"| `{model}` | {_pct(a['accuracy'])} | {_pct(a['defect_recall'])} | "
                     f"{_pct(a['false_alarm_rate'])} | {_pct(b.get('difficulty=hard', {}).get('defect_recall'))} | "
                     f"{_pct(a['unusable'])} | {a.get('median_s', 0):.1f} s |")
    lines.append("")
    for model, rows in results.items():
        b = breakdown(rows)
        lines += [f"## `{model}`", "", "| subset | n | accuracy | defect recall | false alarms |",
                  "|---|---|---|---|---|"]
        for k, m in b.items():
            lines.append(f"| {k} | {m['n']} | {_pct(m['accuracy'])} | {_pct(m.get('defect_recall'))} | "
                         f"{_pct(m.get('false_alarm_rate'))} |")
        a = b["all"]
        u = (usage.get("per_model") or {}).get(model)
        extra = []
        if "confidence_when_right" in a:
            extra.append(f"mean confidence {a['confidence_when_right']:.2f} when right"
                         + (f", {a['confidence_when_wrong']:.2f} when wrong" if "confidence_when_wrong" in a else ""))
        if u and u.get("calls"):
            extra.append(f"{u['prompt_tokens'] / u['calls']:.0f} prompt + {u['completion_tokens'] / u['calls']:.0f} "
                         f"completion tokens per image")
        if extra:
            lines += ["", "; ".join(extra) + "."]
        sheet = error_sheet(set_dir, rows, os.path.join(set_dir, f"errors_{model_slug(model)}.jpg"))
        if sheet:
            lines += ["", f"![images {model} got wrong]({os.path.basename(sheet)})"]
        lines.append("")
    path = os.path.join(set_dir, "report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


# ---------------------------------------------------------------------- cli
def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    mk = sub.add_parser("make-set", help="simulate builds and photograph fixtures (no model calls)")
    mk.add_argument("--out", default="vision_set")
    mk.add_argument("--spec", default=None, help="harness spec (default: the packaged demo_3fork.yaml)")
    mk.add_argument("--seeds", default="0-11", help="e.g. 0-11 or 0,3,5")
    mk.add_argument("--no-randomize", action="store_true", help="nominal layout for every seed")
    mk.add_argument("--no-faults", action="store_true", help="do not botch any steps")
    sc = sub.add_parser("score", help="ask vision models about every image and score them")
    sc.add_argument("--set", dest="set_dir", default="vision_set")
    sc.add_argument("--model", action="append", default=None,
                    help="Token Factory model id; repeat to compare models (default: the picked vision model)")
    sc.add_argument("--workers", type=int, default=4, help="parallel requests")
    sc.add_argument("--limit", type=int, default=None, help="only the first N images (a quick look)")
    rp = sub.add_parser("report", help="rewrite report.md from the results files already in the set")
    rp.add_argument("--set", dest="set_dir", default="vision_set")
    args = ap.parse_args(list(argv) if argv is not None else None)

    if args.cmd == "make-set":
        spec = args.spec or os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                                         "specs", "demo_3fork.yaml")
        if not os.path.exists(spec):
            from ament_index_python.packages import get_package_share_directory
            spec = os.path.join(get_package_share_directory("harness_agent"), "specs", "demo_3fork.yaml")
        labels = make_set(args.out, spec, parse_seeds(args.seeds), randomize=not args.no_randomize,
                          faults=not args.no_faults)
        n_def = sum(not l["truth"] for l in labels)
        print(f"\n{len(labels)} images ({n_def} not seated) in {args.out}/")
        return 0

    results: Dict[str, List[Dict[str, Any]]] = {}
    usage: Dict[str, Any] = {}
    if args.cmd == "score":
        from .llm import TokenFactoryClient
        client = TokenFactoryClient(max_retries=6)          # rides out rate limits on long runs
        models = args.model or [client.vision_model()]
        for model in models:
            print(f"\n{model}:")
            rows = score_model(args.set_dir, model, client=client, workers=args.workers, limit=args.limit)
            results[model] = rows
            a = metrics(rows)
            print(f"  accuracy {_pct(a['accuracy'])}, defect recall {_pct(a['defect_recall'])}, "
                  f"false alarms {_pct(a['false_alarm_rate'])}, unusable {_pct(a['unusable'])}")
        usage = client.usage.as_dict()
    for name in sorted(os.listdir(args.set_dir)):          # earlier runs of other models, for comparison
        if name.startswith("results_") and name.endswith(".jsonl"):
            with open(os.path.join(args.set_dir, name), encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
            if rows and rows[0]["model"] not in results:
                results[rows[0]["model"]] = rows
    path = write_report(args.set_dir, results, usage)
    print(f"\nreport: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
