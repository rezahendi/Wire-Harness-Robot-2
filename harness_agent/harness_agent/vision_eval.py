"""Measure how well a vision model inspects the formboard, against simulator ground truth.

Two steps, so the slow part (simulation) runs once and every model sees the same images:

    # 1. Build harnesses and photograph fixtures along the way: seated, not yet routed,
    #    and deliberately botched (wire released over the fork, wire not pressed in,
    #    connector dropped on its holder). The simulator labels every image. Each
    #    fixture is photographed in both question styles (see vision.py), and a few
    #    labelled examples from a separate build go to refs/ for few-shot prompts.
    python -m harness_agent.vision_eval make-set --out vision_set2 --seeds 0-11

    # 2. Ask one or more vision models about every image and score them.
    python -m harness_agent.vision_eval score --set vision_set2 \\
        --model openbmb/MiniCPM-V-4_5 --style v1 --style v2 --style v2refs

``score`` writes one results file per model and style, and ``report.md`` with accuracy,
defect recall, false alarms, latency, tokens and a sheet of the images each got wrong.
The same images are also scored with perception (the tracked cable keypoints), as the
baseline.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Generator, Iterable, List, Optional, Sequence, Tuple

import numpy as np

FAULTS = ("released_early", "not_pressed", "connector_dropped")
SCORE_STYLES = ("v1", "v2", "v2refs")
REF_CAPTIONS = {
    ("fork", True): "fork seated: the orange wire runs through the gap between the two prongs, "
                    "below the yellow knobs. Answer A yes, B yes.",
    ("fork", False): "fork NOT seated: the gap between the prongs is empty; the wire lies on the board "
                     "next to the fork. Answer A no, B no.",
    ("connector", True): "connector seated: it lies flat, fully inside the green pocket. Answer A yes, B yes.",
    ("connector", False): "connector NOT seated: it is not down in the pocket. Answer A no, B no.",
}


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
def _build_and_snap(spec_path: str, seed: int, randomize: bool, faults: bool, snap_cb) -> Dict[str, Any]:
    """One build with botched steps; calls snap_cb(session, target, stage, fault) along the way."""
    from .session import CellSession
    from .spec import HarnessSpec

    session = CellSession(HarnessSpec.from_yaml(spec_path), seed=seed, randomize=randomize)
    if not session.feasible:
        raise SystemExit(f"{spec_path} fails validation")
    route, cid = session.route, session.connector_id

    def snap(target: str, stage: str, fault: Optional[str] = None) -> None:
        snap_cb(session, target, stage, fault)

    snap(route[0], "start")
    snap(cid, "start")
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
    truth = session.truth()
    session.close()
    return truth


def _observe_target(session, target: str) -> Tuple[str, bool, bool, Dict[str, Any]]:
    """(kind, truth, perceived, perception detail) after parking the arm and settling."""
    if session.perceive()["tcp_height_above_board_mm"] < 150:
        session.retreat()                   # park the arm out of the camera's view
    state = session.perceive(session.settled_obs(0.4))
    truth = session.truth()
    if target == session.connector_id:
        return "connector", bool(truth["connector_seated"]), bool(state["connector"]["in_holder"]), \
            state["connector"]
    return "fork", bool(truth["forks_routed"][target]), bool(state["forks"][target]["wire_in_slot"]), \
        state["forks"][target]


def make_set(out: str, spec_path: str, seeds: Sequence[int], randomize: bool = True,
             faults: bool = True, refs_seed: Optional[int] = 1000, log=print) -> List[Dict[str, Any]]:
    from harness_core.render_util import choose_gl_backend
    choose_gl_backend()

    for sub in ("images", "images_v2"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    labels: List[Dict[str, Any]] = []
    for seed in seeds:
        t0 = time.perf_counter()
        rng = np.random.default_rng(10_000 + seed)
        n_before = len(labels)

        def snap(session, target: str, stage: str, fault: Optional[str]) -> None:
            kind, truth, perceived, detail = _observe_target(session, target)
            view = int(rng.integers(0, 3))
            ident = f"s{seed:03d}_{len(labels) - n_before:02d}_{target}"
            paths = {}
            for style, sub in (("v1", "images"), ("v2", "images_v2")):
                _, image, _ = session.photograph(target, view=view, style=style)
                paths[style] = os.path.join(sub, ident + ".jpg")
                image.save(os.path.join(out, paths[style]), quality=90)
            labels.append({"id": ident, "image": paths["v1"], "image_v2": paths["v2"], "kind": kind,
                           "target": target, "seed": seed, "spec": os.path.basename(spec_path),
                           "stage": stage, "fault": fault, "view": view, "truth": truth,
                           "perceived": perceived, "difficulty": _difficulty(kind, truth, detail),
                           "detail": detail})

        final = _build_and_snap(spec_path, seed, randomize, faults, snap)
        new = labels[n_before:]
        log(f"seed {seed}: {len(new)} images ({sum(not l['truth'] for l in new)} not seated), "
            f"success={final.get('success')} in {time.perf_counter() - t0:.0f} s")
        with open(os.path.join(out, "labels.jsonl"), "w", encoding="utf-8") as f:
            for lab in labels:
                f.write(json.dumps(lab) + "\n")
    if refs_seed is not None:
        make_refs(os.path.join(out, "refs"), spec_path, refs_seed, randomize=randomize, log=log)
    return labels


def make_refs(ref_dir: str, spec_path: str, seed: int, randomize: bool = True, log=print) -> Dict[str, Any]:
    """Labelled example images for few-shot prompts, from a build that is not in the set.

    Picks a seated and a not-seated example of each fixture kind; for the not-seated fork
    it prefers a wire lying close to the fork (the case models get wrong)."""
    os.makedirs(ref_dir, exist_ok=True)
    cands: Dict[Tuple[str, bool], List[Tuple[int, Any, Any]]] = {}

    def snap(session, target: str, stage: str, fault: Optional[str]) -> None:
        kind, truth, _, detail = _observe_target(session, target)
        if kind == "fork" and not truth and stage != "next":
            return                            # botched/failed states can look ambiguous
        rank = 0 if _difficulty(kind, truth, detail) in ("hard", "seated") else 1
        imgs = {style: session.photograph(target, view=0, style=style)[1] for style in ("v1", "v2")}
        cands.setdefault((kind, truth), []).append((rank, imgs, stage))

    _build_and_snap(spec_path, seed, randomize, True, snap)
    spec: Dict[str, List[Dict[str, str]]] = {"v1": [], "v2": []}
    for (kind, truth), items in sorted(cands.items(), key=lambda kv: (kv[0][0], not kv[0][1])):
        rank, imgs, stage = sorted(items, key=lambda it: it[0])[0]
        for style in ("v1", "v2"):
            name = f"{kind}_{'yes' if truth else 'no'}_{style}.jpg"
            imgs[style].save(os.path.join(ref_dir, name), quality=90)
            spec[style].append({"kind": kind, "seated": truth, "image": name,
                                "caption": REF_CAPTIONS[(kind, truth)], "stage": stage})
    with open(os.path.join(ref_dir, "refs.json"), "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=1)
    log(f"references from seed {seed}: " + ", ".join(f"{d['kind']} {'yes' if d['seated'] else 'no'}"
                                                     for d in spec["v2"]))
    return spec


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
    toks = [(r.get("prompt_tokens") or 0, r.get("completion_tokens") or 0) for r in rows]
    if any(p or c for p, c in toks):
        out["prompt_tokens"] = float(np.mean([p for p, _ in toks]))
        out["completion_tokens"] = float(np.mean([c for _, c in toks]))
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
    order = ["all"] + sorted(k for k in groups if k != "all")
    return {k: metrics(groups[k], pred_key) for k in order}


def score_model(set_dir: str, model: str, client=None, style: str = "v2", workers: int = 4,
                limit: Optional[int] = None, log=print) -> List[Dict[str, Any]]:
    from PIL import Image

    from .vision import VisualInspector, load_references

    labels = read_set(set_dir)[:limit] if limit else read_set(set_dir)
    base_style = "v1" if style == "v1" else "v2"
    refs = load_references(os.path.join(set_dir, "refs"), base_style) if style.endswith("refs") else None
    inspector = VisualInspector(client=client, model=model, style=base_style, references=refs)

    def one(lab: Dict[str, Any]) -> Dict[str, Any]:
        path = lab["image"] if base_style == "v1" else lab.get("image_v2", lab["image"])
        with Image.open(os.path.join(set_dir, path)) as im:
            v = inspector.ask(lab["kind"], lab["target"], im.convert("RGB"), view=lab["view"])
        return {**lab, "model": model, "style": style, "scored_image": path, "pred": v.seated,
                "confidence": v.confidence, "evidence": v.evidence, "per_view": v.per_view,
                "seconds": round(v.seconds, 2), "error": v.error, "raw": v.raw,
                "prompt_tokens": v.prompt_tokens, "completion_tokens": v.completion_tokens}

    rows: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for k, row in enumerate(pool.map(one, labels), 1):
            rows.append(row)
            mark = "ok " if row["pred"] is not None and row["pred"] == row["truth"] else (
                "?? " if row["pred"] is None else "XX ")
            log(f"  [{k:3d}/{len(labels)}] {mark}{row['id']:24s} truth={'seated' if row['truth'] else 'NOT'}"
                f"  model={row['pred']} ({row['seconds']:.1f} s)" + (f"  {row['error']}" if row["error"] else ""))
    with open(os.path.join(set_dir, f"results_{model_slug(model)}__{style}.jsonl"), "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return rows


def load_results(set_dir: str) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    """Every results file in the set, keyed by (model, style)."""
    out: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for name in sorted(os.listdir(set_dir)):
        if name.startswith("results_") and name.endswith(".jsonl"):
            with open(os.path.join(set_dir, name), encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
            if rows:
                out[(rows[0]["model"], rows[0].get("style", "v1"))] = rows
    return out


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
        with Image.open(os.path.join(set_dir, r.get("scored_image") or r["image"])) as im:
            im = im.convert("RGB")
            im = im.resize((w, round(im.height * w / im.width)))
        cap = Image.new("RGB", (w, 44), (255, 255, 255))
        d = ImageDraw.Draw(cap)
        said = "no answer" if r["pred"] is None else ("seated" if r["pred"] else "not seated")
        views = r.get("per_view") or {}
        vtxt = ("  views " + " ".join(f"{k}:{'yes' if v else 'no' if v is False else '?'}" for k, v in views.items())
                if views else "")
        d.text((6, 3), f"{r['id']}  truth: {'seated' if r['truth'] else 'NOT seated'}  model: {said}{vtxt}",
               fill=(160, 20, 20), font=_font(14))
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


def write_report(set_dir: str, results: Dict[Any, List[Dict[str, Any]]], usage: Optional[Dict[str, Any]] = None) -> str:
    """report.md for the set. ``results`` is keyed by (model, style) or by model."""
    labels = read_set(set_dir)
    n_def = sum(not l["truth"] for l in labels)
    lines = ["# Visual inspection accuracy", "",
             f"{len(labels)} images from {len({l['seed'] for l in labels})} simulated builds "
             f"({n_def} not seated, {len(labels) - n_def} seated; "
             f"{sum(bool(l.get('fault')) for l in labels)} from deliberately botched steps). "
             "Labels are simulator ground truth. A *defect* is a fixture that is not seated; "
             "*defect recall* is the share of defects the check catches, *false alarms* the "
             "share of good fixtures it rejects. An unusable answer counts as wrong. "
             "Styles: v1 general views and one yes/no question; v2 views through the slot, the "
             "fixture boxed, one question per view; v2refs the same with two labelled examples.", "",
             "| check | style | accuracy | defect recall | false alarms | hard defects caught | unusable | "
             "median latency | tokens in/out |", "|---|---|---|---|---|---|---|---|---|"]
    base = breakdown(labels, "perceived")
    lines.append(f"| perception (cable keypoints) | - | {_pct(base['all']['accuracy'])} | "
                 f"{_pct(base['all']['defect_recall'])} | {_pct(base['all']['false_alarm_rate'])} | "
                 f"{_pct(base.get('difficulty=hard', {}).get('defect_recall'))} | 0% | - | - |")
    keyed = {(k if isinstance(k, tuple) else (k, rows[0].get("style", "v1") if rows else "v1")): rows
             for k, rows in results.items()}
    for (model, style), rows in keyed.items():
        b = breakdown(rows)
        a = b["all"]
        tok = (f"{a['prompt_tokens']:.0f} / {a['completion_tokens']:.0f}" if "prompt_tokens" in a else "-")
        lines.append(f"| `{model}` | {style} | {_pct(a['accuracy'])} | {_pct(a['defect_recall'])} | "
                     f"{_pct(a['false_alarm_rate'])} | {_pct(b.get('difficulty=hard', {}).get('defect_recall'))} | "
                     f"{_pct(a['unusable'])} | {a.get('median_s', 0):.1f} s | {tok} |")
    lines.append("")
    for (model, style), rows in keyed.items():
        b = breakdown(rows)
        lines += [f"## `{model}`, {style}", "", "| subset | n | accuracy | defect recall | false alarms |",
                  "|---|---|---|---|---|"]
        for k, m in b.items():
            lines.append(f"| {k} | {m['n']} | {_pct(m['accuracy'])} | {_pct(m.get('defect_recall'))} | "
                         f"{_pct(m.get('false_alarm_rate'))} |")
        a = b["all"]
        if "confidence_when_right" in a:
            lines += ["", f"Mean confidence {a['confidence_when_right']:.2f} when right"
                      + (f", {a['confidence_when_wrong']:.2f} when wrong" if "confidence_when_wrong" in a else "")
                      + "."]
        sheet = error_sheet(set_dir, rows, os.path.join(set_dir, f"errors_{model_slug(model)}__{style}.jpg"))
        if sheet:
            lines += ["", f"![images {model} ({style}) got wrong]({os.path.basename(sheet)})"]
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
    mk.add_argument("--refs-seed", type=int, default=1000, help="build that supplies the few-shot examples "
                                                                  "(-1: none)")
    mk.add_argument("--no-randomize", action="store_true", help="nominal layout for every seed")
    mk.add_argument("--no-faults", action="store_true", help="do not botch any steps")
    sc = sub.add_parser("score", help="ask vision models about every image and score them")
    sc.add_argument("--set", dest="set_dir", default="vision_set")
    sc.add_argument("--model", action="append", default=None,
                    help="Token Factory model id; repeat to compare models (default: the picked vision model)")
    sc.add_argument("--style", action="append", default=None, choices=SCORE_STYLES,
                    help="question style; repeat to compare (default: v2)")
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
                          faults=not args.no_faults, refs_seed=None if args.refs_seed < 0 else args.refs_seed)
        n_def = sum(not l["truth"] for l in labels)
        print(f"\n{len(labels)} images ({n_def} not seated) in {args.out}/")
        return 0

    if args.cmd == "score":
        from .llm import TokenFactoryClient
        client = TokenFactoryClient(max_retries=6)          # rides out rate limits on long runs
        models = args.model or [client.vision_model()]
        styles = args.style or ["v2"]
        for model in models:
            for style in styles:
                print(f"\n{model}, style {style}:")
                rows = score_model(args.set_dir, model, client=client, style=style, workers=args.workers,
                                   limit=args.limit)
                a = metrics(rows)
                print(f"  accuracy {_pct(a['accuracy'])}, defect recall {_pct(a['defect_recall'])}, "
                      f"false alarms {_pct(a['false_alarm_rate'])}, unusable {_pct(a['unusable'])}")
    path = write_report(args.set_dir, load_results(args.set_dir))
    print(f"\nreport: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
