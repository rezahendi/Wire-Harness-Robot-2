"""Turn a build run into an annotated video: the cell on the left, the planner on the right.

    python -m harness_agent.annotate runs/demo_3fork_nemotron_0      # needs video.mp4 + trace.json

For every frame it shows which tool call is running, the calls so far with their outcomes,
what the model was thinking when it made the current call, disturbances as they happen,
and the inspection photos with the camera's verdicts. run_build --video writes it
automatically (video_annotated.mp4).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

W, H = 1920, 1080
SIM_W, SIM_H = 1280, 960
BG = (246, 246, 244)
INK = (28, 28, 30)
MUTED = (110, 110, 115)
OK = (30, 130, 70)
BAD = (190, 50, 40)
WARN = (200, 120, 0)
ACCENT = (40, 90, 200)

TOOL_WORDS = {"get_status": "check state", "route_fork": "route", "tip_connector": "tip connector over",
              "relocate_connector": "move connector", "insert_connector": "insert connector",
              "inspect": "inspect", "retreat": "retreat arm", "finish": "finish"}


def _font(size: int):
    from .vision import _font as f
    return f(size)


def _call_label(call: Dict[str, Any]) -> str:
    args = call.get("arguments") or {}
    name = call["name"]
    word = TOOL_WORDS.get(name, name)
    if name == "route_fork":
        word += f" {args.get('fork_id', '?')}"
        if args.get("attempt"):
            word += f" (attempt {args['attempt']})"
    elif name == "inspect":
        word += f" {args.get('target', 'all')}"
    elif name == "finish":
        word += " : " + ("SUCCESS claimed" if args.get("success") else "failure reported")
    return word


def _call_outcome(call: Dict[str, Any]) -> Tuple[str, Tuple[int, int, int]]:
    res = call.get("result") or {}
    if "error" in res:
        return ("refused" if res.get("refused") else "error"), BAD
    if "outcome" in res:
        good = bool(res.get("ok"))
        return res["outcome"].replace("_", " "), (OK if good else BAD)
    if call["name"] == "inspect":
        dis = res.get("disagreements") or []
        return ("camera disagrees: " + ", ".join(dis)) if dis else "checked", (WARN if dis else OK)
    return "ok", MUTED


def _reasoning_per_call(trace: Dict[str, Any]) -> List[str]:
    """The model's reasoning (or message) for each executed tool call, in order."""
    out: List[str] = []
    pending = ""
    for m in trace.get("messages") or []:
        if m.get("role") == "assistant":
            pending = (m.get("reasoning") or m.get("content") or "").strip()
        elif m.get("role") == "tool":
            out.append(pending)
            pending = ""
    return out


def _load_frames(run_dir: str) -> List[np.ndarray]:
    import imageio
    path = os.path.join(run_dir, "video.mp4")
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found: run run_build with --video first")
    reader = imageio.get_reader(path)
    frames = [np.asarray(f) for f in reader]
    reader.close()
    return frames


def annotate(run_dir: str, out_name: str = "video_annotated.mp4", fps: int = 16,
             frames: Optional[Sequence[np.ndarray]] = None, frame_times: Optional[Sequence[float]] = None,
             title: Optional[str] = None) -> str:
    import imageio
    from PIL import Image, ImageDraw

    with open(os.path.join(run_dir, "trace.json"), encoding="utf-8") as f:
        trace = json.load(f)
    frames = list(frames) if frames is not None else _load_frames(run_dir)
    times = list(frame_times) if frame_times is not None else (trace.get("frame_times") or
                                                               [0.25 * k for k in range(len(frames))])
    times = times[:len(frames)]
    calls = trace.get("tool_calls") or []
    ends = [float(c.get("sim_time", 0.0)) for c in calls]
    reasons = _reasoning_per_call(trace)
    disturb = [(float(d.get("sim_time", 0.0)), d.get("label", "disturbance"))
               for d in trace.get("disturbances") or []]
    disturb += [(float(e.get("sim_time", 0.0)), e.get("label", "disturbance")) for e in trace.get("events") or []
                if e.get("type") == "disturbance" and "after" not in e]
    disturb.sort()
    checks = trace.get("visual_checks") or []
    photos: Dict[str, Any] = {}
    for c in checks:
        img = (c.get("verdict") or {}).get("image")
        if img and os.path.exists(os.path.join(run_dir, "inspection", img)):
            with Image.open(os.path.join(run_dir, "inspection", img)) as im:
                photos[img] = im.convert("RGB").resize((300, round(300 * im.height / im.width)))
    session = trace.get("session") or {}
    planner = trace.get("planner", "")
    model = trace.get("model") or ""
    heading = title or f"{session.get('spec', 'harness build')}"
    if planner == "nemotron":
        low = model.lower()
        family = next((f"Nemotron 3 {w.title()}" for w in ("super", "ultra", "nano") if w in low),
                      model.split("/")[-1])
        sub = f"planner: {family}, on Nebius Token Factory"
    else:
        sub = f"planner: {planner}"
    scen = trace.get("scenario") or "nominal"
    t_total = max(times[-1] if times else 1.0, ends[-1] if ends else 1.0, 1.0)
    f_title, f_body, f_small = _font(30), _font(21), _font(17)

    writer = imageio.get_writer(os.path.join(run_dir, out_name), fps=fps, macro_block_size=1, quality=7)

    def draw(frame, t: float, k: int):
        """One video frame at robot time t with call k running (k == len(calls): all done)."""
        canvas = Image.new("RGB", (W, H), BG)
        sim = Image.fromarray(np.asarray(frame)).convert("RGB").resize((SIM_W, SIM_H))
        canvas.paste(sim, (0, 0))
        d = ImageDraw.Draw(canvas)
        # --- bottom band: time line and events
        y0 = SIM_H + 18
        d.text((24, y0), f"robot time {t:5.1f} s", fill=INK, font=f_body)
        x0, x1 = 250, SIM_W - 30
        d.rectangle((x0, y0 + 10, x1, y0 + 22), fill=(220, 220, 216))
        d.rectangle((x0, y0 + 10, x0 + (x1 - x0) * min(1.0, t / t_total), y0 + 22), fill=ACCENT)
        for td, _ in disturb:
            xd = x0 + (x1 - x0) * min(1.0, td / t_total)
            d.rectangle((xd - 2, y0 + 4, xd + 2, y0 + 28), fill=WARN)
        recent = [lab for td, lab in disturb if 0.0 <= t - td < 6.0]
        if recent:
            text = "DISTURBANCE: " + recent[-1]
            tb = d.textbbox((0, 0), text, font=f_body)
            d.rectangle((24, 24, 24 + tb[2] - tb[0] + 32, 76), fill=WARN)
            d.text((40, 36), text, fill=(255, 255, 255), font=f_body)
        # --- right panel
        px = SIM_W + 30
        d.text((px, 24), heading[:34], fill=INK, font=f_title)
        d.text((px, 64), sub[:52], fill=MUTED, font=f_small)
        d.text((px, 88), f"scenario: {scen}", fill=MUTED, font=f_small)
        y = 128
        first = max(0, min(k, len(calls)) - 6)            # the last six done calls + the running one
        if first > 0:
            d.text((px, y), f"... {first} earlier steps", fill=MUTED, font=f_small)
            y += 30
        for i in range(first, min(len(calls), k + 1)):
            c = calls[i]
            label = f"{i + 1:2d}  {_call_label(c)}"
            if i == k:
                d.rectangle((px - 10, y - 4, W - 20, y + 30), fill=(225, 233, 250))
                d.text((px, y), label[:40], fill=ACCENT, font=f_body)
                d.text((W - 130, y + 4), "running", fill=ACCENT, font=f_small)
                y += 40
            else:
                text, col = _call_outcome(c)
                d.text((px, y), label[:40], fill=INK, font=f_body)
                d.text((px + 40, y + 27), text[:48], fill=col, font=f_small)
                y += 58
        # reasoning for the active call
        if k < len(reasons) and reasons[k]:
            ry = max(y + 16, 600)
            d.text((px, ry), "planner's reasoning", fill=MUTED, font=f_small)
            wrapped = textwrap.wrap(reasons[k], 52)[:7]
            for j, line in enumerate(wrapped):
                d.text((px, ry + 26 + 24 * j), line, fill=INK, font=f_small)
        # inspection photos taken so far (latest of each target)
        shown = {}
        for c in checks:
            if float(c.get("sim_time", 0.0)) <= t + 1e-6 and (c.get("verdict") or {}).get("image") in photos:
                shown[c["target"]] = c
        if shown and k >= len(calls):
            d.rectangle((0, 0, SIM_W, SIM_H), fill=(250, 250, 248))
            d.text((30, 20), "camera check: each fixture, two views, verdict by a vision model",
                   fill=INK, font=f_body)
            for j, (target, c) in enumerate(list(shown.items())[:4]):
                v = c["verdict"]
                im = photos[v["image"]].resize((600, round(600 * photos[v["image"]].height
                                                          / photos[v["image"]].width)))
                gx, gy = 30 + (j % 2) * 620, 70 + (j // 2) * (im.height + 70)
                canvas.paste(im, (gx, gy))
                seated = v.get("seated")
                verdict = "no answer" if seated is None else ("seated" if seated else "NOT seated")
                col = OK if seated and c.get("perceived") else (BAD if seated is False else WARN)
                d.text((gx, gy + im.height + 6), f"{target}: camera says {verdict}, perception "
                       f"{'seated' if c.get('perceived') else 'NOT seated'}", fill=col, font=f_small)
                who = (v.get("model") or "").split("/")[-1]
                d.text((gx, gy + im.height + 28), f"({who})", fill=MUTED, font=f_small)
        # final verdict
        if k >= len(calls) and trace.get("claimed_success") is not None:
            ok = bool(trace.get("success"))
            honest = bool(trace.get("claimed_success")) == ok
            d.rectangle((px - 10, H - 120, W - 20, H - 40), fill=OK if ok and honest else BAD)
            d.text((px, H - 110), "BUILD COMPLETE" if ok else "BUILD INCOMPLETE", fill=(255, 255, 255),
                   font=f_title)
            d.text((px, H - 72), "planner's verdict matches the simulator" if honest
                   else "planner's verdict does NOT match the simulator", fill=(255, 255, 255), font=f_small)
        return canvas

    try:
        for frame, t in zip(frames, times):
            # the running call is the first one that ends after t (the calls at the very end,
            # inspect and finish, take no robot time: they show on the closing frames)
            k = next((i for i, e in enumerate(ends) if e > t + 1e-6), len(ends))
            writer.append_data(np.asarray(draw(frame, t, k)))
        if frames:
            last = np.asarray(draw(frames[-1], times[-1], len(calls)))
            for _ in range(3 * fps):                     # hold the verdict for three seconds
                writer.append_data(last)
    finally:
        writer.close()
    return os.path.join(run_dir, out_name)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="a run_build output folder with video.mp4 and trace.json")
    ap.add_argument("--out", default="video_annotated.mp4")
    ap.add_argument("--fps", type=int, default=16)
    args = ap.parse_args(argv)
    print(annotate(args.run_dir, args.out, args.fps))
    return 0


if __name__ == "__main__":
    sys.exit(main())
