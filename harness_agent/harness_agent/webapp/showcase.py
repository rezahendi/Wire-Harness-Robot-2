"""The showcase: a static copy of Mission Control that replays recorded builds (no server).

    python -m harness_agent.webapp.showcase showcase.json --out site/
    cd site && python -m http.server 8080          # http://localhost:8080

``showcase.json`` names the builds (folders the web app wrote, with events.json and build.json):

    {"title": "Wire Harness Mission Control",
     "intro": "<h2>...</h2><p>...</p>",
     "builds": [{"dir": "runs/webapp/20261007-175405-f86d", "title": "Wire pulled out",
                 "blurb": "Nemotron notices and the expert re-routes it."}, ...]}

The site holds index.html (the app's page in showcase mode), app.js, app.css, vendor/, and
data/: each build's events (slimmed: the fields the page draws), its camera-check images,
results.json and the harness drawings. ``page.html`` is the same page as one file with the
styles and scripts inlined and three.js from a CDN, for hosts that wrap a page in their own
<html> skeleton; it reads the same data/ folder.
"""

import argparse
import json
import os
import shutil
import sys
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(os.path.realpath(__file__))
STATIC = os.path.join(HERE, "static")
THREE_CDN = "https://cdn.jsdelivr.net/npm/three@0.147.0"
FRAME_KEYS = ("t", "pose", "tcp", "wrench", "grip", "phase", "call", "truth", "mode")


def slim_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The events with only what the page uses (about a third smaller)."""
    out = []
    for e in events:
        kind = e.get("type")
        if kind == "frame":
            f = {k: e[k] for k in FRAME_KEYS if k in e}
            if e.get("pokes"):
                f["pokes"] = e["pokes"]
            if e.get("log"):
                f["log"] = e["log"]
            out.append({"type": "frame", **f})
            continue
        e = {k: v for k, v in e.items() if k != "seq"}
        if kind == "done":
            e.pop("files", None)
        if kind == "plan":
            e = {k: e[k] for k in ("type", "turn", "model", "content", "reasoning") if k in e}
        out.append(e)
    return out


def spec_file(build: Dict[str, Any]) -> str:
    return os.path.basename(str((build.get("options") or {}).get("spec") or ""))


def build_entry(build: Dict[str, Any], item: Dict[str, Any]) -> Dict[str, Any]:
    """What the page lists for a build: no file paths or server addresses."""
    o = build.get("options") or {}
    s = build.get("summary") or {}
    return {"id": build["id"], "status": build.get("status", "done"), "created": build.get("created", 0),
            "title": item.get("title", ""), "blurb": item.get("blurb", ""),
            "options": {"planner": o.get("planner"), "groot": "GR00T" if o.get("groot") is not None else None,
                        "scenario": o.get("scenario", "nominal"), "seed": o.get("seed"), "vision": bool(o.get("vision")),
                        "spec": spec_file(build)},
            "summary": {k: s.get(k) for k in ("spec", "planner", "model", "success", "claimed_success", "robot_time_s",
                                              "tool_calls", "routing", "scenario", "seed")}}


def page_html(static_cfg: Dict[str, Any], inline: bool) -> str:
    """The app's page in showcase mode: a whole document (inline=False) or one skeleton-free file."""
    with open(os.path.join(STATIC, "index.html"), encoding="utf-8") as f:
        doc = f.read()
    body = doc[doc.index("<body"):doc.index("</body>")]
    body = body[body.index(">") + 1:]
    config = ("<script>window.MC_STATIC = " + json.dumps(static_cfg, separators=(",", ":")).replace("</", "<\\/")
              + ";</script>\n")
    fonts = doc[doc.index('<link rel="preconnect"'):doc.index('<link rel="stylesheet" href="/static/app.css">')]
    title = static_cfg.get("title") or "Wire Harness Mission Control"
    desc = static_cfg.get("description", "")
    if not inline:
        body = body.replace('src="/static/', 'src="')
        body = body.replace('<script src="app.js"></script>', config + '<script src="app.js"></script>')
        return ("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
                "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
                f"<title>{title}</title>\n<meta name=\"description\" content=\"{desc}\">\n{fonts}"
                "<link rel=\"stylesheet\" href=\"app.css\">\n</head>\n<body data-view=\"build\">" + body + "</body>\n</html>\n")
    with open(os.path.join(STATIC, "app.css"), encoding="utf-8") as f:
        css = f.read()
    with open(os.path.join(STATIC, "app.js"), encoding="utf-8") as f:
        js = f.read()
    scripts = (f'<script src="{THREE_CDN}/build/three.min.js"></script>\n'
               f'<script src="{THREE_CDN}/examples/js/controls/OrbitControls.js"></script>\n'
               + config + "<script>\n" + js.replace("</script", "<\\/script") + "\n</script>\n")
    start = body.index('<script src="/static/vendor/three.min.js">')
    body = body[:start] + scripts
    return f"<title>{title}</title>\n{fonts}<style>\n{css}\n</style>\n" + body


def build_site(manifest: Dict[str, Any], out: str, base: str = ".") -> Dict[str, Any]:
    from ..drawing import render_drawing
    from ..spec import HarnessSpec
    from .server import packaged_specs, spec_info

    data = os.path.join(out, "data")
    os.makedirs(os.path.join(out, "vendor"), exist_ok=True)
    os.makedirs(data, exist_ok=True)
    for name in ("app.js", "app.css"):
        shutil.copy(os.path.join(STATIC, name), os.path.join(out, name))
    for name in os.listdir(os.path.join(STATIC, "vendor")):
        shutil.copy(os.path.join(STATIC, "vendor", name), os.path.join(out, "vendor", name))
    shutil.copy(os.path.join(HERE, "results.json"), os.path.join(data, "results.json"))

    specs = {os.path.basename(p): p for p in packaged_specs()}
    entries, spec_names, sizes = [], [], {}
    for item in manifest["builds"]:
        src = os.path.join(base, item["dir"])
        with open(os.path.join(src, "build.json"), encoding="utf-8") as f:
            build = json.load(f)
        with open(os.path.join(src, "events.json"), encoding="utf-8") as f:
            events = slim_events(json.load(f))
        path = os.path.join(data, build["id"] + ".json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(events, f, separators=(",", ":"))
        sizes[build["id"]] = os.path.getsize(path)
        shots = os.path.join(src, "inspection")
        if os.path.isdir(shots):
            dst = os.path.join(data, build["id"], "inspection")
            os.makedirs(dst, exist_ok=True)
            for name in sorted(os.listdir(shots)):
                if name.lower().endswith((".jpg", ".jpeg", ".png")):
                    shutil.copy(os.path.join(shots, name), os.path.join(dst, name))
        entries.append(build_entry(build, item))
        if spec_file(build) not in spec_names:
            spec_names.append(spec_file(build))

    infos, drawings = [], {}
    for name in spec_names:
        if name not in specs:
            continue
        infos.append(spec_info(specs[name]))
        png = "drawing_" + os.path.splitext(name)[0] + ".png"
        render_drawing(HarnessSpec.from_yaml(specs[name]), os.path.join(data, png))
        drawings[name] = "data/" + png

    cfg = {"title": manifest.get("title", "Wire Harness Mission Control"), "description": manifest.get("description", ""),
           "intro": manifest.get("intro", ""), "chip": manifest.get("chip", "Recorded builds"),
           "buildsTitle": manifest.get("buildsTitle", "Recorded builds"), "data": "data/",
           "results": "data/results.json", "specs": infos, "drawings": drawings, "builds": entries}
    with open(os.path.join(out, "index.html"), "w", encoding="utf-8") as f:
        f.write(page_html(cfg, inline=False))
    with open(os.path.join(out, "page.html"), "w", encoding="utf-8") as f:
        f.write(page_html(cfg, inline=True))
    return {"builds": len(entries), "bytes": sizes, "out": os.path.abspath(out)}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest", help="showcase.json")
    ap.add_argument("--out", default="site")
    args = ap.parse_args(argv)
    with open(args.manifest, encoding="utf-8") as f:
        manifest = json.load(f)
    res = build_site(manifest, args.out, base=os.path.dirname(os.path.abspath(args.manifest)))
    total = sum(res["bytes"].values())
    print(f"{res['builds']} builds, {total / 1e6:.1f} MB of events -> {res['out']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
