"""Mission control for the wire-harness cell: a small web server and a single-page app.

    pip install fastapi "uvicorn[standard]"
    python -m harness_agent.webapp                       # http://127.0.0.1:8000
    python -m harness_agent.webapp --groot 127.0.0.1:5556 --runs runs/webapp

Pick a harness, check it, start a build with the scripted planner or Nemotron (Token
Factory key in NEBIUS_API_KEY), optionally with GR00T routing the wire (a GR00T policy
server, e.g. on the GPU VM through ``ssh -L 5556:localhost:5556 ...``) and the camera
check; watch it live in 3D with every decision, skill and force; open past builds again.
"""

# No ``from __future__ import annotations`` here: FastAPI reads the endpoint annotations at
# runtime, and WebSocket is imported inside create_app (FastAPI is optional for the package).
import argparse
import asyncio
import glob
import json
import os
import sys
import tempfile
from dataclasses import fields
from typing import Any, Dict, List, Optional

from .builds import BuildManager, BuildOptions

HERE = os.path.dirname(os.path.realpath(__file__))
STATIC = os.path.join(HERE, "static")


def packaged_specs() -> List[str]:
    dirs = [os.path.join(os.path.dirname(os.path.dirname(HERE)), "specs")]
    try:
        from ament_index_python.packages import get_package_share_directory
        dirs.append(os.path.join(get_package_share_directory("harness_agent"), "specs"))
    except Exception:
        pass
    for d in dirs:
        found = sorted(glob.glob(os.path.join(d, "*.yaml")))
        if found:
            return found
    return []


def spec_info(path: str) -> Dict[str, Any]:
    from ..spec import HarnessSpec, has_errors, validate
    spec = HarnessSpec.from_yaml(path)
    issues = validate(spec)
    wire = spec.wires[0] if spec.wires else None
    return {"file": os.path.basename(path), "name": spec.name, "revision": spec.revision,
            "route": list(wire.route) if wire else [], "connector": wire.end if wire else "",
            "feasible": not has_errors(issues),
            "issues": [{"severity": i.severity, "message": i.message} for i in issues]}


def results_file() -> Dict[str, Any]:
    path = os.path.join(HERE, "results.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def create_app(runs: str = "runs/webapp", groot: Optional[str] = None):
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(title="Harness cell mission control")
    manager = BuildManager(runs)
    specs = {os.path.basename(p): p for p in packaged_specs()}
    upload_dir = os.path.join(runs, "_specs")
    drawings = tempfile.mkdtemp(prefix="harness_drawings_")

    def spec_path(name: str) -> str:
        if name in specs:
            return specs[name]
        p = os.path.join(upload_dir, os.path.basename(name))
        if os.path.exists(p):
            return p
        raise HTTPException(404, f"no spec {name!r}")

    @app.get("/")
    def index():
        return FileResponse(os.path.join(STATIC, "index.html"))

    @app.get("/api/status")
    def status():
        reach = None
        if groot is not None:
            from ..groot_client import DEFAULT_PORT, GrootClient
            host, _, port = (groot or f"127.0.0.1:{DEFAULT_PORT}").rpartition(":")
            c = GrootClient(host or "127.0.0.1", int(port), timeout_ms=1500)
            reach = c.ping()
            c.close()
        return {"nebius_key": bool(os.environ.get("NEBIUS_API_KEY")), "groot": groot,
                "groot_reachable": reach, "busy": manager.busy(), "runs": os.path.abspath(runs),
                "groot_setup": {"ensemble": BuildOptions.groot_ensemble, "seat_assist": BuildOptions.groot_seat_assist,
                                "trained_on": "demo_3fork.yaml"}}

    @app.get("/api/specs")
    def list_specs():
        out = []
        uploaded = sorted(n for n in (os.listdir(upload_dir) if os.path.isdir(upload_dir) else [])
                          if n.endswith((".yaml", ".yml")) and n not in specs)
        for name in sorted(specs) + uploaded:
            try:
                info = spec_info(spec_path(name))
            except Exception as exc:                      # a broken upload must not hide the rest
                info = {"file": name, "name": name, "error": str(exc), "feasible": False, "issues": []}
            out.append({**info, "uploaded": name in uploaded})
        return out

    @app.post("/api/specs")
    def upload_spec(body: Dict[str, Any]):
        text = str(body.get("yaml") or "")
        name = os.path.basename(str(body.get("file") or "uploaded.yaml"))
        if not name.endswith((".yaml", ".yml")):
            name += ".yaml"
        os.makedirs(upload_dir, exist_ok=True)
        path = os.path.join(upload_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        try:
            return {**spec_info(path), "uploaded": True}
        except Exception as exc:
            os.remove(path)
            raise HTTPException(400, f"not a harness spec: {str(exc).replace(path, name)}")

    @app.get("/api/specs/{name}/drawing.png")
    def drawing(name: str):
        from ..drawing import render_drawing
        from ..spec import HarnessSpec
        out = os.path.join(drawings, name + ".png")
        if not os.path.exists(out):
            render_drawing(HarnessSpec.from_yaml(spec_path(name)), out)
        return FileResponse(out, media_type="image/png")

    @app.post("/api/builds")
    def start_build(body: Dict[str, Any]):
        if manager.busy():
            raise HTTPException(409, "a build is running; wait for it to finish")
        allowed = {f.name for f in fields(BuildOptions)}
        opts = {k: v for k, v in body.items() if k in allowed}
        opts["spec"] = spec_path(str(body.get("spec") or ""))
        if body.get("use_groot"):
            opts["groot"] = groot if groot is not None else ""
        else:
            opts["groot"] = None
        build = manager.start(BuildOptions(**opts))
        return build.info()

    @app.get("/api/builds")
    def list_builds():
        live = sorted((b.info() for b in manager.builds.values()), key=lambda b: -b["created"])
        return live + manager.past()

    def build_dir(build_id: str) -> str:
        d = os.path.join(runs, os.path.basename(build_id))
        if not os.path.isdir(d):
            raise HTTPException(404, f"no build {build_id!r}")
        return d

    @app.get("/api/builds/{build_id}")
    def build_info(build_id: str):
        if build_id in manager.builds:
            return manager.builds[build_id].info()
        with open(os.path.join(build_dir(build_id), "build.json"), encoding="utf-8") as f:
            return json.load(f)

    @app.get("/api/builds/{build_id}/events")
    def build_events(build_id: str):
        if build_id in manager.builds:          # this server's builds: from memory (the file may still be written)
            return JSONResponse(manager.builds[build_id].events)
        return FileResponse(os.path.join(build_dir(build_id), "events.json"), media_type="application/json")

    @app.get("/api/builds/{build_id}/files/{path:path}")
    def build_file(build_id: str, path: str):
        root = os.path.realpath(build_dir(build_id))
        full = os.path.realpath(os.path.join(root, path))
        if not full.startswith(root + os.sep) or not os.path.isfile(full):
            raise HTTPException(404, "no such file")
        return FileResponse(full)

    @app.get("/api/results")
    def results():
        return results_file()

    @app.websocket("/ws/builds/{build_id}")
    async def build_stream(ws: WebSocket, build_id: str):
        await ws.accept()
        build = manager.builds.get(build_id)
        if build is None:
            await ws.send_json({"type": "failed", "error": f"no running build {build_id}"})
            await ws.close()
            return
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()

        def push(ev: Dict[str, Any]) -> None:
            loop.call_soon_threadsafe(q.put_nowait, ev)

        backlog = build.subscribe(push)
        try:
            for ev in backlog:
                await ws.send_text(json.dumps(ev, default=str))
            if build.status in ("done", "failed") and backlog and backlog[-1]["type"] in ("done", "failed"):
                return
            while True:
                ev = await q.get()
                await ws.send_text(json.dumps(ev, default=str))
                if ev.get("type") in ("done", "failed"):
                    break
        except WebSocketDisconnect:
            pass
        finally:
            build.unsubscribe(push)
            try:
                await ws.close()
            except Exception:
                pass

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--runs", default="runs/webapp", help="where builds are kept")
    ap.add_argument("--groot", nargs="?", const="", default=None, metavar="HOST:PORT",
                    help="offer GR00T routing from this policy server (default 127.0.0.1:5556)")
    args = ap.parse_args(argv)
    try:
        import uvicorn
    except ImportError:
        raise SystemExit('the web app needs: pip install fastapi "uvicorn[standard]"')
    from harness_core.render_util import choose_gl_backend
    choose_gl_backend()
    app = create_app(args.runs, args.groot)
    print(f"mission control on http://{args.host}:{args.port}  (builds in {os.path.abspath(args.runs)})", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
