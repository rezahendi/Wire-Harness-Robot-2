"""The web app's server: its API, the live stream of a build and the replay files.

Needs fastapi and httpx (FastAPI's test client); skipped without them."""

import base64
import json
import os

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from harness_agent.webapp.server import create_app  # noqa: E402

SPECS = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "specs")
SLOW = os.environ.get("HARNESS_SLOW_TESTS") == "1"


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(str(tmp_path / "runs"))) as c:
        yield c


def test_status_specs_and_results(client):
    st = client.get("/api/status").json()
    assert st["groot"] is None and st["busy"] is False and st["groot_setup"]["seat_assist"] == 1.5
    specs = {s["file"]: s for s in client.get("/api/specs").json()}
    assert specs["demo_3fork.yaml"]["feasible"] and specs["demo_3fork.yaml"]["route"] == ["F1", "F2", "F3"]
    assert not specs["demo_infeasible.yaml"]["feasible"]
    res = client.get("/api/results").json()
    assert [h["key"] for h in res["headline"]] == ["system", "hybrid", "planner"]
    assert all(r["k"] <= r["n"] for r in res["rounds"])
    page = client.get("/")
    assert page.status_code == 200 and "Mission Control" in page.text
    assert client.get("/static/app.js").status_code == 200


def test_a_spec_can_be_uploaded_and_drawn(client):
    with open(os.path.join(SPECS, "demo_2fork_stiff.yaml"), encoding="utf-8") as f:
        text = f.read()
    info = client.post("/api/specs", json={"file": "mine.yaml", "yaml": text}).json()
    assert info["file"] == "mine.yaml" and info["feasible"]
    assert "mine.yaml" in [s["file"] for s in client.get("/api/specs").json()]
    png = client.get("/api/specs/mine.yaml/drawing.png")
    assert png.status_code == 200 and png.content[:4] == b"\x89PNG"
    bad = client.post("/api/specs", json={"file": "broken", "yaml": "name: [unclosed"})
    assert bad.status_code == 400
    assert "broken.yaml" not in [s["file"] for s in client.get("/api/specs").json()]
    assert client.get("/api/specs/nope.yaml/drawing.png").status_code == 404


def test_the_live_stream_answers_for_an_unknown_build(client):
    # the websocket route must be reachable (it was refused with 403 when FastAPI could not read
    # its annotations)
    with client.websocket_connect("/ws/builds/nope") as ws:
        ev = ws.receive_json()
    assert ev["type"] == "failed" and "nope" in ev["error"]
    assert client.get("/api/builds/nope").status_code == 404
    assert client.get("/api/builds").json() == []


def test_build_files_stay_inside_the_build_folder(client, tmp_path):
    d = tmp_path / "runs" / "20260101-000000-abcd"
    d.mkdir(parents=True)
    (d / "report.md").write_text("hello")
    (d / "build.json").write_text(json.dumps({"id": d.name, "status": "done", "created": 1.0, "options": {},
                                              "summary": {}, "events": 0}))
    (d / "events.json").write_text("[]")
    assert client.get(f"/api/builds/{d.name}/files/report.md").text == "hello"
    assert client.get(f"/api/builds/{d.name}/files/..%2F..%2Fsecret").status_code == 404
    assert [b["id"] for b in client.get("/api/builds").json()] == [d.name]
    assert client.get(f"/api/builds/{d.name}/events").json() == []


@pytest.mark.skipif(not SLOW, reason="runs a whole scripted build (~1-2 min); set HARNESS_SLOW_TESTS=1")
def test_a_scripted_build_streams_live_and_can_be_replayed(client):
    info = client.post("/api/builds", json={"spec": "demo_3fork.yaml", "planner": "scripted", "seed": 3,
                                            "pace": 0, "use_groot": False}).json()
    assert info["status"] in ("queued", "running")
    assert client.post("/api/builds", json={"spec": "demo_3fork.yaml"}).status_code == 409    # one at a time
    events = []
    with client.websocket_connect(f"/ws/builds/{info['id']}") as ws:
        while True:
            ev = ws.receive_json()
            events.append(ev)
            if ev["type"] in ("done", "failed"):
                break
    kinds = [e["type"] for e in events]
    assert kinds[0] == "scene" and kinds[-1] == "done", events[-1]
    assert kinds.count("frame") > 500 and kinds.count("call_start") == kinds.count("call_end") >= 6
    assert [e["seq"] for e in events] == list(range(len(events)))
    done = events[-1]
    assert done["success"] and done["truth"]["connector_seated"] and "report.md" in done["files"]
    frame = next(e for e in events if e["type"] == "frame")
    scene = events[0]
    assert len(base64.b64decode(frame["pose"])) == len(scene["moving"]) * 7 * 2      # 7 int16 per moving body
    saved = client.get(f"/api/builds/{info['id']}/events").json()
    assert len(saved) == len(events)
    past = client.get("/api/builds").json()
    assert past[0]["id"] == info["id"] and past[0]["summary"]["success"]


def test_the_showcase_is_a_static_copy_without_paths(tmp_path):
    from harness_agent.webapp.showcase import build_site
    run = tmp_path / "runs" / "b1"
    run.mkdir(parents=True)
    events = [
        {"seq": 0, "type": "scene", "geoms": []},
        {"seq": 1, "type": "frame", "t": 0.5, "pose": [0, 0, 0], "tcp": [0, 0, 0.2], "wrench": [0, 0, 1, 0, 0, 0],
         "grip": 0.06, "phase": "idle", "call": None, "truth": {}, "mode": 0, "debug": {"big": [1] * 50}},
        {"seq": 2, "type": "plan", "turn": 1, "model": "m", "content": "route F1", "reasoning": "why",
         "raw": {"usage": 1}},
        {"seq": 3, "type": "done", "success": True, "files": ["/home/someone/runs/b1/report.md"]},
    ]
    (run / "events.json").write_text(json.dumps(events))
    (run / "build.json").write_text(json.dumps({
        "id": "b1", "status": "done", "created": 1.0,
        "options": {"spec": "/home/someone/specs/demo_3fork.yaml", "planner": "nemotron",
                    "groot": "127.0.0.1:5556", "seed": 3, "scenario": "nominal"},
        "summary": {"success": True, "routing": {"groot_routes": 3}}}))
    (run / "inspection").mkdir()
    (run / "inspection" / "F1.jpg").write_bytes(b"\xff\xd8\xff")
    manifest = {"title": "Showcase", "intro": "<h2>Hi</h2>",
                "builds": [{"dir": "runs/b1", "title": "First", "blurb": "GR00T routes"}]}
    site = tmp_path / "site"
    res = build_site(manifest, str(site), base=str(tmp_path))
    assert res["builds"] == 1

    slim = json.loads((site / "data" / "b1.json").read_text())
    frame, plan, done = slim[1], slim[2], slim[3]
    assert "seq" not in frame and "debug" not in frame and frame["tcp"] == [0, 0, 0.2]
    assert set(plan) == {"type", "turn", "model", "content", "reasoning"}
    assert "files" not in done and done["success"] is True
    assert (site / "data" / "b1" / "inspection" / "F1.jpg").exists()
    assert (site / "data" / "drawing_demo_3fork.png").read_bytes()[:4] == b"\x89PNG"
    assert (site / "data" / "results.json").exists() and (site / "vendor" / "three.min.js").exists()

    for name in ("index.html", "page.html"):
        text = (site / name).read_text()
        assert "window.MC_STATIC" in text and "First" in text
        assert "/home/someone" not in text and "127.0.0.1" not in text
    page = (site / "page.html").read_text()
    assert "<body" not in page and "<html" not in page and "cdn.jsdelivr.net/npm/three@0.147.0" in page
    index = (site / "index.html").read_text()
    assert index.startswith("<!doctype html>") and 'src="app.js"' in index and "/static/" not in index
