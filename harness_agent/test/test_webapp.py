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
    assert [h["key"] for h in res["headline"]] == ["system", "hybrid", "assist"]
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
