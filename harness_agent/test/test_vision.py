"""Visual inspection: verdict parsing, the inspection camera, the evidence gate on finish,
and the accuracy bookkeeping of vision_eval. No network: the model is a stub."""

import json
import os

import numpy as np
import pytest

from harness_agent.llm import LLMError
from harness_agent.session import CellSession
from harness_agent.spec import HarnessSpec
from harness_agent.tools import ToolBox
from harness_agent.vision import (FakeVisionClient, VisualInspector, camera_basis, compose,
                                  fixture_corners, parse_verdict, parse_views, project, to_data_url)
from harness_agent.vision_eval import (breakdown, load_results, metrics, model_slug, parse_seeds,
                                       write_report)

SPECS = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "specs")


def spec(name):
    return HarnessSpec.from_yaml(os.path.join(SPECS, name + ".yaml"))


def can_render():
    try:
        import mujoco
        r = mujoco.Renderer(mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>"), 8, 8)
        r.close()
        return True
    except Exception:
        return False


needs_gl = pytest.mark.skipif(not can_render(), reason="no OpenGL for offscreen rendering")


@pytest.mark.parametrize("text,seated,conf", [
    ('{"seated": true, "confidence": 0.8, "evidence": "wire between the prongs"}', True, 0.8),
    ('```json\n{"seated": false, "confidence": 95, "evidence": "on the lips"}\n```', False, 0.95),
    ('<think>maybe {"seated": true}</think> {"seated": "no", "confidence": "0.7"}', False, 0.7),
    ('Looking closer: {"note": 1} then {"seated": "yes"}', True, 0.5),
    ('The wire is seated.', None, 0.0),
    ('{"seated": "unclear", "confidence": 0.4}', None, 0.4),
])
def test_verdicts_are_parsed_from_imperfect_replies(text, seated, conf):
    got, c, _ = parse_verdict(text)
    assert got is seated
    assert c == pytest.approx(conf)


@pytest.mark.parametrize("text,seated,conf,views", [
    ('{"A": "yes", "B": "yes", "evidence": "wire in the gap"}', True, 1.0, {"A": True, "B": True}),
    ('{"A": "no", "B": "no"}', False, 1.0, {"A": False, "B": False}),
    ('{"view_A": "yes", "view_B": "no"}', False, 0.5, {"A": True, "B": False}),   # views disagree: not seated
    ('{"A": "Yes."}', True, 0.5, {"A": True}),
    ('{"seated": false, "confidence": 0.8}', False, 0.8, {}),                     # a v1-style answer
    ('no idea', None, 0.0, {}),
])
def test_per_view_answers_are_combined_conservatively(text, seated, conf, views):
    got, c, _, v = parse_views(text)
    assert got is seated and c == pytest.approx(conf) and v == views


def test_projection_matches_the_free_camera_convention():
    lookat = np.array([0.5, 0.0, 0.05])
    for az, el in ((0.0, -10.0), (135.0, -45.0), (270.0, -88.0)):
        fwd, right, up = camera_basis(az, el)
        assert np.dot(fwd, right) == pytest.approx(0.0, abs=1e-12) and up[2] > 0.0
        uv = project(lookat, lookat, az, el, 0.15, 45.0, 512, 384)[0]
        assert uv == pytest.approx([256.0, 192.0])              # the look-at point is the image centre
        above = project(lookat + [0, 0, 0.01], lookat, az, el, 0.15, 45.0, 512, 384)[0]
        assert above[1] < 192.0                                  # higher in the world, higher in the image
    corners = fixture_corners("fork", (0.5, 0.0, 0.02, 0.3))
    assert corners.shape == (8, 3) and corners[:, 2].min() == pytest.approx(0.02)


def test_views_are_composed_and_encoded():
    a = np.zeros((384, 512, 3), np.uint8)
    b = np.full((192, 256, 3), 200, np.uint8)              # rescaled to the common width
    im = compose([("along the slot", a), ("oblique", b)], "Fork F2")
    assert im.size == (2 * 512 + 6, 384 + 30)
    assert to_data_url(im).startswith("data:image/jpeg;base64,")
    boxed = np.asarray(compose([("A", a, (100, 100, 200, 200)), ("B", b, None)], "Fork F2"))
    assert tuple(boxed[30 + 150, 100]) == (255, 0, 255)          # the highlight box is drawn in view A
    assert (boxed[30:, 518:] == (255, 0, 255)).all(axis=-1).sum() == 0


def test_inspector_reports_unusable_answers_and_failed_calls():
    img = np.zeros((48, 64, 3), np.uint8)
    v = VisualInspector(FakeVisionClient(lambda p: "I think it looks fine")).ask("fork", "F2", img)
    assert v.seated is None and "no usable JSON" in v.error

    class Down(FakeVisionClient):
        def chat(self, *a, **k):
            raise LLMError("HTTP 503 from x: busy")

    v = VisualInspector(Down()).ask("connector", "X1", img)
    assert v.seated is None and v.error.startswith("vision model call failed")
    stub = FakeVisionClient()
    VisualInspector(stub).ask("fork", "F3", img)
    assert "fork F3" in stub.calls[0]["prompt"] and "orange" in stub.calls[0]["prompt"]
    with pytest.raises(ValueError):
        VisualInspector(stub).ask("clamp", "CL1", img)
    refs = {"fork": [(img, "fork seated"), (img, "fork NOT seated")]}
    fs = FakeVisionClient()
    v = VisualInspector(fs, references=refs).ask("fork", "F1", img)
    assert fs.calls[0]["images"] == 3 and v.seated is True and v.style == "v2refs"
    assert v.prompt_tokens == 100


def test_finish_needs_a_fresh_inspection_showing_everything_seated():
    box = ToolBox(CellSession(spec("demo_3fork"), seed=0))
    out = box.call("finish", {"success": True, "report": "done"})
    assert out.get("refused") == "finish" and "inspect" in out["error"]
    box.call("inspect", {"target": "all"})
    out = box.call("finish", {"success": True, "report": "done"})
    assert "F1, F2, F3, X1 not seated" in out["error"]
    box.call("retreat", {})                                 # a physical skill: evidence goes stale
    assert box.last_inspection is None
    assert box.call("finish", {"success": False, "report": "nothing built"})["recorded"]["success"] is False
    assert box.refused_verdicts == 2


def test_success_verdict_needs_the_arm_out_of_the_camera_view():
    box = ToolBox(CellSession(spec("demo_3fork"), seed=0))
    seated = {"forks": {f: {"wire_in_slot": True} for f in ("F1", "F2", "F3")},
              "connector": {"in_holder": True}}
    box.last_inspection = {**seated, "arm_clear": False}
    assert "retreat" in box.call("finish", {"success": True, "report": "ok"})["error"]
    box.last_inspection = {**seated, "arm_clear": True}
    assert box.call("finish", {"success": True, "report": "ok"})["recorded"]["success"] is True


@needs_gl
def test_inspect_photographs_every_fixture_and_logs_truth(tmp_path):
    said = FakeVisionClient(lambda p: '{"seated": false, "confidence": 0.9, "evidence": "empty slot"}')
    session = CellSession(spec("demo_3fork"), seed=0, inspector=VisualInspector(said),
                          inspection_dir=str(tmp_path))
    out = session.inspect("all")
    assert out["method"] == "perception+vision"
    assert set(out["vision"]) == {"F1", "F2", "F3", "X1"}
    assert out["disagreements"] == []                      # nothing is seated yet: both agree
    assert all(c["truth"] is False and c["verdict"]["seated"] is False for c in session.visual_checks)
    assert len(os.listdir(tmp_path)) == 4
    session.inspect("F1")                                  # a second look uses new views
    assert session.visual_checks[-1]["verdict"]["image"].endswith("F1_view1.jpg")
    # a model that sees everything seated disagrees with perception on every fixture
    session.inspector = VisualInspector(FakeVisionClient())
    assert set(session.inspect("all")["disagreements"]) == {"F1", "F2", "F3", "X1"}
    session.close()


def test_seed_lists_and_slugs():
    assert parse_seeds("0-3,7, 9") == [0, 1, 2, 3, 7, 9]
    assert model_slug("openbmb/MiniCPM-V-4_5") == "openbmb__MiniCPM-V-4_5"


def rows(pairs, kind="fork", difficulty="hard"):
    return [{"truth": t, "pred": p, "kind": kind, "difficulty": difficulty if not t else "seated",
             "confidence": 0.9, "seconds": 1.0} for t, p in pairs]


def test_accuracy_counts_unusable_answers_as_wrong():
    m = metrics(rows([(True, True), (True, False), (False, False), (False, None)]))
    assert m["accuracy"] == pytest.approx(0.5)
    assert m["defect_recall"] == pytest.approx(0.5)        # one defect caught, one missed (no answer)
    assert m["false_alarm_rate"] == pytest.approx(0.5)
    assert m["unusable"] == pytest.approx(0.25)
    assert m["precision"] == pytest.approx(0.5)
    b = breakdown(rows([(True, True)]) + rows([(False, False)], kind="connector"))
    assert b["kind=connector"]["n"] == 1 and b["all"]["accuracy"] == 1.0


def test_report_compares_models_with_the_perception_baseline(tmp_path):
    from PIL import Image
    os.makedirs(tmp_path / "images")
    labels = []
    for k, truth in enumerate((True, False, False)):
        Image.new("RGB", (100, 40), (200, 100, 50)).save(tmp_path / "images" / f"i{k}.jpg")
        labels.append({"id": f"i{k}", "image": f"images/i{k}.jpg", "kind": "fork", "target": "F1", "seed": 0,
                       "stage": "x", "fault": "not_pressed" if k == 2 else None, "view": 0, "truth": truth,
                       "perceived": truth, "difficulty": "seated" if truth else "hard", "detail": {}})
    with open(tmp_path / "labels.jsonl", "w") as f:
        f.writelines(json.dumps(lab) + "\n" for lab in labels)
    results = {"m/one": [{**lab, "model": "m/one", "pred": True, "confidence": 0.8, "evidence": "looks seated",
                          "seconds": 0.5, "error": ""} for lab in labels]}
    text = open(write_report(str(tmp_path), results, {})).read()
    assert "| perception (cable keypoints) | - | 100% | 100% | 0% |" in text
    assert "| `m/one` | v1 | 33% | 0% | 0% |" in text
    assert (tmp_path / "errors_m__one__v1.jpg").exists()
    with open(tmp_path / "results_m__one__v2.jsonl", "w") as f:
        f.writelines(json.dumps({**r, "style": "v2"}) + "\n" for r in results["m/one"])
    assert set(load_results(str(tmp_path))) == {("m/one", "v2")}
