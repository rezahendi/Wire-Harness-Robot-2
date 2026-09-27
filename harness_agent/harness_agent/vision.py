"""Visual inspection: a camera looks at each fixture, a vision-language model judges it.

Perception (the tracked cable keypoints and the connector pose) already tells the planner
whether a fork holds the wire. A real formboard is signed off by looking at it, so the
build agent gets a second, independent check: ``InspectionCamera`` renders a fixture from
two angles (the robot's-eye view a wrist camera would have), and ``VisualInspector`` asks
a vision-language model on Nebius Token Factory one narrow question about it, answered in
JSON with a confidence and a sentence of evidence.

Verdicts are logged next to the simulator's ground truth, so the accuracy of the visual
check is measured rather than assumed: see ``vision_eval``. The inspector takes any
image, so the same questions work on photos of a real board.
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .llm import LLMError, TokenFactoryClient

ImageLike = Union[np.ndarray, Any]          # an RGB array or a PIL image (PIL is imported lazily)

# Viewpoints (name, azimuth offset from the fixture axis in deg, elevation in deg, distance
# in m). Repeated inspections of the same fixture step through the variants, so a second
# look really is a second look.
FORK_VIEWS: Tuple[Tuple[Tuple[str, float, float, float], ...], ...] = (
    (("along the slot", 0.0, -28.0, 0.15), ("oblique", 45.0, -42.0, 0.15)),
    (("along the slot, low", 0.0, -18.0, 0.13), ("oblique, other side", -45.0, -38.0, 0.15)),
    (("across the slot", 90.0, -35.0, 0.14), ("steep", 25.0, -62.0, 0.15)),
)
HOLDER_VIEWS: Tuple[Tuple[Tuple[str, float, float, float], ...], ...] = (
    (("side", 60.0, -38.0, 0.17), ("top", 0.0, -88.0, 0.17)),
    (("along the pocket", 0.0, -35.0, 0.16), ("top, closer", 0.0, -88.0, 0.13)),
    (("other side", -60.0, -40.0, 0.17), ("steep", 30.0, -65.0, 0.15)),
)

SIM_APPEARANCE = {
    "fork": "Here the fork is the blue U-shaped clip with two yellow knobs (the snap lips) at the "
            "top of its prongs, and the wire is the orange cable.",
    "connector": "Here the holder is the green pocket on the board and the connector is the black "
                 "block at the end of the orange wire.",
}

FORK_QUESTION = """You are the quality inspector of a wire-harness assembly cell. The image shows fork {target}, a snap-in cable clip on the formboard, from two angles (A and B). {appearance}

Question: is the wire seated in this fork?
Seated: the wire passes through the fork's slot, between its two prongs and below the snap lips at the top.
Not seated: anything else. The wire lies on the board next to the fork, rests on top of the lips, or does not come near the fork.

Use both views. Reply with JSON only:
{{"seated": true or false, "confidence": 0.0 to 1.0, "evidence": "one short sentence on what you see"}}"""

CONNECTOR_QUESTION = """You are the quality inspector of a wire-harness assembly cell. The image shows the holder for connector {target} from two angles (A and B). {appearance}

Question: is the connector seated in its holder?
Seated: the connector lies flat and fully inside the holder's pocket, between its end walls.
Not seated: anything else. The connector lies outside the pocket, sits on a wall or rail, is tilted or stands on end, or is not in view.

Use both views. Reply with JSON only:
{{"seated": true or false, "confidence": 0.0 to 1.0, "evidence": "one short sentence on what you see"}}"""

QUESTIONS = {"fork": FORK_QUESTION, "connector": CONNECTOR_QUESTION}


# ------------------------------------------------------------------ camera
class InspectionCamera:
    """A virtual inspection camera aimed at one fixture at a time.

    It stands on the robot's side of the fixture and looks outwards, the way a camera on
    the robot would, so a parked arm stays behind it. On a real cell this is the wrist
    camera or a pan-tilt camera over the board.
    """

    def __init__(self, sim, width: int = 512, height: int = 384,
                 base_xy: Sequence[float] = (0.0, 0.0)):
        import mujoco
        self._mj = mujoco
        self.sim = sim
        self.width, self.height = width, height
        self.base_xy = np.asarray(base_xy, dtype=float)
        self.renderer = mujoco.Renderer(sim.model, height, width)

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def shot(self, lookat: Sequence[float], azimuth_deg: float, elevation_deg: float,
             distance: float) -> np.ndarray:
        cam = self._mj.MjvCamera()
        cam.type = self._mj.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = np.asarray(lookat, dtype=float)
        cam.azimuth = float(azimuth_deg)
        cam.elevation = float(elevation_deg)
        cam.distance = float(distance)
        self.renderer.update_scene(self.sim.data, camera=cam)
        return self.renderer.render().copy()

    def _outward(self, xy: np.ndarray, az_deg: float) -> float:
        """az or az + 180, whichever looks away from the robot base."""
        bearing = math.atan2(xy[1] - self.base_xy[1], xy[0] - self.base_xy[0])
        return az_deg if math.cos(math.radians(az_deg) - bearing) >= 0.0 else az_deg + 180.0

    def fork_views(self, fork_pose: Sequence[float], fork_cfg, variant: int = 0) -> List[Tuple[str, np.ndarray]]:
        x, y, z, yaw = (float(v) for v in fork_pose)
        look = np.array([x, y, z + fork_cfg.post_height + 0.006])
        out = []
        for name, d_az, el, dist in FORK_VIEWS[variant % len(FORK_VIEWS)]:
            az = self._outward(np.array([x, y]), math.degrees(yaw) + d_az)
            out.append((name, self.shot(look, az, el, dist)))
        return out

    def holder_views(self, holder_pos: Sequence[float], holder_yaw: float,
                     variant: int = 0) -> List[Tuple[str, np.ndarray]]:
        hp = np.asarray(holder_pos, dtype=float)
        look = np.array([hp[0], hp[1], hp[2] + 0.006])
        out = []
        for name, d_az, el, dist in HOLDER_VIEWS[variant % len(HOLDER_VIEWS)]:
            az = self._outward(hp[:2], math.degrees(holder_yaw) + d_az)
            out.append((name, self.shot(look, az, el, dist)))
        return out


def compose(views: Sequence[Tuple[str, ImageLike]], title: str, width: int = 512):
    """Views side by side under a title strip; each view is labelled A, B, ..."""
    from PIL import Image, ImageDraw

    ims = []
    for _, v in views:
        im = v if hasattr(v, "size") and not isinstance(v, np.ndarray) else Image.fromarray(np.asarray(v))
        im = im.convert("RGB")
        if im.width != width:
            im = im.resize((width, max(1, round(im.height * width / im.width))))
        ims.append(im)
    strip, gap = 30, 6
    h = max(im.height for im in ims)
    canvas = Image.new("RGB", (len(ims) * width + (len(ims) - 1) * gap, h + strip), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    font = _font(17)
    small = _font(15)
    draw.text((8, 6), title, fill=(20, 20, 20), font=font)
    for k, ((name, _), im) in enumerate(zip(views, ims)):
        x0 = k * (width + gap)
        canvas.paste(im, (x0, strip))
        label = f"{chr(65 + k)}: {name}"
        box = draw.textbbox((0, 0), label, font=small)
        draw.rectangle((x0 + 4, strip + 4, x0 + 12 + box[2] - box[0], strip + 10 + box[3] - box[1]),
                       fill=(255, 255, 255))
        draw.text((x0 + 8, strip + 5), label, fill=(20, 20, 20), font=small)
    return canvas


def _font(size: int):
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                      # Pillow < 10.1 has one bitmap size only
        return ImageFont.load_default()


def to_data_url(image: ImageLike, quality: int = 88) -> str:
    from PIL import Image
    im = image if not isinstance(image, np.ndarray) else Image.fromarray(image)
    buf = io.BytesIO()
    im.convert("RGB").save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# ----------------------------------------------------------------- verdicts
@dataclass
class VisualVerdict:
    kind: str                         # "fork" or "connector"
    target: str
    seated: Optional[bool]            # None: no usable answer
    confidence: float
    evidence: str
    model: str
    seconds: float
    view: int = 0                     # viewpoint variant used
    image: str = ""                   # where the composite was saved, if it was
    error: str = ""
    raw: str = ""

    def for_planner(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"seated": self.seated, "confidence": round(self.confidence, 2),
                               "evidence": self.evidence}
        if self.error:
            out["error"] = self.error
        return out

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


_OBJ_RE = re.compile(r"\{[^{}]*\}", re.S)
_TRUE = {"true", "yes", "seated", "y", "1"}
_FALSE = {"false", "no", "not seated", "not_seated", "unseated", "n", "0"}


def parse_verdict(text: str) -> Tuple[Optional[bool], float, str]:
    """(seated, confidence, evidence) from a model reply; seated is None if unusable."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    text = text.replace("```json", "```").replace("```", " ")
    for m in reversed(list(_OBJ_RE.finditer(text))):       # the last JSON object is the answer
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or "seated" not in obj:
            continue
        raw = obj.get("seated")
        if isinstance(raw, bool):
            seated: Optional[bool] = raw
        else:
            key = str(raw).strip().lower()
            seated = True if key in _TRUE else False if key in _FALSE else None
        try:
            conf = float(obj.get("confidence", 0.5))
        except (TypeError, ValueError):
            conf = 0.5
        if conf > 1.0:                                     # a percentage
            conf /= 100.0
        return seated, float(np.clip(conf, 0.0, 1.0)), str(obj.get("evidence", ""))[:300]
    return None, 0.0, ""


class VisualInspector:
    """Asks a vision-language model whether a fixture is right.

    The model runs on Token Factory by default. To use one served elsewhere through an
    OpenAI-compatible endpoint (say, a model served with vLLM on a Nebius GPU VM), set
    HARNESS_VISION_BASE_URL (and HARNESS_VISION_API_KEY if it wants one); the planner stays
    on Token Factory.
    """

    def __init__(self, client: Optional[Any] = None, model: Optional[str] = None,
                 appearance: Optional[Dict[str, str]] = None, max_tokens: int = 700,
                 temperature: float = 0.0, extra: Optional[Dict[str, Any]] = None):
        base_url = os.environ.get("HARNESS_VISION_BASE_URL")
        if base_url:
            client = TokenFactoryClient(api_key=os.environ.get("HARNESS_VISION_API_KEY") or "none",
                                        base_url=base_url)
        self.client = client or TokenFactoryClient()
        self.model = model or self.client.vision_model()
        self.appearance = SIM_APPEARANCE if appearance is None else appearance
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.extra = extra or {}

    def prompt(self, kind: str, target: str) -> str:
        return QUESTIONS[kind].format(target=target, appearance=self.appearance.get(kind, "")).replace(
            "  ", " ")

    def ask(self, kind: str, target: str, image: ImageLike, view: int = 0) -> VisualVerdict:
        if kind not in QUESTIONS:
            raise ValueError(f"unknown inspection kind {kind!r}")
        content = [{"type": "text", "text": self.prompt(kind, target)},
                   {"type": "image_url", "image_url": {"url": to_data_url(image)}}]
        t0 = time.perf_counter()
        try:
            msg = self.client.chat(self.model, [{"role": "user", "content": content}],
                                   temperature=self.temperature, max_tokens=self.max_tokens, **self.extra)
        except LLMError as exc:
            return VisualVerdict(kind, target, None, 0.0, "", self.model, time.perf_counter() - t0, view,
                                 error=f"vision model call failed: {str(exc)[:200]}")
        dt = time.perf_counter() - t0
        text = msg.get("content") or ""
        seated, conf, evidence = parse_verdict(text)
        error = ""
        if seated is None:
            error = ("no answer (the model spent its tokens reasoning)" if not text.strip() and msg.get("reasoning")
                     else "no usable JSON verdict in the reply")
        return VisualVerdict(kind, target, seated, conf, evidence, self.model, dt, view, error=error,
                             raw=text[:600])


class FakeVisionClient:
    """Test double for Token Factory: answers inspection questions with ``answer(prompt)``.

    ``answer`` returns the reply text; the default says every fixture is seated.
    """

    def __init__(self, answer: Optional[Callable[[str], str]] = None):
        self.answer = answer or (lambda prompt: '{"seated": true, "confidence": 0.9, "evidence": "stub"}')
        self.calls: List[Dict[str, Any]] = []
        self.usage = type("U", (), {"as_dict": staticmethod(lambda: {"calls": 0})})()

    def vision_model(self) -> str:
        return "fake-vlm"

    def chat(self, model: str, messages: List[Dict[str, Any]], **_: Any) -> Dict[str, Any]:
        prompt = messages[-1]["content"][0]["text"]
        self.calls.append({"model": model, "prompt": prompt})
        return {"role": "assistant", "content": self.answer(prompt), "tool_calls": [], "reasoning": ""}
