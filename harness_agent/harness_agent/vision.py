"""Visual inspection: a camera looks at each fixture, a vision-language model judges it.

Perception (the tracked cable keypoints and the connector pose) already tells the planner
whether a fork holds the wire. A real formboard is signed off by looking at it, so the
build agent gets a second, independent check: ``InspectionCamera`` renders a fixture from
two angles, and ``VisualInspector`` asks a vision-language model one narrow question about
it, answered in JSON.

Two question styles, both kept so they can be compared on the same images (vision_eval):

  v1  two general views (along the slot, oblique); "is the wire seated in this fork?"
  v2  two views *through* the fork's slot from opposite sides, the fixture to check
      marked with a magenta box, and one local question per view ("is there orange wire
      in the gap between the prongs, below the yellow knobs?"). The verdict is ours:
      seated only if both views say yes. Optionally with labelled example images.

Measured on 130 labelled renders, v1 made small models parrot the definition of
"seated" back and catch half the defects; v2 is the fix under test. Verdicts are logged
next to the simulator's ground truth, so accuracy is measured rather than assumed. The
inspector takes any image, so the same questions work on photos of a real board.
"""

from __future__ import annotations

import base64
import inspect
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
Box = Optional[Tuple[float, float, float, float]]
STYLES = ("v1", "v2", "v3")
DEFAULT_STYLE = "v2"
HIGHLIGHT = (255, 0, 255)                   # magenta: no fixture, wire or board has that colour

# Viewpoints (name, azimuth offset in deg, elevation in deg, distance in m), per style and
# fixture kind. Repeated inspections of a fixture step through the variants, so a second
# look really is a second look. v1 turns each view to face away from the robot; v2 turns
# the pair once, so its second view (+180 deg) looks from the opposite side.
VIEWS: Dict[Tuple[str, str], Tuple[Tuple[Tuple[str, float, float, float], ...], ...]] = {
    ("v1", "fork"): (
        (("along the slot", 0.0, -28.0, 0.15), ("oblique", 45.0, -42.0, 0.15)),
        (("along the slot, low", 0.0, -18.0, 0.13), ("oblique, other side", -45.0, -38.0, 0.15)),
        (("across the slot", 90.0, -35.0, 0.14), ("steep", 25.0, -62.0, 0.15)),
    ),
    ("v1", "connector"): (
        (("side", 60.0, -38.0, 0.17), ("top", 0.0, -88.0, 0.17)),
        (("along the pocket", 0.0, -35.0, 0.16), ("top, closer", 0.0, -88.0, 0.13)),
        (("other side", -60.0, -40.0, 0.17), ("steep", 30.0, -65.0, 0.15)),
    ),
    ("v2", "fork"): (
        (("through the slot", 0.0, -8.0, 0.13), ("through the slot, other side", 180.0, -8.0, 0.13)),
        (("through the slot, higher", 0.0, -16.0, 0.12), ("other side, higher", 180.0, -16.0, 0.12)),
        (("through the slot, close", 0.0, -5.0, 0.11), ("other side, close", 180.0, -5.0, 0.11)),
    ),
    ("v2", "connector"): (
        (("from above", 0.0, -88.0, 0.15), ("from the side, low", 90.0, -12.0, 0.15)),
        (("from above, closer", 0.0, -88.0, 0.12), ("end on, low", 0.0, -12.0, 0.15)),
        (("from above", 30.0, -80.0, 0.15), ("other side, low", -90.0, -12.0, 0.15)),
    ),
    # v3: into the slot at 25 deg off its axis, so a wire through the gap is seen crossing it
    # instead of pointing straight at the camera (where it hides the gap it runs through)
    ("v3", "fork"): (
        (("into the slot", 25.0, -8.0, 0.13), ("into the slot, other side", 205.0, -8.0, 0.13)),
        (("into the slot, higher", -25.0, -15.0, 0.12), ("other side, higher", 155.0, -15.0, 0.12)),
        (("into the slot, close", 30.0, -5.0, 0.11), ("other side, close", 210.0, -5.0, 0.11)),
    ),
    ("v3", "connector"): (
        (("from above", 0.0, -88.0, 0.15), ("from the side", 90.0, -20.0, 0.15)),
        (("from above, closer", 0.0, -88.0, 0.12), ("end on", 0.0, -20.0, 0.15)),
        (("from above", 30.0, -80.0, 0.15), ("other side", -90.0, -20.0, 0.15)),
    ),
}

SIM_APPEARANCE = {
    "fork": "Here the fork is the blue U-shaped clip with two yellow knobs (the snap lips) at the "
            "top of its prongs, and the wire is the orange cable.",
    "connector": "Here the holder is the green pocket on the board and the connector is the black "
                 "block at the end of the orange wire.",
}

_JSON_V1 = '{{"seated": true or false, "confidence": 0.0 to 1.0, "evidence": "one short sentence on what you see"}}'
_JSON_V2 = '{{"A": "yes" or "no", "B": "yes" or "no", "evidence": "what you see, in a few words"}}'

QUESTIONS: Dict[Tuple[str, str], str] = {
    ("v1", "fork"): """You are the quality inspector of a wire-harness assembly cell. The image shows fork {target}, a snap-in cable clip on the formboard, from two angles (A and B). {appearance}

Question: is the wire seated in this fork?
Seated: the wire passes through the fork's slot, between its two prongs and below the snap lips at the top.
Not seated: anything else. The wire lies on the board next to the fork, rests on top of the lips, or does not come near the fork.

Use both views. Reply with JSON only:
""" + _JSON_V1,
    ("v1", "connector"): """You are the quality inspector of a wire-harness assembly cell. The image shows the holder for connector {target} from two angles (A and B). {appearance}

Question: is the connector seated in its holder?
Seated: the connector lies flat and fully inside the holder's pocket, between its end walls.
Not seated: anything else. The connector lies outside the pocket, sits on a wall or rail, is tilted or stands on end, or is not in view.

Use both views. Reply with JSON only:
""" + _JSON_V1,
    ("v2", "fork"): """You check fork {target} on a wire-harness formboard. The image has two views of it, A and B, taken from opposite sides. In each view the fork to check is inside the magenta box; ignore everything outside the box. {appearance}

The fork holds the wire when the wire runs through the U-shaped gap between the fork's two prongs, below the yellow knobs. Both views look through that gap.

For each view, answer one question: is there orange wire inside the gap between the two blue prongs, below the yellow knobs? Wire in front of the fork, behind it, on the board next to it, or lying on top of the knobs does not count.

Reply with JSON only:
""" + _JSON_V2,
    ("v2", "connector"): """You check the holder of connector {target} on a wire-harness formboard. The image has two views of it: A from above and B from the side. In each view the holder to check is inside the magenta box; ignore everything outside the box. {appearance}

The connector is seated when it lies flat and fully inside the holder's pocket, between the end walls, down on the pocket floor.

For each view, answer one question: is the black connector lying flat inside the green pocket? A connector next to the holder, resting on a wall or rail, tilted, or standing on end does not count.

Reply with JSON only:
""" + _JSON_V2,
}

QUESTIONS[("v3", "fork")] = """You check fork {target} on a wire-harness formboard. The image has two views of it, A and B, from opposite sides. In each view the fork to check is inside the magenta box; ignore everything outside the box. {appearance}

The fork holds the wire when the wire runs through the U-shaped gap between the fork's two prongs, below the yellow knobs. Both views look into that gap at a slight angle. A wire that runs through the gap is also seen in front of the fork and behind it, because it runs roughly towards you and away from you; that is expected.

For each view, answer one question: does the orange wire go through the gap between the two blue prongs, below the yellow knobs? Answer no if the gap is empty (you see the background through it), or if the wire only lies on the board next to the fork, or lies across the top of the knobs.

Reply with JSON only:
""" + _JSON_V2
QUESTIONS[("v3", "connector")] = """You check the holder of connector {target} on a wire-harness formboard. The image has two views of it: A from above and B from the side. In each view the holder to check is inside the magenta box; ignore everything outside the box. {appearance}

The connector is seated when it lies flat and fully inside the holder's pocket, between the two end walls, down on the pocket floor. The side rails of the pocket are low, so from the side a seated connector sticks out well above them; that is expected.

For each view, answer one question: is the black connector lying flat inside the green pocket, between the end walls? Answer no if it lies next to the holder, rests on top of a wall or rail, is tilted (one end higher than the other), or stands on end.

Reply with JSON only:
""" + _JSON_V2

REFERENCE_INTRO = "Two labelled examples from the same camera come first, then the image to check."


# ---------------------------------------------------------------- geometry
def camera_basis(azimuth_deg: float, elevation_deg: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """forward, right, up of a MuJoCo free camera (it sits at lookat - distance * forward)."""
    az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
    fwd = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= max(np.linalg.norm(right), 1e-12)
    return fwd, right, np.cross(right, fwd)


def project(points: np.ndarray, lookat: Sequence[float], azimuth_deg: float, elevation_deg: float,
            distance: float, fovy_deg: float, width: int, height: int) -> np.ndarray:
    """Pixel coordinates (u right, v down) of world points in a free-camera image."""
    fwd, right, up = camera_basis(azimuth_deg, elevation_deg)
    cam = np.asarray(lookat, dtype=float) - distance * fwd
    d = np.atleast_2d(np.asarray(points, dtype=float)) - cam
    z = np.maximum(d @ fwd, 1e-6)
    f = (height / 2.0) / math.tan(math.radians(fovy_deg) / 2.0)
    return np.stack([width / 2.0 + f * (d @ right) / z, height / 2.0 - f * (d @ up) / z], axis=1)


def fixture_corners(kind: str, pose: Sequence[float], cfg=None) -> np.ndarray:
    """Corners of a box around a fixture in world coordinates.

    fork: pose = (x, y, z_board, yaw), the box spans the foot, prongs and knobs;
    connector: pose = (x, y, z_seat, yaw) of the holder seat, the box spans the holder
    and a seated connector."""
    x, y, z, yaw = (float(v) for v in pose)
    if kind == "fork":
        f = cfg.fork if cfg is not None else None
        top = (f.post_height + f.prong_height + f.lip_radius) if f is not None else 0.061
        half_y = (f.slot_width / 2 + f.prong_thickness + 0.006) if f is not None else 0.0165
        lo = np.array([-0.009, -half_y, 0.0])
        hi = np.array([0.009, half_y, top + 0.002])
    else:
        lo = np.array([-0.020, -0.016, -0.010])
        hi = np.array([0.020, 0.016, 0.009])
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    corners = np.array([[a, b, cc] for a in (lo[0], hi[0]) for b in (lo[1], hi[1]) for cc in (lo[2], hi[2])])
    return corners @ R.T + np.array([x, y, z])


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
        self.fovy = float(sim.model.vis.global_.fovy)
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

    def views(self, kind: str, pose: Sequence[float], cfg=None, variant: int = 0,
              style: str = DEFAULT_STYLE) -> List[Tuple[str, np.ndarray, Box]]:
        """Rendered views of a fixture: (name, image, highlight box or None).

        kind "fork": pose = fork pose (x, y, z_board, yaw); kind "connector": pose =
        (x, y, z_seat, yaw) of the holder seat."""
        x, y, z, yaw = (float(v) for v in pose)
        if kind == "fork":
            post = cfg.fork.post_height if cfg is not None else 0.030
            look = np.array([x, y, z + post + 0.006])
        else:
            look = np.array([x, y, z + 0.006])
        corners = fixture_corners(kind, pose, cfg)
        table = VIEWS[(style, kind)]
        base = self._outward(np.array([x, y]), math.degrees(yaw))
        out = []
        for name, d_az, el, dist in table[variant % len(table)]:
            if style == "v1":
                az = self._outward(np.array([x, y]), math.degrees(yaw) + d_az)
            else:
                az = base + d_az
            img = self.shot(look, az, el, dist)
            box: Box = None
            if style != "v1":
                uv = project(corners, look, az, el, dist, self.fovy, self.width, self.height)
                m = 10.0
                box = (max(0.0, uv[:, 0].min() - m), max(0.0, uv[:, 1].min() - m),
                       min(self.width - 1.0, uv[:, 0].max() + m), min(self.height - 1.0, uv[:, 1].max() + m))
            out.append((name, img, box))
        return out

    # kept for callers of the first version
    def fork_views(self, fork_pose, fork_cfg, variant: int = 0):
        class _Cfg:
            fork = fork_cfg
        return [(n, im) for n, im, _ in self.views("fork", fork_pose, _Cfg, variant, "v1")]

    def holder_views(self, holder_pos, holder_yaw: float, variant: int = 0):
        pose = (holder_pos[0], holder_pos[1], holder_pos[2], holder_yaw)
        return [(n, im) for n, im, _ in self.views("connector", pose, None, variant, "v1")]


def compose(views: Sequence[Tuple[Any, ...]], title: str, width: int = 512):
    """Views side by side under a title strip, labelled A, B, ...; a view given as
    (name, image, box) gets the box drawn around the fixture to check."""
    from PIL import Image, ImageDraw

    ims, boxes = [], []
    for view in views:
        v = view[1]
        box = view[2] if len(view) > 2 else None
        im = v if hasattr(v, "size") and not isinstance(v, np.ndarray) else Image.fromarray(np.asarray(v))
        im = im.convert("RGB")
        scale = width / im.width
        if im.width != width:
            im = im.resize((width, max(1, round(im.height * scale))))
        ims.append(im)
        boxes.append(None if box is None else tuple(c * scale for c in box))
    strip, gap = 30, 6
    h = max(im.height for im in ims)
    canvas = Image.new("RGB", (len(ims) * width + (len(ims) - 1) * gap, h + strip), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    font = _font(17)
    small = _font(15)
    draw.text((8, 6), title, fill=(20, 20, 20), font=font)
    for k, (view, im, box) in enumerate(zip(views, ims, boxes)):
        x0 = k * (width + gap)
        canvas.paste(im, (x0, strip))
        if box is not None:
            draw.rectangle((x0 + box[0], strip + box[1], x0 + box[2], strip + box[3]), outline=HIGHLIGHT, width=3)
        label = f"{chr(65 + k)}: {view[0]}"
        tb = draw.textbbox((0, 0), label, font=small)
        draw.rectangle((x0 + 4, strip + 4, x0 + 12 + tb[2] - tb[0], strip + 10 + tb[3] - tb[1]),
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
    style: str = DEFAULT_STYLE
    per_view: Optional[Dict[str, Optional[bool]]] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    fallback_from: str = ""           # the model that gave no usable answer before this one

    def for_planner(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"seated": self.seated, "confidence": round(self.confidence, 2),
                               "evidence": self.evidence, "model": self.model.split("/")[-1]}
        if self.per_view:
            out["views"] = self.per_view
        if self.error:
            out["error"] = self.error
        return out

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


_OBJ_RE = re.compile(r"\{[^{}]*\}", re.S)
_TRUE = {"true", "yes", "seated", "y", "1"}
_FALSE = {"false", "no", "not seated", "not_seated", "unseated", "n", "0"}


def _yes_no(raw: Any) -> Optional[bool]:
    if isinstance(raw, bool):
        return raw
    key = str(raw).strip().lower().rstrip(".")
    return True if key in _TRUE else False if key in _FALSE else None


def _json_objects(text: str) -> List[Dict[str, Any]]:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    text = text.replace("```json", "```").replace("```", " ")
    out = []
    for m in _OBJ_RE.finditer(text):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def parse_verdict(text: str) -> Tuple[Optional[bool], float, str]:
    """(seated, confidence, evidence) from a style-v1 reply; seated is None if unusable."""
    for obj in reversed(_json_objects(text)):               # the last JSON object is the answer
        if "seated" not in obj:
            continue
        try:
            conf = float(obj.get("confidence", 0.5))
        except (TypeError, ValueError):
            conf = 0.5
        if conf > 1.0:                                       # a percentage
            conf /= 100.0
        return _yes_no(obj.get("seated")), float(np.clip(conf, 0.0, 1.0)), str(obj.get("evidence", ""))[:300]
    return None, 0.0, ""


def parse_views(text: str) -> Tuple[Optional[bool], float, str, Dict[str, Optional[bool]]]:
    """Style v2: per-view yes/no -> (seated, confidence, evidence, per-view answers).

    Seated only if every view that answered says yes; confidence 1 when the two views
    agree, 0.5 when they do not (the verdict is then "not seated")."""
    for obj in reversed(_json_objects(text)):
        keys = {k.strip().upper().replace("VIEW_", "").replace("VIEW ", ""): v for k, v in obj.items()}
        views = {k: _yes_no(keys[k]) for k in ("A", "B") if k in keys}
        answered = {k: v for k, v in views.items() if v is not None}
        if not answered:
            if "seated" in obj:                              # a v1-style answer: take it
                seated, conf, ev = parse_verdict(json.dumps(obj))
                return seated, conf, ev, {}
            continue
        vals = list(answered.values())
        seated = all(vals)
        conf = 1.0 if len(vals) == 2 and vals[0] == vals[1] else 0.5
        return seated, conf, str(obj.get("evidence", ""))[:300], views
    return None, 0.0, "", {}


class VisualInspector:
    """Asks a vision-language model whether a fixture is right.

    The model runs on Token Factory by default. To use one served elsewhere through an
    OpenAI-compatible endpoint (say, a model served with vLLM on a Nebius GPU VM), set
    HARNESS_VISION_BASE_URL (and HARNESS_VISION_API_KEY if it wants one); the planner stays
    on Token Factory.

    ``references`` maps a fixture kind to labelled example images, [(image, caption), ...],
    sent ahead of the image to check (few-shot). ``max_tokens`` is generous because
    reasoning models think before they answer: capped at 700, Kimi K3 ran out on 100 of
    130 images. ``fallback`` answers when this model gives no usable verdict (see
    ``default_inspector`` for the combination that measured best).
    """

    def __init__(self, client: Optional[Any] = None, model: Optional[str] = None,
                 style: str = DEFAULT_STYLE, references: Optional[Dict[str, List[Tuple[Any, str]]]] = None,
                 appearance: Optional[Dict[str, str]] = None, max_tokens: int = 4000,
                 temperature: float = 0.0, extra: Optional[Dict[str, Any]] = None,
                 fallback: Optional["VisualInspector"] = None):
        if style not in STYLES:
            raise ValueError(f"unknown style {style!r}; use one of {STYLES}")
        base_url = os.environ.get("HARNESS_VISION_BASE_URL")
        if base_url:
            client = TokenFactoryClient(api_key=os.environ.get("HARNESS_VISION_API_KEY") or "none",
                                        base_url=base_url)
        self.client = client or TokenFactoryClient()
        self.model = model or self.client.vision_model()
        self.style = style
        self.references = references or {}
        self.appearance = SIM_APPEARANCE if appearance is None else appearance
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.extra = extra or {}
        self.fallback = fallback

    @property
    def label(self) -> str:
        return self.style + ("refs" if self.references else "")

    def describe(self) -> str:
        text = f"{self.model} ({self.label})"
        if self.fallback is not None:
            text += f", falling back to {self.fallback.describe()}"
        return text

    def prompt(self, kind: str, target: str) -> str:
        text = QUESTIONS[(self.style, kind)].format(target=target, appearance=self.appearance.get(kind, ""))
        if self.references.get(kind):
            text = REFERENCE_INTRO + "\n\n" + text
        return text.replace("  ", " ")

    def content(self, kind: str, target: str, image: ImageLike) -> List[Dict[str, Any]]:
        parts: List[Dict[str, Any]] = [{"type": "text", "text": self.prompt(kind, target)}]
        for k, (ref, caption) in enumerate(self.references.get(kind, []), 1):
            parts.append({"type": "text", "text": f"Example {k}: {caption}"})
            parts.append({"type": "image_url", "image_url": {"url": to_data_url(ref)}})
        if self.references.get(kind):
            parts.append({"type": "text", "text": f"The image to check ({target}):"})
        parts.append({"type": "image_url", "image_url": {"url": to_data_url(image)}})
        return parts

    def ask(self, kind: str, target: str, image: ImageLike, view: int = 0) -> VisualVerdict:
        verdict = self._ask(kind, target, image, view)
        if verdict.seated is None and self.fallback is not None:
            fb = self.fallback.ask(kind, target, image, view)
            fb.fallback_from = self.model
            fb.seconds += verdict.seconds
            fb.prompt_tokens += verdict.prompt_tokens
            fb.completion_tokens += verdict.completion_tokens
            return fb
        return verdict

    def _ask(self, kind: str, target: str, image: ImageLike, view: int = 0) -> VisualVerdict:
        if (self.style, kind) not in QUESTIONS:
            raise ValueError(f"unknown inspection kind {kind!r}")
        t0 = time.perf_counter()
        try:
            msg = self.client.chat(self.model, [{"role": "user", "content": self.content(kind, target, image)}],
                                   temperature=self.temperature, max_tokens=self.max_tokens, **self.extra)
        except LLMError as exc:
            return VisualVerdict(kind, target, None, 0.0, "", self.model, time.perf_counter() - t0, view,
                                 error=f"vision model call failed: {str(exc)[:200]}", style=self.label)
        dt = time.perf_counter() - t0
        text = msg.get("content") or ""
        per_view: Dict[str, Optional[bool]] = {}
        if self.style == "v1":
            seated, conf, evidence = parse_verdict(text)
        else:
            seated, conf, evidence, per_view = parse_views(text)
        error = ""
        if seated is None:
            error = ("no answer (the model spent its tokens reasoning)" if not text.strip() and msg.get("reasoning")
                     else "no usable JSON verdict in the reply")
        usage = msg.get("usage") or {}
        return VisualVerdict(kind, target, seated, conf, evidence, self.model, dt, view, error=error,
                             raw=text[:600], style=self.label, per_view=per_view or None,
                             prompt_tokens=int(usage.get("prompt_tokens") or 0),
                             completion_tokens=int(usage.get("completion_tokens") or 0))


def packaged_references_dir() -> Optional[str]:
    """The labelled example images that ship with the package (refs/), if found."""
    here = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "refs")
    if os.path.exists(os.path.join(here, "refs.json")):
        return here
    try:
        from ament_index_python.packages import get_package_share_directory
        share = os.path.join(get_package_share_directory("harness_agent"), "refs")
        return share if os.path.exists(os.path.join(share, "refs.json")) else None
    except Exception:
        return None


# The combination that measured best on 134 labelled renders (vision_eval, style v2):
# Kimi K3 with the labelled examples answers 70% of the images and is right on 98% of those;
# when it runs out of reasoning budget, MiniCPM-V 4.5 (v2, no examples: they made it worse)
# answers in under a second. Together: 96% accuracy, 97% of defects caught, 5% false alarms.
PRIMARY_PREFERENCE = ("kimi-k3",)
FALLBACK_PREFERENCE = ("minicpm-v", "gemma-3")


def default_inspector(client: Optional[Any] = None, model: Optional[str] = None,
                      fallback_model: Optional[str] = None, style: str = DEFAULT_STYLE,
                      refs: Optional[str] = "packaged") -> "VisualInspector":
    """The measured-best camera check with what this key can use.

    ``model`` / ``fallback_model`` override the choice ("none" disables the fallback);
    ``refs`` is a refs/ folder, "packaged" (default) or None."""
    client = client or TokenFactoryClient()
    try:
        models = list(client.list_models())
    except Exception:
        models = []

    def first(prefs: Sequence[str], exclude: Sequence[str] = ()) -> Optional[str]:
        for pat in prefs:
            hits = sorted((m for m in models if pat in m.lower() and m not in exclude), key=len)
            if hits:
                return hits[0]
        return None

    primary = model or os.environ.get("HARNESS_VISION_MODEL") or first(PRIMARY_PREFERENCE) or client.vision_model()
    if fallback_model == "none":
        fb_model = None
    else:
        fb_model = fallback_model or first(FALLBACK_PREFERENCE, exclude=[primary])
    ref_dir = packaged_references_dir() if refs == "packaged" else refs
    references = load_references(ref_dir, style) if ref_dir and style != "v1" else None
    fallback = VisualInspector(client, fb_model, style=style) if fb_model else None
    return VisualInspector(client, primary, style=style, references=references, fallback=fallback)


def load_references(ref_dir: str, style: str = DEFAULT_STYLE) -> Dict[str, List[Tuple[Any, str]]]:
    """Labelled example images written by ``vision_eval make-set`` (refs/refs.json)."""
    from PIL import Image
    with open(os.path.join(ref_dir, "refs.json"), encoding="utf-8") as f:
        spec = json.load(f)
    out: Dict[str, List[Tuple[Any, str]]] = {}
    for item in spec.get(style, []):
        with Image.open(os.path.join(ref_dir, item["image"])) as im:
            out.setdefault(item["kind"], []).append((im.convert("RGB"), item["caption"]))
    return out


class FakeVisionClient:
    """Test double for Token Factory: answers inspection questions with ``answer(prompt)``.

    ``answer`` returns the reply text; the default says every fixture is seated.
    """

    def __init__(self, answer: Optional[Callable[[str], str]] = None, models: Sequence[str] = ()):
        self.answer = answer or (lambda prompt: '{"seated": true, "confidence": 0.9, "evidence": "stub", '
                                                '"A": "yes", "B": "yes"}')
        self.models = list(models)
        self.calls: List[Dict[str, Any]] = []
        self.usage = type("U", (), {"as_dict": staticmethod(lambda: {"calls": 0})})()

    def vision_model(self) -> str:
        return "fake-vlm"

    def list_models(self) -> List[str]:
        return self.models

    def chat(self, model: str, messages: List[Dict[str, Any]], **_: Any) -> Dict[str, Any]:
        content = messages[-1]["content"]
        prompt = content[0]["text"]
        self.calls.append({"model": model, "prompt": prompt,
                           "images": sum(p.get("type") == "image_url" for p in content)})
        takes_model = len(inspect.signature(self.answer).parameters) >= 2
        text = self.answer(prompt, model) if takes_model else self.answer(prompt)
        return {"role": "assistant", "content": text, "tool_calls": [], "reasoning": "",
                "usage": {"prompt_tokens": 100, "completion_tokens": 10}}
