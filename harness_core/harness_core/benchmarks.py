"""Cross-simulator benchmarks for the wire-harness cell.

The point of this module is that MuJoCo and Isaac Sim (or any other engine) are
asked for *traces*, never for numbers: what a benchmark means, how its metrics are
computed and what the analytic reference is lives here, once, so the two engines
cannot be compared with subtly different yardsticks.

An engine implements a small set of rigs (see ``rigs_mujoco.py`` for the reference
implementation, ``isaac/harness_isaac/rigs_isaac.py`` for the Isaac Sim one), each
returning a dict of numpy arrays. This module turns those into metrics.

Benchmarks
----------
cantilever   wire clamped horizontally, drooping under its own weight. Static test
             of bending stiffness against the continuum elastica (below).
sag          wire hanging between two supports, span < length. Static test of the
             floppy regime against the catenary of an inextensible, limp cable.
swing        wire released from horizontal, tip oscillation: frequency and damping.
snap         a wire section pressed into a fork and pulled back out: insertion peak,
             retention force, and whether the fork keeps the wire.
slide        wire dragged along the board: effective friction coefficient.
step_rate    wall-clock cost of the full cell, with and without the wire.
stability    largest timestep at which the cantilever still settles instead of
             exploding, and the residual jitter at the nominal timestep.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Dict, Tuple

import numpy as np

GRAVITY = 9.81


# ------------------------------------------------------------------ specs
@dataclass
class CantileverSpec:
    """Wire clamped horizontally at one end and left to droop."""

    lengths: Tuple[float, ...] = (0.04, 0.06, 0.09, 0.15)
    segment_lengths: Tuple[float, ...] = (0.03, 0.015, 0.0075)
    clamp_height: float = 0.5          # well above the ground plane
    settle_time: float = 4.0
    still_speed: float = 2e-3          # m/s, below this the shape counts as settled


@dataclass
class SagSpec:
    """Wire held at both ends at the same height, span shorter than the wire."""

    length: float = 0.36
    spans: Tuple[float, ...] = (0.30, 0.24, 0.18)
    segment_length: float = 0.015
    clamp_height: float = 0.5
    settle_time: float = 4.0


@dataclass
class SwingSpec:
    """Wire clamped horizontally, released, tip oscillation recorded."""

    length: float = 0.18
    segment_length: float = 0.015
    clamp_height: float = 0.5
    record_time: float = 4.0
    sample_dt: float = 0.002


@dataclass
class SnapSpec:
    """A straight wire section pressed into a fork slot and pulled back out."""

    wire_length: float = 0.12
    segment_length: float = 0.015
    press_speed: float = 0.02          # m/s
    pull_speed: float = 0.02
    press_depth: float = 0.012         # below the slot bottom the hand aims for
    lateral_offsets: Tuple[float, ...] = (0.0, 0.003)   # wire not perfectly centred
    hold_time: float = 0.3


@dataclass
class SlideSpec:
    """Wire lying on the board, dragged sideways by one end."""

    wire_length: float = 0.15
    segment_length: float = 0.015
    speed: float = 0.05
    distance: float = 0.08
    settle_time: float = 1.0


@dataclass
class StepRateSpec:
    """Wall-clock cost of the full cell scene."""

    seconds_of_sim: float = 2.0
    with_wire: bool = True
    without_wire: bool = True


@dataclass
class StabilitySpec:
    """Largest timestep at which the cantilever still settles."""

    timesteps: Tuple[float, ...] = (0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064)
    length: float = 0.15
    segment_length: float = 0.015
    settle_time: float = 6.0
    tolerance: float = 0.002           # m of drift from the finest timestep still counted stable


@dataclass
class BenchmarkSuite:
    cantilever: CantileverSpec = field(default_factory=CantileverSpec)
    sag: SagSpec = field(default_factory=SagSpec)
    swing: SwingSpec = field(default_factory=SwingSpec)
    snap: SnapSpec = field(default_factory=SnapSpec)
    slide: SlideSpec = field(default_factory=SlideSpec)
    step_rate: StepRateSpec = field(default_factory=StepRateSpec)
    stability: StabilitySpec = field(default_factory=StabilitySpec)

    def to_dict(self) -> Dict:
        return {k: asdict(v) for k, v in self.__dict__.items()}


# ------------------------------------------------- wire material properties
def wire_properties(radius: float, density: float, bend_modulus: float) -> Dict[str, float]:
    """Section properties of the simulated bundle (solid circular cross section)."""
    area = math.pi * radius ** 2
    second_moment = math.pi * radius ** 4 / 4.0
    mass_per_length = density * area
    return {
        "area": area,
        "second_moment": second_moment,
        "mass_per_length": mass_per_length,
        "weight_per_length": mass_per_length * GRAVITY,
        "EI": bend_modulus * second_moment,
        # length at which self weight and bending stiffness balance; wires much
        # longer than this hang like string, much shorter behave like a beam
        "gravito_bending_length": (bend_modulus * second_moment
                                   / max(mass_per_length * GRAVITY, 1e-12)) ** (1.0 / 3.0),
    }


# ------------------------------------------------------ analytic references
def elastica_cantilever(length: float, EI: float, weight_per_length: float,
                        n: int = 400) -> Dict[str, float]:
    """Static shape of a heavy cantilever from the planar elastica (large deflection).

    theta(s) is the angle of the tangent to the horizontal, clamped horizontally at
    s = 0. Moment balance against the weight of the piece beyond s gives

        EI theta'' = w (L - s) cos(theta),   theta(0) = 0,   theta'(L) = 0.

    It is integrated from the *free* end, where both conditions are known up to the
    tip angle, and the tip angle is bisected until the clamp comes out horizontal.
    In the small-deflection limit this reproduces w L^4 / (8 EI).
    """
    du = length / n
    u = np.linspace(0.0, length, n + 1)     # distance from the free end

    def integrate(tip_angle: float) -> np.ndarray:
        """phi(u) = theta(L - u), phi'' = (w/EI) u cos(phi), phi(0) = tip_angle, phi'(0) = 0."""
        phi = np.zeros(n + 1)
        dphi = np.zeros(n + 1)
        phi[0] = tip_angle
        acc = lambda uu, ph: weight_per_length * uu * math.cos(ph) / EI
        for i in range(n):
            k1p, k1v = dphi[i], acc(u[i], phi[i])
            k2p, k2v = dphi[i] + 0.5 * du * k1v, acc(u[i] + 0.5 * du, phi[i] + 0.5 * du * k1p)
            k3p, k3v = dphi[i] + 0.5 * du * k2v, acc(u[i] + 0.5 * du, phi[i] + 0.5 * du * k2p)
            k4p, k4v = dphi[i] + du * k3v, acc(u[i] + du, phi[i] + du * k3p)
            phi[i + 1] = phi[i] + du * (k1p + 2 * k2p + 2 * k3p + k4p) / 6.0
            dphi[i + 1] = dphi[i] + du * (k1v + 2 * k2v + 2 * k3v + k4v) / 6.0
        return phi

    lo, hi = -0.5 * math.pi + 1e-6, 0.0     # tip angle: straight down ... horizontal
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if integrate(mid)[-1] > 0.0:        # clamp angle above horizontal: too little droop
            hi = mid
        else:
            lo = mid
    phi = integrate(0.5 * (lo + hi))
    theta = phi[::-1]                       # back to arc length from the clamp
    ds = du
    x = np.concatenate([[0.0], np.cumsum(np.cos(theta[:-1]) * ds)])
    y = np.concatenate([[0.0], np.cumsum(np.sin(theta[:-1]) * ds)])
    small = weight_per_length * length ** 4 / (8.0 * EI)
    return {
        "tip_drop": float(-y[-1]),
        "tip_reach": float(x[-1]),
        "tip_angle": float(theta[-1]),
        "small_deflection_tip_drop": float(small),
        "shape_x": x,
        "shape_y": y,
    }


def catenary_sag(span: float, length: float) -> float:
    """Mid-span sag of an inextensible, perfectly limp cable (no bending stiffness)."""
    if length <= span:
        return 0.0
    lo, hi = 1e-4, 1e4
    for _ in range(200):
        a = 0.5 * (lo + hi)
        arc = 2.0 * a * math.sinh(span / (2.0 * a))
        if arc > length:
            lo = a
        else:
            hi = a
    a = 0.5 * (lo + hi)
    return float(a * (math.cosh(span / (2.0 * a)) - 1.0))


def hanging_chain_frequency(length: float) -> float:
    """First mode of a limp hanging chain [Hz]: omega = 1.2024 sqrt(g/L)."""
    return float(1.2024 * math.sqrt(GRAVITY / length) / (2.0 * math.pi))


def cantilever_beam_frequency(length: float, EI: float, mass_per_length: float) -> float:
    """First bending mode of a clamped-free beam without gravity [Hz]."""
    return float((1.875104 ** 2) * math.sqrt(EI / (mass_per_length * length ** 4))
                 / (2.0 * math.pi))


# ------------------------------------------------------------- analyses
def analyse_cantilever(points: np.ndarray, clamp_point: np.ndarray, EI: float,
                       weight_per_length: float, length: float) -> Dict[str, float]:
    """Metrics from the settled shape (points: (N, 3) along the wire, clamp first)."""
    p = np.asarray(points, dtype=float)
    tip_drop = float(clamp_point[2] - p[-1, 2])
    horizontal = float(np.linalg.norm(p[-1, :2] - clamp_point[:2]))
    ref = elastica_cantilever(length, EI, weight_per_length)
    return {
        "tip_drop": tip_drop,
        "tip_reach": horizontal,
        "elastica_tip_drop": ref["tip_drop"],
        "elastica_tip_reach": ref["tip_reach"],
        "tip_drop_error": tip_drop - ref["tip_drop"],
        "tip_drop_rel_error": (tip_drop - ref["tip_drop"]) / max(ref["tip_drop"], 1e-9),
        "small_deflection_tip_drop": ref["small_deflection_tip_drop"],
    }


def analyse_sag(points: np.ndarray, span: float, length: float) -> Dict[str, float]:
    p = np.asarray(points, dtype=float)
    top = max(p[0, 2], p[-1, 2])
    sag = float(top - np.min(p[:, 2]))
    ref = catenary_sag(span, length)
    return {
        "sag": sag,
        "catenary_sag": ref,
        "sag_error": sag - ref,
        "sag_rel_error": (sag - ref) / max(ref, 1e-9),
    }


def analyse_swing(t: np.ndarray, tip_z: np.ndarray) -> Dict[str, float]:
    """Frequency and damping ratio of the tip oscillation (log decrement on peaks)."""
    t = np.asarray(t, dtype=float)
    z = np.asarray(tip_z, dtype=float)
    z = z - np.mean(z[len(z) // 2:])          # remove the settled offset
    # peaks
    idx = [i for i in range(1, len(z) - 1) if z[i] > z[i - 1] and z[i] >= z[i + 1] and z[i] > 0]
    out = {"n_peaks": float(len(idx))}
    if len(idx) >= 2:
        periods = np.diff(t[idx])
        out["frequency"] = float(1.0 / np.mean(periods))
        amps = np.abs(z[idx])
        good = amps > 1e-5
        if good.sum() >= 2:
            k = np.arange(len(amps))[good]
            slope = np.polyfit(k, np.log(amps[good]), 1)[0]
            delta = -slope
            out["damping_ratio"] = float(delta / math.sqrt(4.0 * math.pi ** 2 + delta ** 2))
        out["settling_time"] = float(t[idx[-1]] - t[idx[0]])
    else:
        out["frequency"] = float("nan")
        out["damping_ratio"] = float("nan")
    out["peak_amplitude"] = float(np.max(np.abs(z)))
    return out


def analyse_snap(trace: Dict) -> Dict[str, float]:
    """Insertion peak, extraction peak and whether the fork kept the wire.

    ``trace`` carries ``z`` (wire height at the fork), ``fz`` (force the fork and wire
    apply to the hand along +z, positive = pushing back), ``phase`` (0 pressing down,
    1 pulling up) and the fork geometry. The peaks are taken *while the wire crosses
    the barbed lips*, which is the physical snap event; forces at the end of the press
    only measure how hard the hand is told to push.
    """
    z = np.asarray(trace["z"], dtype=float)
    fz = np.asarray(trace["fz"], dtype=float)
    phase = np.asarray(trace["phase"])
    lip_z = float(trace["lip_z"])
    lip_r = float(trace["lip_radius"])
    wire_r = float(trace["wire_radius"])
    window = (z > lip_z - 2.0 * lip_r - wire_r) & (z < lip_z + wire_r)
    down = (phase == 0) & window
    up = (phase == 1) & window
    out = {
        "insertion_peak_force": float(np.max(fz[down])) if down.any() else float("nan"),
        "extraction_peak_force": float(np.max(-fz[up])) if up.any() else float("nan"),
        "press_end_force": float(np.max(fz[phase == 0])) if (phase == 0).any() else float("nan"),
        "retained": float(bool(trace["retained"])),
        "seated_height_above_slot": float(trace["wire_z_seated"] - trace["slot_z"]),
        "lateral_offset": float(trace.get("lateral_offset", 0.0)),
    }
    return out


def analyse_slide(fx: np.ndarray, normal_force: float) -> Dict[str, float]:
    fx = np.asarray(fx, dtype=float)
    steady = fx[len(fx) // 3:]
    return {
        "mean_drag_force": float(np.mean(steady)),
        "peak_drag_force": float(np.max(np.abs(fx))),
        "effective_friction": float(np.mean(steady) / max(normal_force, 1e-9)),
        "drag_force_std": float(np.std(steady)),
    }


def analyse_stability(results: Dict[float, Dict[str, float]],
                      tolerance: float = 0.002) -> Dict[str, float]:
    """results: timestep -> {"finite": bool, "tip_drop": float, "swing": float}.

    A timestep counts as usable while the settled droop still agrees with the finest
    timestep's answer; the point where it stops agreeing (or blows up) is the engine's
    practical limit for this wire.
    """
    dts = sorted(results.keys())
    ref = results[dts[0]].get("tip_drop", float("nan"))
    usable = []
    for dt in dts:
        r = results[dt]
        ok = bool(r.get("finite")) and abs(r.get("tip_drop", 1e9) - ref) <= tolerance
        r["usable"] = float(ok)
        if ok:
            usable.append(dt)
        else:
            break
    return {
        "max_usable_timestep": float(max(usable)) if usable else float("nan"),
        "reference_tip_drop": float(ref),
        "n_timesteps": float(len(dts)),
    }


# --------------------------------------------------------------- helpers
def downsample(arr: np.ndarray, n: int = 400) -> list:
    a = np.asarray(arr, dtype=float)
    if a.ndim == 1 and len(a) > n:
        idx = np.linspace(0, len(a) - 1, n).astype(int)
        a = a[idx]
    return a.tolist()


def resample_shape(points: np.ndarray, n: int = 24) -> np.ndarray:
    """Points along the wire resampled to n samples of arc length (for shape overlays)."""
    p = np.asarray(points, dtype=float)
    d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    if d[-1] <= 0:
        return np.repeat(p[:1], n, axis=0)
    s = np.linspace(0.0, d[-1], n)
    return np.stack([np.interp(s, d, p[:, k]) for k in range(3)], axis=1)


def summarise(results: Dict[str, Dict]) -> Dict[str, float]:
    """One flat dict of headline numbers, for the comparison table."""
    out: Dict[str, float] = {}
    for name, r in results.items():
        metrics = r.get("metrics", {})
        if isinstance(metrics, dict):
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    out[f"{name}.{k}"] = float(v)
    return out
