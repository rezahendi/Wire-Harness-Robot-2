"""Run the cross-simulator benchmark suite against one physics engine.

    ros2 run harness_bench run_benchmarks --engine mujoco --out results/mujoco.json
    python3 -m harness_bench.run --engine isaac --out results/isaac.json

The engine only provides the rigs (``harness_core.rigs_mujoco.MujocoRigs``,
``harness_isaac.rigs_isaac.IsaacRigs``); every metric is computed here through
``harness_core.benchmarks`` so the two engines are scored identically.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from typing import Dict, List, Optional

import numpy as np

from harness_core import benchmarks as B
from harness_core.config import CellConfig


def load_rigs(engine: str, cfg: CellConfig, bend_scale: float, friction_scale: float):
    if engine in ("mujoco", "mj"):
        from harness_core.rigs_mujoco import MujocoRigs
        return MujocoRigs(cfg, bend_scale, friction_scale)
    if engine in ("isaac", "isaacsim"):
        from harness_isaac.rigs_isaac import IsaacRigs        # noqa: F401  (needs Isaac Sim)
        return IsaacRigs(cfg, bend_scale, friction_scale)
    if ":" in engine:                                          # module:Class
        mod, _, cls = engine.partition(":")
        import importlib
        return getattr(importlib.import_module(mod), cls)(cfg, bend_scale, friction_scale)
    raise SystemExit(f"unknown engine {engine!r} (use mujoco, isaac or module:Class)")


def _jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def run_suite(rigs, suite: B.BenchmarkSuite, cfg: CellConfig, quick: bool = False,
              only: Optional[List[str]] = None, log=print) -> Dict:
    props = rigs.properties()
    results: Dict[str, Dict] = {}
    want = (lambda name: (only is None or name in only))

    # ---------------------------------------------------------- cantilever
    if want("cantilever"):
        s = suite.cantilever
        lengths = s.lengths[:2] if quick else s.lengths
        segs = s.segment_lengths[:2] if quick else s.segment_lengths
        rows: List[Dict] = []
        for length in lengths:
            for seg in segs:
                if length / seg < 2.5:
                    continue
                t0 = time.perf_counter()
                tr = rigs.cantilever(length, seg, s)
                m = B.analyse_cantilever(tr["points"], tr["clamp"], props["EI"],
                                         props["weight_per_length"], length)
                rows.append({"length": length, "segment_length": seg,
                             "n_segments": tr["n_segments"], "wall_time": time.perf_counter() - t0,
                             "shape": B.resample_shape(tr["points"], 24).tolist(), **m})
                log(f"  cantilever L={1000 * length:5.0f} mm seg={1000 * seg:5.1f} mm: "
                    f"droop {1000 * m['tip_drop']:6.2f} mm vs elastica {1000 * m['elastica_tip_drop']:6.2f} mm "
                    f"({100 * m['tip_drop_rel_error']:+.0f}%)")
        nominal = cfg.wire.segment_length
        head = min(rows, key=lambda r: (abs(r["segment_length"] - nominal), -r["length"])) if rows else {}
        results["cantilever"] = {"rows": rows, "metrics": {
            "tip_drop_rel_error_at_cell_segmentation": head.get("tip_drop_rel_error", float("nan")),
            "tip_drop": head.get("tip_drop", float("nan")),
            "elastica_tip_drop": head.get("elastica_tip_drop", float("nan")),
            "length": head.get("length", float("nan")),
            "segment_length": head.get("segment_length", float("nan")),
        }}

    # ----------------------------------------------------------------- sag
    if want("sag"):
        s = suite.sag
        spans = s.spans[:1] if quick else s.spans
        rows = []
        for span in spans:
            tr = rigs.sag(s.length, span, s.segment_length, s)
            m = B.analyse_sag(tr["points"], tr["span"], s.length)
            rows.append({"span": tr["span"], "length": s.length,
                         "shape": B.resample_shape(tr["points"], 24).tolist(), **m})
            log(f"  sag span={1000 * tr['span']:5.0f} mm: {1000 * m['sag']:6.1f} mm vs "
                f"catenary {1000 * m['catenary_sag']:6.1f} mm ({100 * m['sag_rel_error']:+.0f}%)")
        results["sag"] = {"rows": rows, "metrics": {
            "sag_rel_error": float(np.mean([r["sag_rel_error"] for r in rows])) if rows else float("nan"),
            "sag": rows[0]["sag"] if rows else float("nan"),
            "catenary_sag": rows[0]["catenary_sag"] if rows else float("nan"),
        }}

    # --------------------------------------------------------------- swing
    if want("swing"):
        s = suite.swing
        tr = rigs.swing(s)
        m = B.analyse_swing(tr["t"], tr["tip_z"])
        m["chain_reference_frequency"] = B.hanging_chain_frequency(s.length)
        m["beam_reference_frequency"] = B.cantilever_beam_frequency(
            s.length, props["EI"], props["mass_per_length"])
        results["swing"] = {"metrics": m,
                            "series": {"t": B.downsample(tr["t"], 600),
                                       "tip_z": B.downsample(tr["tip_z"], 600)}}
        log(f"  swing: {m['frequency']:.2f} Hz, damping ratio {m['damping_ratio']:.3f} "
            f"(limp-chain {m['chain_reference_frequency']:.2f} Hz, stiff-beam "
            f"{m['beam_reference_frequency']:.2f} Hz)")

    # ---------------------------------------------------------------- snap
    if want("snap"):
        s = suite.snap
        offsets = s.lateral_offsets[:1] if quick else s.lateral_offsets
        rows, series = [], {}
        for off in offsets:
            tr = rigs.snap(s, off)
            m = B.analyse_snap(tr)
            rows.append(m)
            series[f"offset_{1000 * off:.0f}mm"] = {
                "z": B.downsample(tr["z"], 600), "fz": B.downsample(tr["fz"], 600),
                "phase": B.downsample(np.asarray(tr["phase"], dtype=float), 600)}
            log(f"  snap offset={1000 * off:4.1f} mm: insertion {m['insertion_peak_force']:5.2f} N, "
                f"pull-out {m['extraction_peak_force']:5.2f} N, retained={bool(m['retained'])}")
        results["snap"] = {"rows": rows, "series": series, "metrics": {
            "insertion_peak_force": rows[0]["insertion_peak_force"] if rows else float("nan"),
            "extraction_peak_force": rows[0]["extraction_peak_force"] if rows else float("nan"),
            "seated_height_above_slot": rows[0]["seated_height_above_slot"] if rows else float("nan"),
        }}

    # --------------------------------------------------------------- slide
    if want("slide"):
        s = suite.slide
        tr = rigs.slide(s)
        m = B.analyse_slide(tr["fx"], tr["normal_force"])
        m["nominal_friction"] = tr["nominal_friction"]
        m["normal_force"] = tr["normal_force"]
        m["friction_rel_error"] = (m["effective_friction"] - m["nominal_friction"]) / max(
            m["nominal_friction"], 1e-9)
        results["slide"] = {"metrics": m, "series": {"t": B.downsample(tr["t"], 400),
                                                     "fx": B.downsample(tr["fx"], 400)}}
        log(f"  slide: effective friction {m['effective_friction']:.3f} "
            f"(nominal {m['nominal_friction']:.2f}, {100 * m['friction_rel_error']:+.0f}%)")

    # ----------------------------------------------------------- step rate
    if want("step_rate"):
        results["step_rate"] = rigs.step_rate(suite.step_rate)
        log("  step rate: " + ", ".join(f"{k}={v:.0f}" for k, v in results["step_rate"]["metrics"].items()))

    # ----------------------------------------------------------- stability
    if want("stability"):
        s = suite.stability
        raw = rigs.stability(s)
        m = B.analyse_stability(raw, s.tolerance)
        results["stability"] = {"metrics": m,
                                "rows": [{"timestep": dt, **{k: v for k, v in r.items()}}
                                         for dt, r in sorted(raw.items())]}
        log(f"  stability: usable up to {1000 * m['max_usable_timestep']:.0f} ms timestep")

    return results


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", default="mujoco", help="mujoco | isaac | module:Class")
    ap.add_argument("--out", default="benchmarks.json")
    ap.add_argument("--config", default=None, help="cell YAML (default: built-in config)")
    ap.add_argument("--bend-scale", type=float, default=1.0)
    ap.add_argument("--friction-scale", type=float, default=1.0)
    ap.add_argument("--quick", action="store_true", help="fewer points, for a smoke test")
    ap.add_argument("--only", nargs="*", default=None,
                    help="subset of: cantilever sag swing snap slide step_rate stability")
    args = ap.parse_args(argv)

    cfg = CellConfig.from_yaml(args.config) if args.config else CellConfig()
    suite = B.BenchmarkSuite()
    rigs = load_rigs(args.engine, cfg, args.bend_scale, args.friction_scale)
    props = rigs.properties()
    print(f"engine: {rigs.name}  {rigs.version()}")
    print(f"wire: EI = {props['EI']:.3e} N m^2, {props['mass_per_length'] * 1000:.1f} g/m, "
          f"gravity/bending length = {1000 * props['gravito_bending_length']:.0f} mm")
    t0 = time.perf_counter()
    results = run_suite(rigs, suite, cfg, quick=args.quick, only=args.only)
    out = {
        "engine": rigs.name,
        "version": rigs.version(),
        "host": {"platform": platform.platform(), "python": sys.version.split()[0]},
        "wire_properties": props,
        "cell_config": cfg.to_dict(),
        "suite": suite.to_dict(),
        "results": results,
        "summary": B.summarise(results),
        "wall_time": time.perf_counter() - t0,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(_jsonable(out), f, indent=1)
    print(f"\n{len(results)} benchmarks in {out['wall_time']:.0f} s -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
