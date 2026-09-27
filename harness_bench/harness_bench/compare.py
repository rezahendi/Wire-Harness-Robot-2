"""Turn two (or more) benchmark result files into a comparison report with plots.

    ros2 run harness_bench compare_benchmarks results/mujoco.json results/isaac.json --out report
    python3 -m harness_bench.compare results/*.json --out report

Writes ``report/report.md`` plus PNG figures. Analytic references (elastica, catenary,
limp-chain frequency, nominal friction) are drawn alongside the engines, so the report
says how each engine compares to theory, not only to the other engine.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np

# headline rows of the comparison table: key -> (label, unit, scale, reference key)
HEADLINE = [
    ("cantilever.tip_drop", "Cantilever droop (cell segmentation)", "mm", 1000.0, "cantilever.elastica_tip_drop"),
    ("cantilever.tip_drop_rel_error_at_cell_segmentation", "  ... error vs elastica", "%", 100.0, None),
    ("sag.sag", "Sag at the widest span", "mm", 1000.0, "sag.catenary_sag"),
    ("sag.sag_rel_error", "  ... error vs catenary", "%", 100.0, None),
    ("swing.frequency", "Release-from-horizontal frequency", "Hz", 1.0, "swing.chain_reference_frequency"),
    ("swing.damping_ratio", "Damping ratio", "-", 1.0, None),
    ("snap.insertion_peak_force", "Force to snap the wire into a fork", "N", 1.0, None),
    ("snap.extraction_peak_force", "Force to pull it back out", "N", 1.0, None),
    ("snap.seated_height_above_slot", "Seated height above the slot bottom", "mm", 1000.0, None),
    ("slide.effective_friction", "Effective friction on the board", "-", 1.0, "slide.nominal_friction"),
    ("step_rate.cell_realtime_factor", "Full cell speed", "x real time", 1.0, None),
    ("step_rate.cable48_physics_steps_per_s", "48-segment cable", "steps/s", 1.0, None),
    ("stability.max_usable_timestep", "Largest usable timestep", "ms", 1000.0, None),
]


def _fmt(v: Optional[float], unit: str) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    if unit in ("steps/s",):
        return f"{v:,.0f}"
    if abs(v) >= 100:
        return f"{v:.0f}"
    if abs(v) >= 10:
        return f"{v:.1f}"
    return f"{v:.3f}".rstrip("0").rstrip(".")


def table(runs: List[Dict]) -> str:
    names = [r["engine"] for r in runs]
    lines = ["| quantity | " + " | ".join(names) + " | reference | unit |",
             "|---|" + "---|" * (len(names) + 2)]
    for key, label, unit, scale, ref_key in HEADLINE:
        vals = [r["summary"].get(key) for r in runs]
        if all(v is None for v in vals):
            continue
        ref = next((r["summary"].get(ref_key) for r in runs if ref_key and ref_key in r["summary"]), None)
        cells = [_fmt(None if v is None else v * scale, unit) for v in vals]
        lines.append(f"| {label} | " + " | ".join(cells) + " | "
                     + _fmt(None if ref is None else ref * scale, unit) + f" | {unit} |")
    return "\n".join(lines)


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_cantilever(runs: List[Dict], path: str) -> Optional[str]:
    rows_by_run = [(r["engine"], r["results"].get("cantilever", {}).get("rows", [])) for r in runs]
    if not any(rows for _, rows in rows_by_run):
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    markers = "os^vD"
    for k, (name, rows) in enumerate(rows_by_run):
        segs = sorted({r["segment_length"] for r in rows})
        for j, seg in enumerate(segs):
            sub = sorted([r for r in rows if r["segment_length"] == seg], key=lambda r: r["length"])
            ax.plot([1000 * r["length"] for r in sub], [1000 * r["tip_drop"] for r in sub],
                    marker=markers[k % len(markers)], ls=["-", "--", ":", "-."][j % 4],
                    label=f"{name}, {1000 * seg:.1f} mm segments")
    ref = sorted({(r["length"], r["elastica_tip_drop"]) for _, rows in rows_by_run for r in rows})
    if ref:
        ax.plot([1000 * a for a, _ in ref], [1000 * b for _, b in ref], "k-", lw=2.2,
                label="elastica (continuum)")
    ax.set_xlabel("free wire length [mm]")
    ax.set_ylabel("tip droop [mm]")
    ax.set_title("Clamped wire drooping under its own weight")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_snap(runs: List[Dict], path: str) -> Optional[str]:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    drawn = False
    for r in runs:
        series = r["results"].get("snap", {}).get("series", {})
        for label, s in series.items():
            z = np.array(s["z"]) * 1000.0
            fz = np.array(s["fz"])
            ph = np.array(s["phase"])
            for phase, style in ((0, "-"), (1, "--")):
                m = ph < 0.5 if phase == 0 else ph >= 0.5
                if m.any():
                    ax.plot(z[m], fz[m], style, lw=1.3,
                            label=f"{r['engine']} {label} {'press' if phase == 0 else 'pull'}")
                    drawn = True
    if not drawn:
        plt.close(fig)
        return None
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("wire height at the fork [mm]")
    ax.set_ylabel("force on the hand [N]   (+ = fork pushes back)")
    ax.set_title("Snapping a wire into a fork and pulling it out")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_shapes(runs: List[Dict], path: str) -> Optional[str]:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    drawn = False
    for r in runs:
        rows = r["results"].get("sag", {}).get("rows", [])
        for row in rows:
            p = np.array(row.get("shape", []))
            if p.size:
                ax.plot(100 * (p[:, 0] - p[0, 0]), 100 * (p[:, 2] - p[0, 2]), lw=1.4,
                        label=f"{r['engine']} span {1000 * row['span']:.0f} mm")
                drawn = True
    if not drawn:
        plt.close(fig)
        return None
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("z [cm]")
    ax.set_title("Hanging wire shapes")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_speed(runs: List[Dict], path: str) -> Optional[str]:
    plt = _plt()
    keys = [("cable12_physics_steps_per_s", 12), ("cable24_physics_steps_per_s", 24),
            ("cable48_physics_steps_per_s", 48)]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    drawn = False
    for r in runs:
        m = r["results"].get("step_rate", {}).get("metrics", {})
        xs = [n for k, n in keys if k in m]
        ys = [m[k] for k, n in keys if k in m]
        if xs:
            ax.plot(xs, ys, "o-", label=r["engine"])
            drawn = True
    if not drawn:
        plt.close(fig)
        return None
    ax.set_xlabel("cable segments")
    ax.set_ylabel("physics steps per wall second")
    ax.set_yscale("log")
    ax.set_title("Cost of the cable")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+", help="benchmark JSON files (one per engine)")
    ap.add_argument("--out", default="report", help="output directory")
    ap.add_argument("--title", default="Simulator comparison: wire-harness cell")
    args = ap.parse_args(argv)

    runs = []
    for path in args.results:
        with open(path, "r", encoding="utf-8") as f:
            runs.append(json.load(f))
    os.makedirs(args.out, exist_ok=True)
    figs = [
        plot_cantilever(runs, os.path.join(args.out, "cantilever.png")),
        plot_shapes(runs, os.path.join(args.out, "shapes.png")),
        plot_snap(runs, os.path.join(args.out, "snap.png")),
        plot_speed(runs, os.path.join(args.out, "speed.png")),
    ]
    lines = [f"# {args.title}", ""]
    for r in runs:
        v = r.get("version", {})
        lines.append(f"* **{r['engine']}** {v.get('version', '')} - {v.get('cable', '')} "
                     f"({r.get('host', {}).get('platform', '')})")
    lines += ["", "All numbers are computed by the same analysis code from traces the engines",
              "produce; the reference column is analytic (elastica, catenary, limp chain,",
              "nominal friction coefficient), so each engine can be read against theory.", "",
              table(runs), ""]
    for fig in figs:
        if fig:
            lines += [f"![{os.path.basename(fig)[:-4]}]({os.path.basename(fig)})", ""]
    report = os.path.join(args.out, "report.md")
    with open(report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[:60]))
    print(f"\n-> {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
