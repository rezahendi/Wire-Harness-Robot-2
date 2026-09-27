"""Render a harness spec as a formboard drawing (PNG).

The drawing is what a person on the line would look at: board outline, fixtures with
their ids and coordinates, the wire route with direction arrows, and a title block with
the part numbers. It doubles as input for the drawing-reading model and as the first
panel of the dashboard, and because it is generated from the spec, every drawing comes
with its ground truth.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .spec import HarnessSpec


def render_drawing(spec: HarnessSpec, path: str, dpi: int = 130, show_coords: bool = True,
                   title: Optional[str] = None) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle

    su, sv = spec.board_size_mm
    fig = plt.figure(figsize=(11.0, 6.4))
    ax = fig.add_axes([0.04, 0.22, 0.92, 0.72])
    ink, route_c, fix_c = "#1d2733", "#d9480f", "#1c4f9c"

    ax.add_patch(Rectangle((0, 0), su, sv, facecolor="#f6efe3", edgecolor=ink, lw=1.6))
    for u in np.arange(50, su, 50):
        ax.plot([u, u], [0, sv], color="#e3d8c5", lw=0.5, zorder=0.5)
    for v in np.arange(50, sv, 50):
        ax.plot([0, su], [v, v], color="#e3d8c5", lw=0.5, zorder=0.5)

    w = spec.wires[0] if spec.wires else None
    if w is not None:
        pts = []
        c = spec.clamp(w.start)
        if c:
            pts.append((c.id, np.array(c.at_mm, dtype=float), "clamp"))
        for fid in w.route:
            f = spec.fork(fid)
            if f:
                pts.append((f.id, np.array(f.at_mm, dtype=float), "fork"))
        k = spec.connector(w.end)
        if k:
            pts.append((k.id, np.array(k.holder_at_mm, dtype=float), "holder"))
        xy = np.array([p for _, p, _ in pts])
        ax.plot(xy[:, 0], xy[:, 1], color=route_c, lw=2.2, ls=(0, (6, 3)), zorder=2)
        for i in range(1, len(xy)):
            mid = 0.5 * (xy[i - 1] + xy[i])
            d = xy[i] - xy[i - 1]
            d = d / (np.linalg.norm(d) + 1e-9)
            ax.add_patch(FancyArrowPatch(mid - 8 * d, mid + 8 * d, arrowstyle="-|>",
                                         mutation_scale=14, color=route_c, zorder=3))
        for idx, (name, p, kind) in enumerate(pts):
            # fixture orientation: along the route bisector (forks), towards the next point
            if kind == "fork":
                a = xy[idx] - xy[idx - 1]
                b = xy[idx + 1] - xy[idx]
                t = a / np.linalg.norm(a) + b / np.linalg.norm(b)
                yaw = math.atan2(t[1], t[0])
                _fork_symbol(ax, p, yaw, fix_c, Polygon)
            elif kind == "clamp":
                ax.add_patch(Rectangle(p - 14, 28, 28, facecolor="#4a4f57", edgecolor=ink, lw=1.0, zorder=4))
            else:
                a = xy[idx] - xy[idx - 1]
                yaw = math.atan2(a[1], a[0])
                _holder_symbol(ax, p, yaw, Polygon)
            label = name if not show_coords else f"{name}\n({p[0]:.0f}, {p[1]:.0f})"
            ax.annotate(label, p, xytext=(12, 12), textcoords="offset points", fontsize=9,
                        color=ink, fontweight="bold" if kind != "fork" else "normal",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#c9c2b4", lw=0.6), zorder=6)
    for f in spec.forks:                                  # forks not on the route
        if w is None or f.id not in w.route:
            _fork_symbol(ax, np.array(f.at_mm, dtype=float), 0.0, "#8a94a6", Polygon)
            ax.annotate(f"{f.id} (unused)", f.at_mm, xytext=(10, -16), textcoords="offset points",
                        fontsize=8, color="#8a94a6")

    ax.annotate("", xy=(su / 2, -12), xytext=(su / 2, -60),
                arrowprops=dict(arrowstyle="-|>", color="#6b7280", lw=1.2), annotation_clip=False)
    ax.text(su / 2 + 12, -48, "robot side", fontsize=8, color="#6b7280")
    ax.text(0, -30, "u [mm] ->", fontsize=8, color="#6b7280")
    ax.set_xlim(-20, su + 20)
    ax.set_ylim(-70, sv + 20)
    ax.set_aspect("equal")
    ax.axis("off")

    tb = fig.add_axes([0.04, 0.03, 0.92, 0.15])
    tb.axis("off")
    tb.add_patch(Rectangle((0, 0), 1, 1, transform=tb.transAxes, fill=False, ec=ink, lw=1.2))
    rows = [("Harness", f"{spec.name}   rev {spec.revision}")]
    if w is not None:
        route = " > ".join([w.start] + w.route + [w.end])
        rows += [("Wire", f"{w.id}  {w.part_number}   d = {w.diameter_mm:.1f} mm, "
                          f"{w.stiffness}, slack {w.slack_mm:.0f} mm"),
                 ("Route", route)]
        k = spec.connector(w.end)
        if k is not None:
            rows.append(("Connector", f"{k.id}  {k.part_number}  {k.description}".strip()))
    for i, (key, val) in enumerate(rows):
        y = 0.80 - i * 0.22
        tb.text(0.012, y, key, fontsize=9, color="#6b7280", va="center", transform=tb.transAxes)
        tb.text(0.11, y, val, fontsize=10, color=ink, va="center", transform=tb.transAxes)
    tb.text(0.988, 0.80, f"board {su:.0f} x {sv:.0f} mm", fontsize=9, color="#6b7280",
            ha="right", va="center", transform=tb.transAxes)
    if title:
        fig.suptitle(title, fontsize=12, color=ink)
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return path


def _fork_symbol(ax, p, yaw, color, Polygon) -> None:
    """A U-shaped fork seen from above: two jaws either side of the wire direction."""
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    for side in (+1, -1):
        jaw = np.array([[-5, 7.5], [5, 7.5], [5, 12.5], [-5, 12.5]], dtype=float)
        jaw[:, 1] *= side
        ax.add_patch(Polygon(p + jaw @ R.T, closed=True, facecolor=color, edgecolor="#0d2a57",
                             lw=0.8, zorder=4))
    base = np.array([[-6, -13], [6, -13], [6, 13], [-6, 13]], dtype=float) * np.array([0.5, 1.0])
    ax.add_patch(Polygon(p + base @ R.T, closed=True, facecolor="none", edgecolor=color,
                         lw=0.8, zorder=3.5))


def _holder_symbol(ax, p, yaw, Polygon) -> None:
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    outer = np.array([[-18, -14], [18, -14], [18, 14], [-18, 14]], dtype=float)
    inner = np.array([[-13, -9], [13, -9], [13, 9], [-13, 9]], dtype=float)
    ax.add_patch(Polygon(p + outer @ R.T, closed=True, facecolor="#2f8f5b", edgecolor="#174d31",
                         lw=1.0, zorder=4))
    ax.add_patch(Polygon(p + inner @ R.T, closed=True, facecolor="#e9f5ee", edgecolor="#174d31",
                         lw=0.8, zorder=4.5))
