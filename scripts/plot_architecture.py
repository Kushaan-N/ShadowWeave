"""System/architecture diagram for the ShadowWeave paper.

A boxes-and-arrows figure of the pipeline: a single monocular-depth frame becomes an
egocentric BEV with an explicit shadow mask, the world model completes occupancy + a
calibrated per-cell uncertainty INTO the shadow (the contribution, highlighted), and the
completed grid + uncertainty drive an A* planner and a 9-zone HRTF audio interface for
eyes-free navigation.

Pure matplotlib (no GPU, no model load); writes figures/architecture.{png,svg}.

    python scripts/plot_architecture.py --out figures/
"""

from __future__ import annotations

import argparse
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Okabe-Ito, readable on white and in grayscale.
BLUE = "#0072B2"    # perception / world-model stages
ORANGE = "#D55E00"  # the contribution: completion + calibrated uncertainty in shadow
GREEN = "#009E73"   # downstream interface
GREY = "#555555"


def _box(ax, x, y, w, h, title, sub, edge, fill, title_size=11, sub_size=8.5, lw=1.8):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.006,rounding_size=0.018",
        linewidth=lw, edgecolor=edge, facecolor=fill, zorder=2))
    ax.text(x, y + (0.018 if sub else 0.0), title, ha="center", va="center",
            fontsize=title_size, fontweight="bold", color="#111", zorder=3)
    if sub:
        ax.text(x, y - 0.030, sub, ha="center", va="center",
                fontsize=sub_size, color="#333", zorder=3)


def _arrow(ax, x0, y0, x1, y1, color=GREY, lw=2.0):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=16,
        linewidth=lw, color=color, zorder=1,
        shrinkA=2, shrinkB=2))


def _tint(hexcolor, alpha_over_white):
    """Blend a hex color toward white for a soft fill."""
    h = hexcolor.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    a = alpha_over_white
    r = int(r * a + 255 * (1 - a)); g = int(g * a + 255 * (1 - a)); b = int(b * a + 255 * (1 - a))
    return f"#{r:02x}{g:02x}{b:02x}"


def plot_architecture(out_dir: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 4.6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ym = 0.66             # main perception row
    w, h = 0.185, 0.20    # default box size
    xs = [0.11, 0.335, 0.56, 0.80]

    _box(ax, xs[0], ym, w, h, "Monocular depth", "single frame  (Depth-Anything-V2)",
         BLUE, _tint(BLUE, 0.12))
    _box(ax, xs[1], ym, w, h, "BEV projection", "egocentric occupancy\n+ explicit shadow mask",
         BLUE, _tint(BLUE, 0.12))
    _box(ax, xs[2], ym, w, h, "World model", "U-Net, single frame\n(BCE, EMA)",
         BLUE, _tint(BLUE, 0.12))
    _box(ax, xs[3], ym, 0.205, 0.235,
         "Completed occupancy", "+ calibrated uncertainty\nINTO shadow",
         ORANGE, _tint(ORANGE, 0.16), lw=2.6)

    for a, b in zip(xs[:-1], xs[1:]):
        _arrow(ax, a + w / 2, ym, b - w / 2, ym)

    # Downstream interface row.
    yd = 0.20
    _box(ax, 0.63, yd, 0.20, 0.17, "A* planner", "routes around hidden risk", GREEN, _tint(GREEN, 0.12),
         title_size=10.5)
    _box(ax, 0.855, yd, 0.20, 0.17, "HRTF audio", "9 azimuth zones, eyes-free", GREEN, _tint(GREEN, 0.12),
         title_size=10.5)

    # contribution box -> both downstream consumers
    _arrow(ax, 0.80, ym - 0.235 / 2, 0.63, yd + 0.17 / 2, color=ORANGE)
    _arrow(ax, 0.80, ym - 0.235 / 2, 0.855, yd + 0.17 / 2, color=ORANGE)

    # Labels / framing
    ax.text(0.5, 0.965, "ShadowWeave: calibrated amodal occupancy completion into occluded space",
            ha="center", va="center", fontsize=12.5, fontweight="bold", color="#111")
    ax.annotate("perception  →  world model", xy=(0.335, ym + h / 2 + 0.03),
                ha="center", fontsize=9, color=BLUE, fontweight="bold")
    ax.annotate("eyes-free navigation interface", xy=(0.74, yd - 0.17 / 2 - 0.045),
                ha="center", fontsize=9, color=GREEN, fontweight="bold")
    ax.text(0.80, ym + 0.235 / 2 + 0.03, "contribution", ha="center", fontsize=9,
            color=ORANGE, fontweight="bold")

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        p = out_dir / f"architecture.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"wrote {p}")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Render the ShadowWeave architecture diagram")
    ap.add_argument("--out", type=str, default="figures")
    args = ap.parse_args()
    plot_architecture(pathlib.Path(args.out))


if __name__ == "__main__":
    main()
