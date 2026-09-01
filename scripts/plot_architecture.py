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


def _box(ax, x, y, w, h, title, sub, edge, fill, title_size=9.5, sub_size=8, lw=1.6):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.006,rounding_size=0.018",
        linewidth=lw, edgecolor=edge, facecolor=fill, zorder=2))
    # Title sits in the upper part of the box, subtitle in the lower part, with
    # enough separation that multi-line subtitles never collide with the title.
    ax.text(x, y + (h * 0.22 if sub else 0.0), title, ha="center", va="center",
            fontsize=title_size, fontweight="bold", color="#111", zorder=3)
    if sub:
        ax.text(x, y - h * 0.20, sub, ha="center", va="center",
                fontsize=sub_size, color="#333", zorder=3, linespacing=1.25)


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
    # Drawn close to its printed size (a NeurIPS text block is 5.5 in wide), so the
    # fonts below are roughly what the reader sees — no illegible shrink.
    fig, ax = plt.subplots(figsize=(8.0, 3.4))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ym = 0.68             # main perception row
    w, h = 0.20, 0.26     # default box size (gaps between boxes must fit a real arrow)
    xs = [0.12, 0.365, 0.61, 0.865]

    _box(ax, xs[0], ym, w, h, "Monocular depth", "single frame\n(Depth-Anything-V2)",
         BLUE, _tint(BLUE, 0.12))
    _box(ax, xs[1], ym, w, h, "BEV projection", "egocentric occupancy\n+ explicit shadow mask",
         BLUE, _tint(BLUE, 0.12))
    _box(ax, xs[2], ym, w, h, "World model", "U-Net, single frame\n(BCE, EMA)",
         BLUE, _tint(BLUE, 0.12))
    _box(ax, xs[3], ym, 0.22, 0.30,
         "Completed\noccupancy", "+ calibrated uncertainty\nINTO shadow",
         ORANGE, _tint(ORANGE, 0.16), lw=2.2)

    # Arrow endpoints must use each box's OWN width — the last box is wider (0.245),
    # and using the default width put the arrowhead underneath its fill.
    widths = [w, w, w, 0.22]
    for i in range(len(xs) - 1):
        _arrow(ax, xs[i] + widths[i] / 2, ym, xs[i + 1] - widths[i + 1] / 2, ym)

    # Downstream interface row.
    yd = 0.175
    _box(ax, 0.615, yd, 0.225, 0.22, "A* planner", "routes around\nhidden risk",
         GREEN, _tint(GREEN, 0.12), title_size=9.5)
    _box(ax, 0.875, yd, 0.225, 0.22, "HRTF audio", "9 azimuth zones,\neyes-free",
         GREEN, _tint(GREEN, 0.12), title_size=9.5)

    # contribution box -> both downstream consumers
    _arrow(ax, 0.83, ym - 0.30 / 2, 0.63, yd + 0.22 / 2, color=ORANGE)
    _arrow(ax, 0.875, ym - 0.30 / 2, 0.875, yd + 0.22 / 2, color=ORANGE)

    # Measured latency (slurm job 63538038, GTX 1080 Ti) — fills the lower-left and keeps
    # the figure honest about what the real-time claim covers.
    ax.text(0.235, 0.185,
            "measured latency (GTX 1080 Ti):\n"
            "depth 50.0 ms + completion 4.4 ms\n"
            "= 54.4 ms end-to-end (18 Hz);  pipelined:\n"
            "7 ms p95 perception→planning at 20 Hz\n"
            "on the latest depth frame (audio excluded)",
            ha="center", va="center", fontsize=7.5, color=GREY, linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#f5f5f5",
                      edgecolor="#cccccc", linewidth=0.9))

    # Labels / framing
    ax.annotate("perception  →  world model", xy=(0.365, ym + h / 2 + 0.055),
                ha="center", fontsize=8, color=BLUE, fontweight="bold")
    ax.annotate("eyes-free navigation interface", xy=(0.745, yd - 0.22 / 2 - 0.07),
                ha="center", fontsize=8, color=GREEN, fontweight="bold")
    ax.text(0.865, ym + 0.30 / 2 + 0.05, "contribution", ha="center", fontsize=8,
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
