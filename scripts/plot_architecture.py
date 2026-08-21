"""System/architecture figure for the paper — the one non-data figure.

Draws the pipeline: depth frame -> BEV projector (occupancy + visibility -> shadow
mask) -> single-frame U-Net world model -> completed occupancy + calibrated per-cell
uncertainty -> {A* planner, reactive policy, 9-zone HRTF audio}. Emphasis matches the
paper's framing: the shadow mask defines the evaluated region; the model is
single-frame (completion, not memory); measured latency covers perception-to-planning.

    python scripts/plot_architecture.py            # -> figures/architecture.{png,svg}
"""

from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BLUE, ORANGE, GREEN, GREY = "#0072B2", "#D55E00", "#009E73", "#666666"


def box(ax, x, y, w, h, title, lines, edge, face="#ffffff"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                linewidth=1.6, edgecolor=edge, facecolor=face))
    cx = x + w / 2
    ax.text(cx, y + h - 0.055, title, ha="center", va="top",
            fontsize=10.5, fontweight="bold", color=edge)
    for i, ln in enumerate(lines):
        ax.text(cx, y + h - 0.115 - 0.052 * i, ln, ha="center", va="top",
                fontsize=8.6, color="#222222")


def arrow(ax, x0, y0, x1, y1, color=GREY, label=None, dy=0.03):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=14, linewidth=1.4, color=color))
    if label:
        ax.text((x0 + x1) / 2, max(y0, y1) + dy, label, ha="center",
                fontsize=8.2, color=color, style="italic")


def main() -> None:
    fig, ax = plt.subplots(figsize=(12.5, 3.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box(ax, 0.015, 0.34, 0.14, 0.42, "depth frame",
        ["single monocular", "depth image", "(no history,", "no memory)"], GREY)
    box(ax, 0.215, 0.34, 0.165, 0.42, "BEV projector",
        ["egocentric 96×96, 5 m", "occupancy (observed)", "visibility ray-cast →",
         "explicit shadow mask"], BLUE)
    box(ax, 0.44, 0.34, 0.175, 0.42, "world model (U-Net, 31M)",
        ["single-frame input", "completed occupancy", "@ 1/3/5/10 s", "per-cell probability"],
        ORANGE)
    box(ax, 0.675, 0.55, 0.31, 0.26, "planning (system context)",
        ["A* global (shadow-aware)", "+ reactive local policy"], GREEN)
    box(ax, 0.675, 0.13, 0.31, 0.26, "eyes-free audio (system context)",
        ["9-zone HRTF cues from", "occupancy + uncertainty"], GREEN)

    arrow(ax, 0.155, 0.55, 0.215, 0.55)
    arrow(ax, 0.38, 0.55, 0.44, 0.55)
    arrow(ax, 0.615, 0.60, 0.675, 0.68)
    arrow(ax, 0.615, 0.50, 0.675, 0.26)

    # The claims live on the model output: calibrated uncertainty inside the shadow mask.
    ax.text(0.44, 0.155,
            "evaluated claim: micro shadow-IoU gain vs persistence\n"
            "+ calibrated uncertainty inside the shadow mask (ECE)",
            ha="center", fontsize=8.6, color=ORANGE, style="italic")
    ax.annotate("", xy=(0.5275, 0.34), xytext=(0.50, 0.26),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.2))

    # Latency bracket: perception-to-planning.
    ax.plot([0.215, 0.655], [0.87, 0.87], color=GREY, lw=1.1)
    ax.plot([0.215, 0.215], [0.84, 0.87], color=GREY, lw=1.1)
    ax.plot([0.655, 0.655], [0.84, 0.87], color=GREY, lw=1.1)
    ax.text(0.435, 0.90, "perception → planning: 7 ms p95 (depth acquisition & audio synthesis excluded)",
            ha="center", fontsize=8.4, color=GREY)

    out = pathlib.Path("figures")
    out.mkdir(exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(out / f"architecture.{ext}", dpi=200, bbox_inches="tight")
        print(f"wrote figures/architecture.{ext}")
    plt.close(fig)


if __name__ == "__main__":
    main()
