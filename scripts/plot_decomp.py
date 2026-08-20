"""Figures for the shadow decomposition + calibration analysis.

Reads the per-tier json written by eval_val_decomp.py (one dir per model) and produces:
  1. decomp_curves.png  — per-horizon dynamic_iou / dynamic_recall / static_coverage,
     one line per tier, showing the forecasting signal rising static<moving<debris and
     decaying with horizon (which the pooled shadow-IOU masked).
  2. reliability.png     — reliability diagram at 5s, shadow vs observed, with per-bin
     counts — the persuasive figure for "uncertainty propagation into unobserved space".

    python scripts/plot_decomp.py --dir <WS>/val_decomp/full --out figures/
"""

from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TIERS = ["static", "moving", "debris"]
BG = "#0d0221"
COLORS = {"static": "#7dd3fc", "moving": "#f0abfc", "debris": "#fca5a5"}


def _load(d: pathlib.Path):
    out = {}
    for t in TIERS:
        p = d / f"{t}.json"
        if p.exists():
            out[t] = json.loads(p.read_text())
    return out


def plot_curves(data, out_path):
    metrics = [("dynamic_iou_model", "dynamic IOU (forecasting)"),
               ("dynamic_recall_model", "dynamic recall"),
               ("static_coverage_model", "static coverage (completion)")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), facecolor=BG)
    for ax, (key, title) in zip(axes, metrics):
        for t, d in data.items():
            hs = d["horizons"]
            ys = [d["decomposition"][f"{h}s"][key] for h in hs]
            ax.plot(hs, ys, marker="o", label=t, color=COLORS.get(t, "#fff"))
        ax.set_title(title, color="#7dd3fc")
        ax.set_xlabel("horizon (s)", color="#ccc"); ax.set_facecolor(BG)
        ax.tick_params(colors="#ccc"); ax.grid(alpha=0.15)
        for sp in ax.spines.values():
            sp.set_color("#444")
    axes[0].set_ylabel("score", color="#ccc")
    axes[0].legend(facecolor=BG, edgecolor="#444", labelcolor="#ccc")
    plt.tight_layout()
    fig.savefig(out_path, dpi=130, facecolor=BG)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_reliability(data, out_path):
    # Use the debris (or first available) tier's 5s reliability curve.
    tier = "debris" if "debris" in data else next(iter(data))
    cal = data[tier]["calibration"]["5s"]
    fig, ax = plt.subplots(figsize=(5.5, 5), facecolor=BG)
    ax.plot([0, 1], [0, 1], "--", color="#666", label="perfect")
    for region, color in (("shadow", "#f0abfc"), ("observed", "#7dd3fc")):
        bins = cal[region]["reliability"] or []
        if not bins:
            continue
        xs = [b["conf"] for b in bins]; ys = [b["acc"] for b in bins]
        ax.plot(xs, ys, marker="o", color=color,
                label=f"{region} (ECE={cal[region]['ece']:.3f})")
    ax.set_title(f"reliability @5s — {tier} tier", color="#7dd3fc")
    ax.set_xlabel("predicted probability", color="#ccc")
    ax.set_ylabel("empirical occupancy", color="#ccc")
    ax.set_facecolor(BG); ax.tick_params(colors="#ccc"); ax.grid(alpha=0.15)
    for sp in ax.spines.values():
        sp.set_color("#444")
    ax.legend(facecolor=BG, edgecolor="#444", labelcolor="#ccc")
    plt.tight_layout()
    fig.savefig(out_path, dpi=130, facecolor=BG)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot decomposition + calibration figures")
    ap.add_argument("--dir", type=str, required=True, help="dir with {static,moving,debris}.json")
    ap.add_argument("--out", type=str, default="figures")
    args = ap.parse_args()

    data = _load(pathlib.Path(args.dir))
    if not data:
        raise SystemExit(f"no tier jsons found in {args.dir}")
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    plot_curves(data, out / "decomp_curves.png")
    plot_reliability(data, out / "reliability.png")


if __name__ == "__main__":
    main()
