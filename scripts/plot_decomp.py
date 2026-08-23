"""Figures for the shadow decomposition + calibration analysis.

Reads the per-tier json written by eval_val_decomp.py (one dir per model) and produces:
  1. decomp_curves.png  — per-horizon dynamic_iou / dynamic_recall / static_coverage,
     one line per tier with the persistence baseline dashed, showing the forecasting
     signal decaying with horizon (which the pooled shadow-IOU masked) and that the
     completion signal is flat.
  2. reliability.png     — reliability diagram at 5s, shadow vs observed — the
     persuasive figure for "uncertainty propagation into unobserved space".
  3. shadow_gain_ci.png  — micro shadow gain (left) and nostatic gain (right) vs
     horizon, with 95% bootstrap-CI bands per tier and, if --randgeom is given, the
     randomized-geometry static run overlaid. This is the "every number has an error
     bar" figure, and the left/right split shows the gain is completion (large, CI
     clear of zero) not pure forecasting (nostatic ~0).

Figures are white-background / colorblind-safe and drawn near their printed size
(a NeurIPS column is 5.5 in), so fonts stay legible after LaTeX scaling.

    python scripts/plot_decomp.py --dir <WS>/val_decomp/full --out figures/ \
        --randgeom <WS>/val_decomp/randgeom.json
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TIERS = ["static", "moving", "debris"]
# Okabe-Ito, readable on white and in grayscale.
COLORS = {"static": "#0072B2", "moving": "#D55E00", "debris": "#009E73"}


def _load(d: pathlib.Path):
    out = {}
    for t in TIERS:
        p = d / f"{t}.json"
        if p.exists():
            out[t] = json.loads(p.read_text())
    return out


def _series(d, key):
    hs = d["horizons"]
    ys = [d["decomposition"][f"{h}s"].get(key, float("nan")) for h in hs]
    return hs, ys


def plot_curves(data, out_path):
    # (model_key, persistence_key, title). Persistence is drawn dashed so the reader
    # can see where the model beats the baseline and where it does not (debris
    # dynamic IOU at 3-10s) — hiding that would be indefensible in review.
    panels = [("dynamic_iou_model", "dynamic_iou_persist", "dynamic IOU in shadow\n(forecasting)"),
              ("dynamic_recall_model", None, "dynamic recall in shadow"),
              ("static_coverage_model", "static_coverage_persist",
               "persistent-structure recall\n(amodal completion)")]
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.2))
    for ax, (mkey, pkey, title) in zip(axes, panels):
        for t, d in data.items():
            hs, ys = _series(d, mkey)
            if all(isinstance(y, float) and math.isnan(y) for y in ys):
                continue  # e.g. the static tier has no dynamic cells at all
            ax.plot(hs, ys, marker="o", label=t, color=COLORS.get(t, "#333"))
            if pkey:
                _, yp = _series(d, pkey)
                ax.plot(hs, yp, linestyle="--", marker=".", alpha=0.6,
                        color=COLORS.get(t, "#333"))
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("horizon (s)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("score")
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color="#555", linestyle="--", marker="."))
    labels.append("persistence (dashed)")
    axes[0].legend(handles, labels, fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"wrote {out_path}")


def _gain_series(d, key):
    """(horizons, point, ci_lo, ci_hi) for a gain key with a matching `<key>_ci`."""
    hs = d["horizons"]
    pt, lo, hi = [], [], []
    for h in hs:
        cell = d["decomposition"][f"{h}s"]
        pt.append(cell[key])
        ci = cell.get(f"{key}_ci", [cell[key], cell[key]])
        lo.append(ci[0])
        hi.append(ci[1])
    return hs, pt, lo, hi


def plot_gain_ci(data, randgeom, out_path):
    # Left: micro shadow gain (dominated by completion) — large, CI clear of zero.
    # Right: nostatic gain (persistent structure removed) — the pure-forecasting
    # residual, ~0. Showing both with error bands is the honest headline.
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(8.0, 3.2))
    for key, ax, title in ((("micro_shadow_gain"), axL, "micro shadow gain\n(completion + forecasting)"),
                           (("micro_shadow_gain_nostatic"), axR,
                            "nostatic micro gain\n(persistent structure removed)")):
        for t, d in data.items():
            hs, pt, lo, hi = _gain_series(d, key)
            c = COLORS.get(t, "#333")
            ax.plot(hs, pt, marker="o", label=t, color=c)
            ax.fill_between(hs, lo, hi, color=c, alpha=0.18, linewidth=0)
        if randgeom is not None:
            hs, pt, lo, hi = _gain_series(randgeom, key)
            ax.plot(hs, pt, marker="s", linestyle="--", color="#000",
                    label="static, rand-geom")
            ax.fill_between(hs, lo, hi, color="#000", alpha=0.12, linewidth=0)
        ax.axhline(0.0, color="#999", linewidth=0.8)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("horizon (s)")
        ax.grid(alpha=0.25)
    axL.set_ylabel("shadow IOU gain over persistence")
    axL.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_reliability(data, out_path):
    # Use the debris (or first available) tier's 5s reliability curve.
    tier = "debris" if "debris" in data else next(iter(data))
    cal = data[tier]["calibration"]["5s"]
    fig, ax = plt.subplots(figsize=(3.8, 3.5))
    ax.plot([0, 1], [0, 1], "--", color="#999", label="perfect")
    for region, color in (("shadow", "#D55E00"), ("observed", "#0072B2")):
        bins = cal[region]["reliability"] or []
        if not bins:
            continue
        xs = [b["conf"] for b in bins]
        ys = [b["acc"] for b in bins]
        ax.plot(xs, ys, marker="o", color=color,
                label=f"{region} (ECE={cal[region]['ece']:.3f})")
    ax.set_title(f"reliability @5s — {tier} tier", fontsize=11)
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("empirical occupancy")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot decomposition + calibration figures")
    ap.add_argument("--dir", type=str, required=True, help="dir with {static,moving,debris}.json")
    ap.add_argument("--randgeom", type=str, default=None,
                    help="optional randgeom.json to overlay on the gain-CI figure")
    ap.add_argument("--out", type=str, default="figures")
    args = ap.parse_args()

    data = _load(pathlib.Path(args.dir))
    if not data:
        raise SystemExit(f"no tier jsons found in {args.dir}")
    randgeom = None
    if args.randgeom:
        rp = pathlib.Path(args.randgeom)
        if rp.exists():
            randgeom = json.loads(rp.read_text())
        else:
            print(f"warning: {rp} not found, skipping randgeom overlay")
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plot_curves(data, out / "decomp_curves.png")
    plot_reliability(data, out / "reliability.png")
    plot_gain_ci(data, randgeom, out / "shadow_gain_ci.png")


if __name__ == "__main__":
    main()
