"""Render BEV prediction figures from a trained checkpoint.

Numbers say whether the model is good; pictures say *how* it is wrong — whether it is
blurring, lagging, hallucinating into shadow, or ignoring its input. Writes PNGs you
can drop straight into a write-up.

    python scripts/visualize.py
    python scripts/visualize.py --ckpt checkpoints/world_model/best.pt --n 4
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

from shadowweave.utils import (  # noqa: E402
    config_from_checkpoint,
    get_device,
    load_checkpoint,
    load_config,
)
from shadowweave.world_model.dataset import RolloutDataset  # noqa: E402
from shadowweave.world_model.diffusion import build_world_model  # noqa: E402

NEON = [(0.05, 0.01, 0.13), (0.0, 1.0, 0.6), (1.0, 0.0, 1.0)]


def _cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("shadowweave", NEON)


def render_samples(cfg, model, ds, device, n: int, out_dir: pathlib.Path) -> list[pathlib.Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    horizons = list(cfg.world_model.prediction_horizons)
    cmap = _cmap()
    written = []
    idxs = np.linspace(0, len(ds) - 1, n).astype(int)

    for k, idx in enumerate(idxs):
        sample = ds[int(idx)]
        x = sample["input"].unsqueeze(0).to(device)
        y = sample["target"].numpy()
        with torch.no_grad():
            p = torch.sigmoid(model(x))[0].cpu().numpy()

        n_cols = len(horizons)
        fig, axes = plt.subplots(4, n_cols, figsize=(2.6 * n_cols, 10.5), facecolor="#0d0221")

        obs = sample["input"][0].numpy()
        vis = sample["input"][1].numpy() if x.shape[1] > 1 else np.ones_like(obs)

        def panel(ax, img, title, cm, vmin, vmax, shadow=False):
            ax.imshow(img, cmap=cm, vmin=vmin, vmax=vmax, interpolation="nearest")
            if shadow:
                # Outline the shadow region so it is obvious which predictions are
                # into unobserved space — the part that actually matters.
                ax.contour(1.0 - vis, levels=[0.5], colors="#ff5500", linewidths=0.8)
            ax.set_title(title, color="#00ff99", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            # A visible frame; an almost-empty target otherwise vanishes into the
            # figure background and you cannot tell the grid extent.
            for spine in ax.spines.values():
                spine.set_edgecolor("#2a1a4a")
                spine.set_linewidth(0.8)

        # The observation is the same for every horizon, so show it once rather than
        # repeating it across the row.
        panel(axes[0, 0], obs, "observed (t)", cmap, 0, 1, shadow=True)
        if n_cols > 1:
            panel(axes[0, 1], vis, "visibility (dark = shadow)", "gray", 0, 1)
        for c in range(2, n_cols):
            axes[0, c].axis("off")

        for c, h in enumerate(horizons):
            panel(axes[1, c], y[c], f"truth  t+{h:g}s", cmap, 0, 1)
            panel(axes[2, c], p[c], f"predicted  t+{h:g}s", cmap, 0, 1)
            panel(axes[3, c], (p[c] > 0.5).astype(float) - y[c], "error", "bwr", -1, 1)

        fig.suptitle(
            "ShadowWeave — egocentric BEV (agent at bottom-centre, facing up)\n"
            "row 1: observation + shadow   rows 2-4: truth / prediction / error "
            "(red = false positive, blue = missed)",
            color="#00ff99", fontsize=10,
        )
        plt.tight_layout()
        path = out_dir / f"prediction_{k:02d}.png"
        fig.savefig(path, dpi=130, facecolor="#0d0221")
        plt.close(fig)
        written.append(path)
    return written


def render_curve(ckpt: dict, out_dir: pathlib.Path) -> pathlib.Path | None:
    """Plot whatever metrics the checkpoint recorded."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = ckpt.get("metrics") or {}
    ious = {k: v for k, v in metrics.items() if k.startswith("iou_")}
    if not ious:
        return None

    fig, ax = plt.subplots(figsize=(6, 3.5), facecolor="#0d0221")
    ax.set_facecolor("#0d0221")
    labels = list(ious)
    ax.bar(labels, [ious[k] for k in labels], color="#ff00ff")
    ax.axhline(0.6, color="#00ff99", linestyle="--", label="target 0.60")
    ax.set_ylim(0, 1)
    ax.set_ylabel("IOU", color="#00ff99")
    ax.set_title(f"Validation IOU @ epoch {ckpt.get('epoch', '?')}", color="#00ff99")
    ax.tick_params(colors="#00ff99")
    ax.legend(facecolor="#0d0221", labelcolor="#00ff99")
    plt.tight_layout()
    path = out_dir / "val_iou.png"
    fig.savefig(path, dpi=130, facecolor="#0d0221")
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="Render BEV prediction figures")
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--data", type=str, default=None)
    ap.add_argument("--split", type=str, default="val")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--overrides", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(overrides=args.overrides)
    device = get_device()
    ckpt_path = pathlib.Path(args.ckpt or (pathlib.Path(cfg.world_model.checkpoint_dir) / "best.pt"))
    out_dir = pathlib.Path(args.out or (pathlib.Path(cfg.eval.results_dir) / "figures"))
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ckpt_path.exists():
        print(f"No checkpoint at {ckpt_path}. Train one first:  make train")
        raise SystemExit(1)

    ckpt = load_checkpoint(ckpt_path, map_location=device)
    wm_cfg = config_from_checkpoint(ckpt_path, cfg)
    model = build_world_model(wm_cfg).to(device).eval()
    model.load_state_dict(ckpt["model"])
    print(f"Loaded {ckpt_path} (base_channels={wm_cfg.world_model.base_channels}, "
          f"epoch={ckpt.get('epoch')})")

    try:
        ds = RolloutDataset(cfg, args.data or cfg.data.root, split=args.split)
    except (FileNotFoundError, ValueError) as e:
        print(f"Cannot load {args.split} rollouts: {str(e)[:120]}")
        raise SystemExit(1)

    written = render_samples(cfg, model, ds, device, args.n, out_dir)
    curve = render_curve(ckpt, out_dir)
    for p in written + ([curve] if curve else []):
        print(f"  wrote {p}")


if __name__ == "__main__":
    main()
