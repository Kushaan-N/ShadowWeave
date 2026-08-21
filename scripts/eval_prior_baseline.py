"""Dataset-prior baseline for the shadow-completion claim.

The reviewer attack this answers: "persistence is a weak baseline for *completion* —
maybe a fixed pose-marginal prior (the mean training-set occupancy in the egocentric
frame, no observation needed) completes hidden structure just as well." The randomized-
geometry experiment rules out a memorized *layout*; this rules out a memorized
*marginal*. If the observation-blind prior scores far below the model, completion is
observation-conditional inference, not an average-room lookup.

Pass 1 accumulates the per-horizon mean target over (strided) training frames; pass 2
scores that fixed grid, thresholded exactly like the model (0.5), with the same pooled
micro counts as eval_val_decomp.py (empty-credit-immune). CPU-only; login-node safe.

    python scripts/eval_prior_baseline.py --train-data $WS/rollouts \
        --val-data $WS/rollouts_val_moving --out $WS/val_decomp/prior_moving.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

from shadowweave.utils import load_config  # noqa: E402
from shadowweave.world_model.dataset import RolloutDataset  # noqa: E402


def _ratio(num: int, den: int) -> float:
    return num / den if den > 0 else float("nan")


def build_prior(cfg, root: str, stride: int) -> torch.Tensor:
    """Per-horizon mean target over every `stride`-th training frame. (T, S, S)."""
    ds = RolloutDataset(cfg, root, split="train", augment=False)
    total = None
    n = 0
    for i in range(0, len(ds), stride):
        fi, t = ds._index[i]
        tgt = torch.from_numpy(np.ascontiguousarray(ds._get_map(fi, "target")[t])).float()
        total = tgt if total is None else total + tgt
        n += 1
    print(f"prior built from {n} training frames (stride {stride})")
    return total / n


# The model is scored at 0.5; a marginal prior may never reach 0.5 anywhere, so scoring
# it only there would be a strawman. Sweep thresholds and report the prior at its BEST —
# if even that is far below the model, the comparison is airtight.
THRESHOLDS = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]


def evaluate(cfg, prior: torch.Tensor, root: str, stride: int, max_frames=None) -> dict:
    ds = RolloutDataset(cfg, root, split="val", augment=False)
    T = prior.shape[0]
    horizons = list(cfg.world_model.prediction_horizons)
    pbs = [prior > th for th in THRESHOLDS]              # fixed, observation-blind
    z = lambda: torch.zeros(len(THRESHOLDS), T, dtype=torch.long)
    acc = {k: z() for k in ["static_p", "dyn_i", "dyn_u", "never_p"]}
    dyn_t = torch.zeros(T, dtype=torch.long)
    n_static = n_never = n_shadow = 0
    n_frames = 0
    for i in range(0, len(ds), stride):
        fi, t = ds._index[i]
        target = torch.from_numpy(np.ascontiguousarray(ds._get_map(fi, "target")[t])).float()
        vis = torch.from_numpy(np.ascontiguousarray(
            ds._get_map(fi, "bev_visibility")[t][0])).float()
        tb = target > 0.5                                # (T, S, S)
        shadow = vis < 0.5
        ever, always = tb.any(dim=0), tb.all(dim=0)
        STATIC = shadow & always
        DYNAMIC = shadow & ever & (~always)
        NEVER = shadow & (~ever)
        n_static += int(STATIC.sum()); n_never += int(NEVER.sum())
        n_shadow += int(shadow.sum())
        for h in range(T):
            th_mask = tb[h]
            dyn_t[h] += (th_mask & DYNAMIC).sum()
            for j, pb in enumerate(pbs):
                acc["static_p"][j, h] += (pb[h] & STATIC).sum()
                acc["dyn_i"][j, h] += (pb[h] & th_mask & DYNAMIC).sum()
                acc["dyn_u"][j, h] += ((pb[h] | th_mask) & DYNAMIC).sum()
                acc["never_p"][j, h] += (pb[h] & NEVER).sum()
        n_frames += 1
        if max_frames and n_frames >= max_frames:
            print(f"stopping early after {n_frames} frames (--max-frames)")
            break

    out = {"horizons": horizons, "n_val_frames": n_frames, "val_stride": stride,
           "thresholds": THRESHOLDS,
           "prior_positive_cells_per_threshold_at_5s": {},
           "decomposition": {}}
    h5 = horizons.index(5) if 5 in horizons else 0
    for j, th in enumerate(THRESHOLDS):
        out["prior_positive_cells_per_threshold_at_5s"][str(th)] = int(pbs[j][h5].sum())
    for h, hs in enumerate(horizons):
        per_th = {}
        for j, th in enumerate(THRESHOLDS):
            sp, di, du, np_ = (int(acc[k][j, h]) for k in
                               ("static_p", "dyn_i", "dyn_u", "never_p"))
            per_th[str(th)] = {
                "static_coverage_prior": _ratio(sp, n_static),
                "dynamic_iou_prior": _ratio(di, du),
                "shadow_fp_rate_prior": _ratio(np_, n_never),
                "micro_shadow_iou_prior": _ratio(sp + di, n_static + np_ + du),
                "micro_shadow_iou_nostatic_prior": _ratio(di, du + np_),
            }
        best_th = max(per_th, key=lambda k: (per_th[k]["micro_shadow_iou_prior"]
                                             if per_th[k]["micro_shadow_iou_prior"] == per_th[k]["micro_shadow_iou_prior"] else -1.0))
        out["decomposition"][f"{hs}s"] = {"by_threshold": per_th,
                                          "best_threshold": float(best_th),
                                          "best": per_th[best_th]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Observation-blind dataset-prior baseline")
    ap.add_argument("--train-data", required=True)
    ap.add_argument("--val-data", required=True)
    ap.add_argument("--train-stride", type=int, default=10)
    ap.add_argument("--val-stride", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_config()
    prior = build_prior(cfg, args.train_data, args.train_stride)
    result = evaluate(cfg, prior, args.val_data, args.val_stride, args.max_frames)
    p = pathlib.Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2))
    print(f"wrote {p}")
    for hs, d in result["decomposition"].items():
        best = {k: (round(v, 4) if v == v else "nan") for k, v in d["best"].items()}
        print(hs, "best_th", d["best_threshold"], best)


if __name__ == "__main__":
    main()
