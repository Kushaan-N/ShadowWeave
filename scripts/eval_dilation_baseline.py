"""Observation-conditioned heuristic baseline: dilate observed occupancy into shadow.

The reviewer attack this answers: "the observation-blind prior is a strawman for a
different reason — a simple observation-conditioned rule (grow every observed obstacle
outward into the occluded region) might complete hidden structure as well as the
learned model." Persistence copies the observation; this heuristic extrapolates it.

For each validation frame the prediction is binary_dilation(observed occupancy, r)
for r in 1..6 cells (1 cell ~ 10.4 cm at 96 px / 10 m), scored with the same pooled
micro counts as eval_val_decomp.py, reported at the radius that maximizes micro shadow
IoU — the heuristic's most favorable reading, mirroring the prior baseline protocol.
CPU-only; login-node safe.

    python scripts/eval_dilation_baseline.py --data $WS/rollouts_val_moving \
        --out $WS/val_decomp/dilate_moving.json --stride 3
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch
from scipy import ndimage

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

from shadowweave.utils import load_config  # noqa: E402
from shadowweave.world_model.dataset import RolloutDataset  # noqa: E402

RADII = [1, 2, 3, 4, 5, 6]


def _ratio(num: int, den: int) -> float:
    return num / den if den > 0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description="Dilation heuristic baseline for shadow completion")
    ap.add_argument("--data", required=True)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_config()
    ds = RolloutDataset(cfg, args.data, split="val", augment=False)
    horizons = list(cfg.world_model.prediction_horizons)
    T = len(horizons)

    z = lambda: torch.zeros(len(RADII), T, dtype=torch.long)
    acc = {k: z() for k in ["static_p", "dyn_i", "dyn_u", "never_p"]}
    n_static = n_never = 0
    n_frames = 0
    for i in range(0, len(ds), args.stride):
        fi, t = ds._index[i]
        target = torch.from_numpy(np.ascontiguousarray(ds._get_map(fi, "target")[t])).float()
        occ = np.ascontiguousarray(ds._get_map(fi, "bev_occupancy")[t][0]) > 0.5
        vis = torch.from_numpy(np.ascontiguousarray(
            ds._get_map(fi, "bev_visibility")[t][0])).float()
        preds = [torch.from_numpy(ndimage.binary_dilation(occ, iterations=r)) for r in RADII]
        tb = target > 0.5
        shadow = vis < 0.5
        ever, always = tb.any(dim=0), tb.all(dim=0)
        STATIC = shadow & always
        DYNAMIC = shadow & ever & (~always)
        NEVER = shadow & (~ever)
        n_static += int(STATIC.sum()); n_never += int(NEVER.sum())
        for h in range(T):
            th = tb[h]
            for j, pb in enumerate(preds):
                acc["static_p"][j, h] += (pb & STATIC).sum()
                acc["dyn_i"][j, h] += (pb & th & DYNAMIC).sum()
                acc["dyn_u"][j, h] += ((pb | th) & DYNAMIC).sum()
                acc["never_p"][j, h] += (pb & NEVER).sum()
        n_frames += 1
        if args.max_frames and n_frames >= args.max_frames:
            print(f"stopping early after {n_frames} frames (--max-frames)")
            break

    out = {"horizons": horizons, "n_val_frames": n_frames, "val_stride": args.stride,
           "radii": RADII, "decomposition": {}}
    for h, hs in enumerate(horizons):
        per_r = {}
        for j, r in enumerate(RADII):
            sp, di, du, np_ = (int(acc[k][j, h]) for k in
                               ("static_p", "dyn_i", "dyn_u", "never_p"))
            per_r[str(r)] = {
                "static_coverage_dilate": _ratio(sp, n_static),
                "shadow_fp_rate_dilate": _ratio(np_, n_never),
                "micro_shadow_iou_dilate": _ratio(sp + di, n_static + np_ + du),
            }
        best_r = max(per_r, key=lambda k: per_r[k]["micro_shadow_iou_dilate"])
        out["decomposition"][f"{hs}s"] = {"by_radius": per_r, "best_radius": int(best_r),
                                          "best": per_r[best_r]}
    p = pathlib.Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(f"wrote {p}")
    for hs, d in out["decomposition"].items():
        print(hs, "best_r", d["best_radius"],
              {k: round(v, 4) for k, v in d["best"].items()})


if __name__ == "__main__":
    main()
