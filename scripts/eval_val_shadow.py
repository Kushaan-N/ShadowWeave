"""Shadow-IOU on the FIXED validation set — an apples-to-apples world-model comparison.

The main eval (run_eval.py) drives the PPO policy, so different world models visit
different trajectories and the shadow region being scored differs run-to-run. That is fine
for reporting deployed performance but confounds an ABLATION table: you cannot tell whether
a shadow-IOU change came from the model or from the policy wandering somewhere easier.

This script removes the policy entirely. It scores every model on the identical stored
validation rollouts, using the SAME shadow metric as the rollout eval:

    shadow mask       = visibility < 0.5                 (unobserved cells)
    model shadow IOU  = masked_iou(model.predict(x), target_h, mask)
    persistence       = masked_iou(observed occupancy,  target_h, mask)   # carried forward
    empty             = masked_iou(zeros,               target_h, mask)
    shadow_gain_h     = model_shadow_iou - max(persistence, empty)        # the paper's claim

The mask and baselines are built from the TRUE stored visibility/occupancy, identically for
every model — only model.predict() differs — so the numbers are directly comparable across
the full model and each ablation checkpoint.

    python scripts/eval_val_shadow.py --ckpt $WS/checkpoints/world_model/best.pt --data $WS/rollouts
    python scripts/eval_val_shadow.py --ckpt <ablation>/best.pt --data $WS/rollouts --out cmp.json
    # quick CPU sanity check on a few batches:
    python scripts/eval_val_shadow.py --ckpt ... --data ... --max-batches 4
"""

from __future__ import annotations

import argparse
import json
import pathlib

import shadowweave  # noqa: F401  (sets env before torch import)

import numpy as np
import torch

from shadowweave.eval.metrics import iou, masked_iou
from shadowweave.utils import config_from_checkpoint, get_device, load_checkpoint, load_config
from shadowweave.world_model import build_world_model
from shadowweave.world_model.dataset import RolloutDataset


@torch.no_grad()
def evaluate(ckpt_path: str, data_root: str, split: str, cfg,
             batch_size: int, max_batches: int | None, device) -> dict:
    ckpt = pathlib.Path(ckpt_path)
    # Rebuild the model under the config it was trained with (architecture + input channels
    # both come from the checkpoint, so an ablation ckpt loads with its own smaller input).
    wm_cfg = config_from_checkpoint(ckpt, cfg)
    model = build_world_model(wm_cfg).to(device)
    model.load_state_dict(load_checkpoint(ckpt, map_location=device)["model"])
    model.eval()
    channels = list(wm_cfg.bev.input_channels)
    horizons = list(wm_cfg.world_model.prediction_horizons)
    print(f"loaded {ckpt}  arch={wm_cfg.world_model.architecture}  channels={channels}")

    # augment=False: evaluation must be deterministic (no random mirroring).
    ds = RolloutDataset(wm_cfg, data_root, split=split, augment=False)
    n = len(ds)
    print(f"validation samples: {n}")

    T = len(horizons)
    model_sh = [[] for _ in range(T)]
    persist_sh = [[] for _ in range(T)]
    empty_sh = [[] for _ in range(T)]
    model_all = [[] for _ in range(T)]

    order = list(range(n))
    nb = 0
    for start in range(0, n, batch_size):
        batch_idx = order[start:start + batch_size]
        xs, tgts, occs, viss = [], [], [], []
        for idx in batch_idx:
            fi, t = ds._index[idx]
            xs.append(ds[idx]["input"])
            tgts.append(torch.from_numpy(np.ascontiguousarray(ds._get_map(fi, "target")[t])).float())
            occs.append(torch.from_numpy(np.ascontiguousarray(ds._get_map(fi, "bev_occupancy")[t])).float())
            viss.append(torch.from_numpy(np.ascontiguousarray(ds._get_map(fi, "bev_visibility")[t])).float())
        x = torch.stack(xs).to(device)                      # (B, C, S, S)
        target = torch.stack(tgts).to(device)               # (B, T, S, S)
        occ = torch.stack(occs)[:, 0].to(device)            # (B, S, S) observed occupancy
        vis = torch.stack(viss)[:, 0].to(device)            # (B, S, S) visibility
        mask = (vis < 0.5)                                  # (B, S, S) True = shadow

        pred = model.predict(x)                             # (B, T, S, S) sigmoid probs
        if nb == 0:
            # Guard the multi-checkpoint reruns: if this data dir was generated with a
            # different horizon set than the checkpoint was trained on, pred[:,h] and
            # target[:,h] would silently misalign (or IndexError). Fail loudly instead.
            assert pred.shape[1] == T and target.shape[1] == T, (
                f"horizon count mismatch: pred {tuple(pred.shape)}, target "
                f"{tuple(target.shape)}, expected T={T} from checkpoint prediction_horizons")
        empty = torch.zeros_like(occ)
        for h in range(T):
            model_sh[h].append(float(masked_iou(pred[:, h], target[:, h], mask).mean()))
            persist_sh[h].append(float(masked_iou(occ, target[:, h], mask).mean()))
            empty_sh[h].append(float(masked_iou(empty, target[:, h], mask).mean()))
            model_all[h].append(float(iou(pred[:, h], target[:, h]).mean()))
        nb += 1
        if max_batches is not None and nb >= max_batches:
            print(f"stopping early after {nb} batches (--max-batches)")
            break

    out: dict[str, float] = {}
    for h, hs in enumerate(horizons):
        label = f"{int(hs)}s"
        m = float(np.mean(model_sh[h])); p = float(np.mean(persist_sh[h])); e = float(np.mean(empty_sh[h]))
        best = max(p, e)
        out[f"val_shadow_iou_{label}"] = m
        out[f"val_persistence_shadow_iou_{label}"] = p
        out[f"val_empty_shadow_iou_{label}"] = e
        out[f"val_shadow_gain_{label}"] = m - best
        out[f"val_iou_{label}"] = float(np.mean(model_all[h]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Shadow-IOU on the fixed validation set (no policy)")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--data", type=str, required=True, help="rollout root (expects <root>/<split>/)")
    ap.add_argument("--split", type=str, default="val")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-batches", type=int, default=None, help="cap batches for a quick check")
    ap.add_argument("--out", type=str, default=None, help="also write the metrics as json")
    ap.add_argument("--overrides", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(overrides=args.overrides)
    device = get_device()
    print(f"device: {device}")
    out = evaluate(args.ckpt, args.data, args.split, cfg, args.batch_size, args.max_batches, device)

    print("\n=== fixed-val shadow metrics ===")
    print(f"{'horizon':>8}  {'model':>7}  {'persist':>7}  {'gain':>7}  {'all-iou':>7}")
    for hs in load_config(overrides=args.overrides).world_model.prediction_horizons:
        label = f"{int(hs)}s"
        print(f"{label:>8}  {out[f'val_shadow_iou_{label}']:>7.3f}  "
              f"{out[f'val_persistence_shadow_iou_{label}']:>7.3f}  "
              f"{out[f'val_shadow_gain_{label}']:>+7.3f}  {out[f'val_iou_{label}']:>7.3f}")

    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
