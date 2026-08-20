"""Static/dynamic shadow decomposition + calibration-in-shadow, on the fixed val set.

Two eval-only analyses that the aggregate shadow-IOU cannot deliver, both on identical
stored validation rollouts (no retraining, no regen). Design locked by adversarial review.

A) DECOMPOSITION (A1). Within the shadow mask (vis<0.5), split cells by their occupancy
   pattern ACROSS the stored horizons — valid because every target horizon is rendered at
   the SAME issue pose, so static geometry projects to identical cells at every horizon:
     STATIC  = shadow & occupied at ALL horizons   (persistent hidden structure; walls)
     DYNAMIC = shadow & occupied at SOME-not-all    (transient-in-window; forecasting)
     NEVER   = shadow & occupied at NO horizon      (free hidden space; false-positive test)
   Metrics are MICRO-averaged (pool integer counts, divide once) so the empty-credit trap
   in masked_iou (union==0 -> 1.0) cannot inflate rare sub-masks; empty denominators report
   NaN, never 1.0:
     static_coverage[h] = |pred & STATIC| / |STATIC|          (amodal completion; a recall)
     dynamic_iou[h]     = |pred & tgt & DYNAMIC| / |(pred|tgt) & DYNAMIC|   (true IOU)
     dynamic_recall[h]  = |pred & tgt & DYNAMIC| / |tgt & DYNAMIC|
     shadow_fp_rate[h]  = |pred & NEVER| / |NEVER|             (precision counterweight)
   Persistence (observed occupancy carried forward) and empty are ~0 on every shadow
   sub-mask, so they are reported once as the ~0 floor. DYNAMIC = "transient-in-window"
   (arrivals + departures + pass-throughs), all legitimately beyond persistence.

B) CALIBRATION-IN-SHADOW. Pooled reliability of the model's occupancy probability restricted
   to shadow cells (vs observed cells for contrast, and walls-excluded = shadow & ~STATIC).
   Equal-MASS binning via a fine 1000-bin histogram merged into 15 equal-count macro-bins
   (in-shadow probs cluster near the prior, so equal-width bins would be mostly empty):
     ECE = sum_b (n_b/N) |conf_b - acc_b| ;  also Brier score and the reliability curve.
   Evidence for "uncertainty propagation through unobserved space": ece_shadow close to
   ece_observed => faithful uncertainty in unseen space; the per-horizon curve => propagation.

    python scripts/eval_val_decomp.py --ckpt <ckpt> --data <val root> --out decomp.json
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib

import shadowweave  # noqa: F401

import numpy as np
import torch

from shadowweave.utils import config_from_checkpoint, get_device, load_checkpoint, load_config
from shadowweave.world_model import build_world_model
from shadowweave.world_model.dataset import RolloutDataset

NBIN_FINE = 1000     # fine histogram resolution for equal-mass merging
NBIN_MACRO = 15      # reported reliability bins


def _ratio(num: int, den: int) -> float:
    return float(num) / float(den) if den > 0 else float("nan")


def _equal_mass_ece(count, sum_prob, sum_truth):
    """Merge a fine (prob) histogram into ~NBIN_MACRO equal-count macro-bins and return ECE
    plus the reliability curve. count/sum_prob/sum_truth are (NBIN_FINE,) accumulators."""
    N = int(count.sum())
    if N == 0:
        return {"ece": float("nan"), "n": 0, "bins": []}
    target_per = N / NBIN_MACRO
    out_bins = []
    ece = 0.0
    cur_c = cur_p = cur_t = 0.0
    for i in range(NBIN_FINE):
        cur_c += float(count[i]); cur_p += float(sum_prob[i]); cur_t += float(sum_truth[i])
        # close the macro-bin once it holds ~1/NBIN_MACRO of the mass, or at the last bin
        if (cur_c >= target_per and len(out_bins) < NBIN_MACRO - 1) or i == NBIN_FINE - 1:
            if cur_c <= 0:
                continue
            conf, acc = cur_p / cur_c, cur_t / cur_c
            ece += (cur_c / N) * abs(conf - acc)
            out_bins.append({"conf": conf, "acc": acc, "count": int(cur_c)})
            cur_c = cur_p = cur_t = 0.0
    return {"ece": ece, "n": N, "bins": out_bins}


@torch.no_grad()
def evaluate(ckpt_path, data_root, split, cfg, batch_size, max_batches, device):
    ckpt = pathlib.Path(ckpt_path)
    wm_cfg = config_from_checkpoint(ckpt, cfg)
    model = build_world_model(wm_cfg).to(device)
    model.load_state_dict(load_checkpoint(ckpt, map_location=device)["model"])
    model.eval()
    horizons = [int(h) for h in wm_cfg.world_model.prediction_horizons]
    T = len(horizons)
    print(f"loaded {ckpt}  arch={wm_cfg.world_model.architecture}  horizons={horizons}")

    ds = RolloutDataset(wm_cfg, data_root, split=split, augment=False)
    n = len(ds)
    print(f"validation samples: {n}")

    # --- A1 integer accumulators (micro-average) ---
    z = lambda: torch.zeros(T, dtype=torch.long)
    n_static = torch.zeros((), dtype=torch.long)
    n_dynamic = torch.zeros((), dtype=torch.long)
    n_never = torch.zeros((), dtype=torch.long)
    n_shadow = torch.zeros((), dtype=torch.long)
    acc = {k: z() for k in [
        "static_m", "dyn_i_m", "dyn_u_m", "dyn_t", "never_m",   # model
        "static_p", "dyn_i_p", "dyn_u_p", "never_p",            # persistence
    ]}

    # --- B calibration fine-histograms: region -> horizon -> [count, sum_prob, sum_truth, sum_sq] ---
    regions = ["shadow", "observed", "shadow_nowall"]
    hist = {r: {h: [torch.zeros(NBIN_FINE), torch.zeros(NBIN_FINE), torch.zeros(NBIN_FINE), torch.zeros(())]
                for h in range(T)} for r in regions}

    nb = 0
    for start in range(0, n, batch_size):
        idxs = range(start, min(start + batch_size, n))
        xs, tgts, occs, viss = [], [], [], []
        for i in idxs:
            fi, t = ds._index[i]
            xs.append(ds[i]["input"])
            tgts.append(torch.from_numpy(np.ascontiguousarray(ds._get_map(fi, "target")[t])).float())
            occs.append(torch.from_numpy(np.ascontiguousarray(ds._get_map(fi, "bev_occupancy")[t])).float())
            viss.append(torch.from_numpy(np.ascontiguousarray(ds._get_map(fi, "bev_visibility")[t])).float())
        x = torch.stack(xs).to(device)
        target = torch.stack(tgts).to(device)               # (B,T,S,S)
        occ = torch.stack(occs)[:, 0].to(device)            # (B,S,S)
        vis = torch.stack(viss)[:, 0].to(device)
        pred = model.predict(x)                             # (B,T,S,S) probs
        if nb == 0:
            assert pred.shape[1] == T and target.shape[1] == T, "horizon count mismatch"

        tb = target > 0.5
        shadow = vis < 0.5
        ever = tb.any(dim=1)
        always = tb.all(dim=1)
        STATIC = shadow & always
        DYNAMIC = shadow & ever & (~always)
        NEVER = shadow & (~ever)
        n_static += STATIC.sum().cpu(); n_dynamic += DYNAMIC.sum().cpu()
        n_never += NEVER.sum().cpu(); n_shadow += shadow.sum().cpu()

        pm = pred > 0.5                                     # (B,T,S,S)
        pp = (occ > 0.5).unsqueeze(1).expand(-1, T, -1, -1) # persistence carried forward
        for h in range(T):
            th = tb[:, h]
            mh, ph = pm[:, h], pp[:, h]
            acc["static_m"][h] += (mh & STATIC).sum().cpu()
            acc["dyn_i_m"][h] += (mh & th & DYNAMIC).sum().cpu()
            acc["dyn_u_m"][h] += ((mh | th) & DYNAMIC).sum().cpu()
            acc["dyn_t"][h] += (th & DYNAMIC).sum().cpu()
            acc["never_m"][h] += (mh & NEVER).sum().cpu()
            acc["static_p"][h] += (ph & STATIC).sum().cpu()
            acc["dyn_i_p"][h] += (ph & th & DYNAMIC).sum().cpu()
            acc["dyn_u_p"][h] += ((ph | th) & DYNAMIC).sum().cpu()
            acc["never_p"][h] += (ph & NEVER).sum().cpu()

            # --- calibration accumulation (compute directly on masked vectors) ---
            prob_h = pred[:, h]
            for rkey, rmask in (("shadow", shadow), ("observed", ~shadow),
                                ("shadow_nowall", shadow & (~STATIC))):
                if rmask.sum() == 0:
                    continue
                pr = prob_h[rmask]
                tr = th[rmask].float()
                idx = (pr * NBIN_FINE).long().clamp(0, NBIN_FINE - 1)
                c, sp, st, ssq = hist[rkey][h]
                c.scatter_add_(0, idx.cpu(), torch.ones_like(pr).cpu())
                sp.scatter_add_(0, idx.cpu(), pr.cpu())
                st.scatter_add_(0, idx.cpu(), tr.cpu())
                hist[rkey][h][3] = ssq + ((pr - tr) ** 2).sum().cpu()
        nb += 1
        if max_batches is not None and nb >= max_batches:
            print(f"stopping early after {nb} batches (--max-batches)")
            break

    # --- assemble output ---
    out = {"horizons": horizons,
           "region_fraction": {
               "static": _ratio(int(n_static), int(n_shadow)),
               "dynamic": _ratio(int(n_dynamic), int(n_shadow)),
               "never": _ratio(int(n_never), int(n_shadow)),
               "n_shadow_cells": int(n_shadow)},
           "decomposition": {}, "calibration": {}}

    for h, hs in enumerate(horizons):
        lab = f"{hs}s"
        out["decomposition"][lab] = {
            "static_coverage_model": _ratio(int(acc["static_m"][h]), int(n_static)),
            "static_coverage_persist": _ratio(int(acc["static_p"][h]), int(n_static)),
            "dynamic_iou_model": _ratio(int(acc["dyn_i_m"][h]), int(acc["dyn_u_m"][h])),
            "dynamic_iou_persist": _ratio(int(acc["dyn_i_p"][h]), int(acc["dyn_u_p"][h])),
            "dynamic_recall_model": _ratio(int(acc["dyn_i_m"][h]), int(acc["dyn_t"][h])),
            "shadow_fp_rate_model": _ratio(int(acc["never_m"][h]), int(n_never)),
            "shadow_fp_rate_persist": _ratio(int(acc["never_p"][h]), int(n_never)),
            "n_dynamic_target_cells": int(acc["dyn_t"][h]),
            # Pooled (micro-averaged) shadow IOU assembled from the same counters —
            # immune to the empty-union=1.0 credit that inflates the per-frame macro
            # numbers in eval_summary.json.  inter = TP on STATIC (always-occupied, so
            # every predicted-positive is a TP) + TP on DYNAMIC; union = all target
            # positives + all predicted positives - inter, which reduces to
            # n_static + never_fp + dyn_union.
            "micro_shadow_iou_model": _ratio(
                int(acc["static_m"][h] + acc["dyn_i_m"][h]),
                int(n_static) + int(acc["never_m"][h]) + int(acc["dyn_u_m"][h])),
            "micro_shadow_iou_persist": _ratio(
                int(acc["static_p"][h] + acc["dyn_i_p"][h]),
                int(n_static) + int(acc["never_p"][h]) + int(acc["dyn_u_p"][h])),
            # Same, excluding every persistently-occupied hidden cell (walls, static
            # obstacles, settled debris) — isolates dynamic forecasting from
            # completion of persistent structure.
            "micro_shadow_iou_nostatic_model": _ratio(
                int(acc["dyn_i_m"][h]),
                int(acc["dyn_u_m"][h]) + int(acc["never_m"][h])),
            "micro_shadow_iou_nostatic_persist": _ratio(
                int(acc["dyn_i_p"][h]),
                int(acc["dyn_u_p"][h]) + int(acc["never_p"][h]))}
        cal = {}
        for r in regions:
            c, sp, st, ssq = hist[r][h]
            em = _equal_mass_ece(c, sp, st)
            N = em["n"]
            cal[r] = {"ece": em["ece"], "brier": (float(ssq) / N if N > 0 else float("nan")),
                      "n": N, "reliability": em["bins"] if hs == 5 else None}
        out["calibration"][lab] = cal
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Static/dynamic decomposition + calibration in shadow")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--split", type=str, default="val")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--overrides", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(overrides=args.overrides)
    device = get_device()
    print(f"device: {device}")
    out = evaluate(args.ckpt, args.data, args.split, cfg, args.batch_size, args.max_batches, device)

    print("\n=== decomposition (model; persistence ~0 floor) ===")
    print(f"  shadow composition: static={out['region_fraction']['static']:.3f}  "
          f"dynamic={out['region_fraction']['dynamic']:.3f}  never={out['region_fraction']['never']:.3f}")
    print(f"{'horizon':>7} {'static_cov':>11} {'dyn_iou':>9} {'dyn_recall':>11} {'fp_rate':>9}")
    for hs in out["horizons"]:
        d = out["decomposition"][f"{hs}s"]
        print(f"{str(hs)+'s':>7} {d['static_coverage_model']:>11.3f} {d['dynamic_iou_model']:>9.3f} "
              f"{d['dynamic_recall_model']:>11.3f} {d['shadow_fp_rate_model']:>9.3f}")
    print("\n=== calibration ECE (lower better) ===")
    print(f"{'horizon':>7} {'shadow':>8} {'observed':>9} {'nowall':>8}")
    for hs in out["horizons"]:
        c = out["calibration"][f"{hs}s"]
        print(f"{str(hs)+'s':>7} {c['shadow']['ece']:>8.3f} {c['observed']['ece']:>9.3f} {c['shadow_nowall']['ece']:>8.3f}")

    if args.out:
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
