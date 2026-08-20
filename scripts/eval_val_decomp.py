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


def _bootstrap_ci(num_m, den_m, num_p, den_p, n_boot, seed):
    """Cluster (per-episode) bootstrap CI on a micro-averaged gain (model minus persistence).

    Each argument is a per-episode (E,) integer count array for one horizon; the gain is a
    ratio of pooled sums. Resampling EPISODES with replacement (not frames — frames within an
    episode share geometry/pose and are correlated) and recomputing the pooled ratio gives an
    honest CI. Bootstrap the PAIRED difference within each replicate (model and persistence are
    positively correlated across episodes). Returns [p2.5, p97.5] or None if disabled."""
    if not n_boot:
        return None
    E = len(num_m)
    rng = np.random.default_rng(seed)
    M = rng.multinomial(E, np.full(E, 1.0 / E), size=n_boot)   # (n_boot, E) resample weights
    g = (M @ num_m) / np.maximum(M @ den_m, 1) - (M @ num_p) / np.maximum(M @ den_p, 1)
    return [float(np.percentile(g, 2.5)), float(np.percentile(g, 97.5))]


@torch.no_grad()
def evaluate(ckpt_path, data_root, split, cfg, batch_size, max_batches, device,
             n_boot=10000, boot_seed=0):
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

    # --- A1 integer accumulators, kept PER EPISODE so the gains can carry a per-episode
    # bootstrap CI. Summing over episodes reproduces the global pool bit-for-bit. Episode id
    # is the dataset file index fi; E = number of episode files. ---
    E = max((fi for fi, _ in ds._index), default=-1) + 1
    KEYS = ["static_m", "dyn_i_m", "dyn_u_m", "dyn_t", "never_m",   # model
            "static_p", "dyn_i_p", "dyn_u_p", "never_p"]           # persistence
    per = {k: np.zeros((E, T), dtype=np.int64) for k in KEYS}
    ep_static = np.zeros(E, dtype=np.int64)
    ep_dynamic = np.zeros(E, dtype=np.int64)
    ep_never = np.zeros(E, dtype=np.int64)
    n_shadow_total = 0

    # --- B calibration fine-histograms: region -> horizon -> [count, sum_prob, sum_truth, sum_sq] ---
    regions = ["shadow", "observed", "shadow_nowall"]
    hist = {r: {h: [torch.zeros(NBIN_FINE), torch.zeros(NBIN_FINE), torch.zeros(NBIN_FINE), torch.zeros(())]
                for h in range(T)} for r in regions}

    nb = 0
    for start in range(0, n, batch_size):
        idxs = range(start, min(start + batch_size, n))
        xs, tgts, occs, viss, fis = [], [], [], [], []
        for i in idxs:
            fi, t = ds._index[i]
            fis.append(fi)
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
        fis_arr = np.asarray(fis)
        _cells = lambda m: m.sum(dim=(-2, -1)).cpu().numpy()   # per-sample cell count -> (B,)
        np.add.at(ep_static, fis_arr, _cells(STATIC))
        np.add.at(ep_dynamic, fis_arr, _cells(DYNAMIC))
        np.add.at(ep_never, fis_arr, _cells(NEVER))
        n_shadow_total += int(shadow.sum())

        pm = pred > 0.5                                     # (B,T,S,S)
        pp = (occ > 0.5).unsqueeze(1).expand(-1, T, -1, -1) # persistence carried forward
        for h in range(T):
            th = tb[:, h]
            mh, ph = pm[:, h], pp[:, h]
            def scat(key, mask):
                np.add.at(per[key], (fis_arr, h), _cells(mask))
            scat("static_m", mh & STATIC)
            scat("dyn_i_m", mh & th & DYNAMIC)
            scat("dyn_u_m", (mh | th) & DYNAMIC)
            scat("dyn_t", th & DYNAMIC)
            scat("never_m", mh & NEVER)
            scat("static_p", ph & STATIC)
            scat("dyn_i_p", ph & th & DYNAMIC)
            scat("dyn_u_p", (ph | th) & DYNAMIC)
            scat("never_p", ph & NEVER)

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
    # Point estimates = sum the per-episode arrays over episodes (identical to the old global
    # pool). CIs come from resampling episodes (the rows of per[...]/ep_*).
    tot = {k: per[k].sum(axis=0) for k in KEYS}            # (T,)
    n_static, n_dynamic, n_never = int(ep_static.sum()), int(ep_dynamic.sum()), int(ep_never.sum())
    out = {"horizons": horizons, "n_episodes": int(E), "n_bootstrap": int(n_boot),
           "region_fraction": {
               "static": _ratio(n_static, n_shadow_total),
               "dynamic": _ratio(n_dynamic, n_shadow_total),
               "never": _ratio(n_never, n_shadow_total),
               "n_shadow_cells": int(n_shadow_total)},
           "decomposition": {}, "calibration": {}}

    for h, hs in enumerate(horizons):
        lab = f"{hs}s"
        # micro shadow gain (model minus persistence) and its walls-excluded variant, each
        # with a per-episode bootstrap CI. inter = TP on STATIC + TP on DYNAMIC; union =
        # n_static + never_fp + dyn_union (see the derivation below).
        msi_m = _ratio(int(tot["static_m"][h] + tot["dyn_i_m"][h]),
                       n_static + int(tot["never_m"][h]) + int(tot["dyn_u_m"][h]))
        msi_p = _ratio(int(tot["static_p"][h] + tot["dyn_i_p"][h]),
                       n_static + int(tot["never_p"][h]) + int(tot["dyn_u_p"][h]))
        nsi_m = _ratio(int(tot["dyn_i_m"][h]), int(tot["dyn_u_m"][h]) + int(tot["never_m"][h]))
        nsi_p = _ratio(int(tot["dyn_i_p"][h]), int(tot["dyn_u_p"][h]) + int(tot["never_p"][h]))
        out["decomposition"][lab] = {
            "static_coverage_model": _ratio(int(tot["static_m"][h]), n_static),
            "static_coverage_persist": _ratio(int(tot["static_p"][h]), n_static),
            "dynamic_iou_model": _ratio(int(tot["dyn_i_m"][h]), int(tot["dyn_u_m"][h])),
            "dynamic_iou_persist": _ratio(int(tot["dyn_i_p"][h]), int(tot["dyn_u_p"][h])),
            "dynamic_recall_model": _ratio(int(tot["dyn_i_m"][h]), int(tot["dyn_t"][h])),
            "shadow_fp_rate_model": _ratio(int(tot["never_m"][h]), n_never),
            "shadow_fp_rate_persist": _ratio(int(tot["never_p"][h]), n_never),
            "n_dynamic_target_cells": int(tot["dyn_t"][h]),
            # Pooled (micro-averaged) shadow IOU — immune to the empty-union=1.0 credit that
            # inflates the per-frame macro numbers in eval_summary.json.
            "micro_shadow_iou_model": msi_m,
            "micro_shadow_iou_persist": msi_p,
            "micro_shadow_iou_nostatic_model": nsi_m,
            "micro_shadow_iou_nostatic_persist": nsi_p,
            # Headline gains + per-episode bootstrap CIs. nostatic excludes the memorizable
            # walls, so it does not depend on wall recall — the reviewer-proof number.
            "micro_shadow_gain": msi_m - msi_p,
            "micro_shadow_gain_ci": _bootstrap_ci(
                per["static_m"][:, h] + per["dyn_i_m"][:, h],
                ep_static + per["never_m"][:, h] + per["dyn_u_m"][:, h],
                per["static_p"][:, h] + per["dyn_i_p"][:, h],
                ep_static + per["never_p"][:, h] + per["dyn_u_p"][:, h], n_boot, boot_seed),
            "micro_shadow_gain_nostatic": nsi_m - nsi_p,
            "micro_shadow_gain_nostatic_ci": _bootstrap_ci(
                per["dyn_i_m"][:, h], per["dyn_u_m"][:, h] + per["never_m"][:, h],
                per["dyn_i_p"][:, h], per["dyn_u_p"][:, h] + per["never_p"][:, h], n_boot, boot_seed)}
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
    ap.add_argument("--bootstrap", type=int, default=10000, help="bootstrap resamples for CIs (0 disables)")
    ap.add_argument("--boot-seed", type=int, default=0)
    ap.add_argument("--overrides", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(overrides=args.overrides)
    device = get_device()
    print(f"device: {device}")
    out = evaluate(args.ckpt, args.data, args.split, cfg, args.batch_size, args.max_batches, device,
                   n_boot=args.bootstrap, boot_seed=args.boot_seed)

    print("\n=== decomposition (model; persistence ~0 floor) ===")
    print(f"  shadow composition: static={out['region_fraction']['static']:.3f}  "
          f"dynamic={out['region_fraction']['dynamic']:.3f}  never={out['region_fraction']['never']:.3f}")
    print(f"{'horizon':>7} {'static_cov':>11} {'dyn_iou':>9} {'dyn_recall':>11} {'fp_rate':>9}")
    for hs in out["horizons"]:
        d = out["decomposition"][f"{hs}s"]
        print(f"{str(hs)+'s':>7} {d['static_coverage_model']:>11.3f} {d['dynamic_iou_model']:>9.3f} "
              f"{d['dynamic_recall_model']:>11.3f} {d['shadow_fp_rate_model']:>9.3f}")
    print(f"\n=== micro shadow gain (model - persistence) with {out['n_bootstrap']} per-episode "
          f"bootstrap over n={out['n_episodes']} episodes ===")
    print(f"{'horizon':>7} {'gain':>8} {'95% CI':>18} | {'nostatic':>9} {'95% CI':>18}")
    for hs in out["horizons"]:
        d = out["decomposition"][f"{hs}s"]
        ci = d.get("micro_shadow_gain_ci"); nci = d.get("micro_shadow_gain_nostatic_ci")
        cis = f"[{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else "     n/a"
        ncis = f"[{nci[0]:+.3f}, {nci[1]:+.3f}]" if nci else "     n/a"
        print(f"{str(hs)+'s':>7} {d['micro_shadow_gain']:>+8.3f} {cis:>18} | "
              f"{d['micro_shadow_gain_nostatic']:>+9.3f} {ncis:>18}")
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
