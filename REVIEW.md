# Adversarial review — 2026-08-20

Reviewer-mode audit of the results, metrics, figures, and claims ahead of the
workshop submission. Three independent code audits (data mechanics, eval metrics,
analysis scripts) plus a cross-check of every number in the write-ups against the
JSONs on scratch. Verdicts below; remediations applied are marked ✔.

## What survives attack (the paper can stand on these)

- **No input leakage.** The model input is genuinely sensor-derived
  (`BEVProjector.forward(depth)`); ground-truth geometry is used only for targets
  (`sim/synthetic_data.py`). The core premise is mechanically sound.
- **Fair baselines.** Persistence copies the same observed input frame the model
  gets, in the same issue-time pose the targets are rendered in — no frame handicap
  (`eval/baselines.py`).
- **Honest shadow mask.** Shadow = visibility < 0.5 at issue time t, applied to
  future truth — the right definition for "cells I cannot see now".
- **Disjoint seeds.** Train (0–399), val (900000+), policy eval (10000+); interior
  obstacle layouts are procedurally random per seed.
- **The decomposition eval is correctly micro-averaged** (pooled integer counts;
  no empty-union credit) and its static/dynamic split cannot be contaminated by
  ego-motion (all horizons share the issue-time pose).
- **Positive, robust shadow gain.** +0.32 (rollouts) / +0.35–0.38 (fixed val)
  macro; confirmed by micro-averaged pooled IOU (model 0.61–0.72 vs persistence
  0.30–0.38). The gain metric was designed to cancel the empty-credit inflation and
  it does.
- **Shadow calibration.** ECE in shadow 0.043–0.047 @5s, better than observed
  space — a genuinely good result for the "sonify uncertainty" story.

## Findings (severity-ordered) and remediations

1. **Raw shadow-IOU levels are empty-credit-inflated; shadow>overall is an
   artifact.** `masked_iou` scores empty-vs-empty as 1.0 and the per-frame macro
   mean is dominated by it; persistence shows the same 0.42-in-shadow inflation.
   Never present 0.744 vs 0.424 as the headline. ✔ RESULTS.md rewritten to lead
   with gains + micro-averaged IOU; report.py labels macro IOUs and adds the gain
   targets to the scorecard.
2. **The gain is ~95% amodal completion of persistent structure, not dynamic
   forecasting.** Excluding persistent hidden cells, shadow IOU collapses to
   ~0.01–0.02 for model *and* persistence; on debris, persistence ≥ model on
   dynamic cells beyond 1 s (noise floor, <1.5% of cells; settled debris counts as
   static). ✔ Decomposition now leads the narrative; debris claim explicitly
   disowned; `micro_shadow_iou[_nostatic]_{model,persist}` keys added to
   `eval_val_decomp.py` for future runs.
3. **Fixed room shell shared across train/val** (`mujoco_env.py` `_ROOM_XML_TEMPLATE`,
   walls at ±3 m in every episode) — a memorizable constant that inflates
   completion recall. ✔ Disclosed in RESULTS.md as a known confound. Fix = randomize
   room geometry (see roadmap).
4. **Falling-object metrics are not object-level.** Events are per due
   (prediction, horizon) instance (44,932 ≠ objects); "arrivals" include static
   geometry entering shadow as the agent moves; lead time is the horizon bucket
   ({1,3,5,10} s, floored at 1 s), so 2.62 s mean reflects the horizon schedule.
   ✔ Caveated in RESULTS.md and report.py. Honest fix = per-object event tracking
   on the debris tier only (roadmap).
5. **Collision target fails under both definitions** (0.155 per-step, 0.367
   per-episode vs ≤0.10); scoring only per-step was the flattering pick.
   `path_efficiency` is net-displacement straightness (no goal-optimal path
   exists in this eval). ✔ Both definitions now in the scorecard; relabeled.
6. **"Pipeline latency" excluded the depth network and audio synthesis** (eval
   consumes simulator depth; audio callback is None in eval) and only timed the
   deterministic U-Net at batch 1. ✔ Relabeled "forward-path latency"; scope
   stated in RESULTS.md.
7. **Grid-wide `calibration_error` is dominated by easy free space.** ✔ RESULTS.md
   now cites the per-region ECE from the decomposition instead.
8. **`final.pt` was stamped `epochs-1` regardless of the actual stop epoch**
   (made it look like training ran 100 epochs; it early-stopped at 24, best epoch
   8). ✔ Fixed in `world_model/train.py`; RESULTS.md corrected. The paper
   checkpoint is `best.pt` (epoch 8) — a weights-only copy is preserved at
   `checkpoints/world_model/best_weights.pt` on /work (scratch expires 2026-09-06).
9. **`masked_iou` had no NaN guard** — a NaN-emitting model would have scored 1.0
   on empty shadow regions. ✔ Guard added (identical results for finite
   predictions, so running ablation evals stay comparable).
10. **Provenance gaps.** `results*/` is gitignored, so the eval summaries backing
    the paper were not version-controlled; the "broken PPO" run's artifacts were
    overwritten (numbers survive only in RESULTS.md); untrained-baseline row is a
    30-episode smoke run vs 90 for the others. ✔ eval summaries now tracked via
    gitignore exception; caveats added.
11. **10 s horizon is low-N in rollout eval** (only predictions issued near step 0
    ever come due at 10 s in a 301-step episode) — do not over-read `iou_10s`, and
    `iou_mean` weights horizons equally regardless of N. Caveat when citing.
12. **Figures.** decomp_curves lacked the persistence reference (hiding the debris
    inversion) and listed a NaN "static" series; dark-theme PNGs unsuitable for a
    paper. ✔ `plot_decomp.py` rewritten (persistence dashed, Okabe-Ito palette,
    white background) and figures regenerated. `val_iou.png` shows empty-credit
    macro IOU — context only, don't headline it. Note reliability.png is
    debris-tier @5s only (labeled).

## Verify before submission

- The three per-tier val roots (`rollouts_val_{static,moving,debris}`) are pure
  single-tier — the decomposition trusts the directory name.
- The production per-tier/decomp JSONs were produced without `--max-batches`.
- Diffusion/no-flow comparisons: frame on `shadow_diversity_ratio` + calibration,
  not raw IOU (DDIM-mean is not capacity-matched); check the eval logs for NaNs.

## Roadmap to elevate the paper (priority order)

1. **Randomize room geometry per episode** (size, wall positions, door gaps) with a
   held-out val distribution, and re-measure the shadow gain. Kills the
   memorization confound outright; the single highest-value change.
2. **Report per-episode bootstrap CIs** for the headline gains (n=90 episodes;
   resample episodes, not frames — frames within an episode are correlated). No
   number in the paper currently has an error bar.
3. **Object-level falling-object metric on the debris tier**: one event per spawned
   object, lead time = (first correct flag time − impact time) in seconds,
   continuous. Either fixes the ≥3 s claim honestly or retires it.
4. **Egomotion-compensated persistence baseline + a learned no-visibility-input
   baseline** as stronger comparisons than raw persistence; report the gain against
   the strongest.
5. **Multi-frame memory ablation**: the input is a single frame; adding a short
   occupancy history (or a recurrent state) directly tests "remembering vs
   completing" and likely lifts dynamic IOU in shadow — the paper's weakest number.
6. **Wall-vs-interior split of persistent-structure recall** (shell cells are
   locatable from the known pose): separates "memorized the constant shell" from
   "completed per-scene structure" — pre-empts the strongest remaining review
   attack without regenerating data.
7. **Full-pipeline latency**: time Depth-Anything-V2 + audio synthesis on a real
   frame to make the 20 Hz claim end-to-end, or keep the current scoping language.
8. **Real-footage qualitative figure** via `ingestion/depth.py` (Depth Anything →
   BEV + shadow → forecast) — even 2–3 curated clips would blunt
   "everything is synthetic".
