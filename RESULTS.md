# ShadowWeave — results (full run, 2026-08-18; revised after adversarial review 2026-08-20; geometry-randomization + bootstrap CIs added 2026-08-21)

World model: U-Net, 31M params, trained with early stopping (stopped at epoch 24,
best epoch 8; best val loss 0.300, val IOU@5s 0.665). Input is a **single frame**:
depth-derived occupancy + visibility + one-step flow — no multi-frame memory.
Policy: reactive PPO local agent, 1M steps. Eval: 90 episodes, 301 steps each.
Sources: `results/eval_summary.json` (policy rollouts), `$WS/val_pertier/`,
`$WS/val_decomp/full/` (fixed val, per tier, with bootstrap CIs),
`$WS/val_decomp/randgeom.json` (randomized-geometry val),
`$WS/results_noflow/` and `$WS/val_decomp/novis/` (ablations). Figures: `figures/`.

All shadow-gain point estimates below carry 95% bootstrap CIs from resampling
**episodes** (not frames — frames within an episode are correlated), 10,000 resamples,
paired model-minus-persistence per resample.

## Headline — occupancy prediction into unobserved ("shadow") space

The model predicts occupancy in cells the sensor cannot currently see, beating the
best trivial baseline (persistence = the observed frame carried forward; empty = all
free). **Raw shadow-IOU levels are inflated by empty-vs-empty credit** (`masked_iou`
scores an empty prediction on an empty masked region 1.0, and most shadow is empty),
and the inflation applies to model and baselines alike — so the defensible number is
the **gain**, not the level:

| horizon | shadow gain over best baseline (rollouts) | shadow gain (fixed val, per-tier: static / moving / debris) |
|--------:|:-----------------------------------------:|:------------------------------------------------:|
| 1 s     | **+0.325** | +0.354 / +0.379 / +0.375 |
| 3 s     | **+0.322** | +0.354 / +0.382 / +0.366 |
| 5 s     | **+0.320** | +0.354 / +0.380 / +0.356 |
| 10 s    | **+0.178** | +0.353 / +0.377 / +0.334 |

(The per-tier column above is the macro, per-frame-averaged gain and is empty-credit
inflated; the empty-credit-immune **micro** gain with CIs is in the next section —
+0.31/+0.36/+0.32 @5 s — and is the number we defend.)

Micro-averaged (pooled-count) shadow IOU, which is immune to empty-credit, confirms
the gain (fixed val @5s: model 0.61–0.72 vs persistence 0.30–0.36 depending on tier).
Raw macro levels for reference: model shadow IOU@5s 0.744, persistence 0.424, overall
IOU@5s 0.627 (the shadow > overall ordering is an artifact of the empty-credit
interacting with a sparser mask — persistence shows the same ordering).

## Statistical significance and robustness to room geometry

Micro-averaged shadow gain (empty-credit-immune), fixed val, per tier, with 95%
bootstrap CIs over episodes (@5 s):

| tier (fixed val, @5 s) | micro shadow gain | 95% CI |
|---|:---:|:---:|
| static | +0.306 | [+0.288, +0.327] |
| moving | +0.363 | [+0.349, +0.378] |
| debris | +0.315 | [+0.281, +0.350] |

Every CI is well clear of zero at every horizon (1/3/5/10 s), so the gain is
statistically significant, not sampling noise.

**Geometry-memorization confound — resolved.** The earlier concern was that a fixed
6×6 m wall shell let the model memorize a constant wall prior. We regenerated the data
with **per-episode randomized room geometry** (width and depth each drawn i.i.d. in
[5, 8] m, obstacle counts scaled to hold the ~7% positive fraction), retrained the U-Net
from scratch on it, and re-measured on 80 held-out randomized rooms the model never saw
(`$WS/val_decomp/randgeom.json`). Comparing like tier to like tier (both static):

| static tier, @5 s | micro shadow gain | 95% CI | persistent-structure recall |
|---|:---:|:---:|:---:|
| fixed 6×6 geometry | +0.306 | [+0.288, +0.327] | 0.829 |
| **randomized geometry** | **+0.309** | **[+0.294, +0.324]** | 0.854 |

The gain does **not drop** — the two CIs overlap almost entirely, and completion recall
is if anything slightly higher on random rooms. Memorizing a constant wall shell
therefore contributes ~nothing to the gain; the model completes hidden structure from
the current observation, in room shapes it has never encountered. The randomized-geometry
rollout corroborates this (pooled shadow gain +0.316 @5 s vs +0.320 fixed). This was the
single highest-value robustness item and it passes.

## What the shadow gain actually is: completion, not clairvoyance

Decomposing shadow cells into persistent structure (occupied at every horizon —
walls, static obstacles, settled debris; ~7% of shadow), dynamic (occupied at some
horizons; ≤1.5%), and never-occupied (~92%):

| component (fixed val @5s) | model | persistence |
|---|---|---|
| persistent-structure recall (amodal completion) | 0.83–0.93 | 0.37–0.51 |
| dynamic IOU, moving tier | 0.092 → 0.031 (1s → 10s, decaying) | 0.060 → 0.020 |
| dynamic IOU, debris tier | 0.020 → 0.019 | 0.006 → 0.023 |
| false-positive rate on never-occupied cells | 0.014–0.034 | 0.022–0.027 |

The gain is dominated by **single-frame amodal completion of persistent hidden
structure**, with a genuine but small dynamic-forecasting signal on the moving tier
(decaying with horizon, as real forecasting should). On the debris tier the dynamic
signal is at the noise floor and persistence matches or beats the model beyond 1 s —
dynamic cells there are <1.5% of shadow and debris that settles within 1 s is
counted as persistent structure. We do not claim dynamic forecasting on debris.

Excluding *all* persistent structure, both model and persistence shadow IOU collapse
to ~0.01–0.02, and the residual "nostatic" gain is at the noise floor: fixed static
+0.000, fixed moving +0.007 (CI [+0.002, +0.012]), randomized-geometry static +0.002
(CI [+0.001, +0.003]) @5 s. So the honest claim is amodal completion of hidden
*structure*, not multi-step dynamic forecasting.

**On the wall-memorization confound (resolved):** this used to be flagged as a risk —
that the fixed 6×6 m shell let the model memorize a constant wall prior. The
randomized-geometry experiment above (gain unchanged on never-seen room shapes) shows
it does not: the completion recall transfers to new geometry, so it is per-scene
completion, not a memorized constant.

## Uncertainty calibration in shadow

From the decomposition eval (micro-averaged, 5 s): ECE in shadow 0.043–0.047 —
*better* than in observed space (0.058–0.086) — with Brier 0.022–0.036. The model is
underconfident in the mid-range (predicted 0.4 → empirical ~0.12) but places little
mass there; see `figures/reliability.png` (debris tier). The rollout-eval
`calibration_error` 0.045 is grid-wide (dominated by easy free space) and is not by
itself evidence of calibrated shadow uncertainty — the per-region numbers above are.

On randomized geometry (`randgeom.json`) shadow calibration holds up: shadow ECE
0.050–0.067 vs observed 0.056–0.077, with shadow becoming the better-calibrated region
at the longer 5–10 s horizons — so calibration is not an artifact of the fixed room.

## Input-channel ablations: occupancy alone carries the signal

The world-model input stack is occupancy + visibility + one-step flow. Dropping either
auxiliary channel and retraining from scratch leaves the shadow gain intact:

| variant | shadow gain @5 s (fixed val, micro, static/moving/debris) | rollout shadow gain @5 s |
|---|---|---|
| full (all 3 channels) | +0.306 / +0.363 / +0.315 | +0.320 |
| no visibility (`$WS/val_decomp/novis/`) | +0.309 / +0.360 / +0.313 | — |
| no flow (`$WS/results_noflow/`) | — | +0.344 |

The no-visibility per-tier gains are within CI of the full model (three ways), and the
no-flow rollout gain is if anything slightly higher. Both auxiliary channels are
therefore redundant: the completion signal is carried by the depth-derived occupancy
channel alone. This simplifies the input and removes optical-flow (RAFT) from the
runtime critical path.

## Diffusion world-model variant

`world_model.architecture: diffusion` (conditional DDPM over future BEV occupancy) was
trained as an alternative to the deterministic U-Net; results are pending the final run
and will be framed on sample diversity / calibration in shadow, not raw IOU (BCE and
DDPM optimize different objectives, so IOU is not a fair head-to-head).

## Falling-object anticipation (metric caveats apply)

| metric | value |
|---|---|
| detection rate (per due prediction) | 0.787 |
| false-alarm rate | 0.021 |
| mean lead time (horizon-granular) | 2.62 s |

Caveats a reader must know: an "event" is one due (prediction, horizon) instance,
not one physical object (44,932 instances over 90 episodes); arrivals include
static geometry entering shadow as the agent moves, not only falling debris; and
lead time is the horizon bucket that fired ({1,3,5,10} s), so it is floored at 1 s
and its mean reflects the horizon schedule as much as anticipation. The ≥3 s target
is not met and, under this definition, the number is not comparable to
object-level lead times in the literature.

## Navigation (reactive local agent; A* owns goal-seeking)

| metric | untrained (n=30) | broken PPO (n=90) | **fixed PPO (n=90)** |
|---|---|---|---|
| collision rate, per-episode | 0.60 | 0.878 | **0.367** |
| collision rate, per-step | 0.133 | 0.158 | 0.155 |
| path straightness (net displacement / path length) | 0.848 | 0.226 | **0.773** |

The fixed policy roughly halves per-episode collisions vs the broken run and beats
the untrained baseline; both collision definitions still fail the ≤0.10 target.
"Path efficiency" is net-displacement straightness (there is no goal-optimal path in
this eval), so the untrained agent's 0.85 partly reflects driving straight into
things. The untrained row comes from a 30-episode smoke run; the broken-PPO run's
raw artifacts were overwritten and survive only in these recorded numbers.

## Real-time budget

Forward-path latency (BEV projection + raycaster + world-model forward + orchestrator,
batch 1, deterministic U-Net): p50 5.97 ms, p95 7.00 ms — inside the 50 ms (20 Hz)
budget. This excludes the monocular-depth network (eval consumes simulator depth) and
HRTF audio synthesis, and does not cover the diffusion variant.

## Target scorecard (see report.md for the full table)

Passes: macro IOU@5s, both gain-over-baseline targets (CIs exclude zero at every
horizon), robustness to randomized room geometry, latency, calibration.
Fails: collision rate (both definitions), falling-object lead time.

## Figures (`figures/`)

- `decomp_curves.png` — dynamic IOU / recall and persistent-structure recall by
  horizon, model (solid) vs persistence (dashed), per tier.
- `reliability.png` — reliability @5 s, shadow vs observed, debris tier.
- `prediction_00..05.png` — per-sample panels: observation + shadow map, ground
  truth, forecasts at 1/3/5/10 s, error maps (red = false positive, blue = missed).
- `val_iou.png` — validation macro IOU by horizon vs the 0.60 target (empty-credit
  inflated; context only).
