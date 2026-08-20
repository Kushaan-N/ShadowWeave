# ShadowWeave — results (full run, 2026-08-18; revised after adversarial review 2026-08-20)

World model: U-Net, 31M params, trained with early stopping (stopped at epoch 24,
best epoch 8; best val loss 0.300, val IOU@5s 0.665). Input is a **single frame**:
depth-derived occupancy + visibility + one-step flow — no multi-frame memory.
Policy: reactive PPO local agent, 1M steps. Eval: 90 episodes, 301 steps each.
Sources: `results/eval_summary.json` (policy rollouts), `$WS/val_pertier/`,
`$WS/val_decomp/` (fixed val set). Figures: `figures/`.

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

Micro-averaged (pooled-count) shadow IOU, which is immune to empty-credit, confirms
the gain (fixed val @5s: model 0.61–0.72 vs persistence 0.30–0.36 depending on tier).
Raw macro levels for reference: model shadow IOU@5s 0.744, persistence 0.424, overall
IOU@5s 0.627 (the shadow > overall ordering is an artifact of the empty-credit
interacting with a sparser mask — persistence shows the same ordering).

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

**Known confound:** every episode shares the same fixed 6×6 m outer wall shell
(interior obstacles are procedurally randomized per seed; train/val seeds are
disjoint). Part of the persistent-structure recall is therefore attributable to a
memorizable constant prior rather than per-scene completion. Excluding *all*
persistent structure, both model and persistence shadow IOU collapse to ~0.01–0.02.
Randomizing room geometry is the top item for future work.

## Uncertainty calibration in shadow

From the decomposition eval (micro-averaged, 5 s): ECE in shadow 0.043–0.047 —
*better* than in observed space (0.058–0.086) — with Brier 0.022–0.036. The model is
underconfident in the mid-range (predicted 0.4 → empirical ~0.12) but places little
mass there; see `figures/reliability.png` (debris tier). The rollout-eval
`calibration_error` 0.045 is grid-wide (dominated by easy free space) and is not by
itself evidence of calibrated shadow uncertainty — the per-region numbers above are.

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

Passes: macro IOU@5s, both gain-over-baseline targets, latency, calibration.
Fails: collision rate (both definitions), falling-object lead time.

## Figures (`figures/`)

- `decomp_curves.png` — dynamic IOU / recall and persistent-structure recall by
  horizon, model (solid) vs persistence (dashed), per tier.
- `reliability.png` — reliability @5 s, shadow vs observed, debris tier.
- `prediction_00..05.png` — per-sample panels: observation + shadow map, ground
  truth, forecasts at 1/3/5/10 s, error maps (red = false positive, blue = missed).
- `val_iou.png` — validation macro IOU by horizon vs the 0.60 target (empty-credit
  inflated; context only).
