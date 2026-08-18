# ShadowWeave — results (full run, 2026-08-18)

World model: U-Net, 31M params, early-stopped epoch 24 (best val loss 0.300, val IOU@5s 0.665).
Policy: reactive PPO local agent, 1M steps. Eval: 90 episodes, 301 steps each.
Source: `results/eval_summary.json`. Figures: `figures/`.

## Headline claim — occupancy forecasting into unobserved ("shadow") space

Inside occluded regions the model predicts occupancy the persistence baseline cannot,
because persistence has no information about space the sensor never saw. The empty
baseline scores 0 in shadow by construction.

| horizon | model (shadow IOU) | persistence (shadow) | **gain over best baseline** |
|--------:|:------------------:|:--------------------:|:---------------------------:|
| 1 s     | 0.753              | 0.428                | **+0.325** |
| 3 s     | 0.749              | 0.426                | **+0.322** |
| 5 s     | 0.744              | 0.424                | **+0.320** |
| 10 s    | 0.481              | 0.303                | **+0.178** |

All-region IOU (context; less informative because BEV is ~7% positive): 0.636 / 0.629 /
0.627 / 0.441 vs persistence 0.316 / 0.315 / 0.313 / 0.216.

## Falling-object anticipation

| metric | value |
|---|---|
| detection rate | 0.787 |
| false-alarm rate | 0.021 |
| detection margin (det − fa) | +0.767 |
| mean lead time | 2.62 s |
| calibration error | 0.045 |

## Navigation (reactive local agent; A* owns goal-seeking)

| metric | untrained | broken PPO | **fixed PPO (this run)** |
|---|---|---|---|
| collision_rate (per-episode) | 0.60 | 0.878 | **0.367** |
| collision_rate (per-step) | — | 0.158 | 0.155 |
| path_efficiency | 0.848 | 0.226 | **0.773** |

The fixed policy roughly halves per-episode collisions vs the broken run and beats the
untrained baseline; path_efficiency recovers to ~0.77 (below untrained 0.85 — the
deliberate trade-off of a purely reactive avoider that takes safer, less direct paths).

## Real-time budget

p50 latency 5.97 ms, p95 7.00 ms — well inside the 50 ms (20 Hz) control budget.

## Target scorecard (4 / 6 measured targets met)

| target | value | bar | verdict |
|---|---|---|---|
| World model IOU @ 5s | 0.627 | ≥ 0.60 | PASS |
| Shadow IOU @ 5s | 0.744 | ≥ 0.40 | PASS |
| Collision rate (per-step) | 0.155 | ≤ 0.10 | **FAIL** |
| Falling-object lead time | 2.62 s | ≥ 3.0 | **FAIL** |
| Pipeline latency p95 | 7.0 ms | ≤ 100 | PASS |
| Calibration error | 0.045 | ≤ 0.10 | PASS |

## Figures (`figures/`)

- `prediction_00..05.png` — per-sample panels: observation + shadow map (row 1),
  ground truth (row 2), model forecast at 1/3/5/10 s (row 3), error map (row 4,
  red = false positive, blue = missed). The forecast clearly fills the shadow cones.
- `val_iou.png` — validation IOU by horizon vs the 0.60 target.
