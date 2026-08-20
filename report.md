# ShadowWeave evaluation report

## Targets

| metric | value | target | status |
|---|---|---|---|
| World model IOU @ 5s (macro) | 0.627 | ≥ 0.6 | ✅ |
| Gain over best baseline @ 5s | 0.313 | ≥ 0 | ✅ |
| Shadow IOU @ 5s (macro) | 0.744 | ≥ 0.4 | ✅ |
| Shadow gain over best baseline @ 5s | 0.320 | ≥ 0 | ✅ |
| Collision rate (per-step) | 0.155 | ≤ 0.1 | ❌ |
| Collision rate (per-episode) | 0.367 | ≤ 0.1 | ❌ |
| Falling-object lead time (horizon-granular) | 2.616 | ≥ 3 | ❌ |
| Forward-path latency p95 (ms) | 6.999 | ≤ 100 | ✅ |
| Calibration error (all cells) | 0.045 | ≤ 0.1 | ✅ |

## Prediction quality by horizon

| horizon | model IOU | persistence | gain | shadow IOU |
|---|---|---|---|---|
| 1s | 0.636 | 0.316 | +0.319 | 0.753 |
| 3s | 0.629 | 0.315 | +0.314 | 0.749 |
| 5s | 0.627 | 0.313 | +0.313 | 0.744 |
| 10s | 0.441 | 0.216 | +0.225 | 0.481 |

## All metrics

| key | value |
|---|---|
| `baseline_empty_iou_10s` | 0.0000 |
| `baseline_empty_iou_1s` | 0.0000 |
| `baseline_empty_iou_3s` | 0.0000 |
| `baseline_empty_iou_5s` | 0.0000 |
| `baseline_empty_shadow_iou_10s` | 0.0000 |
| `baseline_empty_shadow_iou_1s` | 0.0000 |
| `baseline_empty_shadow_iou_3s` | 0.0000 |
| `baseline_empty_shadow_iou_5s` | 0.0000 |
| `baseline_persistence_iou_10s` | 0.2158 |
| `baseline_persistence_iou_1s` | 0.3164 |
| `baseline_persistence_iou_3s` | 0.3150 |
| `baseline_persistence_iou_5s` | 0.3133 |
| `baseline_persistence_shadow_iou_10s` | 0.3027 |
| `baseline_persistence_shadow_iou_1s` | 0.4285 |
| `baseline_persistence_shadow_iou_3s` | 0.4263 |
| `baseline_persistence_shadow_iou_5s` | 0.4242 |
| `calibration_error` | 0.0448 |
| `collision_rate` | 0.3667 |
| `collision_rate_per_step` | 0.1554 |
| `falling_detection_margin` | 0.7668 |
| `falling_detection_rate` | 0.7875 |
| `falling_events_detected` | 44932.0000 |
| `falling_false_alarm_rate` | 0.0206 |
| `falling_lead_time_min_s` | 1.0000 |
| `falling_lead_time_s` | 2.6161 |
| `iou_10s` | 0.4411 |
| `iou_1s` | 0.6358 |
| `iou_3s` | 0.6287 |
| `iou_5s` | 0.6268 |
| `iou_mean` | 0.5831 |
| `latency_p50_ms` | 5.9660 |
| `latency_p95_ms` | 6.9989 |
| `model_gain_over_best_baseline_10s` | 0.2253 |
| `model_gain_over_best_baseline_1s` | 0.3194 |
| `model_gain_over_best_baseline_3s` | 0.3137 |
| `model_gain_over_best_baseline_5s` | 0.3134 |
| `model_shadow_gain_over_best_baseline_10s` | 0.1778 |
| `model_shadow_gain_over_best_baseline_1s` | 0.3246 |
| `model_shadow_gain_over_best_baseline_3s` | 0.3223 |
| `model_shadow_gain_over_best_baseline_5s` | 0.3195 |
| `n_episodes` | 90.0000 |
| `path_efficiency` | 0.7728 |
| `shadow_iou_10s` | 0.4805 |
| `shadow_iou_1s` | 0.7531 |
| `shadow_iou_3s` | 0.7487 |
| `shadow_iou_5s` | 0.7437 |
