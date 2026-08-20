"""Evaluation metrics — collision rate, path efficiency, per-horizon IOU, lead time."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch


def iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Per-sample IOU over the trailing spatial dims. Returns (B,) or (B, T).

    An empty prediction on an empty target scores 1.0, not 0.0 — "correctly predicted
    nothing here" is a success, and scoring it as a miss drags the mean down on the
    many BEV frames that are legitimately almost empty.
    """
    p = (pred > threshold)
    t = (target > threshold)
    dims = (-2, -1)
    inter = (p & t).sum(dim=dims).float()
    union = (p | t).sum(dim=dims).float()
    score = torch.where(union > 0, inter / union.clamp_min(1.0), torch.ones_like(inter))

    # A NaN prediction compares False everywhere, so against an empty target it would
    # land in the empty-vs-empty branch and score a perfect 1.0. A model emitting NaN
    # has failed; score it 0 rather than letting it top the table.
    broken = (~torch.isfinite(pred)).any(dim=dims)
    return torch.where(broken, torch.zeros_like(score), score)


def masked_iou(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, threshold: float = 0.5
) -> torch.Tensor:
    """IOU restricted to cells where ``mask`` is true — used to score shadow regions
    separately from directly observed ones, which is the claim worth measuring."""
    p = (pred > threshold) & mask
    t = (target > threshold) & mask
    dims = (-2, -1)
    inter = (p & t).sum(dim=dims).float()
    union = (p | t).sum(dim=dims).float()
    score = torch.where(union > 0, inter / union.clamp_min(1.0), torch.ones_like(inter))

    # Same NaN guard as ``iou``: a NaN prediction compares False everywhere, so on an
    # empty masked union it would score a perfect 1.0 instead of exposing the failure.
    broken = (~torch.isfinite(pred)).any(dim=dims)
    return torch.where(broken, torch.zeros_like(score), score)


class EvalMetrics:
    """Accumulates episode results and computes aggregate metrics.

    Primary method: ``log_episode(...)`` then ``summary() -> dict``
    """

    def __init__(self, horizons: list[float] | None = None) -> None:
        self.horizons = list(horizons or [])
        self.reset()

    def reset(self) -> None:
        self._collisions: list[bool] = []
        self._collision_steps: list[float] = []
        self._path_lengths: list[float] = []
        self._optimal_lengths: list[float] = []
        self._ious: dict[int, list[float]] = defaultdict(list)
        self._shadow_ious: dict[int, list[float]] = defaultdict(list)
        self._lead_times: list[float] = []
        self._latencies_ms: list[float] = []

    def log_episode(
        self,
        had_collision: bool,
        path_length: float,
        optimal_path_length: float,
        collision_rate_per_step: float = 0.0,
    ) -> None:
        self._collisions.append(bool(had_collision))
        self._collision_steps.append(float(collision_rate_per_step))
        self._path_lengths.append(float(path_length))
        self._optimal_lengths.append(float(optimal_path_length))

    def log_prediction(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        shadow_mask: torch.Tensor | None = None,
        horizon_idx: int | None = None,
    ) -> None:
        """Log a full (T, S, S) prediction stack, or one horizon's (S, S)/(1, S, S)
        slice with its index passed via ``horizon_idx``."""
        if pred.ndim == 2:
            pred, target = pred[None], target[None]
        vals = iou(pred, target)
        base = int(horizon_idx) if horizon_idx is not None else 0
        if horizon_idx is None and vals.shape[0] != len(self.horizons):
            # Without the index a single-horizon slice would silently land in
            # bucket 0 — every horizon's score pooled into "iou_1s".
            raise ValueError(
                f"got {vals.shape[0]} horizon(s) but {len(self.horizons)} are "
                "configured — pass horizon_idx when logging a single slice"
            )
        for h in range(vals.shape[0]):
            self._ious[base + h].append(float(vals[h]))
        if shadow_mask is not None:
            m = shadow_mask.expand_as(pred) if shadow_mask.ndim == pred.ndim else shadow_mask
            svals = masked_iou(pred, target, m.bool())
            for h in range(svals.shape[0]):
                self._shadow_ious[base + h].append(float(svals[h]))

    def log_lead_time(self, seconds: float) -> None:
        self._lead_times.append(float(seconds))

    def log_latency(self, ms: float) -> None:
        self._latencies_ms.append(float(ms))

    def _horizon_name(self, h: int) -> str:
        return f"{self.horizons[h]:g}s" if h < len(self.horizons) else f"h{h}"

    def summary(self) -> dict[str, float]:
        out: dict[str, float] = {}
        n = len(self._collisions)
        if n:
            out["collision_rate"] = sum(self._collisions) / n
            out["collision_rate_per_step"] = float(np.mean(self._collision_steps))
            out["path_efficiency"] = float(np.mean([
                opt / max(actual, 1e-6)
                for actual, opt in zip(self._path_lengths, self._optimal_lengths)
            ]))
            out["n_episodes"] = float(n)

        for h, vals in sorted(self._ious.items()):
            out[f"iou_{self._horizon_name(h)}"] = float(np.mean(vals))
        for h, vals in sorted(self._shadow_ious.items()):
            out[f"shadow_iou_{self._horizon_name(h)}"] = float(np.mean(vals))
        if self._ious:
            out["iou_mean"] = float(np.mean([np.mean(v) for v in self._ious.values()]))
        if self._lead_times:
            out["falling_lead_time_s"] = float(np.mean(self._lead_times))
        if self._latencies_ms:
            out["latency_p50_ms"] = float(np.percentile(self._latencies_ms, 50))
            out["latency_p95_ms"] = float(np.percentile(self._latencies_ms, 95))
        return out


if __name__ == "__main__":
    m = EvalMetrics(horizons=[1, 3, 5, 10])

    for _ in range(10):
        m.log_episode(
            had_collision=bool(np.random.rand() < 0.15),
            path_length=float(np.random.uniform(5, 15)),
            optimal_path_length=5.0,
            collision_rate_per_step=float(np.random.rand() * 0.05),
        )
        for _ in range(30):
            target = (torch.rand(4, 96, 96) > 0.93).float()
            # Imperfect but correlated: drop ~25% of true cells and add false ones,
            # so IOU lands strictly between 0 and 1 and the metric is exercised.
            pred = target * (torch.rand(4, 96, 96) > 0.25).float()
            pred = pred + (torch.rand(4, 96, 96) > 0.98).float()
            m.log_prediction(pred.clamp(0, 1), target, shadow_mask=torch.rand(1, 96, 96) > 0.5)
        m.log_lead_time(float(np.random.uniform(2, 6)))
        m.log_latency(float(np.random.uniform(10, 60)))

    print("EvalMetrics summary:")
    for k, v in m.summary().items():
        print(f"  {k}: {v:.4f}")

    # Empty-on-empty must score 1.0, not 0.0.
    z = torch.zeros(1, 8, 8)
    assert float(iou(z, z)) == 1.0, "empty/empty must be a hit"
    print("  empty-vs-empty IOU: 1.0 ✓")
    print("EvalMetrics OK")
