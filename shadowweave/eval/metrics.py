"""Evaluation metrics — collision rate, path efficiency, prediction IOU."""

from __future__ import annotations

import numpy as np


class EvalMetrics:
    """Accumulates episode results and computes aggregate metrics.

    Primary method: ``log_episode(...)`` then ``summary() -> dict``
    """

    def __init__(self) -> None:
        self._collisions:    list[bool]  = []
        self._path_lengths:  list[float] = []
        self._optimal_lengths: list[float] = []
        self._prediction_ious: list[float] = []  # per-timestep IOU at 5s horizon

    def log_episode(
        self,
        had_collision: bool,
        path_length: float,
        optimal_path_length: float,
    ) -> None:
        self._collisions.append(had_collision)
        self._path_lengths.append(path_length)
        self._optimal_lengths.append(optimal_path_length)

    def log_prediction(self, pred: np.ndarray, target: np.ndarray) -> None:
        """Log one (H, W) binary prediction vs target, compute IOU."""
        pred_bin  = (pred > 0.5).astype(bool)
        target_bin = (target > 0.5).astype(bool)
        intersection = (pred_bin & target_bin).sum()
        union = (pred_bin | target_bin).sum()
        iou = float(intersection) / max(float(union), 1.0)
        self._prediction_ious.append(iou)

    def summary(self) -> dict[str, float]:
        n = len(self._collisions)
        if n == 0:
            return {}
        collision_rate = sum(self._collisions) / n
        efficiency = np.mean([
            opt / max(actual, 1e-6)
            for actual, opt in zip(self._path_lengths, self._optimal_lengths)
        ])
        mean_iou = float(np.mean(self._prediction_ious)) if self._prediction_ious else float("nan")
        return {
            "collision_rate":    collision_rate,
            "path_efficiency":   float(efficiency),
            "prediction_iou_5s": mean_iou,
            "n_episodes":        n,
        }

    def reset(self) -> None:
        self.__init__()


if __name__ == "__main__":
    metrics = EvalMetrics()

    # simulate 10 episodes
    for i in range(10):
        metrics.log_episode(
            had_collision=np.random.rand() < 0.15,
            path_length=np.random.uniform(5, 15),
            optimal_path_length=5.0,
        )
        # simulate 30 prediction timesteps per episode
        for _ in range(30):
            metrics.log_prediction(
                np.random.rand(128, 128),
                (np.random.rand(128, 128) > 0.8).astype(np.float32),
            )

    s = metrics.summary()
    print("EvalMetrics summary:")
    for k, v in s.items():
        print(f"  {k}: {v:.4f}")
    print("EvalMetrics OK")
