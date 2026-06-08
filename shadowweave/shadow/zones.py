"""Zone utilities — pool 2D occupancy maps to the 9-cell spatial audio grid.

The 9 zones map left-to-right across the image (horizontal azimuth slices).
Pooling is max-then-mean: take worst-case row in each column strip, then average.
This matches the audio spec where any obstacle in a zone triggers a cue.
"""

from __future__ import annotations

import numpy as np
import torch


_N_ZONES = 9


def pool_to_zones(occupancy_map: np.ndarray, n_zones: int = _N_ZONES) -> np.ndarray:
    """Pool a (H, W) occupancy map to (n_zones,) float32.

    Each zone covers an equal-width horizontal strip. Value = max over rows,
    then mean over columns in that strip (captures worst-case obstacle per direction).
    """
    H, W = occupancy_map.shape
    zones = np.zeros(n_zones, dtype=np.float32)
    strip_w = W / n_zones
    for i in range(n_zones):
        c0 = int(i * strip_w)
        c1 = int((i + 1) * strip_w)
        strip = occupancy_map[:, c0:c1]
        # row-max then col-mean
        zones[i] = strip.max(axis=0).mean() if strip.size > 0 else 0.0
    return zones


def pool_predictions_to_zones(
    pred: np.ndarray, n_zones: int = _N_ZONES
) -> np.ndarray:
    """Pool world-model predictions (T, H, W) → (T, n_zones).

    Feeds into the local agent as a compressed observation.
    """
    T = pred.shape[0]
    out = np.zeros((T, n_zones), dtype=np.float32)
    for t in range(T):
        out[t] = pool_to_zones(pred[t], n_zones)
    return out


def pool_predictions_to_zones_torch(
    pred: torch.Tensor, n_zones: int = _N_ZONES
) -> torch.Tensor:
    """Differentiable version: (B, T, H, W) → (B, T, n_zones) via adaptive avg pool."""
    B, T, H, W = pred.shape
    flat = pred.reshape(B * T, 1, H, W)
    pooled = torch.nn.functional.adaptive_avg_pool2d(flat, (1, n_zones))  # (B*T, 1, 1, n_zones)
    return pooled.reshape(B, T, n_zones)


if __name__ == "__main__":
    import numpy as np

    occ = np.zeros((128, 128), dtype=np.float32)
    occ[:, :43] = 0.9   # left third blocked
    occ[:, 85:] = 0.5   # right third partially blocked
    # centre is clear

    zones = pool_to_zones(occ)
    print("Zone pooling (left=0, front=4, right=8):")
    for i, v in enumerate(zones):
        bar = "#" * int(v * 20)
        print(f"  zone {i}: {v:.3f}  {bar}")
    print("zones.py OK")
