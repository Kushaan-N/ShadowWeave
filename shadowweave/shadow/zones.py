"""Zone utilities — pool 2D occupancy maps to the 9-cell spatial audio grid.

The 9 zones map left-to-right across the map (azimuth slices). Pooling is
max-then-mean: take the worst case along the range axis in each column strip, then
average across the strip. This matches the audio spec, where any obstacle in a zone
triggers a cue regardless of how far along that bearing it sits.

The numpy and torch implementations MUST agree. They previously did not: the torch
version used adaptive_avg_pool2d, which averages over both axes instead of taking a
max along range, so it returned ~0.50 where numpy returned ~0.99 on the same input.
Since PPO built observations with the torch version while the orchestrator used the
numpy version, the policy was trained on one representation and run on another.
``test_contracts.py`` now pins their equivalence.

Adaptive pooling is also unusable here: MPS rejects output sizes that do not divide
the input (a 96-wide BEV into 9 zones), so the whole RL path crashed on Apple silicon.
Explicit strip slicing is exact, device-agnostic, and differentiable.
"""

from __future__ import annotations

import numpy as np
import torch

_N_ZONES = 9


def zone_bounds(width: int, n_zones: int = _N_ZONES) -> list[tuple[int, int]]:
    """Column ranges for each zone. Single source of truth for both backends."""
    strip = width / n_zones
    return [(int(i * strip), max(int((i + 1) * strip), int(i * strip) + 1))
            for i in range(n_zones)]


def pool_to_zones(occupancy_map: np.ndarray, n_zones: int = _N_ZONES) -> np.ndarray:
    """Pool a (H, W) occupancy map to (n_zones,) float32.

    Each zone covers an equal-width column strip. Value = max over rows, then mean
    over columns in that strip (worst-case obstacle per direction).
    """
    _, W = occupancy_map.shape
    zones = np.zeros(n_zones, dtype=np.float32)
    for i, (c0, c1) in enumerate(zone_bounds(W, n_zones)):
        strip = occupancy_map[:, c0:c1]
        zones[i] = strip.max(axis=0).mean() if strip.size else 0.0
    return zones


def pool_predictions_to_zones(pred: np.ndarray, n_zones: int = _N_ZONES) -> np.ndarray:
    """Pool world-model predictions (T, H, W) → (T, n_zones)."""
    out = np.zeros((pred.shape[0], n_zones), dtype=np.float32)
    for t in range(pred.shape[0]):
        out[t] = pool_to_zones(pred[t], n_zones)
    return out


def pool_predictions_to_zones_torch(
    pred: torch.Tensor, n_zones: int = _N_ZONES
) -> torch.Tensor:
    """Differentiable equivalent of ``pool_predictions_to_zones``.

    (B, T, H, W) → (B, T, n_zones), numerically identical to the numpy version.
    """
    if pred.ndim != 4:
        raise ValueError(f"expected (B, T, H, W), got shape {tuple(pred.shape)}")
    W = pred.shape[-1]
    # max along the range axis, then mean across the strip — same order as numpy.
    cols = [pred[..., c0:c1].amax(dim=-2).mean(dim=-1) for c0, c1 in zone_bounds(W, n_zones)]
    return torch.stack(cols, dim=-1)


if __name__ == "__main__":
    occ = np.zeros((128, 128), dtype=np.float32)
    occ[:, :43] = 0.9   # left third blocked
    occ[:, 85:] = 0.5   # right third partially blocked

    zones = pool_to_zones(occ)
    print("Zone pooling (left=0, front=4, right=8):")
    for i, v in enumerate(zones):
        print(f"  zone {i}: {v:.3f}  {'#' * int(v * 20)}")

    # The two backends must agree — they used to differ by 0.50.
    rng = np.random.default_rng(0)
    pred = rng.random((2, 4, 96, 96)).astype(np.float32)
    np_out = np.stack([pool_predictions_to_zones(p) for p in pred])
    t_out = pool_predictions_to_zones_torch(torch.from_numpy(pred)).numpy()
    diff = np.abs(np_out - t_out).max()
    print(f"\nnumpy vs torch max difference: {diff:.2e}  (must be ~0)")
    assert diff < 1e-5, "backends disagree"

    # Differentiable, and works for a width that does not divide by n_zones.
    t = torch.rand(1, 4, 96, 96, requires_grad=True)
    pool_predictions_to_zones_torch(t).sum().backward()
    print(f"backward: OK (grad norm {t.grad.norm():.3f})")
    print("zones.py OK")
