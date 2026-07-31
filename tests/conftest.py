"""Shared fixtures. Tests run small and on CPU so they stay fast in CI."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from shadowweave.utils import load_config


@pytest.fixture(scope="session")
def cfg():
    """Small config so the suite runs in seconds, not minutes."""
    c = load_config()
    c.depth.output_h = 64
    c.depth.output_w = 64
    c.bev.size = 32
    c.world_model.base_channels = 8
    c.shadow.num_rays = 128
    c.shadow.march_steps = 16
    c.shadow.elevation_samples = 5
    return c


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def wall_left_depth(cfg):
    """Depth map with a close wall on the left third, open elsewhere."""
    h, w = cfg.depth.output_h, cfg.depth.output_w
    d = torch.ones(1, 1, h, w) * 0.9
    d[0, 0, :, : w // 3] = 0.15
    return d


@pytest.fixture
def rollout_dir(cfg, tmp_path):
    """A tiny synthetic rollout set in the current on-disk schema."""
    S = cfg.bev.size
    H = len(cfg.world_model.prediction_horizons)
    rng = np.random.default_rng(0)
    for split, n_eps in (("train", 2), ("val", 1)):
        d = tmp_path / split
        d.mkdir(parents=True, exist_ok=True)
        for ep in range(n_eps):
            T = 6
            np.savez(
                d / f"ep{ep:06d}.npz",
                bev_occupancy=rng.random((T, 1, S, S)).astype(np.float32),
                bev_visibility=rng.random((T, 1, S, S)).astype(np.float32),
                bev_flow=rng.random((T, 2, S, S)).astype(np.float32),
                target=(rng.random((T, H, S, S)) > 0.9).astype(np.float32),
                uncertainty_grid=rng.random((T, 9)).astype(np.float32),
                collision=np.zeros(T, dtype=np.bool_),
            )
    return str(tmp_path)
