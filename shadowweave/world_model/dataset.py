"""MuJoCo rollout dataset — loads (shadow_map, velocity, future_occupancy) tuples."""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from omegaconf import DictConfig


class RolloutDataset(Dataset):
    """Loads pre-generated MuJoCo rollout files from disk.

    Expected file layout (one .npz per episode):
        shadow_map:       (T, 1, H, W)  float32
        velocity:         (T, 2, H, W)  float32
        future_occupancy: (T, horizons, H, W)  float32

    Primary method: ``__getitem__(idx)`` → dict of tensors for one timestep.
    """

    def __init__(self, cfg: DictConfig, data_dir: str, split: str = "train") -> None:
        self.cfg = cfg
        self.data_dir = pathlib.Path(data_dir)
        self.files = sorted((self.data_dir / split).glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No .npz files found in {self.data_dir / split}")

        # index: (file_idx, timestep_idx) pairs
        self._index: list[tuple[int, int]] = []
        for fi, f in enumerate(self.files):
            n = np.load(f, mmap_mode="r")["shadow_map"].shape[0]
            for t in range(n):
                self._index.append((fi, t))

        self._cache: dict[int, Any] = {}

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        fi, t = self._index[idx]
        if fi not in self._cache:
            self._cache[fi] = np.load(self.files[fi], mmap_mode="r")
        data = self._cache[fi]
        return {
            "shadow_map":       torch.from_numpy(data["shadow_map"][t]),
            "velocity":         torch.from_numpy(data["velocity"][t]),
            "future_occupancy": torch.from_numpy(data["future_occupancy"][t]),
        }


if __name__ == "__main__":
    from omegaconf import OmegaConf
    import pathlib, tempfile, os

    cfg_path = pathlib.Path(__file__).parents[1] / "configs" / "default.yaml"
    cfg = OmegaConf.load(cfg_path)

    # create a tiny dummy dataset
    with tempfile.TemporaryDirectory() as tmpdir:
        split_dir = pathlib.Path(tmpdir) / "train"
        split_dir.mkdir()
        h, w = cfg.depth.output_h, cfg.depth.output_w
        T_ep, T_hor = 10, len(cfg.world_model.prediction_horizons)
        np.savez(
            split_dir / "ep000.npz",
            shadow_map=np.random.rand(T_ep, 1, h, w).astype(np.float32),
            velocity=np.random.rand(T_ep, 2, h, w).astype(np.float32),
            future_occupancy=np.random.rand(T_ep, T_hor, h, w).astype(np.float32),
        )
        ds = RolloutDataset(cfg, tmpdir)
        sample = ds[0]
        print(f"RolloutDataset len={len(ds)}")
        for k, v in sample.items():
            print(f"  {k}: {v.shape}")
    print("RolloutDataset OK")
