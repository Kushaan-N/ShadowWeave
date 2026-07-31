"""MuJoCo rollout dataset — loads egocentric BEV observation/target pairs.

Loading strategy matters here. ``np.load(path, mmap_mode="r")`` silently ignores
mmap_mode for .npz archives (it only applies to raw .npy), so the previous loader
decompressed an entire episode array on *every* __getitem__ — O(episode) work per
sample. Rollouts are now written uncompressed and each array is extracted once to a
real .npy sidecar that can be genuinely memory-mapped, so a sample costs one page-in
regardless of episode length. With num_workers > 0 this also stops every worker from
holding a full decompressed copy.
"""

from __future__ import annotations

import pathlib
from typing import Optional

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset

_ARRAYS = ("bev_occupancy", "bev_visibility", "bev_flow", "target", "uncertainty_grid")


class RolloutDataset(Dataset):
    """Loads pre-generated MuJoCo rollout files from disk.

    Expected file layout (one .npz per episode):
        bev_occupancy:    (T, 1, S, S) float32
        bev_visibility:   (T, 1, S, S) float32
        bev_flow:         (T, 2, S, S) float32
        target:           (T, H, S, S) float32
        uncertainty_grid: (T, 9)       float32

    Primary method: ``__getitem__(idx)`` → {"input": (C,S,S), "target": (H,S,S)}
    """

    def __init__(
        self,
        cfg: DictConfig,
        data_dir: str,
        split: str = "train",
        augment: bool = False,
    ) -> None:
        self.cfg = cfg
        self.channels = list(cfg.bev.input_channels)
        self.augment = augment
        self.data_dir = pathlib.Path(data_dir)
        self.split_dir = self.data_dir / split
        self.files = sorted(self.split_dir.glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(
                f"No .npz files found in {self.split_dir}. "
                f"Generate them with: python -m shadowweave.sim.synthetic_data --split {split}"
            )

        self.cache_dir = self.split_dir / ".memmap"
        self._materialise()

        self._maps: dict[tuple[int, str], np.memmap] = {}
        self._index: list[tuple[int, int]] = []
        for fi, n in enumerate(self._lengths):
            self._index.extend((fi, t) for t in range(n))

    def _materialise(self) -> None:
        """Extract each .npz field to a .npy sidecar once, so it can be mmapped."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lengths: list[int] = []
        for fi, f in enumerate(self.files):
            stem = self.cache_dir / f.stem
            marker = stem.with_suffix(".ok")
            if marker.exists():
                self._lengths.append(int(np.load(str(stem) + "__len.npy")))
                continue
            with np.load(f) as data:
                if "bev_occupancy" not in data:
                    raise ValueError(
                        f"{f} uses the pre-BEV schema {sorted(data.files)}.\n"
                        "Rollouts generated before the egocentric-BEV change are not "
                        "usable — their targets were world-frame and constant. "
                        f"Delete {self.split_dir} and regenerate with:\n"
                        "  python -m shadowweave.sim.synthetic_data --split "
                        f"{self.split_dir.name} --episodes 400"
                    )
                for key in _ARRAYS:
                    if key in data:
                        np.save(f"{stem}__{key}.npy", data[key])
                n = int(data["bev_occupancy"].shape[0])
            np.save(f"{stem}__len.npy", np.array(n))
            marker.touch()
            self._lengths.append(n)

    def _get_map(self, fi: int, key: str) -> np.memmap:
        cached = self._maps.get((fi, key))
        if cached is None:
            stem = self.cache_dir / self.files[fi].stem
            cached = np.load(f"{stem}__{key}.npy", mmap_mode="r")
            self._maps[(fi, key)] = cached
        return cached

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        fi, t = self._index[idx]

        parts = []
        if "occupancy" in self.channels:
            parts.append(np.asarray(self._get_map(fi, "bev_occupancy")[t]))
        if "visibility" in self.channels:
            parts.append(np.asarray(self._get_map(fi, "bev_visibility")[t]))
        if "flow" in self.channels:
            parts.append(np.asarray(self._get_map(fi, "bev_flow")[t]))

        x = np.concatenate(parts, axis=0).astype(np.float32)
        y = np.asarray(self._get_map(fi, "target")[t]).astype(np.float32)

        if self.augment:
            x, y = self._augment(x, y)

        # np.array(copy) because memmap slices are read-only and torch.from_numpy on a
        # non-writable buffer warns and produces a tensor that cannot be pinned.
        return {"input": torch.from_numpy(np.ascontiguousarray(x)),
                "target": torch.from_numpy(np.ascontiguousarray(y))}

    def _augment(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Mirror left/right. The BEV is egocentric, so this is a valid symmetry —
        but the flow's lateral channel must flip sign with it."""
        if np.random.rand() < 0.5:
            x = x[:, :, ::-1].copy()
            y = y[:, :, ::-1].copy()
            if "flow" in self.channels:
                x[-1] *= -1.0
        return x, y

    def positive_rate(self, max_samples: int = 200) -> float:
        """Fraction of positive target cells — used to sanity-check pos_weight."""
        idxs = np.linspace(0, len(self) - 1, min(max_samples, len(self))).astype(int)
        return float(np.mean([self[i]["target"].mean().item() for i in idxs]))


if __name__ == "__main__":
    import tempfile

    from ..utils import load_config

    cfg = load_config()
    S = cfg.bev.size
    H = len(cfg.world_model.prediction_horizons)

    with tempfile.TemporaryDirectory() as tmpdir:
        split_dir = pathlib.Path(tmpdir) / "train"
        split_dir.mkdir()
        T_ep = 10
        np.savez(
            split_dir / "ep000.npz",
            bev_occupancy=np.random.rand(T_ep, 1, S, S).astype(np.float32),
            bev_visibility=np.random.rand(T_ep, 1, S, S).astype(np.float32),
            bev_flow=np.random.rand(T_ep, 2, S, S).astype(np.float32),
            target=(np.random.rand(T_ep, H, S, S) > 0.93).astype(np.float32),
            uncertainty_grid=np.random.rand(T_ep, 9).astype(np.float32),
        )
        ds = RolloutDataset(cfg, tmpdir)
        sample = ds[0]
        print(f"RolloutDataset len={len(ds)}  channels={ds.channels}")
        for k, v in sample.items():
            print(f"  {k}: {tuple(v.shape)} {v.dtype}")
        print(f"  positive rate: {ds.positive_rate():.4f}")

        from torch.utils.data import DataLoader
        dl = DataLoader(ds, batch_size=4, num_workers=2)
        batch = next(iter(dl))
        print(f"  batched (num_workers=2): {tuple(batch['input'].shape)}")
    print("RolloutDataset OK")
