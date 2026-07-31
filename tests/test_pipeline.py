"""Dataset, checkpoint, and end-to-end pipeline tests.

The dataset cases exist because the original rollouts were fully degenerate — every
velocity entry was 0.0 and the target had std 0.0 across both time and horizons — and
nothing in the codebase would have noticed.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from shadowweave.eval.metrics import EvalMetrics, iou, masked_iou
from shadowweave.utils import (
    config_from_checkpoint,
    load_checkpoint,
    load_config,
    save_checkpoint,
)
from shadowweave.world_model.dataset import RolloutDataset
from shadowweave.world_model.diffusion import build_world_model


class TestDataset:
    def test_loads_and_shapes(self, cfg, rollout_dir):
        ds = RolloutDataset(cfg, rollout_dir, split="train")
        S, H = cfg.bev.size, len(cfg.world_model.prediction_horizons)
        sample = ds[0]
        assert sample["input"].shape == (4, S, S)   # occupancy + visibility + 2x flow
        assert sample["target"].shape == (H, S, S)
        assert len(ds) == 12  # 2 episodes x 6 timesteps

    def test_works_with_dataloader_workers(self, cfg, rollout_dir):
        """Memory-mapped sidecars must survive being forked into workers."""
        ds = RolloutDataset(cfg, rollout_dir, split="train")
        batch = next(iter(DataLoader(ds, batch_size=4, num_workers=2)))
        assert batch["input"].shape[0] == 4

    def test_tensors_are_writable(self, cfg, rollout_dir):
        """torch.from_numpy on a read-only memmap slice yields an unpinnable tensor."""
        t = RolloutDataset(cfg, rollout_dir, split="train")[0]["input"]
        t += 1.0  # must not raise

    def test_rejects_pre_bev_schema(self, cfg, tmp_path):
        """The 100 rollouts already on disk use the old, degenerate schema."""
        d = tmp_path / "train"
        d.mkdir(parents=True)
        S = cfg.bev.size
        np.savez(
            d / "ep000.npz",
            shadow_map=np.zeros((4, 1, S, S), np.float32),
            velocity=np.zeros((4, 2, S, S), np.float32),
            future_occupancy=np.zeros((4, 4, S, S), np.float32),
        )
        with pytest.raises(ValueError, match="pre-BEV schema"):
            RolloutDataset(cfg, str(tmp_path), split="train")

    def test_missing_split_says_how_to_fix_it(self, cfg, tmp_path):
        with pytest.raises(FileNotFoundError, match="synthetic_data"):
            RolloutDataset(cfg, str(tmp_path), split="train")

    def test_augmentation_flips_flow_sign(self, cfg, rollout_dir):
        """A left/right mirror must negate the lateral flow channel."""
        ds = RolloutDataset(cfg, rollout_dir, split="train", augment=False)
        x = ds[0]["input"].numpy()
        flipped, _ = RolloutDataset(cfg, rollout_dir, split="train", augment=True)._augment(
            x.copy(), ds[0]["target"].numpy().copy()
        )
        # Either untouched or mirrored-with-negated-lateral; never mirrored-as-is.
        if not np.allclose(flipped, x):
            assert np.allclose(flipped[-1], -x[-1, :, ::-1])


class TestDatasetNonDegeneracy:
    """Guards against silently regenerating a constant dataset."""

    def _stats(self, cfg, path):
        ds = RolloutDataset(cfg, path, split="train")
        xs = np.stack([ds[i]["input"].numpy() for i in range(len(ds))])
        ys = np.stack([ds[i]["target"].numpy() for i in range(len(ds))])
        return xs, ys

    def test_inputs_vary_over_time(self, cfg, rollout_dir):
        xs, _ = self._stats(cfg, rollout_dir)
        assert xs.std(axis=0).mean() > 1e-4, "input is constant across the episode"

    def test_targets_vary_over_time(self, cfg, rollout_dir):
        _, ys = self._stats(cfg, rollout_dir)
        assert ys.std(axis=0).mean() > 1e-4, "target is constant — model can learn a constant"

    def test_flow_channels_are_not_all_zero(self, cfg, rollout_dir):
        xs, _ = self._stats(cfg, rollout_dir)
        assert np.abs(xs[:, 2:]).max() > 0, "flow is identically zero, as it was originally"


class TestCheckpoints:
    def test_round_trip_preserves_architecture(self, cfg, tmp_path):
        """A bare state_dict lost base_channels, which is what broke eval and RL."""
        cfg = cfg.copy()
        cfg.world_model.base_channels = 8
        model = build_world_model(cfg)
        p = tmp_path / "best.pt"
        save_checkpoint(p, model, cfg, epoch=3, metrics={"val_loss": 0.5})

        loaded_cfg = config_from_checkpoint(p, load_config())
        assert loaded_cfg.world_model.base_channels == 8
        build_world_model(loaded_cfg).load_state_dict(load_checkpoint(p)["model"])

    def test_metadata_survives(self, cfg, tmp_path):
        p = tmp_path / "c.pt"
        save_checkpoint(p, build_world_model(cfg), cfg, epoch=7, metrics={"iou_5s": 0.61})
        ck = load_checkpoint(p)
        assert ck["epoch"] == 7 and ck["metrics"]["iou_5s"] == pytest.approx(0.61)

    def test_legacy_bare_state_dict_is_normalised(self, cfg, tmp_path):
        p = tmp_path / "legacy.pt"
        torch.save(build_world_model(cfg).state_dict(), p)
        ck = load_checkpoint(p)
        assert ck["format_version"] == 1 and ck["config"] is None
        assert config_from_checkpoint(p, cfg) is cfg  # falls back, does not crash


class TestMetrics:
    def test_perfect_prediction_scores_one(self):
        t = (torch.rand(4, 16, 16) > 0.8).float()
        assert torch.allclose(iou(t, t), torch.ones(4))

    def test_empty_on_empty_is_a_hit(self):
        z = torch.zeros(2, 8, 8)
        assert torch.allclose(iou(z, z), torch.ones(2))

    def test_disjoint_prediction_scores_zero(self):
        p = torch.zeros(1, 8, 8); p[0, :4] = 1.0
        t = torch.zeros(1, 8, 8); t[0, 4:] = 1.0
        assert float(iou(p, t)) == 0.0

    def test_masked_iou_ignores_cells_outside_the_mask(self):
        p = torch.zeros(1, 8, 8); p[0, :4] = 1.0
        t = torch.zeros(1, 8, 8)
        mask = torch.zeros(1, 8, 8, dtype=torch.bool); mask[0, 4:] = True
        assert float(masked_iou(p, t, mask)) == 1.0  # the error lies outside the mask

    def test_summary_keys_are_named_by_horizon(self):
        m = EvalMetrics(horizons=[1, 3, 5, 10])
        m.log_prediction(torch.rand(4, 8, 8), (torch.rand(4, 8, 8) > 0.5).float())
        s = m.summary()
        assert {"iou_1s", "iou_3s", "iou_5s", "iou_10s"} <= set(s)


class TestTrainingStep:
    def test_loss_decreases_on_a_fixed_batch(self, cfg):
        """A real optimisation step must reduce loss — catches a dead graph."""
        import torch.nn.functional as F

        torch.manual_seed(0)
        model = build_world_model(cfg)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        S, T = cfg.bev.size, len(cfg.world_model.prediction_horizons)
        x = torch.rand(2, 4, S, S)
        y = (torch.rand(2, T, S, S) > 0.9).float()
        pw = torch.tensor(cfg.world_model.pos_weight)

        first = last = None
        for i in range(12):
            loss = F.binary_cross_entropy_with_logits(model(x), y, pos_weight=pw)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if i == 0:
                first = loss.item()
            last = loss.item()
        assert last < first, f"loss did not decrease ({first:.4f} → {last:.4f})"
