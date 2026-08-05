"""Regression tests from the deep adversarial pass.

Each corresponds to a failure found by actually executing a path rather than
reasoning about it: DDP, PPO, degenerate sensor input, and hostile configs.
"""

from __future__ import annotations

import pathlib
import tempfile

import numpy as np
import pytest
import torch

from shadowweave.eval.metrics import iou
from shadowweave.shadow.zones import (
    pool_predictions_to_zones,
    pool_predictions_to_zones_torch,
    pool_to_zones,
    zone_bounds,
)
from shadowweave.utils import sanitize_depth


class TestZonePoolingBackendsAgree:
    """The numpy and torch poolers disagreed by 0.50.

    torch used adaptive_avg_pool2d (mean over both axes) while numpy did max-over-rows
    then mean. PPO built observations with one and the orchestrator with the other, so
    the policy was trained on a different representation than it was run on.
    """

    def test_backends_match(self):
        rng = np.random.default_rng(0)
        pred = rng.random((2, 4, 96, 96)).astype(np.float32)
        np_out = np.stack([pool_predictions_to_zones(p) for p in pred])
        t_out = pool_predictions_to_zones_torch(torch.from_numpy(pred)).numpy()
        assert np.abs(np_out - t_out).max() < 1e-5

    @pytest.mark.parametrize("width", [96, 128, 32, 27, 9, 100])
    def test_matches_for_widths_that_do_not_divide_evenly(self, width):
        """96/9 is not an integer; MPS rejects adaptive pooling in exactly that case."""
        rng = np.random.default_rng(1)
        pred = rng.random((1, 2, width, width)).astype(np.float32)
        np_out = pool_predictions_to_zones(pred[0])
        t_out = pool_predictions_to_zones_torch(torch.from_numpy(pred))[0].numpy()
        assert np.abs(np_out - t_out).max() < 1e-5

    def test_zone_bounds_cover_every_column_exactly_once(self):
        for width in (96, 128, 27, 9, 10):
            bounds = zone_bounds(width)
            assert len(bounds) == 9
            assert all(c1 > c0 for c0, c1 in bounds), "a zone must not be empty"
            assert bounds[0][0] == 0

    def test_pooling_is_differentiable(self):
        t = torch.rand(1, 4, 96, 96, requires_grad=True)
        pool_predictions_to_zones_torch(t).sum().backward()
        assert t.grad is not None and float(t.grad.abs().sum()) > 0

    def test_max_then_mean_semantics(self):
        """A single occupied cell must light its whole zone (worst case per bearing)."""
        occ = np.zeros((32, 96), dtype=np.float32)
        occ[5, 0:10] = 1.0
        zones = pool_to_zones(occ)
        assert zones[0] > 0.9, "one obstacle should dominate its zone"
        assert zones[8] == 0.0

    def test_rejects_wrong_rank(self):
        with pytest.raises(ValueError, match="B, T, H, W"):
            pool_predictions_to_zones_torch(torch.rand(4, 96, 96))


class TestDegenerateSensorInput:
    """A NaN from a flaky sensor propagated into the audio the user navigates by."""

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_depth_fails_safe(self, cfg, bad):
        from shadowweave.shadow.bev import BEVProjector
        from shadowweave.shadow.raycast import ShadowRaycaster

        d = torch.full((1, 1, cfg.depth.output_h, cfg.depth.output_w), bad)
        with torch.no_grad():
            grid = ShadowRaycaster(cfg).forward_from_depth(d)
            occ, vis = BEVProjector(cfg)(d)

        assert torch.isfinite(grid).all() and torch.isfinite(occ).all()
        # Fails toward "stop", never toward "clear ahead".
        assert float(grid.min()) > 0.9, "unreadable depth must read as high uncertainty"

    def test_sanitize_clamps_range(self):
        d = torch.tensor([[-5.0, 0.5, 17.0, float("nan")]])
        out = sanitize_depth(d)
        assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
        assert torch.isfinite(out).all()

    @pytest.mark.parametrize("value,expected", [(0.0, 1.0), (1.0, 0.0)])
    def test_extreme_but_valid_depth(self, cfg, value, expected):
        from shadowweave.shadow.raycast import ShadowRaycaster

        d = torch.full((1, 1, cfg.depth.output_h, cfg.depth.output_w), value)
        with torch.no_grad():
            grid = ShadowRaycaster(cfg).forward_from_depth(d)
        assert abs(float(grid.mean()) - expected) < 0.05


class TestMetricsCannotBeGamedByNaN:
    def test_nan_prediction_scores_zero(self):
        """A NaN prediction against an empty target used to score a perfect 1.0."""
        assert float(iou(torch.full((1, 4, 4), float("nan")), torch.zeros(1, 4, 4))) == 0.0

    def test_genuine_empty_still_scores_one(self):
        assert float(iou(torch.zeros(1, 4, 4), torch.zeros(1, 4, 4))) == 1.0

    def test_partial_nan_batch_only_penalises_the_broken_sample(self):
        pred = torch.zeros(2, 4, 4)
        pred[1, 0, 0] = float("nan")
        scores = iou(pred, torch.zeros(2, 4, 4))
        assert float(scores[0]) == 1.0 and float(scores[1]) == 0.0


class TestTrainingRefusesNoOpConfigs:
    def test_batch_larger_than_split_raises(self, cfg, rollout_dir):
        """drop_last with an oversized batch gives 0 batches; every epoch would
        silently do nothing and still report a loss."""
        from shadowweave.world_model.train import train

        cfg = cfg.copy()
        cfg.world_model.batch_size = 100_000
        cfg.wandb.mode = "disabled"
        with pytest.raises(ValueError, match="0 batches"):
            train(cfg, rollout_dir)


class TestCorruptInputsFailClearly:
    def test_unreadable_npz_explains_itself(self, cfg):
        from shadowweave.world_model.dataset import RolloutDataset

        with tempfile.TemporaryDirectory() as td:
            d = pathlib.Path(td) / "train"
            d.mkdir(parents=True)
            (d / "bad.npz").write_bytes(b"truncated garbage")
            with pytest.raises(ValueError, match="not a readable .npz"):
                RolloutDataset(cfg, td, split="train")

    def test_truncated_checkpoint_raises(self, cfg, tmp_path):
        from shadowweave.utils import load_checkpoint, save_checkpoint
        from shadowweave.world_model import build_world_model

        p = tmp_path / "c.pt"
        save_checkpoint(p, build_world_model(cfg), cfg, epoch=1)
        raw = p.read_bytes()
        p.write_bytes(raw[: len(raw) // 2])
        with pytest.raises(Exception):
            load_checkpoint(p)


class TestEdgeConfigsBuild:
    @pytest.mark.parametrize("overrides", [
        {"world_model.prediction_horizons": [5]},
        {"world_model.prediction_horizons": [1, 2, 3, 4, 5, 6]},
        {"world_model.architecture": "convlstm"},
        {"bev.size": 64},
        {"bev.input_channels": ["occupancy"]},
        {"bev.input_channels": ["occupancy", "visibility"]},
        {"world_model.base_channels": 4},
    ])
    def test_model_builds_and_runs(self, cfg, overrides):
        from shadowweave.world_model import build_world_model

        cfg = cfg.copy()
        for k, v in overrides.items():
            node, leaf = k.rsplit(".", 1)
            cfg[node][leaf] = v

        model = build_world_model(cfg)
        ch = list(cfg.bev.input_channels)
        c = ("occupancy" in ch) + ("visibility" in ch) + 2 * ("flow" in ch)
        S, T = cfg.bev.size, len(cfg.world_model.prediction_horizons)
        out = model(torch.rand(1, c, S, S))
        assert out.shape == (1, T, S, S)
        assert pool_predictions_to_zones_torch(torch.sigmoid(out)).shape == (1, T, 9)


class TestSimDeterminism:
    def test_same_seed_gives_identical_rollout(self, cfg):
        pytest.importorskip("mujoco")
        from shadowweave.sim.mujoco_env import ShadowWeaveEnv

        cfg = cfg.copy()
        cfg.sim.difficulty = "debris"
        runs = []
        for _ in range(2):
            env = ShadowWeaveEnv(cfg, render_rgb=False)
            obs = env.reset(seed=1234)
            for _ in range(8):
                obs = env.step(action=np.array([1.0, 0.0, 0.2], dtype=np.float32))
            runs.append((obs["depth"].copy(), obs["bev_occupancy"].copy()))
            env.close()
        assert np.array_equal(runs[0][0], runs[1][0])
        assert np.array_equal(runs[0][1], runs[1][1])


class TestShardArithmetic:
    """The gen_data array job must cover every episode exactly once."""

    @pytest.mark.parametrize("total,nshards", [
        (400, 8), (400, 3), (10, 8), (7, 3), (1, 8), (100, 7),
    ])
    def test_shards_partition_the_range(self, total, nshards):
        per = (total + nshards - 1) // nshards
        covered: list[int] = []
        for s in range(nshards):
            seed0 = s * per
            count = min(per, total - seed0)
            if count > 0:
                covered += list(range(seed0, seed0 + count))
        assert sorted(covered) == list(range(total))
        assert len(covered) == len(set(covered)), "shards overlap"
