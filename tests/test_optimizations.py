"""Tests that the throughput optimizations did not change behaviour.

A fast pipeline that computes something subtly different from the slow one is worse
than no optimization at all, so each of these pins equivalence rather than speed.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from shadowweave.eval.baselines import BaselineComparison, LeadTimeTracker, calibration_error
from shadowweave.shadow.bev import bev_flow


class TestBatchedProjectionEquivalence:
    def test_batched_matches_per_frame(self, cfg):
        """Data generation projects a whole episode at once instead of frame by frame."""
        from shadowweave.shadow.bev import BEVProjector
        from shadowweave.shadow.raycast import ShadowRaycaster

        proj, rc = BEVProjector(cfg), ShadowRaycaster(cfg)
        rng = np.random.default_rng(0)
        depths = [rng.random((cfg.depth.output_h, cfg.depth.output_w)).astype(np.float32)
                  for _ in range(7)]

        with torch.no_grad():
            batched = proj(torch.from_numpy(np.stack(depths)).unsqueeze(1))
            per_frame = [proj(torch.from_numpy(d).unsqueeze(0).unsqueeze(0)) for d in depths]

        for i in range(len(depths)):
            assert torch.allclose(batched[0][i, 0], per_frame[i][0][0, 0], atol=1e-6)
            assert torch.allclose(batched[1][i, 0], per_frame[i][1][0, 0], atol=1e-6)

        with torch.no_grad():
            u_batch = rc.forward_from_depth(torch.from_numpy(np.stack(depths)).unsqueeze(1))
            u_single = torch.cat([rc.forward_from_depth(
                torch.from_numpy(d).unsqueeze(0).unsqueeze(0)) for d in depths])
        assert torch.allclose(u_batch, u_single, atol=1e-5)

    def test_vectorised_flow_matches_sequential(self, cfg):
        S = cfg.bev.size
        occ = torch.rand(6, 1, S, S)
        vec = bev_flow(occ[:-1], occ[1:])
        seq = torch.cat([bev_flow(occ[i - 1 : i], occ[i : i + 1]) for i in range(1, len(occ))])
        assert torch.allclose(vec, seq, atol=1e-7)


class TestRenderToggle:
    def test_skipping_rgb_leaves_depth_untouched(self, cfg):
        """RGB is off during data generation; depth and targets must be unaffected."""
        mujoco = pytest.importorskip("mujoco")
        from shadowweave.sim.mujoco_env import ShadowWeaveEnv

        cfg = cfg.copy()
        cfg.sim.difficulty = "static"

        with_rgb = ShadowWeaveEnv(cfg, render_rgb=True)
        a = with_rgb.reset(seed=11)
        with_rgb.close()

        without = ShadowWeaveEnv(cfg, render_rgb=False)
        b = without.reset(seed=11)
        without.close()

        assert np.allclose(a["depth"], b["depth"])
        assert np.allclose(a["bev_occupancy"], b["bev_occupancy"])
        assert a["rgb"].size > b["rgb"].size, "RGB should not have been rendered"


class TestBaselines:
    def test_persistence_beats_empty_on_a_static_scene(self):
        bc = BaselineComparison([1, 3])
        truth = torch.zeros(1, 16, 16)
        truth[0, 4:8, 4:8] = 1.0
        bc.set_observation(truth.clone())          # nothing moved
        bc.log(0, np.zeros((16, 16), np.float32), truth)
        s = bc.summary()
        assert s["baseline_persistence_iou_1s"] > s["baseline_empty_iou_1s"]

    def test_gain_is_negative_when_model_loses_to_persistence(self):
        bc = BaselineComparison([1])
        truth = torch.zeros(1, 16, 16)
        truth[0, 4:8, 4:8] = 1.0
        bc.set_observation(truth.clone())
        bc.log(0, np.zeros((16, 16), np.float32), truth)   # model predicts nothing
        assert bc.summary()["model_gain_over_best_baseline_1s"] < 0

    def test_gain_is_positive_when_model_wins(self):
        bc = BaselineComparison([1])
        truth = torch.zeros(1, 16, 16)
        truth[0, 4:8, 4:8] = 1.0
        bc.set_observation(torch.zeros(1, 16, 16))         # observation is stale
        model = np.zeros((16, 16), np.float32)
        model[4:8, 4:8] = 1.0
        bc.log(0, model, truth)
        assert bc.summary()["model_gain_over_best_baseline_1s"] > 0


class TestLeadTime:
    def _run(self, pred_fn):
        horizons = [1, 3, 5, 10]
        t = LeadTimeTracker(min_new_cells=4)
        occ = np.zeros((32, 32), np.float32)
        for step in range(0, 200, 10):
            if step >= 90:
                occ = np.zeros((32, 32), np.float32)
                occ[10:14, 10:14] = 1.0
            t.observe(step, pred_fn(), {"bev_occupancy": occ}, horizons, fps=30)
        return t.summary()

    def test_targeted_prediction_scores_a_high_margin(self):
        def pred():
            p = np.zeros((4, 32, 32), np.float32)
            p[:, 10:14, 10:14] = 0.9
            return p
        s = self._run(pred)
        assert s["falling_detection_rate"] > 0.9
        assert s["falling_detection_margin"] > 0.9

    def test_predicting_everything_earns_no_margin(self):
        """A blanket-positive model must not look like a good anticipator."""
        s = self._run(lambda: np.ones((4, 32, 32), np.float32))
        assert s["falling_detection_rate"] > 0.9      # trivially detects
        assert s["falling_false_alarm_rate"] > 0.9    # but flags everything
        assert s["falling_detection_margin"] < 0.1    # so the margin exposes it


class TestCalibration:
    def test_confident_and_wrong_scores_badly(self):
        assert calibration_error(torch.full((512,), 0.99), torch.zeros(512)) > 0.9

    def test_confident_and_right_scores_well(self):
        assert calibration_error(torch.full((512,), 0.99), torch.ones(512)) < 0.05


class TestCheckpointDurability:
    def test_save_is_atomic(self, cfg, tmp_path):
        """A job killed mid-save must not leave a truncated checkpoint behind."""
        from shadowweave.utils import load_checkpoint, save_checkpoint
        from shadowweave.world_model.diffusion import build_world_model

        p = tmp_path / "c.pt"
        model = build_world_model(cfg)
        save_checkpoint(p, model, cfg, epoch=1, extra={"ema": {"steps": 5}})
        assert not list(tmp_path.glob("*.tmp*")), "temp file left behind"
        assert load_checkpoint(p)["ema"]["steps"] == 5
