"""Regression tests for defects found in adversarial review.

Each of these reproduces a real bug that shipped and passed the existing suite. They
are written to fail against the previous implementation, not merely to describe the
current one.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from shadowweave.eval.baselines import BaselineComparison, calibration_error
from shadowweave.utils import unwrap_model


class TestSnapshotGateIsContentDerived:
    """The height gate was memoised in a dict keyed by id(snapshot).

    CPython reuses object ids after garbage collection, so a transient snapshot could
    be handed a previous one's gate — reproduced with three different snapshots
    sharing a single id, silently yielding the wrong ground truth.
    """

    def test_gate_matches_its_own_content(self, cfg):
        pytest.importorskip("mujoco")
        from shadowweave.sim.mujoco_env import ShadowWeaveEnv

        cfg = cfg.copy()
        cfg.sim.difficulty = "debris"
        env = ShadowWeaveEnv(cfg, render_rgb=False)
        env.reset(seed=3)
        for _ in range(40):
            env.step()
        snap = env.geom_snapshot()
        assert np.array_equal(snap["active"], env._height_gate(snap))
        env.close()

    def test_changed_heights_change_the_gate(self, cfg):
        pytest.importorskip("mujoco")
        from shadowweave.sim.mujoco_env import ShadowWeaveEnv

        cfg = cfg.copy()
        cfg.sim.difficulty = "static"
        env = ShadowWeaveEnv(cfg, render_rgb=False)
        obs = env.reset(seed=3)
        pos, yaw = obs["agent_pos"].copy(), obs["agent_yaw"]

        snap = env.geom_snapshot()
        before = env.bev_occupancy(pos, yaw, snap).sum()

        lifted = copy.deepcopy(snap)
        lifted["xpos"][:, 2] += 20.0
        lifted["active"] = env._height_gate(lifted)
        assert env.bev_occupancy(pos, yaw, lifted).sum() == 0.0

        # And the original must be unaffected by the second call.
        assert env.bev_occupancy(pos, yaw, snap).sum() == before
        env.close()


class TestPersistenceBaselineIsNotGivenTheAnswer:
    """Persistence was scored with the observation available at DUE time.

    That hands it the answer, making it near-unbeatable and understating the model's
    gain — the comparison would have read as "the world model failed".
    """

    def test_stale_observation_scores_worse_than_the_present_one(self):
        truth = torch.zeros(1, 16, 16)
        truth[0, 8:12, 8:12] = 1.0
        stale = torch.zeros(1, 16, 16)
        stale[0, 0:4, 0:4] = 1.0        # where the object used to be

        fair = BaselineComparison([1])
        fair.log(0, np.zeros((16, 16), np.float32), truth, observed_at_issue=stale)
        fair_score = fair.summary()["baseline_persistence_iou_1s"]

        rigged = BaselineComparison([1])
        rigged.log(0, np.zeros((16, 16), np.float32), truth, observed_at_issue=truth.clone())
        rigged_score = rigged.summary()["baseline_persistence_iou_1s"]

        assert fair_score < rigged_score, (
            "scoring persistence against the present frame inflates it")
        assert fair_score == 0.0 and rigged_score == 1.0

    def test_snapshot_is_a_copy(self):
        """The stored observation must not alias a buffer that is later overwritten."""
        occ = torch.ones(1, 8, 8)
        snap = BaselineComparison.observation_snapshot(occ)
        occ.zero_()
        assert float(snap.sum()) == 64.0


class TestCalibrationIsDeviceSafe:
    def test_runs_on_the_tensor_device(self):
        p = torch.rand(256)
        t = (torch.rand(256) > 0.5).float()
        assert 0.0 <= calibration_error(p, t) <= 1.0

    @pytest.mark.skipif(not torch.backends.mps.is_available() and not torch.cuda.is_available(),
                        reason="no accelerator available")
    def test_non_cpu_tensors_do_not_raise(self):
        dev = torch.device("cuda" if torch.cuda.is_available() else "mps")
        p = torch.rand(256, device=dev)
        t = (torch.rand(256, device=dev) > 0.5).float()
        calibration_error(p, t)  # bin edges must be built on p.device


class TestUnwrapModel:
    """torch.compile prefixes state_dict keys with _orig_mod., which would be saved
    into the checkpoint and make it unloadable by a plain model."""

    def test_plain_model_is_returned_unchanged(self, cfg):
        from shadowweave.world_model.diffusion import build_world_model
        m = build_world_model(cfg)
        assert unwrap_model(m) is m

    def test_strips_a_module_wrapper(self, cfg):
        from shadowweave.world_model.diffusion import build_world_model

        inner = build_world_model(cfg)

        class Wrapper(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.module = m

        assert unwrap_model(Wrapper(inner)) is inner

    def test_strips_nested_wrappers(self, cfg):
        from shadowweave.world_model.diffusion import build_world_model

        inner = build_world_model(cfg)

        class Compiled(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self._orig_mod = m

        class DDPLike(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.module = m

        assert unwrap_model(DDPLike(Compiled(inner))) is inner

    def test_unwrapped_state_dict_has_no_prefix(self, cfg):
        from shadowweave.world_model.diffusion import build_world_model

        inner = build_world_model(cfg)

        class Compiled(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self._orig_mod = m

        wrapped = Compiled(inner)
        assert any(k.startswith("_orig_mod.") for k in wrapped.state_dict())
        assert not any(k.startswith("_orig_mod.") for k in unwrap_model(wrapped).state_dict())


class TestEMACorrectness:
    def test_ema_tracks_then_restores(self, cfg):
        from shadowweave.world_model.diffusion import build_world_model
        from shadowweave.world_model.train import EMA

        model = build_world_model(cfg)
        ema = EMA(model, decay=0.9)
        original = {k: v.clone() for k, v in model.state_dict().items()}

        with torch.no_grad():
            for p in model.parameters():
                p.add_(1.0)
        ema.update(model)

        backup = ema.copy_to(model)
        # EMA lags the live weights, so it must differ from them...
        assert not all(torch.equal(model.state_dict()[k], backup[k]) for k in backup)
        ema.restore(model, backup)
        # ...and restoring must return exactly the live weights.
        for k, v in backup.items():
            assert torch.equal(model.state_dict()[k], v)

    def test_ema_moves_toward_the_weights(self, cfg):
        from shadowweave.world_model.diffusion import build_world_model
        from shadowweave.world_model.train import EMA

        model = build_world_model(cfg)
        ema = EMA(model, decay=0.5)
        key = next(iter(ema.shadow))
        with torch.no_grad():
            model.state_dict()[key].fill_(10.0)
        before = ema.shadow[key].mean().item()
        for _ in range(20):
            ema.update(model)
        after = ema.shadow[key].mean().item()
        assert abs(after - 10.0) < abs(before - 10.0)
