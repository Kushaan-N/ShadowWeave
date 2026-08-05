"""Tests for the conditional diffusion world model.

The point of adding diffusion was not the label: BCE is mean-seeking and blurs
multimodal futures, which in this system are concentrated in the shadow. So the tests
that matter are the ones asserting the model actually samples — that two draws differ,
and that the machinery for measuring *where* they differ is sound.
"""

from __future__ import annotations

import pytest
import torch

from shadowweave.eval.baselines import shadow_diversity
from shadowweave.world_model import DiffusionWorldModel, build_world_model
from shadowweave.world_model.ddpm import cosine_beta_schedule
from shadowweave.world_model.unet import n_input_channels


@pytest.fixture
def dcfg(cfg):
    c = cfg.copy()
    c.world_model.architecture = "diffusion"
    c.world_model.diffusion.timesteps = 50
    c.world_model.diffusion.sample_steps = 3
    c.world_model.diffusion.predict_samples = 2
    return c


def _cond_target(cfg, b=2):
    S = cfg.bev.size
    C = n_input_channels(cfg)
    T = len(cfg.world_model.prediction_horizons)
    return torch.rand(b, C, S, S), (torch.rand(b, T, S, S) > 0.9).float()


class TestSchedule:
    def test_betas_are_valid_probabilities(self):
        b = cosine_beta_schedule(100)
        assert b.shape == (100,)
        assert float(b.min()) > 0 and float(b.max()) < 1

    def test_alphas_cumprod_decreases_monotonically(self):
        a = torch.cumprod(1.0 - cosine_beta_schedule(100), dim=0)
        assert torch.all(a[1:] <= a[:-1] + 1e-6)
        assert float(a[0]) > 0.9, "first step should barely destroy the signal"
        assert float(a[-1]) < 0.1, "last step should be nearly pure noise"


class TestForwardProcess:
    def test_q_sample_endpoints(self, dcfg):
        m = DiffusionWorldModel(dcfg)
        _, y = _cond_target(dcfg)
        x0 = m._to_pm1(y)
        noise = torch.randn_like(x0)

        early = m.q_sample(x0, torch.zeros(x0.shape[0], dtype=torch.long), noise)
        late = m.q_sample(x0, torch.full((x0.shape[0],), m.timesteps - 1, dtype=torch.long), noise)

        # t=0 stays close to the data; t=T is dominated by noise.
        assert (early - x0).abs().mean() < (late - x0).abs().mean()
        assert (late - noise).abs().mean() < (early - noise).abs().mean()

    def test_scaling_round_trips(self, dcfg):
        m = DiffusionWorldModel(dcfg)
        y = (torch.rand(2, 4, 8, 8) > 0.5).float()
        assert torch.allclose(m._from_pm1(m._to_pm1(y)), y, atol=1e-6)


class TestSampling:
    def test_samples_are_not_identical(self, dcfg):
        """If draws collapse, this is a deterministic model wearing a costume."""
        m = DiffusionWorldModel(dcfg).eval()
        cond, _ = _cond_target(dcfg)
        s = m.sample(cond, n_samples=3)
        T = len(dcfg.world_model.prediction_horizons)
        assert s.shape == (3, cond.shape[0], T, dcfg.bev.size, dcfg.bev.size)
        assert float((s[0] - s[1]).abs().mean()) > 1e-4

    def test_output_is_a_probability(self, dcfg):
        m = DiffusionWorldModel(dcfg).eval()
        cond, _ = _cond_target(dcfg)
        p = m.predict(cond, n_samples=2)
        assert float(p.min()) >= 0.0 and float(p.max()) <= 1.0

    def test_sample_statistics_shapes(self, dcfg):
        m = DiffusionWorldModel(dcfg).eval()
        cond, _ = _cond_target(dcfg)
        mean, std = m.sample_statistics(cond, n_samples=3)
        assert mean.shape == std.shape
        assert float(std.min()) >= 0.0

    def test_step_count_is_a_runtime_knob(self, dcfg):
        """Sampling steps trade quality for latency and must not be baked in."""
        m = DiffusionWorldModel(dcfg).eval()
        cond, _ = _cond_target(dcfg)
        assert m.sample(cond, 1, steps=2).shape == m.sample(cond, 1, steps=5).shape

    def test_seeded_sampling_is_reproducible(self, dcfg):
        m = DiffusionWorldModel(dcfg).eval()
        cond, _ = _cond_target(dcfg)
        g1 = torch.Generator().manual_seed(0)
        g2 = torch.Generator().manual_seed(0)
        assert torch.allclose(m.sample(cond, 2, generator=g1), m.sample(cond, 2, generator=g2))


class TestTraining:
    def test_loss_backprops(self, dcfg):
        m = DiffusionWorldModel(dcfg)
        cond, y = _cond_target(dcfg)
        loss = m.loss(cond, y)
        loss.backward()
        assert torch.isfinite(loss)
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())

    def test_loss_decreases_on_a_fixed_batch(self, dcfg):
        """Overfitting one batch is the cheapest proof the objective is wired up."""
        torch.manual_seed(0)
        m = DiffusionWorldModel(dcfg)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        cond, y = _cond_target(dcfg)
        # Average over several t draws: the per-step loss is noisy by construction.
        first = last = None
        for i in range(25):
            loss = m.loss(cond, y)
            opt.zero_grad(); loss.backward(); opt.step()
            if i < 5:
                first = loss.item() if first is None else first + loss.item()
            if i >= 20:
                last = loss.item() if last is None else last + loss.item()
        assert last < first, f"eps-MSE did not decrease ({first/5:.4f} -> {last/5:.4f})"


class TestInterfaceParity:
    """train.py, run_eval.py and train_rl.py all call loss()/predict() without
    knowing which architecture they hold."""

    @pytest.mark.parametrize("arch", ["unet", "convlstm", "diffusion"])
    def test_all_architectures_share_the_interface(self, dcfg, arch):
        c = dcfg.copy()
        c.world_model.architecture = arch
        m = build_world_model(c)
        cond, y = _cond_target(c)

        assert torch.isfinite(m.loss(cond, y))
        p = m.predict(cond)
        T = len(c.world_model.prediction_horizons)
        assert p.shape == (cond.shape[0], T, c.bev.size, c.bev.size)
        assert float(p.min()) >= 0.0 and float(p.max()) <= 1.0

    def test_factory_builds_diffusion(self, dcfg):
        assert isinstance(build_world_model(dcfg), DiffusionWorldModel)

    def test_only_generative_models_expose_sampling(self, dcfg):
        c = dcfg.copy()
        c.world_model.architecture = "unet"
        assert not hasattr(build_world_model(c), "sample_statistics")
        assert hasattr(build_world_model(dcfg), "sample_statistics")


class TestCheckpointRoundTrip:
    def test_diffusion_checkpoint_reloads(self, dcfg, tmp_path):
        from shadowweave.utils import config_from_checkpoint, load_checkpoint, save_checkpoint

        m = build_world_model(dcfg)
        p = tmp_path / "d.pt"
        save_checkpoint(p, m, dcfg, epoch=1)

        cfg2 = config_from_checkpoint(p, dcfg)
        assert cfg2.world_model.architecture == "diffusion"
        build_world_model(cfg2).load_state_dict(load_checkpoint(p)["model"])


class TestShadowDiversityMetric:
    def test_ratio_above_one_when_uncertainty_sits_in_shadow(self):
        std = torch.zeros(1, 16, 16)
        mask = torch.zeros(1, 16, 16, dtype=torch.bool)
        mask[:, :8] = True
        std[:, :8] = 0.4          # disagreement confined to the shadow
        std[:, 8:] = 0.01
        out = shadow_diversity(std, mask)
        assert out["shadow_diversity_ratio"] > 10

    def test_ratio_near_one_when_noise_is_uniform(self):
        std = torch.full((1, 16, 16), 0.3)
        mask = torch.zeros(1, 16, 16, dtype=torch.bool)
        mask[:, :8] = True
        assert abs(shadow_diversity(std, mask)["shadow_diversity_ratio"] - 1.0) < 0.05

    def test_broadcasts_a_per_cell_mask_over_horizons(self):
        """std carries a horizon axis, the mask does not — both are 3-D, so an
        ndim check alone misses the mismatch."""
        std = torch.rand(4, 16, 16)
        mask = torch.zeros(1, 16, 16, dtype=torch.bool)
        mask[:, :8] = True
        assert shadow_diversity(std, mask)  # must not raise

    def test_returns_empty_when_a_side_is_missing(self):
        std = torch.rand(1, 8, 8)
        assert shadow_diversity(std, torch.ones(1, 8, 8, dtype=torch.bool)) == {}
        assert shadow_diversity(std, torch.zeros(1, 8, 8, dtype=torch.bool)) == {}
