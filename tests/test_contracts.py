"""Interface-contract tests.

Every case here corresponds to a place where two modules disagreed about shapes or
semantics and the mismatch only surfaced at runtime — or never surfaced at all.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from shadowweave.agents.global_agent import GlobalAgent
from shadowweave.agents.local_agent import LocalAgent, compute_reward, observation_dim
from shadowweave.agents.orchestrator import Orchestrator
from shadowweave.audio.cues import AUDIO_PARAM_DIM, CueMapper
from shadowweave.audio.hrtf import HRTFEngine
from shadowweave.world_model.diffusion import ConvLSTMCell, build_world_model


class TestOrchestratorContract:
    """step() returned a 3-dim nav action where 27 audio params were expected."""

    def _orch(self, cfg):
        return Orchestrator(cfg, LocalAgent(cfg), GlobalAgent(cfg))

    def test_audio_params_are_27_dim(self, cfg):
        S, T = cfg.bev.size, len(cfg.world_model.prediction_horizons)
        out = self._orch(cfg).step(np.random.rand(9).astype(np.float32) * 0.5,
                                   np.random.rand(T, S, S).astype(np.float32))
        assert out.audio_params.shape == (AUDIO_PARAM_DIM,)
        assert out.nav_action.shape == (3,)

    def test_stop_override_zeroes_navigation(self, cfg):
        S, T = cfg.bev.size, len(cfg.world_model.prediction_horizons)
        out = self._orch(cfg).step(np.ones(9, dtype=np.float32) * 0.99,
                                   np.random.rand(T, S, S).astype(np.float32))
        assert out.is_stop
        assert not out.nav_action.any(), "must not keep steering during a stop"
        # Every zone must sound, but nine coherent same-phase tones sum linearly —
        # full intensity on all of them hard-clipped 76% of the output into a
        # square wave, so the per-zone intensity is 1/N_ZONES.
        assert (out.audio_params[1::3] > 0).all(), "stop pattern must sound in every zone"
        assert float(out.audio_params[1::3].sum()) <= 1.0 + 1e-6

    def test_flattened_prediction_is_rejected(self, cfg):
        """run_eval passed wm_pred.ravel() into a method that indexes (T, S, S)."""
        S, T = cfg.bev.size, len(cfg.world_model.prediction_horizons)
        with pytest.raises(ValueError, match="wm_pred"):
            self._orch(cfg).step(np.random.rand(9).astype(np.float32),
                                 np.random.rand(T, S, S).astype(np.float32).ravel())

    def test_wrong_zone_count_is_rejected(self, cfg):
        S, T = cfg.bev.size, len(cfg.world_model.prediction_horizons)
        with pytest.raises(ValueError, match="uncertainty_grid"):
            self._orch(cfg).step(np.random.rand(5).astype(np.float32),
                                 np.random.rand(T, S, S).astype(np.float32))

    def test_observation_dim_matches_what_orchestrator_feeds(self, cfg):
        """The dimension eval computed (9 + T*H*W) never matched the 9 + T*9 fed."""
        agent = LocalAgent(cfg)
        assert agent.obs_dim == observation_dim(cfg)
        assert agent.obs_dim == 9 + len(cfg.world_model.prediction_horizons) * 9


class TestWorldModelContract:
    def test_convlstm_cell_runs(self, cfg):
        """ConvLSTMCell.forward raised ValueError on every call — the fallback was dead."""
        cell = ConvLSTMCell(4, 8)
        x = torch.rand(2, 4, 16, 16)
        h = torch.zeros(2, 8, 16, 16)
        h2, c2 = cell(x, h, h.clone())
        assert h2.shape == (2, 8, 16, 16) and c2.shape == (2, 8, 16, 16)
        assert torch.isfinite(h2).all()

    @pytest.mark.parametrize("arch", ["unet", "convlstm"])
    def test_both_architectures_share_an_interface(self, cfg, arch):
        cfg = cfg.copy()
        cfg.world_model.architecture = arch
        model = build_world_model(cfg)
        S, T = cfg.bev.size, len(cfg.world_model.prediction_horizons)
        x = torch.rand(2, 4, S, S)
        assert model(x).shape == (2, T, S, S)
        probs = model.predict(x)
        assert float(probs.min()) >= 0.0 and float(probs.max()) <= 1.0

    def test_forward_returns_logits_not_probabilities(self, cfg):
        """Training applies BCEWithLogits; a sigmoid inside forward would double it."""
        model = build_world_model(cfg)
        torch.nn.init.constant_(model.head.bias, 5.0)
        with torch.no_grad():
            out = model(torch.rand(1, 4, cfg.bev.size, cfg.bev.size))
        assert float(out.max()) > 1.0, "output is squashed — forward() must emit logits"

    def test_bad_bev_size_fails_loudly(self, cfg):
        model = build_world_model(cfg)
        with pytest.raises(ValueError, match="divisible by 16"):
            model(torch.rand(1, 4, 30, 30))


class TestPlannerContract:
    def test_reads_the_global_config_key(self, cfg):
        """cfg.agents.global_agent did not exist, so the weight was never applied."""
        assert GlobalAgent(cfg).heuristic_weight == cfg.agents["global"].astar_heuristic_weight

    def test_avoids_obstacles(self, cfg):
        occ = np.zeros((24, 24), dtype=np.float32)
        occ[10:14, :] = 0.99
        occ[10:14, 17:20] = 0.01
        path = GlobalAgent(cfg).plan(occ, (0, 12), (23, 12))
        assert path and path[0] == (0, 12) and path[-1] == (23, 12)
        assert sum(1 for r, c in path if occ[r, c] > 0.5) == 0

    def test_shadow_penalty_changes_the_route(self, cfg):
        occ = np.zeros((24, 24), dtype=np.float32)
        occ[10:14, :] = 0.99
        occ[10:14, 8:11] = 0.01
        occ[10:14, 17:20] = 0.01
        agent = GlobalAgent(cfg)
        shadow = np.zeros((24, 24), dtype=np.float32)
        shadow[:, :12] = 1.0
        plain = agent.plan(occ, (0, 12), (23, 12))
        avoid = agent.plan(occ, (0, 12), (23, 12), shadow=shadow)
        assert sum(1 for _, c in avoid if c < 12) < sum(1 for _, c in plain if c < 12)

    def test_no_path_returns_empty(self, cfg):
        assert GlobalAgent(cfg).plan(np.zeros((8, 8), np.float32), (0, 0), (99, 99)) == []


class TestRewardContract:
    def test_collision_actually_penalises(self, cfg):
        """The old collision flag was True every step, so reward was constant."""
        clear = compute_reward(False, 0.5, cfg)
        hit = compute_reward(True, 0.5, cfg)
        assert hit < clear
        assert hit - clear == pytest.approx(cfg.agents.local.reward_collision_weight)

    def test_shadow_exposure_is_penalised(self, cfg):
        assert compute_reward(False, 0.5, cfg, shadow_exposure=1.0) < compute_reward(False, 0.5, cfg)


class TestAudioContract:
    def test_cue_cap_is_enforced(self, cfg):
        """max_simultaneous_cues existed in config but nothing applied it."""
        mapper = CueMapper(cfg)
        params = mapper.map(np.ones(9, dtype=np.float32) * 0.9, np.zeros(9, dtype=bool))
        assert int((params[1::3] > 0).sum()) <= cfg.agents.local.max_simultaneous_cues

    def test_falling_zone_always_gets_a_slot(self, cfg):
        mapper = CueMapper(cfg)
        u = np.ones(9, dtype=np.float32) * 0.9
        u[4] = 0.15  # lowest uncertainty, but it is the falling one
        falling = np.zeros(9, dtype=bool)
        falling[4] = True
        params = mapper.map(u, falling)
        assert params[4 * 3 + 1] > 0, "a falling object must not be dropped by the cap"

    def test_silence_means_safe(self, cfg):
        params = CueMapper(cfg).map(np.zeros(9, dtype=np.float32), np.zeros(9, dtype=bool))
        assert float(params[1::3].sum()) == 0.0

    def test_left_and_right_are_distinguishable(self, cfg):
        """abs(azimuth) made -90 and +90 produce a bit-identical signal."""
        eng = HRTFEngine(cfg)
        eng.load_hrtfs()

        def render(zone_idx):
            eng._phase[:] = 0.0
            eng._tail = None
            p = np.zeros((9, 3), dtype=np.float32)
            p[zone_idx] = [zone_idx, 0.7, 0.5]
            return eng._synthesise(p)

        left, right = render(1), render(7)
        assert not np.allclose(left, right), "left/right collapsed to the same audio"
        l_bias = np.abs(left[:, 0]).mean() - np.abs(left[:, 1]).mean()
        r_bias = np.abs(right[:, 0]).mean() - np.abs(right[:, 1]).mean()
        assert l_bias > 0 > r_bias

    def test_phase_is_continuous_across_buffers(self, cfg):
        """Restarting the oscillator each buffer produced a click 43x per second."""
        eng = HRTFEngine(cfg)
        eng.load_hrtfs()
        eng._phase[:] = 0.0
        eng._tail = None
        p = np.zeros((9, 3), dtype=np.float32)
        p[4] = [4, 0.7, 0.5]
        b1, b2 = eng._synthesise(p), eng._synthesise(p)
        seam = abs(float(b2[0, 0] - b1[-1, 0]))
        within = float(np.abs(np.diff(b1[:, 0])).max())
        assert seam <= within * 3, f"discontinuity at buffer seam ({seam:.5f} vs {within:.5f})"
