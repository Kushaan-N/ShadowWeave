"""Orchestrator — coordinates local + global agents at 20Hz, handles stop override.

The previous version returned ``local.act(obs)``, a 3-dim navigation vector, from a
method documented and consumed as a 27-dim audio parameter block — its own __main__
asserted ``shape == (27,)`` and failed. CueMapper was never called at all, so nothing
ever converted uncertainty into audio.

``step`` now returns an explicit result object with both, and the audio parameters
genuinely come from CueMapper.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from omegaconf import DictConfig

from ..audio.cues import AUDIO_PARAM_DIM, CueMapper
from ..shadow.zones import pool_predictions_to_zones
from .global_agent import GlobalAgent
from .local_agent import LocalAgent


@dataclass
class OrchestratorOutput:
    """One control tick."""
    audio_params: np.ndarray          # (27,) for HRTFEngine.play
    nav_action: np.ndarray            # (3,) forward / strafe / turn
    is_stop: bool
    max_uncertainty: float
    waypoints: list[tuple[int, int]] = field(default_factory=list)


class Orchestrator:
    """Runs the main 20Hz control loop, merging local and global agent outputs.

    Override: if max uncertainty across all zones > cfg.agents.orchestrator.
    uncertainty_stop_threshold, pause nav suggestions and emit the distinct "stop"
    audio pattern.

    Primary method: ``step(uncertainty_grid, wm_pred) -> OrchestratorOutput``
    """

    def __init__(
        self,
        cfg: DictConfig,
        local_agent: LocalAgent,
        global_agent: GlobalAgent,
        audio_callback: Optional[Callable[[np.ndarray, bool], None]] = None,
        cue_mapper: Optional[CueMapper] = None,
    ) -> None:
        self.cfg = cfg
        self.local = local_agent
        self.global_ = global_agent
        self.audio_callback = audio_callback
        self.cues = cue_mapper or CueMapper(cfg)
        self._waypoints: list[tuple[int, int]] = []
        self._stop_threshold = cfg.agents.orchestrator.uncertainty_stop_threshold
        self._lock = threading.Lock()
        self._T = len(cfg.world_model.prediction_horizons)
        self._n_zones = cfg.shadow.grid_cells

    def update_waypoints(
        self,
        occupancy_pred: np.ndarray,
        start: tuple[int, int],
        goal: tuple[int, int],
        shadow: Optional[np.ndarray] = None,
    ) -> list[tuple[int, int]]:
        """Called by the 2Hz global planning thread."""
        wps = self.global_.plan(occupancy_pred, start, goal, shadow=shadow)
        with self._lock:
            self._waypoints = wps
        return wps

    @property
    def waypoints(self) -> list[tuple[int, int]]:
        with self._lock:
            return list(self._waypoints)

    def step(
        self,
        uncertainty_grid: np.ndarray,
        wm_pred: np.ndarray,
        falling_mask: Optional[np.ndarray] = None,
    ) -> OrchestratorOutput:
        """Single 20Hz step.

        Args:
            uncertainty_grid: (9,) current zone uncertainties
            wm_pred:          (T, S, S) world model predictions — pooled internally
            falling_mask:     optional (9,) bool, zones with a predicted falling object

        Returns:
            OrchestratorOutput
        """
        uncertainty_grid = np.asarray(uncertainty_grid, dtype=np.float32).ravel()
        if uncertainty_grid.shape[0] != self._n_zones:
            raise ValueError(
                f"uncertainty_grid must be ({self._n_zones},), got {uncertainty_grid.shape}"
            )
        wm_pred = np.asarray(wm_pred, dtype=np.float32)
        if wm_pred.ndim != 3:
            raise ValueError(f"wm_pred must be (T, S, S), got shape {wm_pred.shape}")
        expected_t = len(self.cfg.world_model.prediction_horizons)
        if wm_pred.shape[0] != expected_t:
            # A wrong horizon count otherwise surfaces as an opaque matmul-shape
            # RuntimeError from inside the policy network.
            raise ValueError(
                f"wm_pred has {wm_pred.shape[0]} horizons, config declares {expected_t}"
            )

        if falling_mask is None:
            falling_mask = np.zeros(self._n_zones, dtype=bool)

        if not (np.isfinite(uncertainty_grid).all() and np.isfinite(wm_pred).all()):
            # A dead sensor or diverged model is maximum danger, not "all clear":
            # NaN > threshold is False, so without this the corrupt frame would
            # bypass the stop override and steer a blind user on garbage.
            uncertainty_grid = np.where(
                np.isfinite(uncertainty_grid), uncertainty_grid, 1.0
            ).astype(np.float32)
            max_uncertainty = 1.0
        else:
            max_uncertainty = float(uncertainty_grid.max())
        is_stop = max_uncertainty > self._stop_threshold

        if is_stop:
            audio_params = self.cues.map(uncertainty_grid, falling_mask, is_stop=True)
            nav_action = np.zeros(3, dtype=np.float32)
        else:
            wm_zones = pool_predictions_to_zones(wm_pred, self._n_zones).ravel()
            obs = np.concatenate([uncertainty_grid, wm_zones]).astype(np.float32)
            nav_action = self.local.act(obs)
            audio_params = self.cues.map(uncertainty_grid, falling_mask, is_stop=False)

        if self.audio_callback is not None:
            self.audio_callback(audio_params, is_stop)

        return OrchestratorOutput(
            audio_params=audio_params,
            nav_action=nav_action,
            is_stop=is_stop,
            max_uncertainty=max_uncertainty,
            waypoints=self.waypoints,
        )

    def run(
        self,
        frame_source: Callable[[], tuple[np.ndarray, np.ndarray]],
        max_steps: int = 1000,
    ) -> None:
        """Blocking main loop — for testing only. Real use drives step() externally."""
        period = 1.0 / self.cfg.agents.local.control_hz
        for _ in range(max_steps):
            t0 = time.monotonic()
            uncertainty_grid, wm_pred = frame_source()
            self.step(uncertainty_grid, wm_pred)
            remaining = period - (time.monotonic() - t0)
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    from ..utils import load_config

    cfg = load_config()
    S = cfg.bev.size
    T = len(cfg.world_model.prediction_horizons)

    local = LocalAgent(cfg)
    global_ = GlobalAgent(cfg)
    orch = Orchestrator(cfg, local, global_)

    uncertainty = np.random.rand(9).astype(np.float32) * 0.5
    wm_pred = np.random.rand(T, S, S).astype(np.float32)

    out = orch.step(uncertainty, wm_pred)
    print(f"Orchestrator step: is_stop={out.is_stop} max_unc={out.max_uncertainty:.3f}")
    print(f"  audio_params: {out.audio_params.shape}  nav_action: {out.nav_action.shape}")
    assert out.audio_params.shape == (AUDIO_PARAM_DIM,), out.audio_params.shape
    assert out.nav_action.shape == (3,)

    high = np.ones(9, dtype=np.float32) * 0.9
    stop_out = orch.step(high, wm_pred)
    print(f"Orchestrator stop override: is_stop={stop_out.is_stop} "
          f"nav={stop_out.nav_action}  (nav must be zero when stopped)")
    assert stop_out.is_stop and not stop_out.nav_action.any()
    assert stop_out.audio_params.shape == (AUDIO_PARAM_DIM,)

    # Shape contract: a flattened prediction must be rejected, not silently mangled.
    try:
        orch.step(uncertainty, wm_pred.ravel())
        print("  ERROR: flattened wm_pred should have raised")
    except ValueError as e:
        print(f"  rejects malformed wm_pred: {e}")

    occ = np.zeros((32, 32), dtype=np.float32)
    occ[10:20, 8:24] = 0.9
    wps = orch.update_waypoints(occ, (0, 0), (31, 31))
    print(f"  planned {len(wps)} waypoints; orchestrator sees {len(orch.waypoints)}")
    print("Orchestrator OK")
