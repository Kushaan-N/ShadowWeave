"""Orchestrator — coordinates local + global agents at 20Hz, handles stop override."""

from __future__ import annotations

import threading
import time
from typing import Callable, Optional

import numpy as np
from omegaconf import DictConfig

from .local_agent import LocalAgent
from .global_agent import GlobalAgent
from ..shadow.zones import pool_predictions_to_zones


class Orchestrator:
    """Runs the main 20Hz control loop, merging local and global agent outputs.

    Override: if max uncertainty across all zones > cfg.agents.orchestrator.uncertainty_stop_threshold,
    pause nav suggestions and request a distinct "stop" audio pattern.

    Primary method: ``step(uncertainty_grid, wm_pred) -> (audio_params, is_stop)``

    Args:
        uncertainty_grid: (9,) float32 — current shadow uncertainty per zone
        wm_pred:          (T, H, W) float32 — world model predictions at each horizon
    """

    def __init__(
        self,
        cfg: DictConfig,
        local_agent: LocalAgent,
        global_agent: GlobalAgent,
        audio_callback: Optional[Callable[[np.ndarray, bool], None]] = None,
    ) -> None:
        self.cfg = cfg
        self.local = local_agent
        self.global_ = global_agent
        self.audio_callback = audio_callback
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
    ) -> None:
        """Called by the 2Hz global planning thread."""
        wps = self.global_.plan(occupancy_pred, start, goal)
        with self._lock:
            self._waypoints = wps

    def step(
        self,
        uncertainty_grid: np.ndarray,
        wm_pred: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        """Single 20Hz step.

        Args:
            uncertainty_grid: (9,) current zone uncertainties
            wm_pred:          (T, H, W) world model predictions — pooled internally to (T*9,)

        Returns:
            audio_params: (27,) float32
            is_stop:      True if override engaged
        """
        max_uncertainty = float(uncertainty_grid.max())

        if max_uncertainty > self._stop_threshold:
            stop_audio = np.zeros(27, dtype=np.float32)
            if self.audio_callback is not None:
                self.audio_callback(stop_audio, True)
            return stop_audio, True

        # pool world model predictions from (T, H, W) → (T, 9) → flatten to (T*9,)
        wm_zones = pool_predictions_to_zones(wm_pred, self._n_zones).ravel()
        obs = np.concatenate([uncertainty_grid, wm_zones]).astype(np.float32)
        audio_params = self.local.act(obs)

        if self.audio_callback is not None:
            self.audio_callback(audio_params, False)

        return audio_params, False

    def run(
        self,
        frame_source: Callable[[], tuple[np.ndarray, np.ndarray]],
        goal: tuple[int, int] = (15, 15),
        max_steps: int = 1000,
    ) -> None:
        """Blocking main loop — for testing only. Real use drives step() externally."""
        period = 1.0 / self.cfg.agents.local.control_hz
        for _ in range(max_steps):
            t0 = time.monotonic()
            uncertainty_grid, wm_pred = frame_source()
            self.step(uncertainty_grid, wm_pred)
            elapsed = time.monotonic() - t0
            remaining = period - elapsed
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    from omegaconf import OmegaConf
    import pathlib

    cfg_path = pathlib.Path(__file__).parents[1] / "configs" / "default.yaml"
    cfg = OmegaConf.load(cfg_path)

    H, W = cfg.depth.output_h, cfg.depth.output_w
    T = len(cfg.world_model.prediction_horizons)
    obs_dim = cfg.shadow.grid_cells + T * cfg.shadow.grid_cells  # 9 + 36 = 45

    local   = LocalAgent(cfg, obs_dim)
    global_ = GlobalAgent(cfg)
    orch    = Orchestrator(cfg, local, global_)

    uncertainty = np.random.rand(9).astype(np.float32) * 0.5
    wm_pred     = np.random.rand(T, H, W).astype(np.float32)

    audio_params, is_stop = orch.step(uncertainty, wm_pred)
    print(f"Orchestrator step: is_stop={is_stop}, audio_params shape={audio_params.shape}")
    assert audio_params.shape == (27,)

    high_uncertainty = np.ones(9, dtype=np.float32) * 0.9
    _, is_stop = orch.step(high_uncertainty, wm_pred)
    print(f"Orchestrator stop override: is_stop={is_stop}")
    print("Orchestrator OK")
