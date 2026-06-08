"""Maps uncertainty vector → audio parameters.

Special cues:
  - Falling object (rapid vertical displacement in world model): descending Doppler pitch sweep
  - High-uncertainty zone (shadow): low-frequency ambient hum, panned to zone direction
  - Clear path: silence (absence of cue = safe)

All cue frequencies are non-overlapping to avoid perceptual masking.
"""

from __future__ import annotations

import numpy as np
from omegaconf import DictConfig


# Frequency bands per cue type — keep non-overlapping
_SHADOW_HUM_FREQ_RATIO   = 0.5   # 0.5× base_frequency ≈ 100Hz  (low ambient hum)
_OBSTACLE_TONE_FREQ_RATIO = 1.0  # 1.0× ≈ 200Hz
_FALLING_SWEEP_START      = 3.0  # 3.0× ≈ 600Hz, sweeps down to 0.5×
_STOP_FREQ_RATIO          = 4.0  # 4.0× ≈ 800Hz distinct "stop" pattern


class CueMapper:
    """Converts uncertainty grid + world model flags into audio_params for HRTFEngine.

    Primary method: ``map(uncertainty_grid, falling_mask, is_stop) -> audio_params``

    Args:
        uncertainty_grid: (9,) float32, zone uncertainties in [0, 1]
        falling_mask:     (9,) bool, True if world model predicts falling object in zone
        is_stop:          bool, True if orchestrator override engaged

    Returns:
        audio_params: (27,) float32 — (direction, intensity, pitch) × 9 zones
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self._sweep_phase: float = 0.0  # tracks Doppler sweep position

    def map(
        self,
        uncertainty_grid: np.ndarray,
        falling_mask: np.ndarray,
        is_stop: bool = False,
    ) -> np.ndarray:
        params = np.zeros((9, 3), dtype=np.float32)

        if is_stop:
            # all zones emit the "stop" tone at max intensity
            params[:, 0] = np.arange(9, dtype=np.float32)
            params[:, 1] = 1.0
            params[:, 2] = _STOP_FREQ_RATIO - 1.0  # pitch_ratio field is additive offset
            return params.ravel()

        for i in range(9):
            u = float(uncertainty_grid[i])
            if u < 1e-2:
                continue  # silence = safe

            params[i, 0] = float(i)  # direction (zone index, resolved to azimuth in HRTF)
            params[i, 1] = np.clip(u, self.cfg.audio.min_intensity, self.cfg.audio.max_intensity)

            if falling_mask[i]:
                # descending Doppler sweep: pitch starts high, sweeps down over 3s
                self._sweep_phase = (self._sweep_phase + 1.0 / (self.cfg.audio.sample_rate / self.cfg.audio.buffer_size)) % 1.0
                pitch = _FALLING_SWEEP_START * (1.0 - self._sweep_phase) + _SHADOW_HUM_FREQ_RATIO * self._sweep_phase
                params[i, 2] = pitch - 0.5  # offset from base
            else:
                params[i, 2] = _SHADOW_HUM_FREQ_RATIO - 0.5

        return params.ravel()


if __name__ == "__main__":
    from omegaconf import OmegaConf
    import pathlib

    cfg_path = pathlib.Path(__file__).parents[1] / "configs" / "default.yaml"
    cfg = OmegaConf.load(cfg_path)

    mapper = CueMapper(cfg)
    uncertainty = np.array([0.8, 0.1, 0.0, 0.5, 0.9, 0.2, 0.0, 0.3, 0.6], dtype=np.float32)
    falling = np.array([False, False, False, False, True, False, False, False, False])

    params = mapper.map(uncertainty, falling)
    print(f"CueMapper output shape: {params.shape}")
    print(f"  Zone 4 (front, falling): dir={params[4*3]:.1f} int={params[4*3+1]:.2f} pitch={params[4*3+2]:.2f}")

    stop_params = mapper.map(uncertainty, falling, is_stop=True)
    print(f"  Stop override — zone 0 intensity: {stop_params[1]:.2f}")
    print("CueMapper OK")
