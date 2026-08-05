"""Maps uncertainty vector → audio parameters.

Special cues:
  - Falling object (rapid vertical displacement in world model): descending Doppler pitch sweep
  - High-uncertainty zone (shadow): low-frequency ambient hum, panned to zone direction
  - Clear path: silence (absence of cue = safe)

All cue frequencies are non-overlapping to avoid perceptual masking, and no more than
cfg.agents.local.max_simultaneous_cues zones sound at once. That cap exists in the
config for a reason — nine simultaneous tones is unusable for the blind users this is
built for — but nothing enforced it, so every non-silent zone played.
"""

from __future__ import annotations

import numpy as np
from omegaconf import DictConfig

N_ZONES = 9
AUDIO_PARAM_DIM = N_ZONES * 3

# Frequency bands per cue type — keep non-overlapping
_SHADOW_HUM_FREQ_RATIO = 0.5     # 0.5× base_frequency ≈ 100Hz  (low ambient hum)
_OBSTACLE_TONE_FREQ_RATIO = 1.0  # 1.0× ≈ 200Hz
_FALLING_SWEEP_START = 3.0       # 3.0× ≈ 600Hz, sweeps down to 0.5×
_STOP_FREQ_RATIO = 4.0           # 4.0× ≈ 800Hz distinct "stop" pattern


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
        self.max_cues = int(cfg.agents.local.max_simultaneous_cues)
        self.threshold = float(cfg.audio.cue_threshold)
        self._sweep_phase = np.zeros(N_ZONES, dtype=np.float32)

    def _sweep_step(self) -> float:
        """Fraction of the Doppler sweep advanced per audio buffer."""
        buffers_per_sec = self.cfg.audio.sample_rate / self.cfg.audio.buffer_size
        return 1.0 / max(self.cfg.audio.doppler_sweep_duration * buffers_per_sec, 1.0)

    def map(
        self,
        uncertainty_grid: np.ndarray,
        falling_mask: np.ndarray,
        is_stop: bool = False,
    ) -> np.ndarray:
        u = np.asarray(uncertainty_grid, dtype=np.float32).ravel()
        falling = np.asarray(falling_mask, dtype=bool).ravel()
        if u.size != N_ZONES or falling.size != N_ZONES:
            raise ValueError(
                f"expected {N_ZONES}-zone inputs, got uncertainty {u.size} / falling {falling.size}"
            )
        # A non-finite uncertainty means the upstream model broke, not that the path
        # is clear — silence here would read as "safe" to a blind user. Fail loud.
        u = np.where(np.isfinite(u), np.clip(u, 0.0, 1.0), 1.0)
        params = np.zeros((N_ZONES, 3), dtype=np.float32)
        params[:, 0] = np.arange(N_ZONES, dtype=np.float32)

        if is_stop:
            # Nine coherent same-phase tones sum linearly; full intensity on all of
            # them clips 76% of samples into a square wave. 1/N keeps the summed
            # peak at the level of one full-intensity zone.
            params[:, 1] = 1.0 / N_ZONES
            params[:, 2] = _STOP_FREQ_RATIO - 0.5  # decoded as ratio = pitch + 0.5
            return params.ravel()

        # Falling objects are the highest-priority cue and always keep a slot even
        # below the ambient-hum threshold; the remaining slots go to the most
        # uncertain zones.
        priority = u + falling.astype(np.float32) * 10.0
        audible = np.where((u >= self.threshold) | falling)[0]
        if audible.size > self.max_cues:
            audible = audible[np.argsort(-priority[audible])[: self.max_cues]]
        audible_set = set(int(i) for i in audible)

        step = self._sweep_step()
        for i in range(N_ZONES):
            if i not in audible_set:
                self._sweep_phase[i] = 0.0
                continue  # silence = safe

            params[i, 1] = np.clip(
                u[i], self.cfg.audio.min_intensity, self.cfg.audio.max_intensity
            )
            if falling[i]:
                # A falling hazard the model sees clearly has LOW uncertainty — its
                # warning must not inherit that near-zero loudness.
                params[i, 1] = max(params[i, 1], float(self.cfg.audio.falling_min_intensity))
                # Descending Doppler sweep, one phase per zone so two simultaneous
                # falling objects do not share and corrupt each other's sweep.
                self._sweep_phase[i] = (self._sweep_phase[i] + step) % 1.0
                p = self._sweep_phase[i]
                pitch = _FALLING_SWEEP_START * (1.0 - p) + _SHADOW_HUM_FREQ_RATIO * p
                params[i, 2] = pitch - 0.5
            else:
                self._sweep_phase[i] = 0.0
                params[i, 2] = _SHADOW_HUM_FREQ_RATIO - 0.5

        return params.ravel()


if __name__ == "__main__":
    from ..utils import load_config

    cfg = load_config()
    mapper = CueMapper(cfg)
    labels = ["L-far", "L", "L-near", "FL", "F", "FR", "R-near", "R", "R-far"]

    uncertainty = np.array([0.8, 0.1, 0.0, 0.5, 0.9, 0.2, 0.0, 0.3, 0.6], dtype=np.float32)
    falling = np.zeros(9, dtype=bool)
    falling[4] = True

    params = mapper.map(uncertainty, falling)
    print(f"CueMapper output shape: {params.shape}")
    active = [labels[i] for i in range(9) if params[i * 3 + 1] > 0]
    print(f"  active cues: {active}  (cap = {mapper.max_cues})")
    assert len(active) <= mapper.max_cues, "cue cap not enforced"
    assert "F" in active, "falling zone must always keep a slot"

    print(f"  zone F (falling): int={params[4*3+1]:.2f} pitch={params[4*3+2]:.2f}")
    p2 = mapper.map(uncertainty, falling)
    print(f"  sweep advances between buffers: {params[4*3+2]:.4f} → {p2[4*3+2]:.4f}")
    assert p2[4 * 3 + 2] != params[4 * 3 + 2]

    silent = mapper.map(np.zeros(9, dtype=np.float32), np.zeros(9, dtype=bool))
    print(f"  clear path → total intensity {silent[1::3].sum():.3f}  (silence = safe)")
    assert silent[1::3].sum() == 0.0

    stop = mapper.map(uncertainty, falling, is_stop=True)
    print(f"  stop override → all 9 zones at intensity {stop[1::3].min():.3f}")
    assert (stop[1::3] > 0).all(), "stop pattern must sound in every zone"
    assert stop[1::3].sum() <= 1.0 + 1e-6, "coherent stop tones must not clip when summed"

    quiet_falling = mapper.map(np.zeros(9, dtype=np.float32), falling)
    print(f"  falling in a zero-uncertainty zone → int={quiet_falling[4*3+1]:.2f}")
    assert quiet_falling[4 * 3 + 1] >= cfg.audio.falling_min_intensity

    nan_grid = mapper.map(np.full(9, np.nan, dtype=np.float32), np.zeros(9, dtype=bool))
    print(f"  all-NaN grid → max intensity {nan_grid[1::3].max():.2f}  (broken sensor must not sound safe)")
    assert nan_grid[1::3].max() > 0
    print("CueMapper OK")
