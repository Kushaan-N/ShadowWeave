"""HRTF spatial audio engine — MIT KEMAR dataset + sounddevice playback."""

from __future__ import annotations

import pathlib
import threading
from typing import Optional

import numpy as np
from omegaconf import DictConfig


_KEMAR_URL = "https://sound.media.mit.edu/resources/KEMAR.tar.gz"
_ZONE_NAMES = ["left_far", "left", "left_near", "front_left", "front",
               "front_right", "right_near", "right", "right_far"]


class HRTFEngine:
    """Spatialises mono audio signals using HRTF filters from the MIT KEMAR dataset.

    Primary method: ``play(audio_params) -> None``
    Input:  (27,) float32 — (direction_idx, intensity, pitch_ratio) × 9 zones
            direction_idx: index into _ZONE_NAMES (0–8)
            intensity:     0–1, linear amplitude
            pitch_ratio:   0.5–2.0, multiplied against base frequency
    """

    def __init__(self, cfg: DictConfig, hrtf_dir: Optional[str] = None) -> None:
        self.cfg = cfg
        self.hrtf_dir = pathlib.Path(hrtf_dir) if hrtf_dir else pathlib.Path("data/kemar")
        self._hrtfs: dict[str, np.ndarray] = {}  # zone_name → (2, filter_len) stereo HRTF
        self._stream = None
        self._lock = threading.Lock()
        self._current_buffer: Optional[np.ndarray] = None  # (frames, 2) float32

    def load_hrtfs(self) -> None:
        """Load HRTF filters for each zone from the KEMAR dataset."""
        if not self.hrtf_dir.exists():
            print(f"[HRTFEngine] KEMAR data not found at {self.hrtf_dir}.")
            print(f"  Download from {_KEMAR_URL} and extract to {self.hrtf_dir}")
            print("  Using identity (no spatialisation) until KEMAR data is available.")
            for zone in _ZONE_NAMES:
                self._hrtfs[zone] = np.array([[1.0], [1.0]], dtype=np.float32)
            return

        # load .wav HRTF impulse responses per azimuth angle
        from scipy.io import wavfile
        zone_azimuths = {
            zone: getattr(self.cfg.audio.zones, zone)
            for zone in _ZONE_NAMES
        }
        for zone, az in zone_azimuths.items():
            hrtf_path = self._kemar_path_for_azimuth(az)
            if hrtf_path is not None and hrtf_path.exists():
                sr, data = wavfile.read(hrtf_path)
                self._hrtfs[zone] = data.T.astype(np.float32) / 32768.0
            else:
                self._hrtfs[zone] = np.array([[1.0], [1.0]], dtype=np.float32)

    def _kemar_path_for_azimuth(self, azimuth: int) -> Optional[pathlib.Path]:
        """Map azimuth to nearest KEMAR measurement file."""
        # KEMAR is at elevation 0, azimuth measured in 5° increments
        snapped = round(azimuth / 5) * 5
        fname = f"elev0/H0e{abs(snapped):03d}a.wav"
        return self.hrtf_dir / fname

    def open_stream(self) -> None:
        """Open sounddevice output stream in callback mode."""
        try:
            import sounddevice as sd
            self._stream = sd.OutputStream(
                samplerate=self.cfg.audio.sample_rate,
                blocksize=self.cfg.audio.buffer_size,
                channels=2,
                dtype="float32",
                callback=self._audio_callback,
            )
            self._stream.start()
        except ImportError:
            raise ImportError("sounddevice required — pip install sounddevice")

    def close_stream(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def play(self, audio_params: np.ndarray) -> None:
        """Synthesise and enqueue audio for the next buffer fill.

        audio_params: (27,) = (direction, intensity, pitch) × 9 zones
        """
        params = audio_params.reshape(9, 3)
        buf = self._synthesise(params)
        with self._lock:
            self._current_buffer = buf

    def _synthesise(self, params: np.ndarray) -> np.ndarray:
        """Mix zone tones into a stereo buffer of length buffer_size."""
        n = self.cfg.audio.buffer_size
        sr = self.cfg.audio.sample_rate
        t = np.arange(n, dtype=np.float32) / sr
        out = np.zeros((n, 2), dtype=np.float32)

        for i, zone in enumerate(_ZONE_NAMES):
            _, intensity, pitch_ratio = params[i]
            if intensity < 1e-3:
                continue
            freq = self.cfg.audio.base_frequency * float(pitch_ratio + 0.5)
            tone = intensity * np.sin(2 * np.pi * freq * t).astype(np.float32)
            hrtf = self._hrtfs.get(zone, np.array([[1.0], [1.0]], dtype=np.float32))
            from scipy.signal import fftconvolve
            left  = fftconvolve(tone, hrtf[0])[:n]
            right = fftconvolve(tone, hrtf[1])[:n]
            out[:, 0] += left
            out[:, 1] += right

        # clip to avoid distortion
        return np.clip(out, -1.0, 1.0)

    def _audio_callback(self, outdata: np.ndarray, frames: int, time_info, status) -> None:
        with self._lock:
            buf = self._current_buffer
        if buf is not None and len(buf) >= frames:
            outdata[:] = buf[:frames]
        else:
            outdata[:] = 0.0


if __name__ == "__main__":
    from omegaconf import OmegaConf
    import pathlib

    cfg_path = pathlib.Path(__file__).parents[1] / "configs" / "default.yaml"
    cfg = OmegaConf.load(cfg_path)

    engine = HRTFEngine(cfg)
    engine.load_hrtfs()

    dummy_params = np.zeros(27, dtype=np.float32)
    # zone 4 (front), intensity=0.5, pitch=1.0
    dummy_params[4 * 3 + 1] = 0.5
    dummy_params[4 * 3 + 2] = 0.5

    buf = engine._synthesise(dummy_params.reshape(9, 3))
    print(f"HRTFEngine synthesised buffer: {buf.shape}, range=[{buf.min():.3f}, {buf.max():.3f}]")
    print("HRTFEngine OK (no audio device needed for stub test)")
