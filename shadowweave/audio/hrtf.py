"""HRTF spatial audio engine — MIT KEMAR dataset + sounddevice playback.

Two audible bugs are fixed here:

  * KEMAR measurements only exist for one side of the head; the left side is obtained
    by SWAPPING the two ear channels. The old code took ``abs(azimuth)`` and used the
    file as-is, so a cue at −90° (hard left) was bit-identical to one at +90° (hard
    right). Left/right was completely unresolvable — fatal for the one thing this
    system exists to do.

  * The oscillator restarted at ``t = 0`` every buffer, so each 1024-sample block
    began with a phase discontinuity. At 200Hz that is an audible click 43 times a
    second. Phase is now carried across buffers per zone.
"""

from __future__ import annotations

import pathlib
import threading
from typing import Optional

import numpy as np
from omegaconf import DictConfig

from .cues import AUDIO_PARAM_DIM, N_ZONES

_KEMAR_URL = "https://sound.media.mit.edu/resources/KEMAR/compact.zip"
_ZONE_NAMES = ["left_far", "left", "left_near", "front_left", "front",
               "front_right", "right_near", "right", "right_far"]
_IDENTITY_HRTF = np.array([[1.0], [1.0]], dtype=np.float32)


class HRTFEngine:
    """Spatialises mono audio signals using HRTF filters from the MIT KEMAR dataset.

    Primary method: ``play(audio_params) -> None``
    Input:  (27,) float32 — (direction_idx, intensity, pitch_ratio) × 9 zones
    """

    def __init__(self, cfg: DictConfig, hrtf_dir: Optional[str] = None) -> None:
        self.cfg = cfg
        self.hrtf_dir = pathlib.Path(hrtf_dir or cfg.audio.hrtf_dir)
        if not self.hrtf_dir.is_absolute():
            # Resolve against the repo root, not the CWD — the dashboard and tests
            # are not guaranteed to run from the repo root.
            self.hrtf_dir = pathlib.Path(__file__).resolve().parents[2] / self.hrtf_dir
        self._hrtfs: dict[str, np.ndarray] = {}
        self._stream = None
        self._lock = threading.Lock()
        self._params: Optional[np.ndarray] = None
        self._phase = np.zeros(N_ZONES, dtype=np.float64)
        self._spatialised = False
        # Convolution tail from the previous buffer, so filters do not restart cold.
        self._tail: Optional[np.ndarray] = None

    @property
    def is_spatialised(self) -> bool:
        """False when running on panning filters (no KEMAR data present)."""
        return self._spatialised

    def load_hrtfs(self) -> None:
        """Load HRTF filters for each zone from the KEMAR dataset."""
        if not self.hrtf_dir.exists():
            print(f"[HRTFEngine] KEMAR data not found at {self.hrtf_dir}.")
            print(f"  Download from {_KEMAR_URL} and extract to {self.hrtf_dir}")
            print("  Falling back to amplitude panning until KEMAR data is available.")
            for zone in _ZONE_NAMES:
                self._hrtfs[zone] = self._pan_filter(float(self.cfg.audio.zones[zone]))
            self._spatialised = False
            return

        from scipy.io import wavfile

        loaded = 0
        for zone in _ZONE_NAMES:
            az = float(self.cfg.audio.zones[zone])
            path = self._kemar_path_for_azimuth(az)
            if path is not None and path.exists():
                try:
                    sr_file, data = wavfile.read(path)
                except Exception as exc:
                    # One corrupt file must not abort the load and leave the
                    # remaining zones without lateralisation.
                    print(f"[HRTFEngine] unreadable HRTF {path}: {exc} — pan fallback for '{zone}'")
                    self._hrtfs[zone] = self._pan_filter(az)
                    continue
                if int(sr_file) != int(self.cfg.audio.sample_rate):
                    print(f"[HRTFEngine] {path.name} is {sr_file}Hz but engine runs at "
                          f"{self.cfg.audio.sample_rate}Hz — spatial cues will be shifted")
                hrtf = np.asarray(data, dtype=np.float32).T / 32768.0
                if hrtf.ndim == 1:
                    print(f"[HRTFEngine] {path.name} is mono — zone '{zone}' has no interaural cue")
                    hrtf = np.stack([hrtf, hrtf])
                # KEMAR files are measured on the RIGHT side. For a left-side azimuth
                # the ipsilateral/contralateral ears swap.
                if az < 0:
                    hrtf = hrtf[::-1].copy()
                self._hrtfs[zone] = hrtf
                loaded += 1
            else:
                self._hrtfs[zone] = self._pan_filter(az)
        self._spatialised = loaded > 0
        print(f"[HRTFEngine] loaded {loaded}/{len(_ZONE_NAMES)} KEMAR filters from {self.hrtf_dir}")

    def _pan_filter(self, azimuth_deg: float) -> np.ndarray:
        """Constant-power stereo pan — a usable stand-in when KEMAR is absent.

        Still gives real left/right separation, unlike the previous identity filter
        which made all nine directions sound the same.
        """
        # Normalise by the widest configured zone, not ±90° — clipping at 90 made
        # left_far (−135°) bit-identical to left (−90°).
        span = max(abs(float(a)) for a in self.cfg.audio.zones.values()) or 90.0
        x = np.clip(azimuth_deg / span, -1.0, 1.0)
        angle = (x + 1.0) * (np.pi / 4.0)  # 0 → hard left, pi/2 → hard right
        return np.array([[float(np.cos(angle))], [float(np.sin(angle))]], dtype=np.float32)

    def _kemar_path_for_azimuth(self, azimuth: float) -> Optional[pathlib.Path]:
        """Map azimuth to the nearest KEMAR measurement file (elevation 0, 5° steps)."""
        # Wrap into [-180, 180] BEFORE abs() — the compact set only has 0–180°, so
        # e.g. −190° must resolve to 170°, not a nonexistent 190° file.
        wrapped = ((azimuth + 180.0) % 360.0) - 180.0
        snapped = int(round(abs(wrapped) / 5.0) * 5)
        return self.hrtf_dir / f"elev0/H0e{snapped:03d}a.wav"

    def open_stream(self) -> None:
        """Open sounddevice output stream in callback mode."""
        try:
            import sounddevice as sd
        except ImportError:
            raise ImportError("sounddevice required — pip install sounddevice")
        self._stream = sd.OutputStream(
            samplerate=self.cfg.audio.sample_rate,
            blocksize=self.cfg.audio.buffer_size,
            channels=2,
            dtype="float32",
            callback=self._audio_callback,
        )
        self._stream.start()

    def close_stream(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def play(self, audio_params: np.ndarray) -> None:
        """Publish the latest cue parameters; the stream callback synthesises from them.

        Synthesis happens per callback rather than here: the cue update rate (~20Hz)
        and the device callback rate (~43Hz at 1024/44100) are not locked, so a
        buffer queued from this thread would be replayed with a phase seam — an
        audible click — on every callback that lands between updates.
        """
        params = np.asarray(audio_params, dtype=np.float32).reshape(N_ZONES, 3)
        with self._lock:
            self._params = params.copy()

    def _synthesise(self, params: np.ndarray, n: Optional[int] = None) -> np.ndarray:
        """Mix zone tones into a stereo buffer of n samples (default buffer_size)."""
        from scipy.signal import fftconvolve

        if n is None:
            n = int(self.cfg.audio.buffer_size)
        if n <= 0:
            raise ValueError(f"buffer length must be positive, got {n}")
        sr = float(self.cfg.audio.sample_rate)
        if not self._hrtfs:
            self.load_hrtfs()

        out = np.zeros((n, 2), dtype=np.float32)
        max_tail = 0
        tails: list[tuple[np.ndarray, np.ndarray]] = []

        for i, zone in enumerate(_ZONE_NAMES):
            _, intensity, pitch_ratio = params[i]
            # NaN passes every < comparison, flows into the buffer, and — worse —
            # poisons the stored phase for all future buffers. Treat as silent.
            if not (np.isfinite(intensity) and np.isfinite(pitch_ratio)) or intensity < 1e-3:
                self._phase[i] = 0.0
                continue

            freq = float(self.cfg.audio.base_frequency) * float(pitch_ratio + 0.5)
            # Continue from the stored phase instead of restarting at t=0, which was
            # producing a discontinuity — an audible click — at every buffer boundary.
            omega = 2.0 * np.pi * freq / sr
            phases = self._phase[i] + omega * np.arange(n, dtype=np.float64)
            self._phase[i] = float((phases[-1] + omega) % (2.0 * np.pi))
            tone = (float(intensity) * np.sin(phases)).astype(np.float32)

            hrtf = self._hrtfs.get(zone, _IDENTITY_HRTF)
            left = fftconvolve(tone, hrtf[0])
            right = fftconvolve(tone, hrtf[1])
            out[:, 0] += left[:n]
            out[:, 1] += right[:n]
            if left.size > n:
                tails.append((left[n:], right[n:]))
                max_tail = max(max_tail, left.size - n)

        # Overlap-add the previous buffer's filter tail.
        if self._tail is not None:
            k = min(len(self._tail), n)
            out[:k] += self._tail[:k]
        if tails:
            merged = np.zeros((max_tail, 2), dtype=np.float32)
            for l, r in tails:
                merged[: l.size, 0] += l
                merged[: r.size, 1] += r
            self._tail = merged
        else:
            self._tail = None

        return np.clip(out * float(self.cfg.audio.master_gain), -1.0, 1.0)

    def _audio_callback(self, outdata: np.ndarray, frames: int, time_info, status) -> None:
        with self._lock:
            params = self._params
        if params is None:
            outdata[:] = 0.0
            return
        # Synthesising exactly `frames` samples here keeps oscillator phase
        # continuous at every callback regardless of the cue update rate, and
        # handles devices that request a block size other than buffer_size.
        outdata[:] = self._synthesise(params, frames)


if __name__ == "__main__":
    from ..utils import load_config

    cfg = load_config()
    engine = HRTFEngine(cfg)
    engine.load_hrtfs()
    print(f"  spatialised via KEMAR: {engine.is_spatialised}")

    def one_zone(idx: int, intensity: float = 0.6) -> np.ndarray:
        p = np.zeros(AUDIO_PARAM_DIM, dtype=np.float32)
        p[idx * 3] = idx
        p[idx * 3 + 1] = intensity
        p[idx * 3 + 2] = 0.5
        return p.reshape(N_ZONES, 3)

    buf = engine._synthesise(one_zone(4))
    print(f"HRTFEngine buffer: {buf.shape}, range=[{buf.min():.3f}, {buf.max():.3f}]")

    # Hard left vs hard right must not be identical.
    engine._phase[:] = 0.0
    engine._tail = None
    left_buf = engine._synthesise(one_zone(1))    # 'left', azimuth -90
    engine._phase[:] = 0.0
    engine._tail = None
    right_buf = engine._synthesise(one_zone(7))   # 'right', azimuth +90
    l_bias = float(np.abs(left_buf[:, 0]).mean() - np.abs(left_buf[:, 1]).mean())
    r_bias = float(np.abs(right_buf[:, 0]).mean() - np.abs(right_buf[:, 1]).mean())
    print(f"  L-zone channel bias (L-R): {l_bias:+.4f}   R-zone: {r_bias:+.4f}")
    assert l_bias > 0 > r_bias, "left and right zones must differ in channel balance"
    assert not np.allclose(left_buf, right_buf), "left/right collapsed to the same signal"

    # Phase continuity across buffers: no discontinuity at the seam.
    engine._phase[:] = 0.0
    engine._tail = None
    b1 = engine._synthesise(one_zone(4))
    b2 = engine._synthesise(one_zone(4))
    seam = abs(float(b2[0, 0] - b1[-1, 0]))
    within = float(np.abs(np.diff(b1[:, 0])).max())
    print(f"  seam step={seam:.5f} vs max within-buffer step={within:.5f}  (seam must not spike)")
    assert seam <= within * 3, "phase discontinuity at buffer boundary"
    print("HRTFEngine OK (no audio device needed for this test)")
