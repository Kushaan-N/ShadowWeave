"""Webcam / sim camera abstraction running at target FPS."""

from __future__ import annotations

import time
from typing import Generator

import numpy as np
from omegaconf import DictConfig


class Camera:
    """Wraps either a live webcam or a MuJoCo sim camera into a unified frame iterator.

    Primary method: ``stream()`` — yields (rgb, depth) pairs at the configured FPS.
    When ``source="webcam"``, depth is None (filled downstream by DepthEstimator).
    When ``source="sim"``, depth is the ground-truth array from the sim renderer.
    """

    def __init__(self, cfg: DictConfig, source: str = "webcam") -> None:
        self.cfg = cfg
        self.source = source
        self._cap = None  # cv2.VideoCapture, lazily opened

    def open(self) -> None:
        if self.source == "webcam":
            try:
                import cv2
                self._cap = cv2.VideoCapture(0)
                self._cap.set(cv2.CAP_PROP_FPS, self.cfg.sim.fps)
            except ImportError:
                raise ImportError("cv2 required for webcam source — pip install opencv-python")
        # sim source is driven externally; no handle needed

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def stream(self) -> Generator[tuple[np.ndarray, np.ndarray | None], None, None]:
        """Yield (rgb H×W×3 uint8, depth H×W float32 | None) at cfg.sim.fps."""
        period = 1.0 / self.cfg.sim.fps
        self.open()
        try:
            while True:
                t0 = time.monotonic()
                frame, depth = self._grab_frame()
                yield frame, depth
                elapsed = time.monotonic() - t0
                remaining = period - elapsed
                if remaining > 0:
                    time.sleep(remaining)
        finally:
            self.close()

    def _grab_frame(self) -> tuple[np.ndarray, np.ndarray | None]:
        if self.source == "webcam":
            if self._cap is None or not self._cap.isOpened():
                raise RuntimeError("Camera not opened — call open() first")
            import cv2
            ret, frame = self._cap.read()
            if not ret:
                raise RuntimeError("Webcam read failed")
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), None
        raise ValueError(f"Unknown source '{self.source}'; use webcam or inject frames via sim")


if __name__ == "__main__":
    from omegaconf import OmegaConf
    import pathlib

    cfg_path = pathlib.Path(__file__).parents[1] / "configs" / "default.yaml"
    cfg = OmegaConf.load(cfg_path)

    print("Camera stub demo — generating 5 dummy frames")
    h, w = cfg.depth.output_h, cfg.depth.output_w
    for i in range(5):
        dummy_rgb = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        dummy_depth = np.random.rand(h, w).astype(np.float32)
        print(f"  frame {i}: rgb={dummy_rgb.shape}, depth={dummy_depth.shape}")
    print("Camera stub OK")
