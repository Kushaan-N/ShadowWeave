"""Depth Anything V2 wrapper — monocular RGB → 128×128 float32 depth map."""

from __future__ import annotations

import numpy as np
import torch
from omegaconf import DictConfig


class DepthEstimator:
    """Wraps Depth Anything V2 from HuggingFace into a simple forward call.

    Primary method: ``forward(rgb) -> depth_map``
    Input:  H×W×3 uint8 numpy array
    Output: 128×128 float32 numpy array, values in [0, 1] (normalised inverse depth)
    """

    def __init__(self, cfg: DictConfig, device: str = "cuda") -> None:
        self.cfg = cfg
        self.device = device
        self._model = None
        self._transform = None

    def load(self) -> None:
        """Lazy-load the model so import-time is fast."""
        try:
            from transformers import pipeline
            self._pipe = pipeline(
                "depth-estimation",
                model=f"depth-anything/{self.cfg.depth.model}",
                device=0 if self.device == "cuda" and torch.cuda.is_available() else -1,
            )
        except ImportError:
            raise ImportError("transformers required — pip install transformers")

    def forward(self, rgb: np.ndarray) -> np.ndarray:
        """Return 128×128 float32 depth map from an RGB frame."""
        if self._pipe is None:
            self.load()
        from PIL import Image
        pil = Image.fromarray(rgb)
        result = self._pipe(pil)
        depth = np.array(result["depth"], dtype=np.float32)
        depth = self._resize(depth)
        # normalise to [0, 1]
        dmin, dmax = depth.min(), depth.max()
        if dmax > dmin:
            depth = (depth - dmin) / (dmax - dmin)
        return depth

    def _resize(self, depth: np.ndarray) -> np.ndarray:
        from PIL import Image
        h, w = self.cfg.depth.output_h, self.cfg.depth.output_w
        pil = Image.fromarray(depth).resize((w, h), Image.BILINEAR)
        return np.array(pil, dtype=np.float32)


if __name__ == "__main__":
    from omegaconf import OmegaConf
    import pathlib

    cfg_path = pathlib.Path(__file__).parents[1] / "configs" / "default.yaml"
    cfg = OmegaConf.load(cfg_path)

    print("DepthEstimator stub demo — random depth map (no model loaded)")
    h, w = cfg.depth.output_h, cfg.depth.output_w
    dummy_rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    dummy_depth = np.random.rand(h, w).astype(np.float32)
    print(f"  input rgb: {dummy_rgb.shape}")
    print(f"  output depth: {dummy_depth.shape}, range [{dummy_depth.min():.3f}, {dummy_depth.max():.3f}]")
    print("DepthEstimator stub OK")
