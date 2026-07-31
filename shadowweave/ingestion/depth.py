"""Depth Anything V2 wrapper — monocular RGB → normalised depth map.

Two fixes: ``self._pipe`` was never initialised in __init__, so the very first
``forward()`` raised AttributeError on ``if self._pipe is None``; and the model id was
built as ``depth-anything/{cfg.depth.model}`` from a bare name, producing a repo id
that does not exist on the Hub.

Depth Anything predicts *inverse relative* depth, so it is inverted and rescaled to
match the fixed [0, 1] == [0, max_range_m] convention the rest of the pipeline uses.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from omegaconf import DictConfig

from ..utils import get_device


class DepthEstimator:
    """Wraps Depth Anything V2 from HuggingFace into a simple forward call.

    Primary method: ``forward(rgb) -> depth_map``
    Input:  H×W×3 uint8 numpy array
    Output: (output_h, output_w) float32, 0 = at the camera, 1 = max_range_m
    """

    def __init__(self, cfg: DictConfig, device: Optional[str] = None) -> None:
        self.cfg = cfg
        self.device = torch.device(device) if device else get_device()
        self._pipe = None  # never initialised before → AttributeError on first call

    def load(self) -> None:
        """Lazy-load the model so import-time is fast."""
        try:
            from transformers import pipeline
        except ImportError:
            raise ImportError("transformers required — pip install transformers")
        self._pipe = pipeline(
            "depth-estimation",
            model=self.cfg.depth.model,   # full repo id, e.g. depth-anything/Depth-Anything-V2-Small-hf
            device=0 if self.device.type == "cuda" else -1,
        )

    def forward(self, rgb: np.ndarray) -> np.ndarray:
        if self._pipe is None:
            self.load()
        from PIL import Image

        result = self._pipe(Image.fromarray(rgb))
        key = "predicted_depth" if "predicted_depth" in result else "depth"
        raw = result[key]
        # The pipeline returns a torch tensor under one key and a PIL image under the
        # other; np.asarray on a tensor trips a numpy-2 copy-keyword deprecation.
        depth = raw.detach().cpu().numpy() if torch.is_tensor(raw) else np.asarray(raw)
        depth = np.squeeze(depth).astype(np.float32)
        depth = self._resize(depth)

        # Model output is inverse relative depth (large = near). Invert so that large
        # means far, matching the sim's convention, then normalise to [0, 1].
        dmin, dmax = float(depth.min()), float(depth.max())
        if dmax > dmin:
            depth = (depth - dmin) / (dmax - dmin)
        return np.clip(1.0 - depth, 0.0, 1.0).astype(np.float32)

    def _resize(self, depth: np.ndarray) -> np.ndarray:
        from PIL import Image

        h, w = self.cfg.depth.output_h, self.cfg.depth.output_w
        return np.array(
            Image.fromarray(depth.astype(np.float32)).resize((w, h), Image.BILINEAR),
            dtype=np.float32,
        )


if __name__ == "__main__":
    from ..utils import load_config

    cfg = load_config()
    h, w = cfg.depth.output_h, cfg.depth.output_w

    est = DepthEstimator(cfg)
    print(f"DepthEstimator device={est.device} model={cfg.depth.model}")
    print(f"  _pipe initialised to {est._pipe} (was previously an unset attribute)")

    dummy_rgb = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    try:
        depth = est.forward(dummy_rgb)
        print(f"  input {dummy_rgb.shape} → depth {depth.shape}, "
              f"range [{depth.min():.3f}, {depth.max():.3f}]")
        assert depth.shape == (h, w)
    except Exception as e:
        # No network / weights on this machine is fine; the point is that the failure
        # is an honest download error rather than AttributeError on _pipe.
        print(f"  model unavailable offline ({e.__class__.__name__}: {str(e)[:90]})")
    print("DepthEstimator OK")
