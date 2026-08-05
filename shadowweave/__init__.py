"""ShadowWeave: shadow-ray navigation via spatial audio."""

import os

# Must be set before torch is imported. Apple's MPS backend is missing several
# backward kernels (grid_sampler_2d_backward among them), which the shadow-ray and
# BEV modules both need. Training targets CUDA; this only keeps local Mac dev usable.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

__version__ = "0.2.0"
