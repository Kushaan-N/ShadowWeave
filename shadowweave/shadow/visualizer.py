"""Real-time shadow lattice renderer — plotly 3D surface, neon colour scheme."""

from __future__ import annotations

import numpy as np
from omegaconf import DictConfig


class ShadowVisualizer:
    """Renders the 9-cell uncertainty grid as a 3D plotly surface.

    Primary method: ``update(uncertainty_grid) -> plotly.graph_objects.Figure``
    Input:  (9,) float32 numpy array
    Output: plotly Figure (can be embedded in Gradio or shown standalone)
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self._fig = None

    def update(self, uncertainty_grid: np.ndarray) -> "plotly.graph_objects.Figure":
        """Render a 3×3 heatmap surface from the 9-cell uncertainty grid."""
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError("plotly required — pip install plotly")

        grid = uncertainty_grid.reshape(3, 3)
        # x/y positions for zone labels
        zone_labels = [
            ["L-far", "L", "L-near"],
            ["FL",    "F", "FR"],
            ["R-near","R", "R-far"],
        ]

        fig = go.Figure(data=[
            go.Surface(
                z=grid,
                colorscale=[[0, "#0d0221"], [0.5, "#00ff99"], [1, "#ff00ff"]],
                showscale=True,
                cmin=0, cmax=1,
            )
        ])
        fig.update_layout(
            title="Shadow Uncertainty Lattice",
            scene=dict(
                xaxis_title="Azimuth",
                yaxis_title="Elevation",
                zaxis_title="Uncertainty",
                bgcolor="#0d0221",
            ),
            paper_bgcolor="#0d0221",
            font_color="#00ff99",
            margin=dict(l=0, r=0, t=30, b=0),
        )
        return fig


if __name__ == "__main__":
    from omegaconf import OmegaConf
    import pathlib

    cfg_path = pathlib.Path(__file__).parents[1] / "configs" / "default.yaml"
    cfg = OmegaConf.load(cfg_path)

    viz = ShadowVisualizer(cfg)
    dummy = np.random.rand(9).astype(np.float32)
    print(f"ShadowVisualizer input: {dummy.round(3)}")
    try:
        fig = viz.update(dummy)
        print(f"  Figure type: {type(fig).__name__}")
    except ImportError as e:
        print(f"  plotly not installed: {e}")
    print("ShadowVisualizer stub OK")
