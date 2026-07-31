"""Real-time shadow lattice renderer — plotly, neon colour scheme.

The 9 cells are azimuth slices, not a 3x3 spatial tile. The old renderer reshaped
them to (3, 3) and labelled the second axis "Elevation", which invents a dimension
the data does not have and scrambles the left→right ordering the audio zones depend
on. It is now drawn as a polar fan over the field of view, which is what the numbers
actually mean, with an optional BEV panel for the occupancy grid.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from omegaconf import DictConfig

ZONE_LABELS = ["L-far", "L", "L-near", "FL", "F", "FR", "R-near", "R", "R-far"]


class ShadowVisualizer:
    """Renders the 9-cell uncertainty grid as a polar fan.

    Primary method: ``update(uncertainty_grid) -> plotly.graph_objects.Figure``
    Input:  (9,) float32 numpy array
    Output: plotly Figure (embeddable in Gradio or shown standalone)
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.n_zones = cfg.shadow.grid_cells
        self.fov = float(cfg.sim.camera_fov)

    def update(self, uncertainty_grid: np.ndarray) -> "object":
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError("plotly required — pip install plotly")

        u = np.asarray(uncertainty_grid, dtype=np.float32).ravel()
        n = self.n_zones
        width = self.fov / n
        # Zone 0 is the leftmost slice of the FOV; plotly polar measures theta
        # counter-clockwise from east, so 90 deg is straight ahead.
        centres = [90.0 + self.fov / 2 - width * (i + 0.5) for i in range(n)]

        fig = go.Figure(
            go.Barpolar(
                r=u,
                theta=centres,
                width=[width] * n,
                marker=dict(
                    color=u,
                    colorscale=[[0, "#0d0221"], [0.5, "#00ff99"], [1, "#ff00ff"]],
                    cmin=0, cmax=1,
                    colorbar=dict(title="Uncertainty"),
                ),
                hovertext=[f"{ZONE_LABELS[i]}: {u[i]:.3f}" for i in range(n)],
                hoverinfo="text",
            )
        )
        fig.update_layout(
            title="Shadow Uncertainty Lattice (agent at centre, front = up)",
            polar=dict(
                bgcolor="#0d0221",
                radialaxis=dict(range=[0, 1], showticklabels=True, tickfont=dict(color="#00ff99")),
                angularaxis=dict(
                    tickmode="array",
                    tickvals=centres,
                    ticktext=ZONE_LABELS[:n],
                    tickfont=dict(color="#00ff99", size=9),
                    direction="counterclockwise",
                ),
            ),
            paper_bgcolor="#0d0221",
            font_color="#00ff99",
            margin=dict(l=0, r=0, t=40, b=0),
        )
        return fig

    def bev_figure(
        self,
        occupancy: np.ndarray,
        visibility: Optional[np.ndarray] = None,
        waypoints: Optional[list[tuple[int, int]]] = None,
    ) -> "object":
        """Egocentric BEV heatmap with shadow shading and the planned path."""
        import plotly.graph_objects as go

        occupancy = np.asarray(occupancy)
        fig = go.Figure()
        fig.add_trace(go.Heatmap(
            z=occupancy, zmin=0, zmax=1,
            colorscale=[[0, "#0d0221"], [0.5, "#00ff99"], [1, "#ff00ff"]],
            name="occupancy", colorbar=dict(title="Occ"),
        ))
        if visibility is not None:
            shadow = 1.0 - np.asarray(visibility)
            fig.add_trace(go.Contour(
                z=shadow, showscale=False, contours=dict(start=0.5, end=1.0, size=0.5),
                line=dict(color="#ff5500", width=1), opacity=0.35, name="shadow",
            ))
        if waypoints:
            ys, xs = zip(*waypoints)
            fig.add_trace(go.Scatter(x=list(xs), y=list(ys), mode="lines+markers",
                                     line=dict(color="#00ffff", width=2),
                                     marker=dict(size=4), name="path"))
        S = occupancy.shape[0]
        fig.add_trace(go.Scatter(x=[S // 2], y=[S - 1], mode="markers",
                                 marker=dict(size=12, color="#ffffff", symbol="triangle-up"),
                                 name="agent"))
        fig.update_layout(
            title="Egocentric BEV (agent at bottom-centre, facing up)",
            paper_bgcolor="#0d0221", plot_bgcolor="#0d0221", font_color="#00ff99",
            margin=dict(l=0, r=0, t=40, b=0),
            yaxis=dict(autorange="reversed", scaleanchor="x"),
        )
        return fig


if __name__ == "__main__":
    from ..utils import load_config

    cfg = load_config()
    viz = ShadowVisualizer(cfg)

    dummy = np.array([0.9, 0.8, 0.3, 0.1, 0.05, 0.1, 0.4, 0.7, 0.85], dtype=np.float32)
    print(f"ShadowVisualizer input: {dummy.round(3)}")
    try:
        fig = viz.update(dummy)
        print(f"  polar fan figure: {type(fig).__name__}, {len(fig.data)} trace(s)")

        S = cfg.bev.size
        occ = np.zeros((S, S), dtype=np.float32)
        occ[20:30, 30:60] = 1.0
        vis = np.ones((S, S), dtype=np.float32)
        vis[:30] = 0.0
        bev = viz.bev_figure(occ, vis, waypoints=[(i, S // 2) for i in range(S - 1, 40, -4)])
        print(f"  BEV figure: {len(bev.data)} traces (occupancy, shadow, path, agent)")
    except ImportError as e:
        print(f"  plotly not installed: {e}")
    print("ShadowVisualizer OK")
