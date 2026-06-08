"""Gradio split-screen demo dashboard.

Three panels:
  Left:   raw camera feed from MuJoCo (or webcam)
  Right:  real-time 3D shadow lattice (plotly, updates at 5Hz)
  Bottom: agent trajectory overlay on 2D occupancy map + uncertainty heatmap
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np
from omegaconf import DictConfig, OmegaConf

from ..shadow.visualizer import ShadowVisualizer


class DashboardState:
    """Shared mutable state updated by the pipeline, read by Gradio callbacks."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.rgb:        Optional[np.ndarray] = None    # (H, W, 3) uint8
        self.uncertainty: np.ndarray = np.zeros(9, dtype=np.float32)
        self.occupancy:   Optional[np.ndarray] = None  # (H, W) float32
        self.waypoints:   list[tuple[int, int]] = []


def build_dashboard(cfg: DictConfig, state: DashboardState) -> "gradio.Blocks":
    """Construct and return the Gradio app (call .launch() on the result)."""
    try:
        import gradio as gr
    except ImportError:
        raise ImportError("gradio required — pip install gradio")

    visualizer = ShadowVisualizer(cfg)

    def get_camera_frame():
        with state.lock:
            rgb = state.rgb
        if rgb is None:
            return np.zeros((cfg.depth.output_h, cfg.depth.output_w, 3), dtype=np.uint8)
        return rgb

    def get_shadow_lattice():
        with state.lock:
            u = state.uncertainty.copy()
        return visualizer.update(u)

    def get_occupancy_map():
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        with state.lock:
            occ = state.occupancy
            wps = list(state.waypoints)
            u   = state.uncertainty.copy()

        fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor="#0d0221")
        fig.patch.set_facecolor("#0d0221")

        ax1 = axes[0]
        ax1.set_facecolor("#0d0221")
        if occ is not None:
            ax1.imshow(occ, cmap="hot", vmin=0, vmax=1)
        if wps:
            ys, xs = zip(*wps)
            ax1.plot(xs, ys, "c-o", markersize=3, linewidth=1.5)
        ax1.set_title("Occupancy + Trajectory", color="#00ff99")
        ax1.tick_params(colors="#00ff99")

        ax2 = axes[1]
        ax2.set_facecolor("#0d0221")
        ax2.bar(range(9), u, color="#ff00ff")
        ax2.set_ylim(0, 1)
        ax2.set_xticks(range(9))
        ax2.set_xticklabels(["L-far","L","L-near","FL","F","FR","R-near","R","R-far"],
                             rotation=45, color="#00ff99", fontsize=7)
        ax2.set_title("Zone Uncertainty", color="#00ff99")
        ax2.tick_params(colors="#00ff99")

        plt.tight_layout()
        return fig

    with gr.Blocks(title="ShadowWeave", theme=gr.themes.Base()) as demo:
        gr.Markdown("## ShadowWeave — Spatial Audio Navigation")
        with gr.Row():
            cam_out    = gr.Image(label="Camera Feed", height=320)
            shadow_out = gr.Plot(label="Shadow Lattice")
        occ_out = gr.Plot(label="Occupancy Map + Agent Trajectory")

        demo.load(get_camera_frame, outputs=cam_out,    every=1.0 / cfg.sim.fps)
        demo.load(get_shadow_lattice, outputs=shadow_out, every=0.2)  # 5Hz
        demo.load(get_occupancy_map,  outputs=occ_out,   every=0.2)

    return demo


if __name__ == "__main__":
    import pathlib

    cfg_path = pathlib.Path(__file__).parents[1] / "configs" / "default.yaml"
    cfg = OmegaConf.load(cfg_path)

    state = DashboardState()
    # inject dummy data so the dashboard renders without a live pipeline
    state.rgb = np.random.randint(0, 255, (cfg.depth.output_h, cfg.depth.output_w, 3), dtype=np.uint8)
    state.uncertainty = np.random.rand(9).astype(np.float32)
    state.occupancy   = (np.random.rand(cfg.depth.output_h, cfg.depth.output_w) > 0.85).astype(np.float32)
    state.waypoints   = [(i, i) for i in range(0, cfg.depth.output_h, 8)]

    demo = build_dashboard(cfg, state)
    print("Launching dashboard on http://localhost:7860")
    demo.launch(server_port=7860)
