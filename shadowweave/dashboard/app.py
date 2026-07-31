"""Gradio split-screen demo dashboard.

Panels:
  camera feed  |  shadow uncertainty fan
  egocentric BEV (occupancy + shadow + planned path)  |  zone bar chart

Refresh is driven by ``gr.Timer``. The previous version used
``demo.load(..., every=...)``, which was removed in Gradio 5 — with the Gradio 6
installed here the dashboard would not have refreshed at all.

Run a live pipeline behind it with:
    python -m shadowweave.dashboard.app --live
"""

from __future__ import annotations

import argparse
import threading
import time
from typing import Optional

import numpy as np
from omegaconf import DictConfig

from ..shadow.visualizer import ZONE_LABELS, ShadowVisualizer


class DashboardState:
    """Shared mutable state updated by the pipeline, read by Gradio callbacks."""

    def __init__(self, n_zones: int = 9) -> None:
        self.lock = threading.Lock()
        self.rgb: Optional[np.ndarray] = None
        self.uncertainty: np.ndarray = np.zeros(n_zones, dtype=np.float32)
        self.bev_occupancy: Optional[np.ndarray] = None
        self.bev_visibility: Optional[np.ndarray] = None
        self.waypoints: list[tuple[int, int]] = []
        self.is_stop: bool = False
        self.fps: float = 0.0

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "rgb": None if self.rgb is None else self.rgb.copy(),
                "uncertainty": self.uncertainty.copy(),
                "bev_occupancy": None if self.bev_occupancy is None else self.bev_occupancy.copy(),
                "bev_visibility": None if self.bev_visibility is None else self.bev_visibility.copy(),
                "waypoints": list(self.waypoints),
                "is_stop": self.is_stop,
                "fps": self.fps,
            }


def build_dashboard(cfg: DictConfig, state: DashboardState) -> "object":
    """Construct and return the Gradio app (call .launch() on the result)."""
    try:
        import gradio as gr
    except ImportError:
        raise ImportError("gradio required — pip install gradio")

    visualizer = ShadowVisualizer(cfg)
    S = cfg.bev.size

    def get_camera_frame():
        rgb = state.snapshot()["rgb"]
        if rgb is None:
            return np.zeros((cfg.depth.output_h, cfg.depth.output_w, 3), dtype=np.uint8)
        return rgb

    def get_shadow_lattice():
        return visualizer.update(state.snapshot()["uncertainty"])

    def get_bev():
        s = state.snapshot()
        occ = s["bev_occupancy"]
        if occ is None:
            occ = np.zeros((S, S), dtype=np.float32)
        return visualizer.bev_figure(occ, s["bev_visibility"], s["waypoints"])

    def get_zone_bars():
        import plotly.graph_objects as go

        s = state.snapshot()
        u = s["uncertainty"]
        fig = go.Figure(go.Bar(x=ZONE_LABELS[: len(u)], y=u, marker_color="#ff00ff"))
        fig.add_hline(y=cfg.agents.orchestrator.uncertainty_stop_threshold,
                      line=dict(color="#ff3333", dash="dash"))
        fig.update_layout(
            title="Zone Uncertainty", yaxis=dict(range=[0, 1]),
            paper_bgcolor="#0d0221", plot_bgcolor="#0d0221", font_color="#00ff99",
            margin=dict(l=0, r=0, t=40, b=0),
        )
        return fig

    def get_status():
        s = state.snapshot()
        flag = "⛔ STOP — uncertainty above threshold" if s["is_stop"] else "✅ Navigating"
        return f"{flag}   |   pipeline {s['fps']:.1f} Hz   |   max zone {s['uncertainty'].max():.3f}"

    # Gradio 6 moved `theme` from the Blocks constructor to launch().
    with gr.Blocks(title="ShadowWeave") as demo:
        gr.Markdown("## ShadowWeave — Spatial Audio Navigation")
        status = gr.Markdown("…")
        with gr.Row():
            cam_out = gr.Image(label="Camera Feed", height=320)
            shadow_out = gr.Plot(label="Shadow Lattice")
        with gr.Row():
            bev_out = gr.Plot(label="Egocentric BEV")
            bars_out = gr.Plot(label="Zone Uncertainty")

        # gr.Timer replaces demo.load(every=...), removed in Gradio 5.
        fast = gr.Timer(1.0 / cfg.sim.fps)
        slow = gr.Timer(0.2)  # 5Hz, the dashboard render target
        fast.tick(get_camera_frame, outputs=cam_out)
        slow.tick(get_shadow_lattice, outputs=shadow_out)
        slow.tick(get_bev, outputs=bev_out)
        slow.tick(get_zone_bars, outputs=bars_out)
        slow.tick(get_status, outputs=status)

        demo.load(get_camera_frame, outputs=cam_out)
        demo.load(get_shadow_lattice, outputs=shadow_out)
        demo.load(get_bev, outputs=bev_out)
        demo.load(get_zone_bars, outputs=bars_out)

    return demo


def run_pipeline(cfg: DictConfig, state: DashboardState, stop_event: threading.Event) -> None:
    """Drive the real MuJoCo → shadow → orchestrator loop behind the dashboard."""
    import torch

    from ..agents.global_agent import GlobalAgent
    from ..agents.local_agent import LocalAgent
    from ..agents.orchestrator import Orchestrator
    from ..shadow.bev import BEVProjector, bev_flow
    from ..shadow.raycast import ShadowRaycaster
    from ..sim.mujoco_env import ShadowWeaveEnv
    from ..utils import get_device

    device = get_device()
    env = ShadowWeaveEnv(cfg)
    projector = BEVProjector(cfg).to(device).eval()
    raycaster = ShadowRaycaster(cfg).to(device).eval()
    orch = Orchestrator(cfg, LocalAgent(cfg).to(device).eval(), GlobalAgent(cfg))

    obs = env.reset(seed=0)
    prev_bev = None
    S = cfg.bev.size
    T = len(cfg.world_model.prediction_horizons)

    while not stop_event.is_set():
        t0 = time.perf_counter()
        d = torch.from_numpy(obs["depth"]).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            occ, vis = projector(d)
            unc = raycaster.forward_from_depth(d)[0].cpu().numpy()
            _ = bev_flow(prev_bev, occ) if prev_bev is not None else None
            prev_bev = occ

        out = orch.step(unc, np.zeros((T, S, S), dtype=np.float32))

        occ_np = occ[0, 0].cpu().numpy()
        vis_np = vis[0, 0].cpu().numpy()
        wps = orch.update_waypoints(occ_np, (S - 1, S // 2), (2, S // 2), shadow=1.0 - vis_np)

        with state.lock:
            state.rgb = obs["rgb"]
            state.uncertainty = unc
            state.bev_occupancy = occ_np
            state.bev_visibility = vis_np
            state.waypoints = wps
            state.is_stop = out.is_stop
            state.fps = 1.0 / max(time.perf_counter() - t0, 1e-6)

        obs = env.step(action=out.nav_action)
    env.close()


def main() -> None:
    from ..utils import load_config

    ap = argparse.ArgumentParser(description="ShadowWeave dashboard")
    ap.add_argument("--live", action="store_true", help="drive a real MuJoCo pipeline behind it")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--overrides", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(overrides=args.overrides)
    state = DashboardState(cfg.shadow.grid_cells)
    stop_event = threading.Event()

    if args.live:
        threading.Thread(target=run_pipeline, args=(cfg, state, stop_event), daemon=True).start()
        print("Live pipeline thread started")
    else:
        S = cfg.bev.size
        state.rgb = np.random.randint(0, 255, (cfg.depth.output_h, cfg.depth.output_w, 3), dtype=np.uint8)
        state.uncertainty = np.array([0.9, 0.8, 0.3, 0.1, 0.05, 0.1, 0.4, 0.7, 0.85], dtype=np.float32)
        occ = np.zeros((S, S), dtype=np.float32)
        occ[20:30, 30:60] = 1.0
        state.bev_occupancy = occ
        vis = np.ones((S, S), dtype=np.float32)
        vis[:30] = 0.0
        state.bev_visibility = vis
        state.waypoints = [(i, S // 2) for i in range(S - 1, 40, -4)]

    demo = build_dashboard(cfg, state)
    print(f"Launching dashboard on http://localhost:{args.port}")
    try:
        import gradio as gr
        demo.launch(server_port=args.port, share=args.share, theme=gr.themes.Base())
    finally:
        stop_event.set()


if __name__ == "__main__":
    main()
