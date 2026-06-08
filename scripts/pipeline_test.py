"""End-to-end pipeline smoke test.

Runs 30 frames through: MuJoCo → GeometricOccupancyField → ShadowRaycaster → CueMapper
Prints per-zone uncertainty as an ASCII bar chart each frame.
No trained models required.

Usage:
    /opt/homebrew/anaconda3/envs/shadowweave/bin/python scripts/pipeline_test.py
"""

from __future__ import annotations

import pathlib
import sys
import time

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

from shadowweave.sim.mujoco_env import ShadowWeaveEnv
from shadowweave.shadow.raycast import ShadowRaycaster
from shadowweave.audio.cues import CueMapper

ZONE_LABELS = ["L-far", "L    ", "L-near", "FL   ", "F    ", "FR   ", "R-near", "R    ", "R-far"]
BAR_WIDTH = 30


def render_frame(frame_idx: int, uncertainty: np.ndarray, is_stop: bool) -> None:
    print(f"\n─── Frame {frame_idx:03d} {'[STOP]' if is_stop else '      '} ──────────────────────────")
    for label, u in zip(ZONE_LABELS, uncertainty):
        filled = int(u * BAR_WIDTH)
        colour = "\033[91m" if u > 0.7 else ("\033[93m" if u > 0.4 else "\033[92m")
        reset  = "\033[0m"
        bar = colour + "█" * filled + reset + "░" * (BAR_WIDTH - filled)
        print(f"  {label}: {bar} {u:.3f}")


def occupancy_to_depth(occupancy: np.ndarray) -> np.ndarray:
    """Derive a synthetic depth map from the GT occupancy grid.

    For each pixel column, the depth = (row of nearest obstacle from top) / H.
    Free columns get depth = 1.0 (open space, far away).
    This gives raycaster.forward_from_depth() a meaningful signal without MuJoCo depth rendering.
    """
    H, W = occupancy.shape
    depth = np.ones((H, W), dtype=np.float32)
    for c in range(W):
        col = occupancy[:, c]
        hits = np.where(col > 0.5)[0]
        if hits.size > 0:
            nearest_row = hits[0]
            # assign depth proportional to row index: top row → near, bottom → far
            frac = nearest_row / H
            depth[:, c] = frac
    return depth


def main() -> None:
    cfg_path = pathlib.Path(__file__).parents[1] / "shadowweave" / "configs" / "default.yaml"
    cfg = OmegaConf.load(cfg_path)

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Pipeline test — device: {device}")
    print("Note: MuJoCo depth rendering is broken on macOS (ARB_clip_control missing).")
    print("      Using GT-occupancy→synthetic-depth as the depth source.\n")

    env       = ShadowWeaveEnv(cfg)
    raycaster = ShadowRaycaster(cfg).to(device)
    cue_mapper = CueMapper(cfg)

    # Run all three difficulty tiers for 10 frames each
    latencies: list[float] = []

    for difficulty in ["static", "moving", "debris"]:
        cfg.sim.difficulty = difficulty
        print(f"\n{'═'*50}")
        print(f"  Difficulty: {difficulty.upper()}")
        print(f"{'═'*50}")
        env.reset(seed=42)

        for frame_idx in range(10):
            t0 = time.perf_counter()

            obs = env.step()
            # Synthesise depth from GT occupancy (stand-in for Depth Anything V2)
            depth_np = occupancy_to_depth(obs["occupancy"])

            # depth → shadow → uncertainty
            depth_t = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                grid = raycaster.forward_from_depth(depth_t)
            uncertainty = grid[0].cpu().numpy()

            # uncertainty → audio cues
            falling_mask = np.zeros(9, dtype=bool)
            # simple heuristic: zone is "falling" if vertical velocity is large
            vel = obs["velocity"]
            if vel.shape[0] > 0:
                max_downward = vel[:, 2].min()   # most negative z = fastest falling
                if max_downward < -0.5:
                    # flag the zone with highest uncertainty as the falling zone
                    falling_mask[int(uncertainty.argmax())] = True

            is_stop  = bool(uncertainty.max() > cfg.agents.orchestrator.uncertainty_stop_threshold)
            audio_params = cue_mapper.map(uncertainty, falling_mask, is_stop)

            latency_ms = (time.perf_counter() - t0) * 1000
            latencies.append(latency_ms)

            render_frame(frame_idx, uncertainty, is_stop)
            print(f"  latency: {latency_ms:.1f} ms  |  depth range: [{depth_np.min():.2f}, {depth_np.max():.2f}]")
            print(f"  active cues: {int((uncertainty > 0.1).sum())}/9  |  audio intensity: {audio_params[1::3].max():.3f}")

    env.close()

    p50  = np.percentile(latencies, 50)
    p95  = np.percentile(latencies, 95)
    p_max = max(latencies)
    print(f"\n{'═'*50}")
    print(f"  Latency summary (camera→shadow→cues, excl. audio output):")
    print(f"    p50={p50:.1f}ms  p95={p95:.1f}ms  max={p_max:.1f}ms  (target <100ms)")
    status = "✓ PASS" if p95 < 100 else "✗ FAIL"
    print(f"  {status}")
    print(f"{'═'*50}")


if __name__ == "__main__":
    main()
