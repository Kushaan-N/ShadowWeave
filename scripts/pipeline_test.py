"""End-to-end pipeline smoke test.

Runs MuJoCo → BEVProjector → ShadowRaycaster → Orchestrator → CueMapper → HRTFEngine
and prints per-zone uncertainty as an ASCII bar chart, plus a latency budget.
No trained models required.

The previous version synthesised depth from the ground-truth occupancy grid because
MuJoCo depth rendering was assumed broken; the env now returns real metric depth (and
falls back to mj_ray rather than to a fabricated map), so this measures the real path.

Usage:
    python scripts/pipeline_test.py
    python scripts/pipeline_test.py --frames 30 --difficulty debris
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

# Before torch: shadowweave.__init__ sets PYTORCH_ENABLE_MPS_FALLBACK, which torch
# reads once at import — set afterwards it does nothing.
import shadowweave  # noqa: F401,E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from shadowweave.agents.global_agent import GlobalAgent  # noqa: E402
from shadowweave.agents.local_agent import LocalAgent  # noqa: E402
from shadowweave.agents.orchestrator import Orchestrator  # noqa: E402
from shadowweave.audio.hrtf import HRTFEngine  # noqa: E402
from shadowweave.shadow.bev import BEVProjector, bev_flow  # noqa: E402
from shadowweave.shadow.raycast import ShadowRaycaster  # noqa: E402
from shadowweave.sim.mujoco_env import ShadowWeaveEnv  # noqa: E402
from shadowweave.utils import get_device, load_config  # noqa: E402

ZONE_LABELS = ["L-far", "L    ", "L-near", "FL   ", "F    ", "FR   ", "R-near", "R    ", "R-far"]
BAR_WIDTH = 30


def render_frame(idx: int, uncertainty: np.ndarray, is_stop: bool, shadow_frac: float) -> None:
    tag = "\033[91m[STOP]\033[0m" if is_stop else "      "
    print(f"\n─── Frame {idx:03d} {tag}  shadow={shadow_frac:.2f} ──────────────────")
    for label, u in zip(ZONE_LABELS, uncertainty):
        filled = int(u * BAR_WIDTH)
        colour = "\033[91m" if u > 0.7 else ("\033[93m" if u > 0.4 else "\033[92m")
        bar = colour + "█" * filled + "\033[0m" + "░" * (BAR_WIDTH - filled)
        print(f"  {label}: {bar} {u:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="ShadowWeave pipeline smoke test")
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--difficulty", type=str, default=None,
                    choices=["static", "moving", "debris"])
    ap.add_argument("--quiet", action="store_true", help="latency summary only")
    ap.add_argument("--warmup", type=int, default=3,
                    help="frames excluded from the latency budget (lazy imports, "
                         "FFT plan setup and shader compilation land on the first call)")
    ap.add_argument("--overrides", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(overrides=args.overrides)
    device = get_device()
    print(f"Pipeline test — device: {device}")

    env = ShadowWeaveEnv(cfg)
    projector = BEVProjector(cfg).to(device).eval()
    raycaster = ShadowRaycaster(cfg).to(device).eval()
    orch = Orchestrator(cfg, LocalAgent(cfg).to(device).eval(), GlobalAgent(cfg))
    hrtf = HRTFEngine(cfg)
    hrtf.load_hrtfs()

    S = cfg.bev.size
    T = len(cfg.world_model.prediction_horizons)
    tiers = [args.difficulty] if args.difficulty else list(cfg.eval.difficulty_tiers)

    stage_ms: dict[str, list[float]] = {k: [] for k in
                                        ["sim", "bev", "shadow", "orchestrator", "audio", "total"]}
    n_stop = 0

    for difficulty in tiers:
        cfg.sim.difficulty = difficulty
        print(f"\n{'═' * 52}\n  Difficulty: {difficulty.upper()}\n{'═' * 52}")
        obs = env.reset(seed=42)
        prev_bev = None

        for frame_idx in range(args.frames):
            t_start = time.perf_counter()

            t0 = time.perf_counter()
            depth_t = torch.from_numpy(obs["depth"]).unsqueeze(0).unsqueeze(0).to(device)
            stage_ms["sim"].append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            with torch.no_grad():
                occ, vis = projector(depth_t)
                flow = (bev_flow(prev_bev, occ) if prev_bev is not None
                        else torch.zeros(1, 2, S, S, device=device))
                prev_bev = occ
            stage_ms["bev"].append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            with torch.no_grad():
                uncertainty = raycaster.forward_from_depth(depth_t)[0].cpu().numpy()
            stage_ms["shadow"].append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            # No trained world model in a smoke test; zeros keep the contract honest.
            out = orch.step(uncertainty, np.zeros((T, S, S), dtype=np.float32))
            stage_ms["orchestrator"].append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            buf = hrtf._synthesise(out.audio_params.reshape(9, 3))
            stage_ms["audio"].append((time.perf_counter() - t0) * 1000)

            stage_ms["total"].append((time.perf_counter() - t_start) * 1000)
            n_stop += int(out.is_stop)

            shadow_frac = float(1.0 - vis.mean().item())
            if not args.quiet:
                render_frame(frame_idx, uncertainty, out.is_stop, shadow_frac)
                active = int((out.audio_params[1::3] > 0).sum())
                print(f"  latency: {stage_ms['total'][-1]:5.1f} ms  |  "
                      f"active cues: {active}/{cfg.agents.local.max_simultaneous_cues}  |  "
                      f"peak audio: {np.abs(buf).max():.3f}  |  "
                      f"clearance: {obs['min_clearance']:.2f}m"
                      f"{'  COLLISION' if obs['collision'] else ''}")

            obs = env.step(action=out.nav_action)

    env.close()

    # Lazy imports, scipy's FFT plan and shader compilation all land on the first
    # call and can cost hundreds of ms. Including them makes the reported percentile
    # depend on --frames rather than on the pipeline, so warmup is excluded from the
    # steady-state budget and reported on its own line.
    warm = min(args.warmup, len(stage_ms["total"]) - 1)
    cold = stage_ms["total"][:warm]
    steady = {k: v[warm:] for k, v in stage_ms.items()}

    print(f"\n{'═' * 52}")
    print(f"  Latency budget (camera → audio buffer), {warm} warmup frame(s) excluded")
    for stage in ["sim", "bev", "shadow", "orchestrator", "audio"]:
        v = steady[stage]
        print(f"    {stage:14s} p50={np.percentile(v, 50):6.2f}ms  p95={np.percentile(v, 95):6.2f}ms")
    total = steady["total"]
    p50, p95 = np.percentile(total, 50), np.percentile(total, 95)
    print(f"    {'TOTAL':14s} p50={p50:6.2f}ms  p95={p95:6.2f}ms  max={max(total):6.2f}ms")
    if cold:
        print(f"    {'(warmup)':14s} first-frame cost {max(cold):.0f}ms — one-off, not steady state")

    n = len(total)
    shadow_hz = 1000.0 / max(np.percentile(steady["shadow"], 50), 1e-6)
    print(f"\n  frames={n} scored ({warm} warmup)  stop-overrides={n_stop}  "
          f"shadow-ray throughput={shadow_hz:.0f} Hz")
    checks = [
        ("full pipeline latency < 100ms", p95 < 100),
        ("shadow-ray throughput >= 15Hz", shadow_hz >= 15),
    ]
    ok = True
    for label, passed in checks:
        print(f"  {'✓ PASS' if passed else '✗ FAIL'}  {label}")
        ok &= passed
    print(f"{'═' * 52}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
