"""Measure throughput and memory before committing cluster hours.

Answers the questions you otherwise discover an hour into a job: what batch size fits,
how long an epoch will take, whether AMP is actually helping on this GPU, and whether
data loading or compute is the bottleneck.

    python scripts/benchmark.py                  # model fwd/bwd sweep
    python scripts/benchmark.py --data data/rollouts
    python scripts/benchmark.py --stages         # per-stage inference latency
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

from shadowweave.utils import count_parameters, get_device, load_config  # noqa: E402
from shadowweave.world_model.diffusion import build_world_model  # noqa: E402


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _time_it(fn, device, warmup: int = 3, iters: int = 10) -> float:
    for _ in range(warmup):
        fn()
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync(device)
    return (time.perf_counter() - t0) / iters


def bench_model(cfg, device, batch_sizes: list[int], amp_modes: list[bool]) -> None:
    S = cfg.bev.size
    C = 4
    print(f"\nWorld model  ({cfg.world_model.architecture}, "
          f"base_channels={cfg.world_model.base_channels}, bev={S}x{S})")

    model = build_world_model(cfg).to(device)
    print(f"  parameters: {count_parameters(model)/1e6:.1f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    pw = torch.tensor(cfg.world_model.pos_weight, device=device)

    print(f"\n  {'batch':>6} {'amp':>5} {'fwd+bwd':>10} {'samples/s':>11} {'peak mem':>10}")
    for amp in amp_modes:
        if amp and device.type != "cuda":
            continue
        for bs in batch_sizes:
            try:
                x = torch.rand(bs, C, S, S, device=device)
                y = (torch.rand(bs, len(cfg.world_model.prediction_horizons), S, S,
                                device=device) > 0.93).float()
                scaler = torch.amp.GradScaler("cuda", enabled=amp)

                def step():
                    with torch.autocast("cuda", enabled=amp):
                        loss = torch.nn.functional.binary_cross_entropy_with_logits(
                            model(x), y, pos_weight=pw)
                    opt.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()

                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                dt = _time_it(step, device, iters=6)
                mem = (f"{torch.cuda.max_memory_allocated()/1e9:.2f} GB"
                       if device.type == "cuda" else "n/a")
                print(f"  {bs:>6} {str(amp):>5} {dt*1000:>9.1f}ms {bs/dt:>11.1f} {mem:>10}")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  {bs:>6} {str(amp):>5} {'OOM':>10}")
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    break
                raise


def bench_data(cfg, data_root: str, workers: list[int]) -> None:
    from torch.utils.data import DataLoader

    from shadowweave.world_model.dataset import RolloutDataset

    try:
        ds = RolloutDataset(cfg, data_root, split="train")
    except (FileNotFoundError, ValueError) as e:
        print(f"\nData loading: skipped ({str(e)[:80]})")
        return

    print(f"\nData loading  ({len(ds)} samples in {data_root})")
    print(f"  {'workers':>8} {'samples/s':>11}")
    for nw in workers:
        dl = DataLoader(ds, batch_size=cfg.world_model.batch_size, num_workers=nw,
                        shuffle=True, drop_last=True)
        it = iter(dl)
        next(it)  # pay worker startup outside the timed region
        t0, n = time.perf_counter(), 0
        for _ in range(8):
            try:
                n += next(it)["input"].shape[0]
            except StopIteration:
                break
        dt = time.perf_counter() - t0
        print(f"  {nw:>8} {n/max(dt,1e-9):>11.1f}")
    print("  (compare against the model's samples/s above — the smaller number is the ceiling)")


def bench_stages(cfg, device) -> None:
    """Per-stage inference latency for the real-time budget."""
    from shadowweave.shadow.bev import BEVProjector
    from shadowweave.shadow.raycast import ShadowRaycaster

    h, w = cfg.depth.output_h, cfg.depth.output_w
    depth = torch.rand(1, 1, h, w, device=device)
    proj = BEVProjector(cfg).to(device).eval()
    rc = ShadowRaycaster(cfg).to(device).eval()
    model = build_world_model(cfg).to(device).eval()
    stack = torch.rand(1, 4, cfg.bev.size, cfg.bev.size, device=device)

    print("\nInference latency (batch 1)")
    with torch.no_grad():
        for name, fn in [
            ("bev projection", lambda: proj(depth)),
            ("shadow raycast", lambda: rc.forward_from_depth(depth)),
            ("world model", lambda: model(stack)),
        ]:
            dt = _time_it(fn, device, iters=20)
            print(f"  {name:16s} {dt*1000:7.2f} ms  ({1/dt:7.0f} Hz)")
    print(f"  target: full pipeline < 100ms, shadow-ray >= {cfg.shadow.target_hz} Hz")


def main() -> None:
    ap = argparse.ArgumentParser(description="ShadowWeave throughput benchmark")
    ap.add_argument("--data", type=str, default=None, help="rollout root to benchmark loading")
    ap.add_argument("--batches", type=int, nargs="*", default=[8, 16, 32, 64, 128])
    ap.add_argument("--workers", type=int, nargs="*", default=[0, 2, 4, 8])
    ap.add_argument("--stages", action="store_true", help="per-stage inference latency only")
    ap.add_argument("--overrides", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(overrides=args.overrides)
    device = get_device()
    print(f"ShadowWeave benchmark — device={device}, torch={torch.__version__}")
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(0)
        print(f"  {p.name}, {p.total_memory/1e9:.1f} GB, capability {p.major}.{p.minor}")

    if args.stages:
        bench_stages(cfg, device)
        return

    bench_model(cfg, device, args.batches, amp_modes=[False, True])
    bench_stages(cfg, device)
    if args.data:
        bench_data(cfg, args.data, args.workers)

    print("\nPick the largest batch that fits with headroom, then set "
          "world_model.batch_size and num_workers accordingly.")


if __name__ == "__main__":
    main()
