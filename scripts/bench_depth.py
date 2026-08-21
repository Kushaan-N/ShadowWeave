"""Benchmark the monocular-depth network on its own — the piece the 7 ms forward-path
latency deliberately excludes.

The paper's latency claim covers BEV -> world model -> orchestrator; in deployment a
depth network (cfg.depth.model, Depth-Anything-V2-Small) precedes it. This measures
that stage in isolation so the end-to-end story can be stated honestly: either depth
fits the 50 ms budget too, or it runs asynchronously at its own rate while the
completion path consumes the latest depth frame at 20 Hz.

    python scripts/bench_depth.py --n 100 --width 640 --height 480
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

from shadowweave.ingestion.depth import DepthEstimator  # noqa: E402
from shadowweave.utils import load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark the monocular depth stage")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    args = ap.parse_args()

    cfg = load_config()
    est = DepthEstimator(cfg)
    est.load()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    name = torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"
    rgb = (np.random.rand(args.height, args.width, 3) * 255).astype(np.uint8)

    for _ in range(args.warmup):
        est.forward(rgb)
    if dev == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(args.n):
        t0 = time.perf_counter()
        est.forward(rgb)
        if dev == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)

    times = np.array(times)
    print(f"depth model : {cfg.depth.model}")
    print(f"device      : {name}")
    print(f"input       : {args.height}x{args.width} RGB, n={args.n}")
    print(f"latency     : p50 {np.percentile(times, 50):.1f} ms   "
          f"p95 {np.percentile(times, 95):.1f} ms   mean {times.mean():.1f} ms")
    budget = 50.0
    print(f"vs 50 ms    : {'FITS' if np.percentile(times, 95) <= budget else 'EXCEEDS'} "
          f"the 20 Hz budget on this hardware")


if __name__ == "__main__":
    main()
