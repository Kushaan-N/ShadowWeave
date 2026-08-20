"""Training loop for the ShadowWeave world model.

Built for a SLURM GPU node: AMP, multi-worker loading, optional DDP across the GPUs
in an allocation, resume-from-checkpoint so a job can survive a time limit, and IOU
(the metric the project is actually judged on) tracked per horizon every epoch.

Launch:
    python -m shadowweave.world_model.train                       # single GPU
    torchrun --nproc_per_node=4 -m shadowweave.world_model.train  # DDP
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import pathlib
import time
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from ..eval.metrics import iou
from ..utils import (
    count_parameters,
    unwrap_model,
    get_device,
    load_checkpoint,
    load_config,
    save_checkpoint,
    seed_everything,
)
from .dataset import RolloutDataset
from .unet import build_world_model


def _dist_info() -> tuple[int, int, int]:
    """(rank, world_size, local_rank) from torchrun / SLURM env vars."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size(), int(os.environ.get("LOCAL_RANK", 0))
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        return int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ.get("LOCAL_RANK", 0))
    return 0, 1, 0


def _maybe_init_distributed() -> tuple[int, int, int]:
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        if not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend)
        rank, world, local = _dist_info()
        if torch.cuda.is_available():
            torch.cuda.set_device(local)
        return rank, world, local
    return 0, 1, 0


class EMA:
    """Exponential moving average of the weights.

    Averaged weights are consistently a little better than the final SGD iterate at
    essentially no training cost, which matters when a full run costs cluster hours.
    The decay is warmed up so the average is not dominated by the random init.
    """

    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = decay
        self.steps = 0
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items() if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.steps += 1
        d = min(self.decay, (1 + self.steps) / (10 + self.steps))
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(d).add_(v.detach().float(), alpha=1 - d)

    def copy_to(self, model: torch.nn.Module) -> dict[str, torch.Tensor]:
        """Swap EMA weights in, returning the originals so they can be restored."""
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()
                  if k in self.shadow}
        model.load_state_dict({**model.state_dict(),
                               **{k: v.to(dtype=backup[k].dtype) for k, v in self.shadow.items()}})
        return backup

    def restore(self, model: torch.nn.Module, backup: dict[str, torch.Tensor]) -> None:
        model.load_state_dict({**model.state_dict(), **backup})

    def state_dict(self) -> dict:
        return {"decay": self.decay, "steps": self.steps, "shadow": self.shadow}

    def load_state_dict(self, state: dict) -> None:
        self.decay, self.steps, self.shadow = state["decay"], state["steps"], state["shadow"]


class _TrainStep(torch.nn.Module):
    """Routes the world model's loss through ``forward`` so DistributedDataParallel's
    gradient all-reduce hooks fire.

    DDP arms its reducer inside ``forward()`` (``Reducer.prepare_for_backward``).
    Calling ``model.loss(x, y)`` directly — the architecture-agnostic entry point the
    rest of the codebase relies on — bypasses ``forward`` entirely, so with more than
    one rank no gradient is ever averaged: each rank trains on its own sampler shard
    and rank 0 ships a silently quarter-data model. Wrapping the loss call in a module
    whose ``forward`` *is* the loss, and putting DDP around that, restores syncing
    without any world model needing to know DDP exists.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.model.loss(x, y)


def _lr_at(step: int, cfg: DictConfig, total_steps: int) -> float:
    """Linear warmup then cosine decay. Warmup matters here because pos_weight makes
    the first few batches produce very large gradients."""
    warm = cfg.world_model.warmup_steps
    base = cfg.world_model.lr
    if step < warm:
        return base * (step + 1) / max(warm, 1)
    progress = (step - warm) / max(total_steps - warm, 1)
    return base * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def train(cfg: DictConfig, data_dir: str, resume: Optional[str] = None) -> dict[str, float]:
    rank, world_size, local_rank = _maybe_init_distributed()
    is_main = rank == 0

    seed_everything(cfg.seed + rank, deterministic=cfg.deterministic)
    # Pin to this rank's GPU only when there is one. Hardcoding cuda:{local_rank}
    # whenever world_size > 1 breaks any CPU/gloo distributed run outright.
    if world_size > 1 and torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = get_device()
    use_amp = bool(cfg.world_model.amp) and device.type == "cuda"

    if is_main:
        print(f"Training WorldModel on {device} (world_size={world_size}, amp={use_amp})")

    ckpt_dir = pathlib.Path(cfg.world_model.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    wandb = None
    if is_main and cfg.wandb.mode != "disabled":
        try:
            import wandb as _wandb
            # Persist the run id so a requeued job continues the same run instead of
            # littering the project with a fresh run per requeue.
            id_file = ckpt_dir / "wandb_run_id.txt"
            run_id = id_file.read_text().strip() if id_file.exists() else _wandb.util.generate_id()
            id_file.write_text(run_id)
            _wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                # The env var must win: passing mode= explicitly overrides
                # WANDB_MODE=offline, so a keyless cluster node would error out
                # of wandb entirely and record nothing to sync later.
                mode=os.environ.get("WANDB_MODE", cfg.wandb.mode),
                id=run_id,
                resume="allow",
                config=OmegaConf.to_container(cfg, resolve=True),
            )
            wandb = _wandb
        except Exception as e:
            print(f"wandb unavailable ({e.__class__.__name__}) — logging to stdout only")

    train_ds = RolloutDataset(cfg, data_dir, split="train", augment=True)
    val_ds = RolloutDataset(cfg, data_dir, split="val")
    if is_main:
        print(f"  train={len(train_ds)} samples, val={len(val_ds)} samples")

    train_sampler = DistributedSampler(train_ds) if world_size > 1 else None
    pin = device.type == "cuda"
    nw = cfg.world_model.num_workers
    train_dl = DataLoader(
        train_ds, batch_size=cfg.world_model.batch_size, shuffle=train_sampler is None,
        sampler=train_sampler, num_workers=nw, pin_memory=pin, drop_last=True,
        persistent_workers=nw > 0, prefetch_factor=4 if nw > 0 else None,
    )
    val_dl = DataLoader(
        val_ds, batch_size=cfg.world_model.batch_size, shuffle=False,
        num_workers=nw, pin_memory=pin, persistent_workers=nw > 0,
    )

    # drop_last with a batch larger than the split yields zero batches, so every
    # epoch would silently do nothing and still report a loss of 0.0. Fail here
    # rather than after hours of "training".
    if len(train_dl) == 0:
        raise ValueError(
            f"train split has {len(train_ds)} samples but batch_size="
            f"{cfg.world_model.batch_size} with drop_last=True yields 0 batches. "
            f"Generate more rollouts or lower world_model.batch_size."
        )
    if len(val_dl) == 0:
        raise ValueError(f"val split has {len(val_ds)} samples — too few to evaluate.")

    model = build_world_model(cfg).to(device)
    if is_main:
        print(f"  Parameters: {count_parameters(model)/1e6:.1f}M")
    if cfg.world_model.compile and hasattr(torch, "compile"):
        model = torch.compile(model)
    # Wrap the loss (not the model) in DDP — see _TrainStep. Optimizer, grad clip and
    # backward all go through train_module so the reducer sees every gradient; EMA,
    # validation and checkpointing keep operating on the raw core via unwrap_model.
    train_module: torch.nn.Module = _TrainStep(model)
    if world_size > 1:
        train_module = DistributedDataParallel(
            train_module, device_ids=[local_rank] if device.type == "cuda" else None
        )

    opt = torch.optim.AdamW(
        train_module.parameters(), lr=cfg.world_model.lr, weight_decay=cfg.world_model.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    pos_weight = torch.tensor(cfg.world_model.pos_weight, device=device)

    raw = unwrap_model(model)
    ema = EMA(raw, cfg.world_model.ema_decay) if cfg.world_model.ema else None

    iou_key = f"iou_{cfg.eval.iou_horizon:g}s"
    start_epoch, best_iou, best_val, bad_epochs, global_step = 0, -1.0, float("inf"), 0, 0
    epoch = cfg.world_model.epochs - 1  # final.pt stamp if the loop body never runs
    if resume and pathlib.Path(resume).exists():
        state = load_checkpoint(resume, map_location=device)
        unwrap_model(model).load_state_dict(state["model"])
        if "optimizer" in state:
            opt.load_state_dict(state["optimizer"])
        if "scaler" in state and use_amp:
            scaler.load_state_dict(state["scaler"])
        if ema is not None and "ema" in state:
            ema.load_state_dict(state["ema"])
        start_epoch = state.get("epoch", 0) + 1
        # Patience must survive a requeue, or early stopping can never fire on a job
        # that requeues before early_stop_patience epochs elapse.
        bad_epochs = int(state.get("bad_epochs", 0))
        # The best score lives in best.pt, not in the last.pt we resume from. Reading
        # the resume file's metrics would reset the bar to the latest (possibly worse)
        # epoch, letting a subsequent mediocre epoch overwrite the true best.
        best_ckpt = ckpt_dir / "best.pt"
        if best_ckpt.exists():
            bm = load_checkpoint(best_ckpt, map_location="cpu").get("metrics", {})
            best_iou = float(bm.get(iou_key, -1.0))
            best_val = float(bm.get("val_loss", float("inf")))
        # Without this the LR schedule restarts from warmup on every requeue, so a
        # job that gets requeued a few times never leaves the warmup ramp.
        global_step = start_epoch * len(train_dl)
        if is_main:
            print(f"  Resumed from {resume} at epoch {start_epoch} "
                  f"(step {global_step}, best {iou_key}={best_iou:.3f})")

    total_steps = max(cfg.world_model.epochs * len(train_dl), 1)
    summary: dict[str, float] = {}

    for epoch in range(start_epoch, cfg.world_model.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        train_loss = 0.0
        t0 = time.time()

        for i, batch in enumerate(train_dl):
            x = batch["input"].to(device, non_blocking=True)
            y = batch["target"].to(device, non_blocking=True)

            lr = _lr_at(global_step, cfg, total_steps)
            for g in opt.param_groups:
                g["lr"] = lr

            with torch.autocast("cuda", enabled=use_amp):
                # Each architecture owns its objective: weighted BCE + horizon
                # smoothness for the U-Net, epsilon-MSE for the diffusion model.
                # Called through the DDP-wrapped module so gradients are all-reduced.
                loss = train_module(x, y)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(train_module.parameters(), cfg.world_model.grad_clip)
            scaler.step(opt)
            scaler.update()
            if ema is not None:
                ema.update(raw)

            train_loss += loss.item()
            global_step += 1

            if is_main and i % cfg.world_model.log_interval == 0:
                print(f"    epoch {epoch+1} step {i}/{len(train_dl)} loss={loss.item():.4f} lr={lr:.2e}",
                      flush=True)

        train_loss /= max(len(train_dl), 1)

        # ── validation ────────────────────────────────────────────────
        # Validate the averaged weights — they are what gets shipped.
        ema_backup = ema.copy_to(raw) if ema is not None else None
        model.eval()
        val_loss = 0.0
        n_h = len(cfg.world_model.prediction_horizons)
        iou_sum = torch.zeros(n_h, device=device)
        n_val = 0
        with torch.no_grad():
            for batch in val_dl:
                x = batch["input"].to(device, non_blocking=True)
                y = batch["target"].to(device, non_blocking=True)
                with torch.autocast("cuda", enabled=use_amp):
                    val_loss += float(raw.loss(x, y))
                # IOU comes from predict(), not from raw logits: for the diffusion
                # model those two differ (sampling vs a sigmoid), and IOU is the
                # number the checkpoint is selected on.
                iou_sum += iou(raw.predict(x).float(), y).sum(dim=0)
                n_val += x.shape[0]
        val_loss /= max(len(val_dl), 1)
        ious = (iou_sum / max(n_val, 1)).cpu()

        if world_size > 1:
            t = torch.tensor([val_loss], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.AVG)
            val_loss = float(t.item())
            iou_t = ious.to(device)
            dist.all_reduce(iou_t, op=dist.ReduceOp.AVG)
            ious = iou_t.cpu()

        horizons = list(cfg.world_model.prediction_horizons)
        iou_log = {f"iou_{h:g}s": float(ious[k]) for k, h in enumerate(horizons)}
        target_iou = iou_log.get(f"iou_{cfg.eval.iou_horizon:g}s", float(ious.mean()))

        if is_main:
            elapsed = time.time() - t0
            iou_str = "  ".join(f"{k}={v:.3f}" for k, v in iou_log.items())
            print(f"Epoch {epoch+1:03d}/{cfg.world_model.epochs}  train={train_loss:.4f}  "
                  f"val={val_loss:.4f}  {iou_str}  ({elapsed:.1f}s)", flush=True)

            if wandb is not None:
                wandb.log({"train_loss": train_loss, "val_loss": val_loss,
                           "epoch": epoch, "lr": _lr_at(global_step, cfg, total_steps), **iou_log})

            raw_model = unwrap_model(model)
            metrics = {"val_loss": val_loss, "train_loss": train_loss, **iou_log}
            # Select on IOU — the metric the project is judged on. val_loss is an
            # adequate proxy for the U-Net (weighted BCE tracks IOU) but for the
            # diffusion model it is epsilon-MSE, orthogonal to sample quality, so
            # selecting on it ships an essentially arbitrary epoch.
            improved = target_iou > best_iou
            if improved:
                best_iou, best_val, bad_epochs = target_iou, val_loss, 0
            else:
                bad_epochs += 1
            extra = {"bad_epochs": bad_epochs}
            if ema is not None:
                extra["ema"] = ema.state_dict()
            save_checkpoint(ckpt_dir / "last.pt", raw_model, cfg, epoch=epoch,
                            optimizer=opt, scaler=scaler if use_amp else None,
                            metrics=metrics, extra=extra)
            if improved:
                save_checkpoint(ckpt_dir / "best.pt", raw_model, cfg, epoch=epoch,
                                optimizer=opt, scaler=scaler if use_amp else None,
                                metrics=metrics, extra=extra)
            summary = metrics

        # Put the live weights back before the next epoch trains on them.
        if ema is not None and ema_backup is not None:
            ema.restore(raw, ema_backup)

        # Every rank must agree on stopping or the collective ops deadlock.
        stop = torch.tensor([1 if bad_epochs >= cfg.world_model.early_stop_patience else 0],
                            device=device)
        if world_size > 1:
            dist.broadcast(stop, src=0)
        if bool(stop.item()):
            if is_main:
                print(f"Early stopping at epoch {epoch+1} ({bad_epochs} epochs without improvement)")
            break

    if is_main:
        raw_model = unwrap_model(model)
        if ema is not None:
            ema.copy_to(raw_model)  # ship the averaged weights
        save_checkpoint(ckpt_dir / "final.pt", raw_model, cfg, epoch=epoch,
                        metrics=summary)
        print(f"Training done. Best val loss: {best_val:.4f}  IOU@{cfg.eval.iou_horizon}s: {best_iou:.3f}")
        if wandb is not None:
            wandb.finish()

    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the ShadowWeave world model")
    ap.add_argument("--data", type=str, default=None)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--overrides", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(overrides=args.overrides)
    train(cfg, args.data or cfg.data.root, resume=args.resume)


if __name__ == "__main__":
    main()
