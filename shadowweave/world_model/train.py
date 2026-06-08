"""Training loop for the ShadowWeave world model."""

from __future__ import annotations

import pathlib
import time

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from .dataset import RolloutDataset
from .diffusion import WorldModel


def train(cfg: DictConfig, data_dir: str) -> None:
    import torch
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training WorldModel on {device}")

    wandb = None
    try:
        import wandb as _wandb
        _wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        wandb = _wandb
    except Exception:
        print("wandb unavailable — logging to stdout only")

    train_ds = RolloutDataset(cfg, data_dir, split="train")
    val_ds   = RolloutDataset(cfg, data_dir, split="val")
    # pin_memory not supported on MPS
    pin = device == "cuda"
    train_dl = DataLoader(train_ds, batch_size=cfg.world_model.batch_size, shuffle=True,  num_workers=0, pin_memory=pin)
    val_dl   = DataLoader(val_ds,   batch_size=cfg.world_model.batch_size, shuffle=False, num_workers=0, pin_memory=pin)

    model = WorldModel(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.world_model.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.world_model.epochs)

    ckpt_dir = pathlib.Path(cfg.world_model.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(cfg.world_model.epochs):
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for batch in train_dl:
            shadow = batch["shadow_map"].to(device)
            vel    = batch["velocity"].to(device)
            target = batch["future_occupancy"].to(device)

            pred = model(shadow, vel)
            bce  = F.binary_cross_entropy(pred, target)

            # physics consistency: penalise sudden large displacements in predicted occupancy
            physics_reg = (pred[:, 1:] - pred[:, :-1]).abs().mean()
            loss = cfg.world_model.bce_weight * bce + cfg.world_model.physics_reg_weight * physics_reg

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item()

        train_loss /= len(train_dl)
        scheduler.step()

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dl:
                shadow = batch["shadow_map"].to(device)
                vel    = batch["velocity"].to(device)
                target = batch["future_occupancy"].to(device)
                pred   = model(shadow, vel)
                val_loss += F.binary_cross_entropy(pred, target).item()
        val_loss /= len(val_dl)

        elapsed = time.time() - t0
        print(f"Epoch {epoch+1:03d}/{cfg.world_model.epochs}  train={train_loss:.4f}  val={val_loss:.4f}  ({elapsed:.1f}s)")

        if wandb is not None:
            wandb.log({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_dir / "best.pt")

    torch.save(model.state_dict(), ckpt_dir / "final.pt")
    print(f"Training done. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    import sys
    import pathlib

    cfg_path = pathlib.Path(__file__).parents[1] / "configs" / "default.yaml"
    cfg = OmegaConf.load(cfg_path)

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./data/rollouts"
    train(cfg, data_dir)
