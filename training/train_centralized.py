import os
import sys
import time
import argparse
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.inferers import sliding_window_inference

sys.path.append(str(Path(__file__).parent.parent))
from data.mama_mia_dataset import build_centralized_loaders, DATA_ROOT
from models.unet_3d import UNet3D
from training.losses import DiceCELoss


# ─────────────────────────────────────────────
# 1. ARGUMENT PARSING
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    PROJECT_ROOT = str(Path(__file__).parent.parent)
    parser.add_argument("--config",      type=str, default=os.path.join(PROJECT_ROOT, "configs", "config.yaml"))
    parser.add_argument("--data_root",   type=str, default=DATA_ROOT)
    parser.add_argument("--output_dir",  type=str, default="outputs/unet3d")
    parser.add_argument("--epochs",      type=int, default=None)
    parser.add_argument("--batch_size",  type=int, default=None)
    parser.add_argument("--lr",          type=float, default=None)
    parser.add_argument("--cache_rate",  type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--persistent_cache_dir", type=str, default=None,
                        help="Cache preprocessed volumes to disk (recommended on Kaggle)")
    parser.add_argument("--resume",      type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--smoke_test",  action="store_true",
                        help="Run 2 epochs on 4 cases to verify pipeline end-to-end")
    return parser.parse_args()


# ─────────────────────────────────────────────
# 2. CONFIG LOADER
# ─────────────────────────────────────────────

def load_config(config_path: str, args) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # CLI args override config file values
    if args.epochs      is not None: cfg["training"]["epochs"]      = args.epochs
    if args.batch_size  is not None: cfg["training"]["batch_size"]  = args.batch_size
    if args.lr          is not None: cfg["training"]["lr"]          = args.lr
    if args.cache_rate  is not None: cfg["data"]["cache_rate"]      = args.cache_rate
    if args.num_workers is not None: cfg["data"]["num_workers"]     = args.num_workers

    return cfg


# ─────────────────────────────────────────────
# 3. VALIDATION — SLIDING WINDOW INFERENCE
# ─────────────────────────────────────────────

def validate(model, val_loader, dice_metric, post_pred, post_label, patch_size, device):
    model.eval()
    dice_metric.reset()

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            # Sliding window inference: tiles the full volume into overlapping
            # patches, runs model on each, stitches results back together
            preds = sliding_window_inference(
                inputs=images,
                roi_size=patch_size,
                sw_batch_size=1,
                predictor=model,
                overlap=0.5,
            )

            preds_bin  = post_pred(preds)    # argmax → binary prediction
            labels_bin = post_label(labels)  # ensure binary label

            dice_metric(y_pred=preds_bin, y=labels_bin)

    mean_dice = dice_metric.aggregate().item()
    dice_metric.reset()
    return mean_dice


# ─────────────────────────────────────────────
# 4. TRAINING LOOP
# ─────────────────────────────────────────────

def train(cfg, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    patch_size  = tuple(cfg["data"]["patch_size"])
    epochs      = cfg["training"]["epochs"]
    lr          = cfg["training"]["lr"]
    batch_size  = cfg["training"]["batch_size"]
    cache_rate  = cfg["data"]["cache_rate"]
    num_workers = cfg["data"]["num_workers"]
    val_every   = cfg["training"]["val_every"]

    # ── Data ──────────────────────────────────
    split_csv = os.path.join(args.data_root, "train_test_splits.csv")
    if not os.path.exists(split_csv):
        split_csv = None
        print("Warning: train_test_splits.csv not found — using all data for train")

    if args.smoke_test:
        print("SMOKE TEST MODE — 4 cases, 2 epochs")
        epochs      = 2
        cache_rate  = 0.0
        num_workers = 0
        val_every   = 1

    train_loader, val_loader = build_centralized_loaders(
        data_root=args.data_root,
        split_csv=split_csv,
        cache_rate=cache_rate,
        num_workers=num_workers,
        batch_size=batch_size,
        max_cases=4 if args.smoke_test else None,
        persistent_cache_dir=args.persistent_cache_dir or "",
    )

    # ── Model ─────────────────────────────────
    model = UNet3D(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        base_filters=cfg["model"]["base_filters"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: UNet3D | Parameters: {total_params:,}")

    # ── Loss, Optimizer, Scheduler ────────────
    criterion = DiceCELoss(
        dice_weight=cfg["training"]["dice_weight"],
        ce_weight=cfg["training"]["ce_weight"],
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler    = GradScaler("cuda", enabled=device.type == "cuda")  # mixed precision (GPU only)

    # ── Metrics ───────────────────────────────
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    post_pred   = AsDiscrete(argmax=True, to_onehot=2)
    post_label  = AsDiscrete(to_onehot=2)

    # ── Resume from checkpoint ─────────────────
    start_epoch  = 0
    best_dice    = 0.0
    train_log    = []

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_dice   = ckpt.get("best_dice", 0.0)
        print(f"Resumed from epoch {start_epoch} | Best Dice so far: {best_dice:.4f}")

    # ── Training ──────────────────────────────
    print(f"\nStarting training: {epochs} epochs | LR: {lr} | Batch: {batch_size}")
    print("=" * 60)

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss  = 0.0
        epoch_dice  = 0.0
        epoch_ce    = 0.0
        num_batches = 0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:03d}/{epochs}",
                    leave=False, ncols=110, unit="batch", file=sys.stdout)

        for batch in pbar:
            # RandCropByPosNegLabeld returns list of patches — flatten into batch
            if isinstance(batch["image"], list):
                images = torch.cat(batch["image"], dim=0).to(device)
                labels = torch.cat(batch["label"], dim=0).to(device)
            else:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)

            optimizer.zero_grad()

            with autocast("cuda", enabled=device.type == "cuda"):
                preds = model(images)
                total_loss, dice_loss, ce_loss = criterion(preds, labels)

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss  += total_loss.item()
            epoch_dice  += dice_loss.item()
            epoch_ce    += ce_loss.item()
            num_batches += 1

            pbar.set_postfix({
                "loss": f"{total_loss.item():.4f}",
                "dice": f"{dice_loss.item():.4f}",
                "ce":   f"{ce_loss.item():.4f}",
            })

        pbar.close()

        scheduler.step()

        avg_loss = epoch_loss / num_batches
        avg_dice = epoch_dice / num_batches
        avg_ce   = epoch_ce   / num_batches
        elapsed  = time.time() - t0
        lr_now   = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch+1:03d}/{epochs} | "
              f"Loss: {avg_loss:.4f} | Dice loss: {avg_dice:.4f} | CE loss: {avg_ce:.4f} | "
              f"LR: {lr_now:.2e} | Time: {elapsed:.0f}s")

        log_entry = {
            "epoch": epoch + 1,
            "loss": avg_loss,
            "dice_loss": avg_dice,
            "ce_loss": avg_ce,
            "lr": lr_now,
        }

        # ── Validation ────────────────────────
        if (epoch + 1) % val_every == 0:
            val_dice = validate(model, val_loader, dice_metric,
                                post_pred, post_label, patch_size, device)
            print(f"  → Val Dice: {val_dice:.4f}  (best: {best_dice:.4f})")
            log_entry["val_dice"] = val_dice

            if val_dice > best_dice:
                best_dice = val_dice
                torch.save({
                    "epoch":      epoch,
                    "model":      model.state_dict(),
                    "optimizer":  optimizer.state_dict(),
                    "scheduler":  scheduler.state_dict(),
                    "best_dice":  best_dice,
                    "config":     cfg,
                }, output_dir / "best_model.pth")
                print(f"  *** NEW BEST MODEL saved — Epoch {epoch+1} | Val Dice: {best_dice:.4f} → {output_dir}/best_model.pth ***")

        train_log.append(log_entry)

        # Save latest checkpoint every 10 epochs for resuming
        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch":      epoch,
                "model":      model.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "scheduler":  scheduler.state_dict(),
                "best_dice":  best_dice,
                "config":     cfg,
            }, output_dir / "latest_checkpoint.pth")

    # ── Save training log ─────────────────────
    import json
    with open(output_dir / "train_log.json", "w") as f:
        json.dump(train_log, f, indent=2)

    print(f"\nTraining complete. Best Val Dice: {best_dice:.4f}")
    print(f"Outputs saved to: {output_dir}")


# ─────────────────────────────────────────────
# 5. ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    cfg  = load_config(args.config, args)
    train(cfg, args)
