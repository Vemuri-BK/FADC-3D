"""Kaggle entry point: python -m fadc_av.run --help.

Uses the existing cached preprocessing/augmentation and batch Dice+CE loss.
Epoch-local loader seeds allow deterministic epoch-boundary resume; exact
cross-device/library reproducibility is not promised. No mid-epoch resume.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from monai.data import DataLoader, list_data_collate
from monai.data.utils import worker_init_fn
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism
from tqdm import tqdm

from data.mama_mia_dataset import PreprocessedDataset
from training.losses import DiceCELoss
from .model import build_model


def inventory(root, config, *, check_arrays=True):
    """Fail closed on wrong split counts, duplicate patients or malformed cache."""
    root = Path(root)
    cases, manifest = {}, []
    for split in ("train", "val"):
        files = sorted((root / split).glob("*.npz"))
        if len(files) != config["expected_" + split] or not files:
            raise ValueError(f"{split}: found {len(files)}, expected {config['expected_' + split]}")
        cases[split] = []
        for path in tqdm(files, desc=f"{split}: validate volumes" if check_arrays else f"{split}: check file metadata", unit="case", file=sys.stdout):
            if check_arrays:
                with np.load(path, allow_pickle=False) as data:
                    x, y = data["image"], data["label"]
                    if x.ndim != 4 or x.shape[0] != 2 or y.shape != (1, *x.shape[1:]):
                        raise ValueError(f"Invalid two-channel 3D image/mask shape: {path}")
                    if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isin(y, [0, 1]).all():
                        raise ValueError(f"Nonfinite image or nonbinary mask: {path}")
                    if split == "train" and any(a < b for a, b in zip(x.shape[1:], config["patch_size"])):
                        raise ValueError(f"Volume smaller than training patch: {path}")
            cases[split].append({"patient_id": path.stem, "npz_path": str(path.resolve())})
            manifest.append({"split": split, "patient_id": path.stem,
                             "relative_path": f"{split}/{path.name}", "bytes": path.stat().st_size})
    train_ids = {c["patient_id"].casefold() for c in cases["train"]}
    val_ids = {c["patient_id"].casefold() for c in cases["val"]}
    if train_ids & val_ids:
        raise ValueError("Patient IDs overlap between train and validation")
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return cases, manifest, digest


def file_stats(cases):
    """Cheap freshness check for an unchanged, read-only Kaggle dataset mount.

    Size/mtime checks are not cryptographic content verification.
    """
    return {f"{split}/{case['patient_id']}": [Path(case['npz_path']).stat().st_size,
                                             Path(case['npz_path']).stat().st_mtime_ns]
            for split, items in cases.items() for case in items}


def loaders(root, cases, cfg, epoch):
    result = []
    for offset, split in enumerate(("train", "val")):
        ds = PreprocessedDataset(root, cases[split], is_train=split == "train", patch_size=cfg["patch_size"])
        seed = cfg["seed"] + epoch * 2 + offset
        if ds.transform is not None:
            ds.transform.set_random_state(seed=seed)
        result.append(DataLoader(ds, batch_size=cfg["batch_size"] if split == "train" else 1,
                                 shuffle=split == "train", num_workers=cfg["num_workers"],
                                 collate_fn=list_data_collate, worker_init_fn=worker_init_fn,
                                 generator=torch.Generator().manual_seed(seed),
                                 persistent_workers=False, pin_memory=torch.cuda.is_available()))
    return result


def atomic_save(state, path):
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temp)
    os.replace(temp, path)


def overlap_metrics(tp, fp, fn):
    """Foreground metrics from pooled counts; undefined denominators return None."""
    return {"dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "iou": tp / (tp + fp + fn) if tp + fp + fn else None,
            "sensitivity": tp / (tp + fn) if tp + fn else None,
            "precision": tp / (tp + fp) if tp + fp else None}


def step(model, batch, optimizer, scaler, device, *, return_metrics=False):
    x, y = batch["image"].to(device), batch["label"].to(device)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        logits = model(x)
    # Reductions over large 3D volumes must run in float32.
    loss, dice_loss, ce_loss = DiceCELoss()(logits.float(), y)
    if not torch.isfinite(loss):
        raise RuntimeError("Nonfinite loss")
    result = float(loss.detach())
    if return_metrics:
        with torch.no_grad():
            pred, target = logits.argmax(1).bool(), y[:, 0].bool()
            result = {"loss": float(loss.detach()), "dice_loss": float(dice_loss.detach()),
                      "ce_loss": float(ce_loss.detach()), "tp": int((pred & target).sum()),
                      "fp": int((pred & ~target).sum()), "fn": int((~pred & target).sum())}
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    named_grads = [(n, p.grad) for n, p in model.named_parameters() if p.grad is not None]
    finite = torch.stack([torch.isfinite(g).all() for _, g in named_grads]).all()
    if not finite:
        bad = [n for n, g in named_grads if not torch.isfinite(g).all()]
        if not scaler.is_enabled():
            raise RuntimeError(f"Nonfinite gradients without AMP scaling: {bad[:8]}")
        old_scale = scaler.get_scale()
        # unscale_ recorded the overflow: step skips the unsafe optimizer update.
        # Do not clip inf/NaN gradients, which would hide or propagate the fault.
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        optimizer.amp_skips = getattr(optimizer, "amp_skips", 0) + 1
        optimizer.consecutive_amp_skips = getattr(optimizer, "consecutive_amp_skips", 0) + 1
        print(f"AMP update skipped: scale {old_scale:g} -> {scaler.get_scale():g}; "
              f"nonfinite gradients in {bad[:8]}", flush=True)
        if optimizer.consecutive_amp_skips >= 8:
            raise RuntimeError("Eight consecutive AMP overflows; stop for numerical diagnosis")
        return result
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    scaler.step(optimizer)
    scaler.update()
    optimizer.consecutive_amp_skips = 0
    return result


@torch.no_grad()
def evaluate(model, loader, cfg, device):
    model.eval()
    rows = []
    for batch in tqdm(loader, desc="Full-volume validation", file=sys.stdout, unit="case"):
        # Keep full volumes and assembled logits on CPU; only windows use GPU.
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = sliding_window_inference(
                batch["image"].float(), cfg["patch_size"], cfg["sw_batch_size"], model,
                overlap=cfg["val_overlap"], mode="constant", sw_device=device, device="cpu")
        pred, target = logits.argmax(1).bool(), batch["label"][:, 0].bool()
        tp = int((pred & target).sum())
        fp = int((pred & ~target).sum())
        fn = int((~pred & target).sum())
        # Match MONAI ignore_empty=True for foreground Dice; report empties.
        dice = 2 * tp / (2 * tp + fp + fn) if tp + fn else None
        rows.append({"patient_id": batch["patient_id"][0], "dice": dice,
                     "iou": tp / (tp + fp + fn + 1e-6),
                     "sensitivity": tp / (tp + fn + 1e-6), "false_positive_voxels": fp})
    valid = [r["dice"] for r in rows if r["dice"] is not None]
    if not valid:
        raise ValueError("Validation contains no foreground cases")
    return float(np.mean(valid)), rows


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--config", default="fadc_av/experiment.json")
    p.add_argument("--cache-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--mode", choices=("preflight", "train", "evaluate"), default="preflight")
    p.add_argument("--resume", help="Trusted checkpoint created by this runner")
    p.add_argument("--preflight-report", help="Reuse a successful preflight without decompressing all volumes")
    p.add_argument("--allow-cpu", action="store_true", help="Local synthetic verification only")
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("Enable a Kaggle GPU before running")
    set_determinism(seed=cfg["seed"])
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    if args.mode == "preflight":
        if args.preflight_report:
            raise ValueError("Preflight must perform its own full checks")
        Path(args.output).mkdir(parents=True, exist_ok=True)
        (Path(args.output) / "preflight.json").write_text(json.dumps({"passed": False}))
    print(f"Starting {args.mode} on {device}", flush=True)
    if args.preflight_report:
        report = json.loads(Path(args.preflight_report).read_text())
        if report.get("passed") is not True or report.get("config") != cfg:
            raise ValueError("Preflight missing success or configuration mismatch")
        if report.get("cache_root") != str(Path(args.cache_root).resolve()):
            raise ValueError("Preflight has no matching cache root; rerun preflight with this launcher")
        cases, manifest, fingerprint = inventory(args.cache_root, cfg, check_arrays=False)
        if (report.get("split_fingerprint") != fingerprint
                or report.get("file_stats") != file_stats(cases)):
            raise ValueError("Dataset files changed since preflight; rerun preflight")
        print("Preflight reused: no volume decompression scan. Building model...", flush=True)
    else:
        cases, manifest, fingerprint = inventory(args.cache_root, cfg)
    checked_stats = file_stats(cases)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if args.mode == "train" and (out / "last.pt").exists() and not args.resume:
        raise ValueError("Output already contains a run; use --resume or a fresh directory")
    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    warmup = cfg["warmup_epochs"]
    if 0 < warmup < cfg["epochs"]:
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg["epochs"] - warmup, eta_min=1e-6)
        ], milestones=[warmup])
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg["epochs"], eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start, best, history = 0, -1.0, []
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        if ckpt["config"] != cfg or ckpt["split_fingerprint"] != fingerprint:
            raise ValueError("Checkpoint configuration or split fingerprint mismatch")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        optimizer.amp_skips = ckpt.get("amp_skips", 0)
        start, best, history = ckpt["epoch"], ckpt["best_dice"], ckpt["history"]
        random.setstate(ckpt["python_rng"])
        np.random.set_state(ckpt["numpy_rng"])
        torch.set_rng_state(ckpt["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
        # Preserve the previously selected model when resuming into a new session.
        if args.mode == "train" and best >= 0 and not (out / "best.pt").exists():
            previous_best = Path(args.resume).parent / "best.pt"
            if not previous_best.is_file():
                raise ValueError("Resume requires the previous best.pt alongside last.pt")
            saved_best = torch.load(previous_best, map_location="cpu", weights_only=False)
            if (saved_best["config"] != cfg or saved_best["split_fingerprint"] != fingerprint
                    or saved_best["best_dice"] != best or saved_best["epoch"] > start):
                raise ValueError("Previous best.pt does not match the resumed run")
            shutil.copy2(previous_best, out / "best.pt")
    if args.mode == "evaluate":
        if not args.resume:
            raise ValueError("Evaluation requires --resume pointing to best.pt")
        _, val = loaders(args.cache_root, cases, cfg, start)
        score, rows = evaluate(model, val, cfg, device)
        (out / "evaluation.json").write_text(json.dumps({"mean_dice": score, "cases": rows}, indent=2))
        print(f"Checkpoint validation Dice: {score:.6f}")
        return
    provenance = {"config": cfg, "split_fingerprint": fingerprint, "manifest": manifest,
                  "torch": torch.__version__, "device": str(device),
                  "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  "git_status": subprocess.check_output(["git", "status", "--porcelain"], text=True),
                  "operator": ("legacy plain 3D U-Net" if cfg["variant"] == "baseline" else
                               f"{cfg['variant']}; four bands; d=1,2,3; AdaKern; fixed temperature=1")}
    (out / f"{args.mode}_provenance.json").write_text(json.dumps(provenance, indent=2))
    if args.mode == "preflight":
        print("Preflight: loading first training batch...", flush=True)
        train, val = loaders(args.cache_root, cases, cfg, 0)
        model.train()
        batch = next(iter(train))
        losses = [step(model, batch, optimizer, scaler, device) for _ in tqdm(range(2), desc="Preflight optimizer steps", file=sys.stdout)]
        # One actual whole-volume validation and nonidentity model reload.
        score, _ = evaluate(model, [next(iter(val))], cfg, device)
        model.eval()
        x = batch["image"][:1].to(device)
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            before = model(x).float().cpu()
        atomic_save({"model": model.state_dict()}, out / "preflight_model.pt")
        model.load_state_dict(torch.load(out / "preflight_model.pt", map_location=device, weights_only=False)["model"])
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            after = model(x).float().cpu()
        torch.testing.assert_close(before, after)
        report = {"passed": True, "losses": losses, "one_case_dice": score,
                  "config": cfg, "split_fingerprint": fingerprint,
                  "cache_root": str(Path(args.cache_root).resolve()), "file_stats": checked_stats,
                  "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else None}
        (out / "preflight.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        return
    for epoch in range(start, cfg["epochs"]):
        t0 = time.time()
        lr_used = optimizer.param_groups[0]["lr"]
        skips_before = getattr(optimizer, "amp_skips", 0)
        train, val = loaders(args.cache_root, cases, cfg, epoch)
        model.train()
        print(f"Epoch {epoch+1}/{cfg['epochs']}: {len(train)} batches; loading data...", flush=True)
        losses = []
        batch_metrics = []
        with tqdm(train, desc=f"Epoch {epoch+1}/{cfg['epochs']}", file=sys.stdout, unit="batch") as progress:
            for batch in progress:
                batch_metrics.append(step(model, batch, optimizer, scaler, device, return_metrics=True))
                losses.append(batch_metrics[-1]["loss"])
                progress.set_postfix(loss=f"{losses[-1]:.4f}", mean_loss=f"{np.mean(losses):.4f}",
                                     best_dice=f"{best:.4f}" if best >= 0 else "not validated")
        training_minutes = (time.time() - t0) / 60
        scheduler.step()
        row = {"epoch": epoch + 1, "loss": float(np.mean(losses)), "lr": scheduler.get_last_lr()[0],
               "amp_skips_total": getattr(optimizer, "amp_skips", 0), "amp_scale": scaler.get_scale()}
        row.update({"training_minutes": training_minutes, "validation_minutes": 0.0,
                    "lr_used": lr_used, "amp_skips_epoch": getattr(optimizer, "amp_skips", 0) - skips_before,
                    "train_dice_loss": float(np.mean([m["dice_loss"] for m in batch_metrics])),
                    "train_ce_loss": float(np.mean([m["ce_loss"] for m in batch_metrics])),
                    "val_dice": None, "val_iou": None, "val_sensitivity": None})
        counts = [sum(m[k] for m in batch_metrics) for k in ("tp", "fp", "fn")]
        row.update({f"train_patch_{k}": v for k, v in overlap_metrics(*counts).items()})
        improved = False
        if (epoch + 1) % cfg["val_every"] == 0 or epoch + 1 == cfg["epochs"]:
            val_start = time.time()
            score, metrics = evaluate(model, val, cfg, device)
            row["val_dice"] = score
            row["val_iou"] = float(np.mean([m["iou"] for m in metrics]))
            row["val_sensitivity"] = float(np.mean([m["sensitivity"] for m in metrics]))
            row["validation_minutes"] = (time.time() - val_start) / 60
            improved = score > best
            best = max(best, score)
            (out / f"validation_{epoch+1:03d}.json").write_text(json.dumps(metrics, indent=2))
        row["seconds"] = time.time() - t0
        row["epoch_minutes"] = row["seconds"] / 60
        row["best_val_dice"] = best if best >= 0 else None
        history.append(row)
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                 "epoch": epoch + 1, "best_dice": best, "history": history, "config": cfg,
                 "amp_skips": getattr(optimizer, "amp_skips", 0),
                 "split_fingerprint": fingerprint, "python_rng": random.getstate(),
                 "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}
        atomic_save(state, out / "last.pt")
        if improved:
            atomic_save(state, out / "best.pt")
            print(f"New best model saved: epoch {epoch+1}, validation Dice {best:.6f}", flush=True)
        (out / "train_log.json").write_text(json.dumps(history, indent=2))
        with (out / "train_log.csv").open("w", newline="") as handle:
            fields = list(dict.fromkeys(k for entry in history for k in entry))
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(history)
        print(f"Epoch {epoch+1}/{cfg['epochs']} | training {training_minutes:.2f} min | "
              f"validation {row['validation_minutes']:.2f} min | loss {row['loss']:.4f} | "
              f"training-patch Dice {row['train_patch_dice']} | validation Dice {row['val_dice']}", flush=True)
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
