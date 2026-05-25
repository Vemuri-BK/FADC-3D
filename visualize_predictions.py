"""
Standalone prediction visualization for the 2-channel MAMA-MIA pipeline.

Loads a trained checkpoint (best_model.pth) + the preprocessed .npz cache,
runs sliding-window inference on one validation case per collection
(DUKE / ISPY1 / ISPY2 / NACT), and saves a 4-column grid PNG:

    post-contrast MRI | ground truth (green) | prediction (cyan) | TP/FP/FN map

Designed to slot directly into the paper as Figure 4 (Qualitative Results).

USAGE (Kaggle, inference-only — no GPU quota for training):
    !python /kaggle/working/FADC-3D/visualize_predictions.py \
        --ckpt   /kaggle/working/outputs/unet3d_2ch_100ep/best_model.pth \
        --cache  /kaggle/input/datasets/bharathkumarvemuri/mama-mia-preprocessed-cache-2ch \
        --model  unet3d \
        --out    /kaggle/working/outputs/baseline_predictions.png

For FADC variants, pass --model unet3d_fadc_mid (or _encoder/_bottleneck/_full).

The checkpoint's saved config provides in_channels / out_channels / base_filters /
patch_size, so no extra flags are required.
"""

import argparse
import os
import sys
import numpy as np
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Local imports — script lives in repo root
sys.path.insert(0, str(Path(__file__).parent))
from models.unet_3d import UNet3D
from models.unet_3d_fadc import UNet3DFADC
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete


COLLECTIONS = ["DUKE", "ISPY1", "ISPY2", "NACT"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",  required=True, help="Path to best_model.pth")
    p.add_argument("--cache", required=True,
                   help="Root of preprocessed cache (contains val/*.npz)")
    p.add_argument("--model", default="unet3d",
                   choices=["unet3d", "unet3d_fadc",
                            "unet3d_fadc_encoder", "unet3d_fadc_bottleneck",
                            "unet3d_fadc_mid"])
    p.add_argument("--out",   default="outputs/predictions.png",
                   help="Output PNG path")
    p.add_argument("--n_per_collection", type=int, default=1,
                   help="How many validation cases to plot per collection")
    p.add_argument("--overlap", type=float, default=0.25,
                   help="Sliding-window overlap (higher = better at boundaries, slower)")
    return p.parse_args()


def build_model(model_name, cfg, device):
    kwargs = dict(
        in_channels  = cfg["model"]["in_channels"],
        out_channels = cfg["model"]["out_channels"],
        base_filters = cfg["model"]["base_filters"],
    )
    placement_map = {
        "unet3d_fadc":            "full",
        "unet3d_fadc_encoder":    "encoder",
        "unet3d_fadc_bottleneck": "bottleneck",
        "unet3d_fadc_mid":        "mid",
    }
    if model_name in placement_map:
        return UNet3DFADC(**kwargs, fadc_placement=placement_map[model_name]).to(device)
    return UNet3D(**kwargs).to(device)


def pick_cases(val_dir: Path, n_per_collection: int) -> list[dict]:
    """Pick first N alphabetical val cases per collection for stable, reproducible figures."""
    selected = []
    for col in COLLECTIONS:
        matches = sorted(val_dir.glob(f"{col.lower()}_*.npz"))
        for p in matches[:n_per_collection]:
            selected.append({"patient_id": p.stem, "collection": col, "path": p})
    return selected


def run_inference(model, npz_path, patch_size, device, overlap):
    data = np.load(npz_path)
    image = torch.from_numpy(data["image"].astype(np.float32)).unsqueeze(0).to(device)
    label = torch.from_numpy(data["label"].astype(np.float32)).unsqueeze(0)

    assert image.shape[1] == 2, (
        f"Expected 2-channel cache, got {image.shape[1]} channels in {npz_path}"
    )

    with torch.no_grad():
        logits = sliding_window_inference(
            image, patch_size, sw_batch_size=4, predictor=model, overlap=overlap
        )
    pred_cls = AsDiscrete(argmax=True)(logits[0]).cpu().numpy()
    pred_fg  = (pred_cls[0] == 1).astype(np.float32)
    label_fg = label[0, 0].numpy()
    image_post = image[0, 1].cpu().numpy()  # channel 1 = post-contrast

    tp    = (pred_fg * label_fg).sum()
    denom = pred_fg.sum() + label_fg.sum()
    dice  = float(2 * tp / (denom + 1e-6))
    return image_post, label_fg, pred_fg, dice


def find_best_slice(label_fg, fallback_idx):
    """Axial slice with the most ground-truth tumor voxels."""
    sums = label_fg.sum(axis=(0, 1))
    return int(sums.argmax()) if sums.max() > 0 else fallback_idx


def make_figure(results, model_name, val_dice, epoch, out_path):
    n = len(results)
    fig, axes = plt.subplots(n, 4, figsize=(20, 5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = [
        "MRI Input (post-contrast)",
        "Ground Truth (green)",
        "Prediction (cyan)",
        "TP / FP / FN",
    ]

    for i, r in enumerate(results):
        img, gt, pred, z = r["image"], r["gt"], r["pred"], r["z"]
        img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)

        img_s  = img_norm[:, :, z].T
        gt_s   = gt[:, :, z].T
        pred_s = pred[:, :, z].T

        axes[i, 0].imshow(img_s, cmap="gray", origin="lower")
        axes[i, 0].set_ylabel(
            f"{r['patient_id']}\n({r['collection']})\nDice = {r['dice']:.3f}",
            fontsize=10,
        )

        axes[i, 1].imshow(img_s, cmap="gray", origin="lower")
        gt_rgba = np.zeros((*gt_s.shape, 4))
        gt_rgba[gt_s > 0] = [0.0, 1.0, 0.0, 0.55]
        axes[i, 1].imshow(gt_rgba, origin="lower")

        axes[i, 2].imshow(img_s, cmap="gray", origin="lower")
        pred_rgba = np.zeros((*pred_s.shape, 4))
        pred_rgba[pred_s > 0] = [0.0, 0.85, 1.0, 0.55]
        axes[i, 2].imshow(pred_rgba, origin="lower")

        axes[i, 3].imshow(img_s, cmap="gray", origin="lower")
        overlay = np.zeros((*gt_s.shape, 4))
        overlay[(gt_s > 0)  & (pred_s > 0)] = [1.00, 1.00, 0.00, 0.65]
        overlay[(gt_s == 0) & (pred_s > 0)] = [1.00, 0.20, 0.20, 0.65]
        overlay[(gt_s > 0)  & (pred_s == 0)] = [0.20, 1.00, 0.20, 0.65]
        axes[i, 3].imshow(overlay, origin="lower")

        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=12, fontweight="bold", pad=8)

    legend_handles = [
        Patch(facecolor="yellow", label="True Positive  (TP)"),
        Patch(facecolor="red",    label="False Positive (FP)"),
        Patch(facecolor="lime",   label="False Negative (FN)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               fontsize=11, bbox_to_anchor=(0.5, -0.01), framealpha=0.95)

    title = f"{model_name} - Best Model (Val Dice: {val_dice:.4f} | Epoch {epoch})"
    plt.suptitle(title, fontsize=14, fontweight="bold", y=1.005)

    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(args.ckpt, map_location=device)
    cfg  = ckpt["config"]
    val_dice = ckpt["best_dice"]
    epoch    = ckpt["epoch"] + 1
    patch_size = tuple(cfg["data"]["patch_size"])
    print(f"Checkpoint: epoch {epoch}, Val Dice {val_dice:.4f}, patch {patch_size}")

    model = build_model(args.model, cfg, device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Model: {args.model}")

    val_dir = Path(args.cache) / "val"
    assert val_dir.exists(), f"Validation cache dir not found: {val_dir}"

    cases = pick_cases(val_dir, args.n_per_collection)
    if not cases:
        raise SystemExit(f"No val .npz files matched any of {COLLECTIONS} in {val_dir}")
    print(f"Selected {len(cases)} cases: {[c['patient_id'] for c in cases]}")

    results = []
    for c in cases:
        img, gt, pred, dice = run_inference(model, c["path"], patch_size, device, args.overlap)
        z = find_best_slice(gt, img.shape[2] // 2)
        print(f"  {c['patient_id']:25s}  Dice={dice:.4f}  slice z={z}")
        results.append({**c, "image": img, "gt": gt, "pred": pred, "dice": dice, "z": z})

    model_label = {
        "unet3d":                 "3D U-Net Baseline",
        "unet3d_fadc":            "FADC-Full",
        "unet3d_fadc_encoder":    "FADC-Encoder",
        "unet3d_fadc_bottleneck": "FADC-Bottleneck",
        "unet3d_fadc_mid":        "FADC-Mid",
    }[args.model]

    make_figure(results, model_label, val_dice, epoch, args.out)
    mean_dice = np.mean([r["dice"] for r in results])
    print(f"Mean Dice across {len(results)} sampled cases: {mean_dice:.4f}")


if __name__ == "__main__":
    main()
