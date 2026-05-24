"""
Offline preprocessing script — run locally on your machine with MAMA-MIA data.

Applies all deterministic transforms (load, resample, normalize, crop, pad) to
every case and saves as compressed .npz files indexed by patient_id.
The Kaggle notebook then loads these directly, skipping expensive resampling.

Usage:
    python scripts/preprocess_to_cache.py \
        --data_root "C:/Users/bhara/Desktop/MAMA_MIA_COMPLETE" \
        --out_dir   "C:/Users/bhara/Desktop/mama_mia_cache" \
        --n_jobs 4

Estimated time: ~3-5 hours with 4 workers on a CPU-only machine.
Output size:    ~4-6 GB compressed (uploadable as Kaggle dataset).
"""

import argparse
import os
import sys
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from data.mama_mia_dataset import discover_cases

PATCH_SIZE = (128, 128, 64)


# ── Worker — must be module-level for Windows multiprocessing pickle ──────────

def _is_valid_npz(path: str) -> bool:
    """Return True if the .npz file exists and loads without error."""
    try:
        d = np.load(path)
        _ = d["image"].shape
        _ = d["label"].shape
        return True
    except Exception:
        return False


def _process_one(task: tuple) -> tuple[str, str]:
    """Process one case and save {patient_id}.npz with 2-channel image
    (pre-contrast + post-contrast phase 1). Returns (patient_id, status)."""
    patient_id, image_pre, image_post, label_path, out_path, verify = task

    if Path(out_path).exists():
        if not verify or _is_valid_npz(out_path):
            return patient_id, "skipped"
        # File exists but is corrupted — delete and reprocess
        Path(out_path).unlink(missing_ok=True)

    try:
        from monai.transforms import (
            Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
            Spacingd, ScaleIntensityRangePercentilesd, CropForegroundd,
            SpatialPadd, ConcatItemsd, DeleteItemsd, EnsureTyped,
        )

        img_keys = ["image_pre", "image_post"]
        all_keys = img_keys + ["label"]

        transform = Compose([
            LoadImaged(keys=all_keys),
            EnsureChannelFirstd(keys=all_keys),
            Orientationd(keys=all_keys, axcodes="RAS"),
            Spacingd(
                keys=all_keys,
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "bilinear", "nearest"),
            ),
            ScaleIntensityRangePercentilesd(
                keys=img_keys, lower=1, upper=99,
                b_min=0.0, b_max=1.0, clip=True,
            ),
            CropForegroundd(keys=all_keys, source_key="image_post"),
            ConcatItemsd(keys=img_keys, name="image", dim=0),
            DeleteItemsd(keys=img_keys),
            SpatialPadd(keys=["image", "label"], spatial_size=PATCH_SIZE),
            EnsureTyped(keys=["image", "label"]),
        ])

        data = transform({
            "image_pre":  image_pre,
            "image_post": image_post,
            "label":      label_path,
        })

        # float16 for image (normalized [0,1] — fine precision), uint8 for binary label
        image_arr = data["image"].numpy().astype(np.float16)
        label_arr = data["label"].numpy().astype(np.uint8)

        np.savez_compressed(out_path, image=image_arr, label=label_arr)
        return patient_id, "done"

    except Exception as e:
        return patient_id, f"ERROR: {e}"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str,
                        default=r"C:\Users\bhara\Desktop\MAMA_MIA_COMPLETE")
    parser.add_argument("--out_dir",   type=str,
                        default=r"C:\Users\bhara\Desktop\mama_mia_cache")
    parser.add_argument("--split_csv", type=str, default=None,
                        help="Path to train_test_splits.csv (auto-detected if omitted)")
    parser.add_argument("--n_jobs",    type=int, default=4,
                        help="Parallel workers (default 4, use 1 to debug)")
    parser.add_argument("--verify",    action="store_true",
                        help="Check existing .npz files and reprocess any that are corrupted")
    args = parser.parse_args()

    data_root = args.data_root
    out_dir   = Path(args.out_dir)

    split_csv = args.split_csv
    if split_csv is None:
        auto = os.path.join(data_root, "train_test_splits.csv")
        split_csv = auto if os.path.exists(auto) else None

    train_out = out_dir / "train"
    val_out   = out_dir / "val"
    train_out.mkdir(parents=True, exist_ok=True)
    val_out.mkdir(parents=True, exist_ok=True)

    train_cases = discover_cases(data_root, split_csv, split="train")
    val_cases   = discover_cases(data_root, split_csv, split="test")
    all_cases   = [(c, train_out) for c in train_cases] + \
                  [(c, val_out)   for c in val_cases]

    print(f"Train: {len(train_cases)} | Val: {len(val_cases)} | Total: {len(all_cases)}")
    print(f"Output dir: {out_dir}")
    print(f"Workers: {args.n_jobs}")
    if args.verify:
        print("Mode: VERIFY — existing files will be checked and corrupted ones reprocessed")

    tasks = [
        (c["patient_id"], c["image_pre"], c["image_post"], c["label"],
         str(dst / f"{c['patient_id']}.npz"), args.verify)
        for c, dst in all_cases
    ]

    done = skipped = errors = 0
    error_list = []

    if args.n_jobs == 1:
        for task in tqdm(tasks, desc="Preprocessing", unit="case"):
            pid, status = _process_one(task)
            if status == "done":        done    += 1
            elif status == "skipped":   skipped += 1
            else:                       errors  += 1; error_list.append((pid, status))
    else:
        with ProcessPoolExecutor(max_workers=args.n_jobs) as pool:
            futures = {pool.submit(_process_one, t): t[0] for t in tasks}
            with tqdm(total=len(tasks), desc="Preprocessing", unit="case") as pbar:
                for future in as_completed(futures):
                    pid, status = future.result()
                    if status == "done":      done    += 1
                    elif status == "skipped": skipped += 1
                    else:                     errors  += 1; error_list.append((pid, status))
                    pbar.update(1)
                    pbar.set_postfix(done=done, skip=skipped, err=errors)

    print(f"\nDone: {done} | Skipped (already exist): {skipped} | Errors: {errors}")
    if error_list:
        print("Failed cases:")
        for pid, msg in error_list:
            print(f"  {pid}: {msg}")

    # Print total size
    total_bytes = sum(f.stat().st_size for f in out_dir.rglob("*.npz"))
    print(f"Total cache size: {total_bytes / 1e9:.2f} GB")
    print(f"\nUpload '{out_dir}' to Kaggle as a new dataset.")
    print("Then set PREPROCESSED_CACHE_DIR in the notebook to its Kaggle path.")


if __name__ == "__main__":
    main()
