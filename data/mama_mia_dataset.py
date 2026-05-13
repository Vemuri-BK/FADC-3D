import os
import json
import pandas as pd
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from monai.data import CacheDataset, PersistentDataset, DataLoader as MonaiLoader
from monai.data.utils import list_data_collate
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    Orientationd,
    ScaleIntensityRangePercentilesd,
    CropForegroundd,
    SpatialPadd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    EnsureTyped,
    ToTensord,
)


DATA_ROOT = r"C:\Users\bhara\Desktop\MAMA_MIA_COMPLETE"


def _safe_collate(batch):
    """numpy→tensor conversion before list_data_collate.

    PersistentDataset loads cached items as numpy arrays. PyTorch's default_collate
    (called inside list_data_collate) can't handle numpy, so we convert first.
    """
    def convert(item):
        if isinstance(item, dict):
            return {k: convert(v) for k, v in item.items()}
        if isinstance(item, list):
            return [convert(i) for i in item]
        if isinstance(item, np.ndarray):
            return torch.as_tensor(item.copy())
        return item

    return list_data_collate([convert(b) for b in batch])

COLLECTIONS = ["DUKE", "ISPY1", "ISPY2", "NACT"]

# FL client assignment — one collection per client
CLIENT_MAP = {
    "DUKE":  0,
    "ISPY1": 1,
    "ISPY2": 2,
    "NACT":  3,
}


# ─────────────────────────────────────────────
# 1. FILE DISCOVERY
# ─────────────────────────────────────────────

def discover_cases(data_root: str, split_csv: str = None, split: str = "train") -> list[dict]:
    """
    Scan the images folder and match each patient to their expert segmentation.
    Returns a list of dicts: {image, label, patient_id, collection, client_id}

    If split_csv is given, filter to only train or test cases.
    """
    data_root = Path(data_root)
    images_dir = data_root / "images"
    seg_dir    = data_root / "segmentations" / "expert"

    # Load official train/test split if provided
    # CSV format: two columns — train_split, test_split — each row has one train and one test patient ID
    split_ids = None
    if split_csv:
        df = pd.read_csv(split_csv)
        col = "train_split" if split == "train" else "test_split"
        split_ids = set(df[col].dropna().astype(str).str.lower())

    cases = []
    for patient_folder in sorted(images_dir.iterdir()):
        if not patient_folder.is_dir():
            continue

        folder_name = patient_folder.name          # e.g. DUKE_001
        collection  = folder_name.split("_")[0]    # e.g. DUKE
        patient_id  = folder_name.lower()          # e.g. duke_001

        if collection not in COLLECTIONS:
            continue

        # Filter by split
        if split_ids is not None and patient_id not in split_ids:
            continue

        # Post-contrast phase 1 as input (_0001)
        # Support both .nii.gz (local) and .nii (Kaggle unzips to .nii)
        image_path = patient_folder / f"{patient_id}_0001.nii.gz"
        if not image_path.exists():
            image_path = patient_folder / f"{patient_id}_0001.nii"

        seg_path = seg_dir / f"{patient_id}.nii.gz"
        if not seg_path.exists():
            seg_path = seg_dir / f"{patient_id}.nii"

        if not image_path.exists() or not seg_path.exists():
            continue

        cases.append({
            "image":       str(image_path),
            "label":       str(seg_path),
            "patient_id":  patient_id,
            "collection":  collection,
            "client_id":   CLIENT_MAP[collection],
        })

    return cases


# ─────────────────────────────────────────────
# 2. TRANSFORMS
# ─────────────────────────────────────────────

PATCH_SIZE = (128, 128, 64)   # spatial patch fed to the model
POS_NEG_RATIO = 1             # 1:1 — half patches centered on tumor, half random


def get_train_transforms():
    return Compose([
        # Load NIfTI files from disk
        LoadImaged(keys=["image", "label"]),

        # Add channel dim: (H,W,D) → (1,H,W,D)
        EnsureChannelFirstd(keys=["image", "label"]),

        # Reorient all scans to RAS standard axes
        Orientationd(keys=["image", "label"], axcodes="RAS"),

        # Resample to 1x1x1 mm isotropic spacing
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),   # bilinear for image, nearest for mask
        ),

        # Clip to 1st-99th percentile then scale to [0, 1]
        ScaleIntensityRangePercentilesd(
            keys=["image"],
            lower=1, upper=99,
            b_min=0.0, b_max=1.0,
            clip=True,
        ),

        # Remove black background — crop tight around non-zero voxels
        CropForegroundd(keys=["image", "label"], source_key="image"),

        # Ensure volume is at least patch size — some small volumes shrink below 128³ after resampling
        SpatialPadd(keys=["image", "label"], spatial_size=PATCH_SIZE),

        # Random patch: 50% centered on tumor, 50% random background
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=PATCH_SIZE,
            pos=POS_NEG_RATIO,
            neg=POS_NEG_RATIO,
            num_samples=2,      # 2 patches per volume per iteration
            image_key="image",
            image_threshold=0,
        ),

        # Augmentation — simulate scanner variation across sites
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),

        EnsureTyped(keys=["image", "label"]),
    ])


def get_val_transforms():
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRangePercentilesd(
            keys=["image"],
            lower=1, upper=99,
            b_min=0.0, b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        EnsureTyped(keys=["image", "label"]),
    ])


def get_rand_train_transforms():
    """Random-only transforms applied at training time to preprocessed volumes."""
    return Compose([
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=PATCH_SIZE,
            pos=POS_NEG_RATIO,
            neg=POS_NEG_RATIO,
            num_samples=2,
            image_key="image",
            image_threshold=0,
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
        EnsureTyped(keys=["image", "label"]),
    ])


# ─────────────────────────────────────────────
# 3b. PREPROCESSED CACHE DATASET
# ─────────────────────────────────────────────

class PreprocessedDataset(Dataset):
    """
    Loads preprocessed .npz files (float16 image, uint8 label) produced by
    scripts/preprocess_to_cache.py and applies random augmentations on the fly.

    Each __getitem__ returns a list of 2 patch dicts (from RandCrop num_samples=2)
    — same structure CacheDataset returns, so _safe_collate works unchanged.
    """

    def __init__(self, cache_dir: str, cases: list[dict], is_train: bool = True):
        self.cache_dir = Path(cache_dir)
        self.cases     = cases
        self.transform = get_rand_train_transforms() if is_train else None

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        patient_id = self.cases[idx]["patient_id"]
        npz_path   = self.cache_dir / f"{patient_id}.npz"

        data_npz = np.load(npz_path)
        item = {
            "image": torch.from_numpy(data_npz["image"].astype(np.float32)),
            "label": torch.from_numpy(data_npz["label"].astype(np.float32)),
        }

        if self.transform is not None:
            item = self.transform(item)

        return item


# ─────────────────────────────────────────────
# 3. DATASET BUILDERS
# ─────────────────────────────────────────────

def build_centralized_loaders(
    data_root: str = DATA_ROOT,
    split_csv: str = None,
    cache_rate: float = 0.0,
    num_workers: int = 2,
    batch_size: int = 2,
    max_cases: int = None,
    persistent_cache_dir: str = "",
    preprocessed_cache_dir: str = "",
):
    """
    Returns (train_loader, val_loader) using all collections combined.

    preprocessed_cache_dir: path to .npz cache built by scripts/preprocess_to_cache.py.
      Fastest option — skips all NIfTI loading and resampling on Kaggle.
      Expected layout: <cache_dir>/train/{patient_id}.npz, <cache_dir>/val/{patient_id}.npz
    persistent_cache_dir: MONAI PersistentDataset cache (preprocess once to disk per session).
    cache_rate=0.0: no caching — slowest but works anywhere.
    """
    train_cases = discover_cases(data_root, split_csv, split="train")
    val_cases   = discover_cases(data_root, split_csv, split="test")

    if max_cases is not None:
        train_cases = train_cases[:max_cases]
        val_cases   = val_cases[:max_cases]

    print(f"Train cases: {len(train_cases)} | Val cases: {len(val_cases)}")

    if preprocessed_cache_dir:
        train_cache = os.path.join(preprocessed_cache_dir, "train")
        val_cache   = os.path.join(preprocessed_cache_dir, "val")
        print(f"Using preprocessed .npz cache: {preprocessed_cache_dir}")
        train_ds = PreprocessedDataset(train_cache, train_cases, is_train=True)
        val_ds   = PreprocessedDataset(val_cache,   val_cases,   is_train=False)
    elif persistent_cache_dir:
        train_cache = os.path.join(persistent_cache_dir, "train")
        val_cache   = os.path.join(persistent_cache_dir, "val")
        os.makedirs(train_cache, exist_ok=True)
        os.makedirs(val_cache,   exist_ok=True)
        print(f"Using PersistentDataset cache: {persistent_cache_dir}")
        train_ds = PersistentDataset(
            data=train_cases,
            transform=get_train_transforms(),
            cache_dir=train_cache,
        )
        val_ds = PersistentDataset(
            data=val_cases,
            transform=get_val_transforms(),
            cache_dir=val_cache,
        )
    else:
        train_ds = CacheDataset(
            data=train_cases,
            transform=get_train_transforms(),
            cache_rate=cache_rate,
            num_workers=num_workers,
        )
        val_ds = CacheDataset(
            data=val_cases,
            transform=get_val_transforms(),
            cache_rate=cache_rate,
            num_workers=num_workers,
        )

    train_loader = MonaiLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, collate_fn=_safe_collate,
                               pin_memory=True, persistent_workers=num_workers > 0)
    val_loader   = MonaiLoader(val_ds,   batch_size=1,          shuffle=False,
                               num_workers=num_workers, collate_fn=_safe_collate,
                               pin_memory=True, persistent_workers=num_workers > 0)

    return train_loader, val_loader


def build_client_loaders(
    client_id: int,
    data_root: str = DATA_ROOT,
    split_csv: str = None,
    cache_rate: float = 0.0,
    num_workers: int = 2,
    batch_size: int = 2,
):
    """
    Returns (train_loader, val_loader) for a single FL client (one collection).
    """
    collection = [k for k, v in CLIENT_MAP.items() if v == client_id][0]

    train_cases = [c for c in discover_cases(data_root, split_csv, split="train")
                   if c["collection"] == collection]
    val_cases   = [c for c in discover_cases(data_root, split_csv, split="test")
                   if c["collection"] == collection]

    print(f"Client {client_id} ({collection}) — Train: {len(train_cases)} | Val: {len(val_cases)}")

    train_ds = CacheDataset(data=train_cases, transform=get_train_transforms(),
                            cache_rate=cache_rate, num_workers=num_workers)
    val_ds   = CacheDataset(data=val_cases,   transform=get_val_transforms(),
                            cache_rate=cache_rate, num_workers=num_workers)

    train_loader = MonaiLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = MonaiLoader(val_ds,   batch_size=1,          shuffle=False, num_workers=num_workers)

    return train_loader, val_loader


# ─────────────────────────────────────────────
# 4. SMOKE TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import time

    print("=" * 50)
    print("Discovering cases...")
    cases = discover_cases(DATA_ROOT)
    print(f"Total cases found: {len(cases)}")

    by_collection = {}
    for c in cases:
        by_collection.setdefault(c["collection"], 0)
        by_collection[c["collection"]] += 1
    for col, count in sorted(by_collection.items()):
        print(f"  {col}: {count} cases")

    print("\nTesting transforms on 1 case (CPU, no cache)...")
    from monai.data import CacheDataset
    from monai.data import DataLoader as MonaiLoader

    test_cases = cases[:2]
    ds = CacheDataset(data=test_cases, transform=get_train_transforms(),
                      cache_rate=0.0, num_workers=0)

    t0 = time.time()
    sample = ds[0]
    elapsed = time.time() - t0

    print(f"  Time to load + preprocess 1 case: {elapsed:.1f}s")
    print(f"  Patches returned: {len(sample)}")
    for i, patch in enumerate(sample):
        print(f"  Patch {i} — image: {patch['image'].shape} | label: {patch['label'].shape} "
              f"| range: [{patch['image'].min():.3f}, {patch['image'].max():.3f}] "
              f"| tumor voxels: {int(patch['label'].sum())}")
    print("\nSmoke test passed.")
