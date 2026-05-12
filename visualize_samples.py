import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

DATA_ROOT = r"C:\Users\bhara\Desktop\MAMA_MIA_COMPLETE"

SAMPLES = [
    {
        "name": "DUKE_001",
        "collection": "DUKE",
        "orientation": "Axial",
        "scanner": "1.5T/3T Siemens/GE",
        "image": os.path.join(DATA_ROOT, "images", "DUKE_001", "duke_001_0001.nii.gz"),
        "seg":   os.path.join(DATA_ROOT, "segmentations", "expert", "duke_001.nii.gz"),
    },
    {
        "name": "ISPY1_1001",
        "collection": "ISPY1",
        "orientation": "Sagittal",
        "scanner": "1.5T GE/Siemens/Philips",
        "image": os.path.join(DATA_ROOT, "images", "ISPY1_1001", "ispy1_1001_0001.nii.gz"),
        "seg":   os.path.join(DATA_ROOT, "segmentations", "expert", "ispy1_1001.nii.gz"),
    },
    {
        "name": "ISPY2_100899",
        "collection": "ISPY2",
        "orientation": "Axial",
        "scanner": "1.5T/3T GE/Siemens/Philips",
        "image": os.path.join(DATA_ROOT, "images", "ISPY2_100899", "ispy2_100899_0001.nii.gz"),
        "seg":   os.path.join(DATA_ROOT, "segmentations", "expert", "ispy2_100899.nii.gz"),
    },
    {
        "name": "NACT_01",
        "collection": "NACT",
        "orientation": "Sagittal",
        "scanner": "1.5T GE",
        "image": os.path.join(DATA_ROOT, "images", "NACT_01", "nact_01_0001.nii.gz"),
        "seg":   os.path.join(DATA_ROOT, "segmentations", "expert", "nact_01.nii.gz"),
    },
]


def load_nifti(path):
    nii = nib.load(path)
    data = nii.get_fdata()
    spacing = nii.header.get_zooms()[:3]
    return data, spacing


def normalize(img):
    p1, p99 = np.percentile(img, 1), np.percentile(img, 99)
    img = np.clip(img, p1, p99)
    img = (img - p1) / (p99 - p1 + 1e-8)
    return img


def find_tumor_center(seg):
    """Find the slice index with most tumor voxels in each dimension."""
    centers = []
    for dim in range(3):
        counts = seg.sum(axis=tuple(i for i in range(3) if i != dim))
        if counts.max() > 0:
            centers.append(int(np.argmax(counts)))
        else:
            centers.append(seg.shape[dim] // 2)
    return centers  # [ax0_center, ax1_center, ax2_center]


def overlay_seg(img_slice, seg_slice, alpha=0.4):
    """Return RGB image with red segmentation overlay."""
    rgb = np.stack([img_slice, img_slice, img_slice], axis=-1)
    mask = seg_slice > 0
    rgb[mask, 0] = np.clip(rgb[mask, 0] + alpha, 0, 1)
    rgb[mask, 1] = np.clip(rgb[mask, 1] - alpha * 0.5, 0, 1)
    rgb[mask, 2] = np.clip(rgb[mask, 2] - alpha * 0.5, 0, 1)
    return rgb


def visualize_sample(sample, ax_row, fig):
    img, spacing = load_nifti(sample["image"])
    seg, _ = load_nifti(sample["seg"])

    img = normalize(img)
    centers = find_tumor_center(seg)

    # Three views: dim0 slice, dim1 slice, dim2 slice
    views = [
        (img[centers[0], :, :],  seg[centers[0], :, :],  "View along axis 0"),
        (img[:, centers[1], :],  seg[:, centers[1], :],  "View along axis 1"),
        (img[:, :, centers[2]],  seg[:, :, centers[2]],  "View along axis 2"),
    ]

    has_tumor = seg.max() > 0
    tumor_vol = int(seg.sum())

    for col, (img_slice, seg_slice, view_label) in enumerate(views):
        ax = ax_row[col]
        img_slice = np.rot90(img_slice)
        seg_slice = np.rot90(seg_slice)

        if has_tumor:
            rgb = overlay_seg(img_slice, seg_slice)
            ax.imshow(rgb, cmap=None, aspect='equal')
        else:
            ax.imshow(img_slice, cmap='gray', aspect='equal')

        ax.set_title(view_label, fontsize=8, pad=2)
        ax.axis('off')

    # Label the row
    ax_row[0].set_ylabel(
        f"{sample['name']}\n{sample['collection']} | {sample['orientation']}\n"
        f"Shape: {img.shape} | Spacing: {tuple(round(s,2) for s in spacing)}\n"
        f"Tumor voxels: {tumor_vol}",
        fontsize=7, rotation=0, labelpad=120, va='center'
    )


def main():
    n_rows = len(SAMPLES)
    fig, axes = plt.subplots(n_rows, 3, figsize=(12, n_rows * 4))
    fig.suptitle("MAMA-MIA Dataset — One Sample Per Collection\n(Post-contrast phase 1, red = expert tumor segmentation)",
                 fontsize=12, fontweight='bold', y=1.01)

    for i, sample in enumerate(SAMPLES):
        print(f"Loading {sample['name']}...")
        visualize_sample(sample, axes[i], fig)

    patch = mpatches.Patch(color='red', alpha=0.7, label='Expert tumor segmentation')
    fig.legend(handles=[patch], loc='lower right', fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "outputs", "dataset_samples.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved to: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
