#!/usr/bin/env python3
# Before/After RandCropByPosNegLabeld (headless, RAM-light, return-type safe)

import argparse
from pathlib import Path
import gc
import numpy as np

# --- force non-GUI to avoid ^C/KeyboardInterrupt from plt.show() ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    RandCropByPosNegLabeld, Orientationd, Spacingd
)

def show_overlay(ax, img2d, lbl2d, title):
    ax.imshow(img2d, cmap="gray", interpolation="nearest")
    if lbl2d is not None:
        ax.imshow((lbl2d > 0).astype(np.float32), alpha=0.35, cmap="autumn", interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.axis("off")

def normalize_crops(cropped, keys=("image","label")):
    """
    Normalize MONAI RandCropByPosNegLabeld output to list[dict].
    Supports:
      - list[dict] (already normalized)
      - dict[str, list]  (convert to list[dict])
    """
    # Case 1: already list[dict]
    if isinstance(cropped, (list, tuple)):
        # Ensure each element is a dict with the expected keys
        if len(cropped) > 0 and isinstance(cropped[0], dict):
            return list(cropped)
        else:
            raise TypeError(f"Got list/tuple but elements aren't dicts: elem type {type(cropped[0]) if cropped else 'EMPTY'}")

    # Case 2: dict[str, list]
    if isinstance(cropped, dict):
        # infer length from any key
        example_key = next(iter(cropped))
        n = len(cropped[example_key])
        out = []
        for i in range(n):
            item = {k: cropped[k][i] for k in keys if k in cropped}
            out.append(item)
        return out

    raise TypeError(f"Unexpected crop return type: {type(cropped)}")

def main():
    ap = argparse.ArgumentParser(description="Before/After for RandCropByPosNegLabeld (lightweight & robust)")
    ap.add_argument("--image", required=True, help="image NIfTI path")
    ap.add_argument("--label", required=True, help="label NIfTI path")
    ap.add_argument("--roi_x", type=int, default=96)
    ap.add_argument("--roi_y", type=int, default=96)
    ap.add_argument("--roi_z", type=int, default=96)
    ap.add_argument("--num_samples", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("before_after_randcrop_light.png"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reorient", action="store_true", help="Apply Orientation+Spacing (slightly more RAM/CPU).")
    ap.add_argument("--pixdim", type=float, nargs=3, default=[1.0, 1.0, 1.0], help="Spacing (x y z) if --reorient.")
    ap.add_argument("--no_float16", action="store_true", help="Keep float32 image (default: downcast to float16).")
    args = ap.parse_args()

    data = {"image": args.image, "label": args.label}

    # Minimal, memory-conscious pipeline
    pre = [
        LoadImaged(keys=["image", "label"], image_only=False),
        EnsureChannelFirstd(keys=["image", "label"]),
    ]
    if args.reorient:
        pre += [
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=args.pixdim, mode=("bilinear", "nearest")),
        ]
    pre = Compose(pre)

    item = pre(data)  # dict with numpy arrays (C, Z, Y, X)

    # Downcast to save RAM
    img = item["image"]
    lbl = item["label"]
    if img.dtype != np.float16 and not args.no_float16:
        img = img.astype(np.float16, copy=False)
    if lbl.dtype != np.uint8:
        lbl = (lbl > 0).astype(np.uint8, copy=False)

    # Pick an axial slice with label content if possible
    zsum = lbl[0].sum(axis=(1, 2))
    if (zsum > 0).any():
        nz = np.where(zsum > 0)[0]
        z = int(nz[len(nz) // 2])
    else:
        z = img.shape[1] // 2

    # Sampler (deterministic)
    cropper = RandCropByPosNegLabeld(
        keys=["image", "label"],
        label_key="label",
        spatial_size=(args.roi_x, args.roi_y, args.roi_z),
        pos=1, neg=1,
        num_samples=args.num_samples,
        image_key="image",
        image_threshold=0,
    )
    cropper.set_random_state(seed=args.seed)
    raw_crops = cropper({"image": img, "label": lbl})
    crops = normalize_crops(raw_crops, keys=("image","label"))

    # --- Plot (no plt.show) ---
    n = min(args.num_samples, len(crops))
    rows, cols = ((2, 3) if n <= 4 else (((n + 1) + 2) // 3, 3))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 + 3 * (rows - 1)))
    axes = np.atleast_1d(axes).ravel()

    # Original slice
    show_overlay(axes[0], img[0, z], lbl[0, z], f"Original (z={z})")

    # Crops (center z per crop)
    for i in range(n):
        cimg = crops[i]["image"]  # (C, Z, Y, X)
        clbl = crops[i]["label"]
        # ensure numpy
        cimg = np.asarray(cimg)
        clbl = np.asarray(clbl)
        cz = cimg.shape[1] // 2
        show_overlay(axes[i + 1], cimg[0, cz], clbl[0, cz], f"Crop {i+1} (cz={cz})")

    # Hide extra axes
    for j in range(n + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    fig.suptitle("Before / After RandCropByPosNegLabeld (lightweight, headless-safe)", y=1.02)
    plt.subplots_adjust(top=0.9)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")

    # Free memory
    del img, lbl, crops, raw_crops, item
    gc.collect()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting cleanly.")
