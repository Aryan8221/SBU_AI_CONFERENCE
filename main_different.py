#!/usr/bin/env python3
import os
import math
import argparse
import random
from pathlib import Path
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

from monai.data import Dataset, DataLoader, decollate_batch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped,
    ConcatItemsd, RandFlipd, RandRotate90d, RandAffined,
    RandGaussianNoised, RandGaussianSmoothd, RandScaleIntensityd,
    RandShiftIntensityd, Rand3DElasticd, RandBiasFieldd,
    RandCropByPosNegLabeld, SpatialPadd, Orientationd, Spacingd
)
from monai.networks.nets import UNet
from monai.metrics import DiceMetric
from monai.metrics import DiceMetric, ConfusionMatrixMetric
from monai.transforms import AsDiscrete
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism

# -----------------------
# Utilities
# -----------------------
def make_cases(root: Path, indices: List[int]) -> List[Dict]:
    """Builds a list of dicts with paths to PET/CT/label for given indices."""
    cases = []
    for i in indices:
        pet_p = root / "PET" / f"PT{i}.nii.gz"
        ct_p  = root / "CT"  / f"CT{i}.nii.gz"
        lbl_p = root / "LABELS" / f"{i}_seg.nii.gz"
        if not (pet_p.exists() and ct_p.exists() and lbl_p.exists()):
            raise FileNotFoundError(f"Missing files for index {i}: "
                                    f"{pet_p.exists()=}, {ct_p.exists()=}, {lbl_p.exists()=}")
        cases.append({"pet": str(pet_p), "ct": str(ct_p), "label": str(lbl_p), "idx": i})
    return cases


def kfold_split(indices: List[int], num_folds: int, seed: int = 42) -> List[Tuple[List[int], List[int]]]:
    """Return list of (train_idx, val_idx) splits."""
    rng = random.Random(seed)
    idxs = indices[:]
    rng.shuffle(idxs)
    folds = []
    fold_size = math.ceil(len(idxs) / num_folds)
    for k in range(num_folds):
        val_idx = idxs[k*fold_size:(k+1)*fold_size]
        train_idx = [x for x in idxs if x not in val_idx]
        if not val_idx:
            continue
        folds.append((train_idx, val_idx))
    return folds


def build_transforms(roi: Tuple[int,int,int], spacing: Tuple[float,float,float] = None):
    """
    Build MONAI transforms.
    - We assume images are already oriented/resampled. If you need to enforce spacing/orientation again,
      uncomment Orientationd/Spacingd lines below and set 'spacing'.
    """
    # Common pre-transform (load + channel-first + concat PET/CT -> 'image')
    base = [
        LoadImaged(keys=["pet", "ct", "label"]),
        EnsureChannelFirstd(keys=["pet", "ct", "label"]),
        # If needed to enforce orientation/spacing again, uncomment:
        # Orientationd(keys=["pet", "ct", "label"], axcodes="RAS"),
        # Spacingd(keys=["pet", "ct"], pixdim=spacing, mode=("bilinear","bilinear")),
        # Spacingd(keys=["label"], pixdim=spacing, mode="nearest"),
        ConcatItemsd(keys=["pet", "ct"], name="image", dim=0),  # 2-channel input
        EnsureTyped(keys=["image", "label"]),
    ]

    # Data augmentation for training
    train_aug = [
        SpatialPadd(keys=["image", "label"], spatial_size=roi),  # pad small Z if needed
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=roi,
            pos=2, neg=1,  # bias toward positives
            num_samples=2,
            image_key="image",
            image_threshold=0.0,
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.2, max_k=3, spatial_axes=(0,1)),
        RandAffined(
            keys=["image", "label"],
            prob=0.20,
            rotate_range=(0.1, 0.1, 0.05),
            scale_range=(0.1, 0.1, 0.0),
            mode=("bilinear","nearest"),
        ),
        Rand3DElasticd(
            keys=["image", "label"],
            prob=0.10,
            sigma_range=(5, 7),  # Gaussian smoothing of the displacement field (voxels)
            magnitude_range=(50, 150),  # displacement magnitude (larger => more warp)
            rotate_range=(0.0, 0.0, 0.0),
            translate_range=(0.0, 0.0, 0.0),
            scale_range=(0.0, 0.0, 0.0),
            mode=("bilinear", "nearest"),
            padding_mode="zeros",
        ),
        RandBiasFieldd(keys=["image"], prob=0.15),
        RandGaussianNoised(keys=["image"], prob=0.10, mean=0.0, std=0.01),
        RandGaussianSmoothd(keys=["image"], prob=0.10, sigma_x=(0.5,1.5), sigma_y=(0.5,1.5), sigma_z=(0.0,0.5)),
        RandScaleIntensityd(keys=["image"], factors=0.10, prob=0.25),
        RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.25),
    ]

    # Validation transforms: just load+concat, maybe pad to ROI for sliding window convenience
    val_aug = [
        SpatialPadd(keys=["image", "label"], spatial_size=roi),
    ]

    return Compose(base + train_aug), Compose(base + val_aug)


def create_loaders(data_root: Path, train_ids: List[int], val_ids: List[int],
                   roi: Tuple[int,int,int], batch_size: int, num_workers: int):
    train_files = make_cases(data_root, train_ids)
    val_files   = make_cases(data_root, val_ids)

    train_t, val_t = build_transforms(roi=roi)

    train_ds = Dataset(data=train_files, transform=train_t)
    val_ds   = Dataset(data=val_files, transform=val_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


def build_model() -> nn.Module:
    # Standard 3D UNet (small) — tweak channels/depths if you have more VRAM
    model = UNet(
        spatial_dims=3,
        in_channels=2,          # PET+CT
        out_channels=2,         # background/foreground
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm="INSTANCE",
        dropout=0.0,
    )
    return model


# -----------------------
# Training / Validation
# -----------------------
def validate(model, loader, device, roi, amp: bool):
    model.eval()
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    dice_perclass = DiceMetric(include_background=True, reduction="none")

    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(device)
            seg = batch["label"].to(device)

            with autocast(enabled=amp):
                logits = sliding_window_inference(
                    inputs=img, roi_size=roi, sw_batch_size=1, predictor=model, overlap=0.25, mode="gaussian"
                )
                probs = torch.softmax(logits, dim=1)

            # decollate to per-sample tensors
            seg_list  = decollate_batch(seg)    # each: [B,1,D,H,W] or [1,D,H,W]
            pred_list = decollate_batch(probs)  # each: [2,D,H,W]

            # make one-hot GT for Dice
            seg_ohe = []
            for s in seg_list:
                s_ = s
                if s_.ndim == 5:   # [B,1,D,H,W]
                    s_ = s_.squeeze(0)
                if s_.shape[0] != 1:
                    s_ = s_.unsqueeze(0)
                s_idx = s_.squeeze(0).long()  # [D,H,W]
                seg_ohe.append(
                    torch.nn.functional.one_hot(s_idx, num_classes=2).permute(3,0,1,2).float()
                )  # [2,D,H,W]

            dice_metric(y_pred=pred_list, y=seg_ohe)
            dice_perclass(y_pred=pred_list, y=seg_ohe)

    mean_dice = dice_metric.aggregate().item()
    perclass_dice = dice_perclass.aggregate().cpu().numpy().tolist()

    dice_metric.reset(); dice_perclass.reset()
    return mean_dice, perclass_dice



def train_one_fold(fold_id: int, data_root: Path, train_ids: List[int], val_ids: List[int],
                   args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    fold_dir = Path(args.out_dir) / f"fold{fold_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    # ROI choice: balanced patch size for 168x168 in-plane; adjust depth to your VRAM
    roi = (args.roi_x, args.roi_y, args.roi_z)

    train_loader, val_loader = create_loaders(
        data_root=data_root,
        train_ids=train_ids, val_ids=val_ids,
        roi=roi,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )

    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = GradScaler(enabled=args.amp)

    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True, include_background=True, lambda_dice=1.0, lambda_ce=1.0)

    best_dice = -1.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            img = batch["image"].to(device)   # [B,2,D,H,W]
            seg = batch["label"].to(device)   # [B,1,D,H,W] (integer mask)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=args.amp):
                logits = model(img)           # [B,2,D,H,W]
                loss = loss_fn(logits, seg)

            if args.amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= max(1, len(train_loader))

        # Validate
        val_dice, val_dice_pc = validate(model, val_loader, device, roi, amp=args.amp)
        print(f"[Fold {fold_id}] Epoch {epoch}/{args.epochs} | "
              f"train_loss={epoch_loss:.4f} | val_dice={val_dice:.4f} | "
              f"per-class dice={val_dice_pc}")

        # Checkpointing
        if val_dice > best_dice:
            best_dice = val_dice
            patience_counter = 0
            ckpt_path = fold_dir / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_dice": best_dice,
                "roi": roi,
            }, ckpt_path)
            print(f"  -> Saved best checkpoint: {ckpt_path} (dice={best_dice:.4f})")
        else:
            patience_counter += 1

        if args.early_stop > 0 and patience_counter >= args.early_stop:
            print(f"Early stopping at epoch {epoch} (no improvement for {patience_counter} epochs).")
            break

    return best_dice


# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser("PET/CT segmentation with K-fold CV (indices 1..11 only)")
    parser.add_argument("--data_root", type=str, required=True, help="Root folder containing PET/, CT/, LABELS/")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for folds/checkpoints")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds (<= 11)")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--roi_x", type=int, default=160)
    parser.add_argument("--roi_y", type=int, default=160)
    parser.add_argument("--roi_z", type=int, default=64)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--early_stop", type=int, default=20, help="Stop if no val Dice improvement for N epochs (0=disable)")
    args = parser.parse_args()

    set_determinism(seed=args.seed)

    data_root = Path(args.data_root)
    # Use ONLY indices 1..11
    all_indices = list(range(1, 12))
    splits = kfold_split(all_indices, num_folds=args.folds, seed=args.seed)

    print(f"Running {len(splits)} folds on indices {all_indices}")
    best_per_fold = []
    for fold_id, (train_ids, val_ids) in enumerate(splits, start=1):
        print(f"\n=== Fold {fold_id}/{len(splits)} ===")
        print(f"Train: {sorted(train_ids)} | Val: {sorted(val_ids)}")
        best = train_one_fold(fold_id, data_root, train_ids, val_ids, args)
        best_per_fold.append(best)

    print("\nCV summary (Dice):")
    for i, d in enumerate(best_per_fold, start=1):
        print(f"  Fold {i}: {d:.4f}")
    if best_per_fold:
        print(f"  Mean: {sum(best_per_fold)/len(best_per_fold):.4f}")


if __name__ == "__main__":
    main()
