# save as sanity_check_labels_as_preds.py
# Standalone script: builds the validation loader from a MONAI Decathlon-style JSON,
# then computes a "sanity Dice" by feeding the labels as predictions.

import argparse
import torch
from monai import data, transforms
from monai.data import load_decathlon_datalist
from monai.data.utils import no_collation
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.utils.enums import MetricReduction
from monai.data import decollate_batch


def build_val_loader(data_dir, json_list, workers):
    """
    Build a MONAI validation DataLoader from a Decathlon-style JSON split.
    Expects the JSON to have a "validation" section with dicts containing "image" and "label" keys.
    """
    datalist_json = f"{data_dir.rstrip('/')}/{json_list}"
    val_files = load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)

    # Minimal, label-preserving transforms (no random augs). Keep channel-first & types.
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"], image_only=True),
            transforms.EnsureChannelFirstd(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.ToTensord(keys=["image", "label"]),
            transforms.EnsureTyped(keys=["image"], dtype=torch.float32, track_meta=False),
            transforms.EnsureTyped(keys=["label"], dtype=torch.long, track_meta=False),
        ]
    )

    val_ds = data.Dataset(data=val_files, transform=val_transform)

    # normal collation is fine; using default batch_size=1 ensures decollate consistency
    val_loader = data.DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )
    return val_loader


def sanity_check_label_as_pred(val_loader, out_channels, include_background=False, device=None):
    """
    Feed ground-truth labels as predictions to verify metric pipeline.
    Returns a single scalar Dice (mean over classes and batch).
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    dice_metric = DiceMetric(
        include_background=include_background,
        reduction=MetricReduction.MEAN,  # mean over classes and batch
        get_not_nans=True,
    )
    post_label = AsDiscrete(to_onehot=out_channels)

    dice_metric.reset()
    with torch.no_grad():
        for batch in val_loader:
            # batch is a dict with "label" (B, D, H, W) or (B, 1, D, H, W) depending on transforms
            target = batch["label"].to(device)

            # decollate to list of tensors (one per sample), then one-hot
            labels_list = decollate_batch(target)
            labels_onehot = [post_label(x) for x in labels_list]

            # perfect prediction: y_pred == y
            dice_metric(y_pred=labels_onehot, y=labels_onehot)

    acc, _ = dice_metric.aggregate()  # tensor of per-class Dice (already reduced over batch)
    return float(acc.mean().cpu())    # single scalar


def parse_args():
    p = argparse.ArgumentParser(description="Sanity-check Dice by passing labels as predictions (standalone).")
    p.add_argument("--data_dir", type=str, required=True, help="Root dataset directory")
    p.add_argument("--json_list", type=str, required=True, help="Datalist JSON path relative to data_dir")
    p.add_argument("--out_channels", type=int, required=True, help="Number of classes (including background)")
    p.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    p.add_argument("--gpu", type=int, default=0, help="CUDA device index (if available)")
    p.add_argument("--include_background", action="store_true", help="Include background in Dice (default: False)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Build val loader
    val_loader = build_val_loader(args.data_dir, args.json_list, args.workers)

    # Select device
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    # Run sanity check
    val_acc = sanity_check_label_as_pred(
        val_loader=val_loader,
        out_channels=args.out_channels,
        include_background=args.include_background,
        device=device,
    )

    print(f"Sanity Dice (labels used as predictions) -> {val_acc:.6f}")
