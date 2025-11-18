# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import os
import shutil
import time

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from torch.cuda.amp import GradScaler, autocast
from monai.inferers import sliding_window_inference
from utils import AverageMeter
import SimpleITK as sitk
from pathlib import Path
import nibabel as nib
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    ResampleToMatchd,
)

from monai.data import decollate_batch

def _to_tensor(x):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    return x.long()


def compute_segmentation_metrics(
    y_pred,
    y_true,
    num_classes=None,
    ignore_index=None,
    eps=1e-8,
):
    """
    y_pred, y_true: integer label maps, same shape (3D or 4D with batch).
    """
    y_pred = _to_tensor(y_pred)
    y_true = _to_tensor(y_true)

    y_pred_flat = y_pred.view(-1)
    y_true_flat = y_true.view(-1)

    if num_classes is None:
        num_classes = int(max(y_pred_flat.max(), y_true_flat.max()).item() + 1)

    device = y_pred_flat.device

    tp = torch.zeros(num_classes, device=device, dtype=torch.float64)
    fp = torch.zeros(num_classes, device=device, dtype=torch.float64)
    fn = torch.zeros(num_classes, device=device, dtype=torch.float64)
    tn = torch.zeros(num_classes, device=device, dtype=torch.float64)

    for c in range(num_classes):
        pred_c = (y_pred_flat == c)
        true_c = (y_true_flat == c)

        tp[c] = (pred_c & true_c).sum()
        fp[c] = (pred_c & ~true_c).sum()
        fn[c] = (~pred_c & true_c).sum()
        tn[c] = (~pred_c & ~true_c).sum()

    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = (2 * precision * recall + eps) / (precision + recall + eps)

    mask = torch.ones(num_classes, dtype=torch.bool, device=device)
    if ignore_index is not None and 0 <= ignore_index < num_classes:
        mask[ignore_index] = False

    return {
        "per_class": {
            "dice": dice.cpu().numpy(),
            "iou": iou.cpu().numpy(),
            "precision": precision.cpu().numpy(),
            "recall": recall.cpu().numpy(),
            "f1": f1.cpu().numpy(),
        },
        "mean_over_classes": {
            "dice": dice[mask].mean().item(),
            "iou": iou[mask].mean().item(),
            "precision": precision[mask].mean().item(),
            "recall": recall[mask].mean().item(),
            "f1": f1[mask].mean().item(),
        },
    }

def compute_metrics_pred_to_label_from_paths(label_path, pred_path, num_classes, ignore_index=0):
    """
    Load prediction & label from disk, resample prediction -> label space,
    then compute segmentation metrics.
    """
    xforms = Compose([
        LoadImaged(keys=["label", "pred"]),
        EnsureChannelFirstd(keys=["label", "pred"]),
        Orientationd(keys=["label", "pred"], axcodes="RAS"),
        ResampleToMatchd(keys="pred", key_dst="label", mode="nearest"),
    ])

    data = xforms({"label": label_path, "pred": pred_path})
    # [1, Z, Y, X] -> [Z, Y, X]
    label = data["label"][0].cpu().numpy()
    pred  = data["pred"][0].cpu().numpy()

    metrics = compute_segmentation_metrics(
        y_pred=pred,
        y_true=label,
        num_classes=num_classes,
        ignore_index=ignore_index,
    )
    return metrics


def save_val_nifti(logits, epoch, idx, args, batch_data):
    # logits: [B, C, Z, Y, X]
    pred = torch.argmax(logits, dim=1).detach().cpu().numpy()

    out_dir = Path(args.logdir) / "val_preds"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"PREDICTION NIFTI: {pred.shape}")

    B = pred.shape[0]
    meta_all = batch_data["image_meta_dict"]

    out_paths = []

    for b in range(B):
        pred_arr = pred[b].astype(np.uint8)

        # handle batch_size=1 vs >1
        if isinstance(meta_all, (list, tuple)):
            meta = meta_all[b]
        else:
            meta = meta_all

        affine = meta.get("original_affine", None)
        if affine is None:
            affine = meta.get("affine", None)

        if torch.is_tensor(affine):
            affine = affine.cpu().numpy()

        if affine.ndim == 3:  # very defensive, if shape like (1,4,4) / (B,4,4)
            affine = affine[b]

        src_path = meta.get("filename_or_obj", f"case_{idx:03d}_b{b}")
        stem = Path(str(src_path)).stem

        nii = nib.Nifti1Image(pred_arr, affine)
        out_path = out_dir / f"{stem}_ep{epoch + 1:04d}_b{b}.nii.gz"
        nib.save(nii, out_path)
        out_paths.append(str(out_path))

    return out_paths


def train_epoch(model, loader, optimizer, scaler, epoch, loss_func, args, logger=None, model_inferrer=None):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    for idx, batch_data in enumerate(loader):
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data["image"], batch_data["label"]
        data, target = data.to(device), target.to(device)
        for param in model.parameters():
            param.grad = None
        with autocast(enabled=args.amp):
            # print(f"DATA SHAPE: {data.shape}")
            logits = model(data)
            print(f"DATA SHAPE TRAIN: {logits.shape}")
            loss = loss_func(logits, target)
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        run_loss.update(loss.item(), n=args.batch_size)

        logger.info(
            f"Epoch {epoch + 1}/{args.max_epochs} {idx + 1}/{len(loader)}\tloss: {run_loss.avg:.4f}\ttime {time.time() - start_time:.2f}s"
        )
        start_time = time.time()
    for param in model.parameters():
        param.grad = None
    return run_loss.avg


def val_epoch(
    model,
    loader,
    epoch,
    acc_func,          # not used anymore, but kept for signature compatibility
    args,
    post_label=None,
    post_pred=None,
    logger=None,
    model_inferrer=None
):
    model.eval()
    run_acc = AverageMeter()  # we’ll store mean Dice over classes
    start_time = time.time()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    num_classes = args.out_channels       # make sure this exists in args
    ignore_index = 0                     # change if your background is different

    global_case_idx = 0  # just a running index for nicer logs

    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data["image"], batch_data["label"]

            data = data.to(device)

            with autocast(enabled=args.amp):
                if model_inferrer is not None:
                    logits = model_inferrer(data)
                    print(f"DATA SHAPE VAL: {logits.shape}")
                else:
                    logits = model(data)

            # 1) Save predictions as NIfTI and get file paths
            pred_paths = save_val_nifti(
                logits=logits,
                epoch=epoch,
                idx=idx,
                args=args,
                batch_data=batch_data
            )

            # 2) Get corresponding label paths from meta dict
            label_meta = batch_data["label_meta_dict"]
            if isinstance(label_meta, (list, tuple)):
                label_paths = [m["filename_or_obj"] for m in label_meta]
            else:
                label_paths = [label_meta["filename_or_obj"]]

            # 3) For each case in this batch: prediction -> label metrics
            batch_dices = []
            for lp, pp in zip(label_paths, pred_paths):
                metrics_case = compute_metrics_pred_to_label_from_paths(
                    label_path=lp,
                    pred_path=pp,
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                )

                per_class = metrics_case["per_class"]
                mean_over = metrics_case["mean_over_classes"]

                # ---- LOG EVERYTHING HERE ----
                logger.info(
                    f"[VAL][epoch {epoch + 1}] case {global_case_idx} "
                    f"label_path={lp} pred_path={pp}"
                )
                logger.info(
                    f"[VAL][epoch {epoch + 1}] case {global_case_idx} "
                    f"per_class_dice={per_class['dice']} "
                    f"per_class_iou={per_class['iou']}"
                )
                logger.info(
                    f"[VAL][epoch {epoch + 1}] case {global_case_idx} "
                    f"per_class_precision={per_class['precision']} "
                    f"per_class_recall={per_class['recall']} "
                    f"per_class_f1={per_class['f1']}"
                )
                logger.info(
                    f"[VAL][epoch {epoch + 1}] case {global_case_idx} "
                    f"mean_over_classes="
                    f"dice={mean_over['dice']:.4f}, "
                    f"iou={mean_over['iou']:.4f}, "
                    f"precision={mean_over['precision']:.4f}, "
                    f"recall={mean_over['recall']:.4f}, "
                    f"f1={mean_over['f1']:.4f}"
                )
                # ------------------------------

                mean_dice = mean_over["dice"]
                batch_dices.append(mean_dice)

                global_case_idx += 1

            # mean over this batch (cases in the current loader iteration)
            batch_dices = np.array(batch_dices, dtype=np.float32)
            batch_mean = float(batch_dices.mean())

            run_acc.update(batch_mean, n=len(batch_dices))

            logger.info(
                f"Val {epoch + 1}/{args.max_epochs} "
                f"{idx + 1}/{len(loader)}\t"
                f"batch_mean_dice (pred->label): {batch_mean:.4f} "
                f"running_mean_dice: {run_acc.avg:.4f}\t"
                f"time {time.time() - start_time:.2f}s"
            )
            start_time = time.time()

    return run_acc.avg


def save_checkpoint(model, epoch, args, filename="model.pt", best_acc=0, optimizer=None, scheduler=None, logger=None):
    state_dict = model.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    logger.info(f"Saving checkpoint {filename}")


def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    acc_func,
    args,
    model_inferrer=None,
    scheduler=None,
    start_epoch=0,
    post_label=None,
    post_pred=None,
    logger=None
):
    scaler = None
    if args.amp:
        scaler = GradScaler()
    val_acc_max = 0.0
    for epoch in range(start_epoch, args.max_epochs):
        logger.info(f'{time.ctime()}\tEpoch: {epoch}')
        epoch_time = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, scaler=scaler, epoch=epoch, loss_func=loss_func, args=args, logger=logger, model_inferrer=model_inferrer
        )
        logger.info(
            f"Final training  {epoch + 1}/{args.max_epochs}\tloss: {train_loss:.4f}\ttime {time.time() - epoch_time:.2f}s"
        )
        b_new_best = False
        if (epoch + 1) % args.val_every == 0:
            epoch_time = time.time()
            val_avg_acc = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                acc_func=acc_func,
                args=args,
                post_label=post_label,
                post_pred=post_pred,
                logger=logger,
                model_inferrer=model_inferrer
            )

            val_avg_acc = np.mean(val_avg_acc)

            logger.info(
                f"Final validation  {epoch + 1}/{args.max_epochs}\tacc:{val_avg_acc}\ttime {time.time() - epoch_time:.2f}s"
            )
            if val_avg_acc > val_acc_max:
                logger.info("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                val_acc_max = val_avg_acc
                b_new_best = True
                if args.logdir is not None and args.save_checkpoint:
                    save_checkpoint(
                        model, epoch, args, best_acc=val_acc_max, optimizer=optimizer, scheduler=scheduler, logger=logger
                    )
            if args.logdir is not None and args.save_checkpoint:
                save_checkpoint(model, epoch, args, best_acc=val_acc_max, filename="model_final.pt", logger=logger)
                if b_new_best:
                    logger.info("Copying to pet-ct-fold1.pt new best model!!!!")
                    shutil.copyfile(os.path.join(args.logdir, "model_final.pt"), os.path.join(args.logdir, "model.pt"))

        if scheduler is not None:
            scheduler.step()

    logger.info(f"Training Finished !, Best Accuracy: {val_acc_max}")

    return val_acc_max
