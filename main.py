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

import argparse
import os

# from model import ResNet3D
from model_inception_enc_dec import InceptionDecoder3D
from logger import setup_logger

from monai.inferers import sliding_window_inference
from functools import partial
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from trainer import run_training
from data_utils import get_loader
from monai.networks.nets import SwinUNETR, UNet, SegResNet


from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.utils.enums import MetricReduction

parser = argparse.ArgumentParser(description="ResNet pipeline")
parser.add_argument("--checkpoint", default=None, help="start training from saved checkpoint")
parser.add_argument("--exp", default="test", type=str, help="directory to save the tensorboard logs")
parser.add_argument("--data_dir", default="/dataset/dataset0/", type=str, help="dataset directory")
parser.add_argument("--json_list", default="dataset_0.json", type=str, help="dataset json file")
parser.add_argument("--save_checkpoint", action="store_true", help="save checkpoint during training")
parser.add_argument("--max_epochs", default=5000, type=int, help="max number of training epochs")
parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
parser.add_argument("--optim_lr", default=1e-4, type=float, help="optimization learning rate")
parser.add_argument("--optim_name", default="adamw", type=str, help="optimization algorithm")
parser.add_argument("--reg_weight", default=1e-5, type=float, help="regularization weight")
parser.add_argument("--momentum", default=0.99, type=float, help="momentum")
parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
parser.add_argument("--val_every", default=50, type=int, help="validation frequency")
parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
parser.add_argument("--dist-url", default="tcp://127.0.0.1:23456", type=str, help="distributed url")
parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
parser.add_argument("--norm_name", default="instance", type=str, help="normalization name")
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=12, type=int, help="number of output channels")
parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")
parser.add_argument("--overlap", default=0.5, type=int, help="sliding inferrer overlap")
parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--smooth_dr", default=1e-6, type=float, help="constant added to dice denominator to avoid nan")
parser.add_argument("--smooth_nr", default=0.0, type=float, help="constant added to dice numerator to avoid zero")
parser.add_argument("--use_ssl_pretrained", action="store_true", help="use self-supervised pretrained weights")
parser.add_argument("--ssl_pretrained_path", default="./pretrained_models/model_swinvit.pt", type=str, help="path to the visual representation")
parser.add_argument("--gpu", default=0, type=int, help="gpu ID")
parser.add_argument("--squared_dice", action="store_true", help="use squared Dice")


def main():
    args.amp = not args.noamp
    args.logdir = "./swin_unetr_center1_results/" + args.exp
    os.makedirs(args.logdir, exist_ok=True)
    logger = setup_logger(args)

    main_worker(args=args, logger=logger)


def main_worker(args, logger):

    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)
    # torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True
    args.test_mode = False
    loader = get_loader(args)
    logger.info(f'gpu {args.gpu}')
    logger.info(f"Batch size is: {args.batch_size}\tepochs: {args.max_epochs}")

    # model = InceptionDecoder3D(in_channels=2, out_channels=12)
    model = SwinUNETR(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        feature_size=48,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=args.use_checkpoint,
    ).to("cuda")

    # model = UNet(
    #     spatial_dims=3,
    #     in_channels=args.in_channels,  # e.g. 2 for PET+CT
    #     out_channels=args.out_channels,  # num classes
    #     channels=(16, 32, 64, 128, 256),  # you can tune this
    #     strides=(2, 2, 2, 2),  # downsampling per level
    #     num_res_units=2,
    # ).to("cuda")

    # model = SegResNet(
    #     spatial_dims=3,
    #     in_channels=args.in_channels,
    #     out_channels=args.out_channels,
    #     init_filters=32,  # base filters
    #     dropout_prob=0.0,
    # ).to("cuda")

    model_inferrer = partial(
        sliding_window_inference,
        roi_size=(args.roi_x, args.roi_y, args.roi_z),
        sw_batch_size=4,
        predictor=model,
        overlap=args.overlap,
    )

    if args.squared_dice:
        dice_loss = DiceCELoss(
            to_onehot_y=True, softmax=True, squared_pred=True, smooth_nr=args.smooth_nr, smooth_dr=args.smooth_dr
        )
    else:
        dice_loss = DiceCELoss(to_onehot_y=True, softmax=True)
    post_label = AsDiscrete(to_onehot=args.out_channels, n_classes=args.out_channels)
    post_pred = AsDiscrete(argmax=True, to_onehot=args.out_channels, n_classes=args.out_channels)
    dice_acc = DiceMetric(include_background=True, reduction=MetricReduction.MEAN, get_not_nans=True)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters count {pytorch_total_params}")

    best_acc = 0
    start_epoch = 0

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in checkpoint["state_dict"].items():
            new_state_dict[k.replace("backbone.", "")] = v
        model.load_state_dict(new_state_dict, strict=False)
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"]
        if "best_acc" in checkpoint:
            best_acc = checkpoint["best_acc"]
        logger.info("=> loaded checkpoint '{}' (epoch {}) (bestacc {})".format(args.checkpoint, start_epoch, best_acc))

    if args.use_ssl_pretrained:
        try:
            model_dict = torch.load(args.ssl_pretrained_path)
            state_dict = model_dict["state_dict"]
            # fix potential differences in state dict keys from pre-training to
            # fine-tuning
            if "module." in list(state_dict.keys())[0]:
                logger.info("Tag 'module.' found in state dict - fixing!")
                for key in list(state_dict.keys()):
                    state_dict[key.replace("module.", "")] = state_dict.pop(key)
            if "swin_vit" in list(state_dict.keys())[0]:
                logger.info("Tag 'swin_vit' found in state dict - fixing!")
                for key in list(state_dict.keys()):
                    state_dict[key.replace("swin_vit", "swinViT")] = state_dict.pop(key)
            # We now load model weights, setting param `strict` to False, i.e.:
            # this load the encoder weights (Swin-ViT, SSL pre-trained), but leaves
            # the decoder weights untouched (CNN UNet decoder).
            model.load_state_dict(state_dict, strict=False)
            logger.info(f"Using pretrained self-supervised Swin UNETR backbone weights: {args.ssl_pretrained_path}")
        except ValueError:
            raise ValueError("Self-supervised pre-trained weights not available for" + str(args.model_name))

    model.cuda(args.gpu)

    if args.optim_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.optim_lr, momentum=args.momentum, nesterov=True, weight_decay=args.reg_weight
        )
    else:
        raise ValueError("Unsupported Optimization Procedure: " + str(args.optim_name))

    scheduler = None
    accuracy = run_training(
        model=model,
        train_loader=loader[0],
        val_loader=loader[1],
        optimizer=optimizer,
        loss_func=dice_loss,
        acc_func=dice_acc,
        model_inferrer=model_inferrer,
        # model_inferrer=None,
        args=args,
        scheduler=scheduler,
        start_epoch=start_epoch,
        post_label=post_label,
        post_pred=post_pred,
        logger=logger,
    )
    return accuracy


if __name__ == "__main__":
    args = parser.parse_args()
    args.exp = "pet-center1-fold0"
    args.data_dir = "/content/drive/MyDrive/SBU_AI_Conference/NEW_DATA"
    args.json_list = "center_folds/center1_pet_only/fold0.json"
    # args.use_ssl_pretrained = True
    args.ssl_pretrained_path = "/content/drive/MyDrive/SBU_AI_Conference/pretrained_model/model_bestValRMSE.pt"
    args.in_channels = 1
    args.out_channels = 12
    args.max_epochs = 500
    args.val_every = 100
    args.roi_x = 64
    args.roi_y = 64
    args.roi_z = 64
    args.workers = 4
    args.overlap = 0.8
    # args.optim_lr = 0.00001
    args.use_checkpoint = True
    args.save_checkpoint = True
    # args.noamp = True
    main()
