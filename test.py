import argparse
import os

from model import ResNet3D
from model_inception_enc_dec import InceptionDecoder3D
import nibabel as nib
import numpy as np
import torch
from data_utils import get_loader
from utils import dice, resample_3d
import pandas as pd

from monai.inferers import sliding_window_inference

class_map = {
    0: 'background',
    1: 'lacrimal_glands',
    2: 'parotid_glands',
    3: 'tubarial_gland',
    4: 'sublingual_gland',
    5: 'submandibular_glands',
    6: 'spleen',
    7: 'liver',
    8: 'small_intestine',
    9: 'kidneys',
    10: 'bladder',
    11: 'lesions'
}

parser = argparse.ArgumentParser(description="ResNet18 segmentation pipeline")
parser.add_argument(
    "--pretrained_dir", default="./pretrained_models/", type=str, help="pretrained checkpoint directory"
)
parser.add_argument("--data_dir", default="/dataset/dataset0/", type=str, help="dataset directory")
parser.add_argument("--exp_name", default="test1", type=str, help="experiment name")
parser.add_argument("--json_list", default="dataset_0.json", type=str, help="dataset json file")
parser.add_argument(
    "--pretrained_model_name",
    default="swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt",
    type=str,
    help="pretrained model name",
)
parser.add_argument("--feature_size", default=48, type=int, help="feature size")
parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
parser.add_argument("--in_channels", default=2, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=12, type=int, help="number of output channels")
parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--workers", default=4, type=int, help="number of workers")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")
parser.add_argument("--save", action="store_true", help="save segmentation output")


def main():
    args.test_mode = True
    output_directory = "./outputs/" + args.exp_name
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    val_loader = get_loader(args)
    pretrained_dir = args.pretrained_dir
    model_name = args.pretrained_model_name
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pretrained_pth = os.path.join(pretrained_dir, model_name)

    model = InceptionDecoder3D(in_channels=2, out_channels=12)

    if os.path.exists(pretrained_pth):
        model_dict = torch.load(pretrained_pth)["state_dict"]
        model.load_state_dict(model_dict)
    else:
        print(f"Pretrained model not found at {pretrained_pth}, initializing from scratch.")

    model.eval()
    model.to(device)

    with torch.no_grad():
        dice_list_case = []
        classwise_dice_scores = []
        for i, batch in enumerate(val_loader):
            val_inputs, val_labels = (batch["image"].cuda(0), batch["label"].cuda(0))
            original_affine = batch["label_meta_dict"]["affine"][0].numpy()
            _, _, h, w, d = val_labels.shape
            target_shape = (h, w, d)
            img_name = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-1]
            print("Inference on case {}".format(img_name))

            val_outputs = sliding_window_inference(
                val_inputs, (args.roi_x, args.roi_y, args.roi_z), 4, model, overlap=args.infer_overlap, mode="gaussian"
            )

            val_outputs = torch.softmax(val_outputs, 1).cpu().numpy()  # Calculate probabilities
            val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)[0]  # Get argmax class
            val_labels = val_labels.cpu().numpy()[0, 0, :, :, :]
            val_outputs = resample_3d(val_outputs, target_shape)
            dice_list_sub = []
            row = {}
            row['filename'] = img_name
            for i in range(0, args.out_channels):
                organ_Dice = dice(val_outputs == i, val_labels == i)
                dice_list_sub.append(organ_Dice)
                row[class_map[i]] = organ_Dice
            mean_dice = np.mean(dice_list_sub)
            print("Mean Organ Dice: {}".format(mean_dice))
            dice_list_case.append(mean_dice)
            classwise_dice_scores.append(row)
            if args.save:
                nib.save(
                    nib.Nifti1Image(val_outputs.astype(np.uint8), original_affine), os.path.join(output_directory, img_name)
                )
        df = pd.DataFrame.from_dict(classwise_dice_scores, orient='columns')
        df_float = df.select_dtypes(include="float64")
        df.loc["overall"] = df_float.mean(axis=0)
        df_float = df.select_dtypes(include="float64")
        df["mean_dice"] = df_float.mean(axis=1)
        df.to_csv(os.path.join(output_directory, f'scores-{args.exp_name}.csv'))
        print("Overall Mean Dice: {}".format(np.mean(dice_list_case)))


if __name__ == "__main__":

    args = parser.parse_args()
    args.data_dir = 'masked_data'
    args.json_list = 'fold0.json'
    args.pretrained_dir = "result/pet-ct-fold0"
    args.exp_name = "pet-ct-fold0"
    args.pretrained_model_name = "model.pt"
    # args.save = True
    main()

