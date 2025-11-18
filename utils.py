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

import numpy as np
import scipy.ndimage as ndimage
from scipy.spatial.distance import directed_hausdorff


def resample_3d(img, target_size):
    imx, imy, imz = img.shape
    tx, ty, tz = target_size
    zoom_ratio = (float(tx) / float(imx), float(ty) / float(imy), float(tz) / float(imz))
    img_resampled = ndimage.zoom(img, zoom_ratio, order=0, prefilter=False)
    return img_resampled


def dice(x, y):
    intersect = np.sum(np.sum(np.sum(x * y)))
    y_sum = np.sum(np.sum(np.sum(y)))
    if y_sum == 0:
        return 0.0
    x_sum = np.sum(np.sum(np.sum(x)))
    return 2 * intersect / (x_sum + y_sum)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


def recall(pred, target):
    """
    Calculate the recall for a binary segmentation mask.

    Args:
        pred (np.ndarray): Predicted binary mask.
        target (np.ndarray): Ground truth binary mask.

    Returns:
        float: Recall score.
    """
    true_positive = np.sum((pred == 1) & (target == 1))
    false_negative = np.sum((pred == 0) & (target == 1))

    if true_positive + false_negative == 0:
        return 0.0

    return true_positive / (true_positive + false_negative)


def precision(pred, target):
    """
    Calculate the precision for a binary segmentation mask.

    Args:
        pred (np.ndarray): Predicted binary mask.
        target (np.ndarray): Ground truth binary mask.

    Returns:
        float: Precision score.
    """
    true_positive = np.sum((pred == 1) & (target == 1))
    false_positive = np.sum((pred == 1) & (target == 0))

    if true_positive + false_positive == 0:
        return 0.0

    return true_positive / (true_positive + false_positive)


def hausdorff_distance(pred, target, percentile=100):
    """
    Calculate the Hausdorff Distance or Hausdorff Distance at a given percentile (e.g., HD95).

    Args:
        pred (np.ndarray): Predicted binary mask.
        target (np.ndarray): Ground truth binary mask.
        percentile (int): Percentile of the Hausdorff Distance to calculate (default 100 for full HD).

    Returns:
        float: Hausdorff Distance or Hausdorff Distance at given percentile.
    """
    pred_points = np.transpose(np.nonzero(pred))
    target_points = np.transpose(np.nonzero(target))

    if len(pred_points) == 0 or len(target_points) == 0:
        return float('inf')  # If no foreground is found, return infinity.

    # Compute distances from pred to target and vice versa
    forward_hd = directed_hausdorff(pred_points, target_points)[0]
    backward_hd = directed_hausdorff(target_points, pred_points)[0]

    # Full Hausdorff distance is the maximum of both directions
    hd_value = max(forward_hd, backward_hd)

    if percentile == 100:
        return hd_value

    # Calculate distances for HD95 or any other percentile
    forward_distances = np.array([np.min(np.linalg.norm(p - target_points, axis=1)) for p in pred_points])
    backward_distances = np.array([np.min(np.linalg.norm(t - pred_points, axis=1)) for t in target_points])

    all_distances = np.concatenate([forward_distances, backward_distances])

    return np.percentile(all_distances, percentile)

