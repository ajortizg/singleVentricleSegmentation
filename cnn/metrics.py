import torch
from scipy.ndimage import distance_transform_edt, binary_erosion, generate_binary_structure
import numpy as np


def ravd(pred, gt) -> torch.Tensor:
    """
    Relative absolute volume difference.
    """
    vol_pred = torch.count_nonzero(pred, dim=(1, 2, 3, 4))
    vol_gt = torch.count_nonzero(gt, dim=(1, 2, 3, 4))
    return (vol_pred - vol_gt) / vol_gt


def surface_voxels(x, connectivity=1) -> torch.Tensor:
    """
    Extract only 1-pixel border line of objects
    """
    device = x.device
    bs, ch, *_ = x.shape
    x = x.detach().cpu().numpy().astype(np.bool)
    footprint = generate_binary_structure(3, connectivity)
    border = np.zeros_like(x)
    for b in range(bs):
        for c in range(ch):
            border[b, c] = x[b, c] ^ binary_erosion(x[b, c], structure=footprint, iterations=1)
    return torch.from_numpy(border).float().to(device)


def _surface_distance(pred, gt, connectivity=1):
    footprint = generate_binary_structure(pred.ndim, connectivity)
    pred = pred.detach().cpu().numpy().astype(np.bool)
    gt = gt.detach().cpu().numpy().astype(np.bool)

    # extract only 1-pixel border line of objects
    pred_border = pred ^ binary_erosion(pred, structure=footprint, iterations=1)
    gt_border = gt ^ binary_erosion(gt, structure=footprint, iterations=1)

    # compute average surface distance
    dt = distance_transform_edt(~gt_border, sampling=None)
    sds = dt[pred_border]
