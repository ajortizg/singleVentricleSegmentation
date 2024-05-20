from torchvision.utils import draw_segmentation_masks, make_grid
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os.path as osp
import torch
import os
from typing import Optional, Union

# Use 'Agg' backend for matplotlib to work in environments without a display
matplotlib.use('Agg')
plt.rcParams["savefig.bbox"] = 'tight'


def write_image_mask_overlay(
    img3d: Union[np.ndarray, torch.Tensor],
    mask3d: Optional[Union[np.ndarray, torch.Tensor]],
    filename: str,
    alpha: float,
    color: list = [1, 1, 0],
    aspect_ratio: float = 1.7
):
    """
    Saves a series of images and their corresponding masks overlapped, as a single image file.
    The shape of the data must be [z, y, x].

    Args:
        img3d (Union[np.ndarray, torch.Tensor]): 3D array of images (can be numpy array or torch tensor).
        mask3d (Optional[Union[np.ndarray, torch.Tensor]]): 3D array of masks (can be numpy array or torch tensor). Can be None.
        filename (str): The name of the file to save the image as.
        save_dir (str): Directory where the file will be saved.
        alpha (float): Transparency level for the mask overlay (between 0 and 1).
        color (list, optional): RGB color values for the overlay mask.
        aspect_ratio (float, optional): Aspect ratio to determine the layout of subplots.

    Returns:
        None
    """
    # Convert torch Tensors to numpy arrays if needed
    if isinstance(img3d, torch.Tensor):
        img3d = img3d.cpu().detach().numpy()
    img3d = img3d.astype(np.float32)

    if isinstance(mask3d, torch.Tensor):
        mask3d = mask3d.cpu().detach().numpy()
    if mask3d is not None:
        mask3d = mask3d.astype(bool)

    # Determine the number of columns and rows for the subplots
    NZ = img3d.shape[0]
    cols = int(NZ / aspect_ratio)
    if (NZ % cols > 0):
        cols += 1
    rows = math.ceil(NZ / cols)

    fig, axs = plt.subplots(rows, cols, constrained_layout=True, figsize=(18, 10), dpi=4)
    fig.patch.set_facecolor('black')
    fig.suptitle('file: {}'.format(osp.basename(filename)), fontsize=16)
    for z, ax in enumerate(axs.flat):
        if z < NZ:
            img = cv2.cvtColor(img3d[z], cv2.COLOR_GRAY2BGR)
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            if mask3d is not None:
                mask = mask3d[z]
                img = overlay(img, mask, alpha, color)
            ax.imshow(img)
            ax.set_title(f'layer {z}')
            ax.axis('off')
        else:
            ax.axis('off')
    plt.savefig(filename, dpi=100)
    plt.close('all')


def overlay(
    img: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.35,
    color: list = [1, 1, 0]
) -> np.ndarray:
    """
    Merges an image with its corresponding mask with a specified alpha transparency level and color.

    Args:
        img (numpy.ndarray): 2D or 3D array of image pixels.
        mask (numpy.ndarray): 2D array of mask pixels.
        alpha (float, optional): Transparency level for the mask overlay (between 0 and 1).
        color (list, optional): RGB color values for the overlay mask.

    Returns:
        numpy.ndarray: Merged image with overlay mask.
    """
    res = np.zeros_like(img, dtype=np.float32)
    for c in range(3):
        res[..., c] = img[..., c] * (1 - alpha) + alpha * color[c] * 255
    img[mask] = res[mask]
    return img.astype(np.uint8)
