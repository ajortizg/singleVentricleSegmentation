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


def save_image_mask_overlay(
    img3d: Union[np.ndarray, torch.Tensor],
    mask3d: Optional[Union[np.ndarray, torch.Tensor]],
    filename: str,
    save_dir: str,
    alpha: float,
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
        aspect_ratio (float, optional): Aspect ratio to determine the layout of subplots. Defaults to 1.7.

    Returns:
        None
    """
    # Convert torch Tensors to numpy arrays if needed
    if isinstance(img3d, torch.Tensor):
        img3d = img3d.cpu().detach().numpy()

    if isinstance(mask3d, torch.Tensor):
        mask3d = mask3d.cpu().detach().numpy()

    # Determine the number of columns and rows for the subplots
    NZ = img3d.shape[0]
    cols = int(NZ / aspect_ratio)
    if (NZ % cols > 0):
        cols += 1
    rows = math.ceil(NZ / cols)

    fig, axs = plt.subplots(rows, cols, constrained_layout=True, figsize=(18, 10), dpi=4)
    fig.suptitle('file: {}'.format(os.path.basename(filename)), fontsize=16)
    for z, ax in enumerate(axs.flat):
        if z < NZ:
            img = img3d[z]
            ax.imshow(img, cmap='gray', interpolation='none')
            if mask3d is not None:
                mask = mask3d[z]
                ax.imshow(mask, cmap='jet', alpha=alpha, interpolation='none')
            ax.set_title(f'layer {z}')
            ax.axis('off')
        else:
            ax.axis('off')
    path_name = osp.join(save_dir, filename)
    plt.savefig(path_name, dpi=100)
    plt.close('all')
