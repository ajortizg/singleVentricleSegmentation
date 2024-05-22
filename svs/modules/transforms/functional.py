from scipy.ndimage import binary_erosion
import cv2
import torch
import numpy as np
from typing import Tuple


def add_leading_dims(n: int, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """
    Adds `n` leading dimensions to the input array or tensor.

    Args:
        n (int): Number of leading dimensions to add.
        x (np.ndarray | torch.Tensor): Input array or tensor to which leading dimensions will be added.

    Returns:
        np.ndarray | torch.Tensor: The input array or tensor with `n` additional leading dimensions.

    Raises:
        TypeError: If the input is not a numpy array or torch tensor.
    """
    if isinstance(x, np.ndarray):
        for _ in range(n):
            x = np.expand_dims(x, axis=0)
    elif isinstance(x, torch.Tensor):
        for _ in range(n):
            x = x.unsqueeze(0)
    else:
        raise TypeError("Only np.ndarray and torch.Tensor are supported types by add_leading_dims")
    return x


def add_dim_at(axis: int, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """
    Adds a new dimension at the specified axis of the input array or tensor.

    Args:
        axis (int): The axis at which to add the new dimension.
        x (np.ndarray | torch.Tensor]: Input array or tensor.

    Returns:
        np.ndarray | torch.Tensor: The input array or tensor with a new dimension added at the specified axis.

    Raises:
        TypeError: If the input is not a numpy array or torch tensor.
    """
    if isinstance(x, np.ndarray):
        x = np.expand_dims(x, axis=axis)
    elif isinstance(x, torch.Tensor):
        x = x.unsqueeze(axis)
    else:
        raise TypeError("Only np.ndarray and torch.Tensor are supported types by add_dim_at")
    return x


def reorder_axes(axes: Tuple[int], x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """
    Reorders the axes of the input array or tensor.

    Args:
        axes (Tuple[int]): The new order of axes.
        x (np.ndarray | torch.Tensor]): Input array or tensor.

    Returns:
        np.ndarray | torch.Tensor: The input array or tensor with reordered axes.

    Raises:
        TypeError: If the input is not a numpy array or torch tensor.
    """
    if isinstance(x, np.ndarray):
        x = np.transpose(x, axes)
    elif isinstance(x, torch.Tensor):
        x = torch.permute(x, axes)
    else:
        raise TypeError("Only np.ndarray and torch.Tensor are supported types by reorder_axes")
    return x


def to_float_tensor(x: np.ndarray) -> torch.Tensor:
    """
    Converts a numpy array to a float tensor.

    Args:
        x (np.ndarray): Input numpy array.

    Returns:
        torch.Tensor: Float tensor converted from the input numpy array.
    """
    return torch.from_numpy(x).float()


def erode(x: np.ndarray | torch.Tensor, iters: int = 1) -> np.ndarray:
    """
    Applies morphological erosion (slice-wise) to a 3D volumetric data.

    Args:
        x (np.ndarray | torch.Tensor): The input volume with shape (depth, height, width).
        iters (int): The number of iterations for the erosion operation.

    Returns:
        np.ndarray: The eroded borders of the input volume.
    """
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()

    nz = x.shape[0]
    borders = np.zeros_like(x)

    for z in range(nz):
        mask = x[z]
        borders[z] = (mask - cv2.erode(mask, kernel=None, borderValue=0, iterations=iters))
    return borders
    # x = x.astype(bool)
    # x_e = binary_erosion(x, iterations=iters)
    # borders = x & ~x_e  # Calculate the borders by subtracting the eroded volume from the original volume
    # return borders.astype(np.uint8)
