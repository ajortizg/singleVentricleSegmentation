import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def plot_slices(X, str="", block=True):
    """
    X:  numpy array with shape [Z,Y,X]
    """
    fig, _ = plt.subplots(2, X.shape[0] // 2)
    plt.suptitle(f"{str} {X.shape}")
    for i, ax in enumerate(fig.get_axes()):
        ax.imshow(X[i, :, :], cmap="gray", origin="lower")
    plt.show(block=block)


def create_grid(NZ, NY, NX):
    zz, yy, xx = np.meshgrid(np.arange(NZ), np.arange(NY),
                             np.arange(NX), indexing="ij")
    return np.stack((xx, yy, zz), axis=3).astype(np.float64)



def scale_grid(grid):
    # scale grid to [-1,1]
    if grid.ndim == 4:
        NZ, NY, NX, _ = grid.shape
        grid[:, :, :, 0] = 2.0 * grid[:, :, :, 0] / max(NX-1, 1) - 1.0
        grid[:, :, :, 1] = 2.0 * grid[:, :, :, 1] / max(NY-1, 1) - 1.0
        grid[:, :, :, 2] = 2.0 * grid[:, :, :, 2] / max(NZ-1, 1) - 1.0
    elif grid.ndim == 3:
        NY, NX, _ = grid.shape
        grid[:, :, 0] = 2.0 * grid[:, :, 0] / max(NX-1, 1) - 1.0
        grid[:, :, 1] = 2.0 * grid[:, :, 1] / max(NY-1, 1) - 1.0

    return grid


def warp_grid(vol, grid, mode="bilinear"):
    """
    mode = {bilinear, nearest}
    """
    grid = scale_grid(grid)
    new_coords = torch.from_numpy(grid)

    new_coords.unsqueeze_(dim=0)
    vol = torch.from_numpy(vol)
    vol.unsqueeze_(dim=0).unsqueeze_(dim=0)

    vol_w = F.grid_sample(vol, new_coords, align_corners=True,
                          mode=mode, padding_mode="zeros")
    return vol_w.squeeze().numpy()
