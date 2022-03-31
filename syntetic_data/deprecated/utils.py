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


def compute_optical_flow(grid, voxels, R, S, t):
    NZ, NY, NX = grid.shape[:3]
    # grid_t = np.zeros((NZ, NY, NX, 3), dtype=np.float64)
    grid_t = np.copy(grid)
    of = np.zeros((NZ, NY, NX, 3), dtype=np.float64)
    R_left = np.linalg.inv(R)
    noise = np.random.rand(NZ, NY, NX, 3)

    for z in range(NZ):
        for y in range(NY):
            for x in range(NX):
                if voxels[z, y, x]:
                    grid_t[z, y, x, :] = S@(R@grid[z, y, x, :]) + t
                    of[z, y, x, :] = grid_t[z, y, x, :] - grid[z, y, x, :]
                # print(grid[z, y, x, :], grid_t[z, y, x, :],
                #       of[z, y, x, :], grid[z, y, x, :] + of[z, y, x, :])
    return of


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


def warp_grid(vol, grid):
    grid = scale_grid(grid)
    new_coords = torch.from_numpy(grid)

    new_coords.unsqueeze_(dim=0)
    vol = torch.from_numpy(vol)
    vol.unsqueeze_(dim=0).unsqueeze_(dim=0)

    vol_w = F.grid_sample(vol, new_coords, align_corners=True,
                          mode="bilinear", padding_mode="zeros")
    return vol_w.squeeze().numpy()
