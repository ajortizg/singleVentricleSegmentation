import numpy as np
import matplotlib.pyplot as plt
import sys
from objects import *
import transforms as T
import torch.nn.functional as F
import torch

np.set_printoptions(precision=2, suppress=True)


def plot_slices(X, str="", block=True):
    """
    X:  numpy array with shape [Z,Y,X]
    """
    fig, _ = plt.subplots(2, X.shape[0] // 2)
    plt.suptitle(f"{str} {X.shape}")
    for i, ax in enumerate(fig.get_axes()):
        ax.imshow(X[i, :, :], cmap="gray", origin="lower")
        # ax.imshow(X[:, :, i], cmap="gray")
    plt.show(block=block)


def create_grid(NZ, NY, NX):
    zz, yy, xx = np.meshgrid(np.arange(NZ), np.arange(NY),
                             np.arange(NX), indexing="ij")
    return np.stack((xx, yy, zz), axis=3).astype(np.float64)


def compute_optical_flow(grid, R, t):
    NZ, NY, NX = grid.shape[:3]
    grid_t = np.zeros((NZ, NY, NX, 3), dtype=np.float64)
    of = np.zeros((NZ, NY, NX, 3), dtype=np.float64)
    R_left = np.linalg.inv(R)

    for x in range(NX):
        for y in range(NY):
            for z in range(NZ):
                grid_t[z, y, x, :] = R@grid[z, y, x, :] + t
                of[z, y, x, :] = grid_t[z, y, x, :] - grid[z, y, x, :]
                print(grid[z, y, x, :], grid_t[z, y, x, :],
                      of[z, y, x, :], grid[z, y, x, :] + of[z, y, x, :])
    return of


# Dimension of the volume
NX, NY, NZ = 32, 44, 16
grid = create_grid(NZ, NY, NX)

ellipsoid = Ellipsoid(NX//2, NY//2, NZ//2, 8, 16, 5)
ellipsoid.create_voxels(grid)

# voxelarray = sphere.voxels | cube.voxels
voxelarray = ellipsoid.voxels

# Voxel color
alpha = 0.3
voxel_color = np.empty([NZ, NY, NX, 4])
voxel_color[ellipsoid.voxels] = [0, 1, 0, alpha]  # green
# voxel_color[cube.voxels] = [0, 0, 1, alpha*2]  # blue

# 3D plot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.voxels(np.transpose(voxelarray, (2, 1, 0)),
          facecolors=np.transpose(voxel_color, (2, 1, 0, 3)),
          edgecolor='k', linewidth=0.5)
ax.set(xlabel='X', ylabel='Y', zlabel='Z')
plot_slices(ellipsoid.gray_values, str="gray", block=False)

of = compute_optical_flow(grid, T.rotz(8), np.array([1, 0, 2]))
gray_w = ellipsoid.warp(grid, of)
plot_slices(gray_w, str="gray_w", block=True)

# voxelarray_w = ndimage.map_coordinates(voxelarray,
#                                        [new_cords[:, :, :, 2],
#                                         new_cords[:, :, :, 1],
#                                         new_cords[:, :, :, 0]],
#                                        order=3, mode="constant")
# voxelarray_w[voxelarray_w < 0.5] = 0
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.voxels(np.transpose(voxelarray_w, (2, 1, 0)),
#           edgecolor='k', linewidth=0.5)
# ax.set(xlabel='X', ylabel='Y', zlabel='Z')
plt.show()
