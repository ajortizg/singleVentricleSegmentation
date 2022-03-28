import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import ndimage
import sys
from objects import *
import transforms as T
import torch.nn.functional as F
import scipy


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


def scale_grid(grid):
    # scale grid to [-1,1]
    NZ, NY, NX, _ = grid.shape
    grid[:, :, :, 0] = 2.0 * grid[:, :, :, 0] / max(NX-1, 1) - 1.0
    grid[:, :, :, 1] = 2.0 * grid[:, :, :, 1] / max(NY-1, 1) - 1.0
    grid[:, :, :, 2] = 2.0 * grid[:, :, :, 2] / max(NZ-1, 1) - 1.0
    return grid


def warp(vol, grid, of):
    grid = torch.from_numpy(grid[:, :, :, :3])
    of = torch.from_numpy(of)

    new_coords = scale_grid(grid - of)
    new_coords.unsqueeze_(dim=0)
    vol = torch.from_numpy(vol)
    vol.unsqueeze_(dim=0).unsqueeze_(dim=0)

    vol_w = F.grid_sample(vol, new_coords, align_corners=True,
                          mode="bilinear", padding_mode="zeros")
    vol_w = vol_w.squeeze_().numpy()
    return vol_w


def create_grid(NZ, NY, NX):
    zz, yy, xx = np.meshgrid(np.arange(NZ), np.arange(NY),
                             np.arange(NX), indexing="ij")
    return np.stack((xx, yy, zz), axis=3)


# Dimension of the volume
NX, NY, NZ = 32, 44, 16
grid = create_grid(NZ, NY, NX)

# sphere = Sphere(NX//2, NY//2, NZ//2, 9)
# sphere.create_voxels(grid)

# cube = Cube(NX//3, NY//2, NZ//2, 12, 6, NZ)
# cube.create_voxels(grid)

ellipsoid = Ellipsoid(NX//2, NY//2, NZ//2, 18, 7, 2)
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

# voxel value
# voxel_val = np.where(cube.voxels == True, 0.5,
#                      np.where(sphere.voxels == True, 1, 0))
voxel_val = np.where(ellipsoid.voxels == True, 0.5, 0)
plot_slices(voxel_val, str="voxelarray", block=False)


# small rotation
# small translation
# compression and gray value change
grid_w = np.zeros_like(grid)
of = np.zeros([NZ, NY, NX, 3])
# R = T.rotz(20) * T.rotx(16) * T.roty(20)
R = np.eye(3,3)
t = np.array([1, 0, 0])

R_left = np.linalg.inv(R)

for x in range(NX):
    for y in range(NY):
        for z in range(NZ):
            # grid_w[z, y, x, :] = R@grid[z, y, x, :]
            grid_w[z, y, x, :] = R@grid[z, y, x, :] + t
            of[z, y, x, :] = grid_w[z, y, x, :] - grid[z, y, x, :]
            # print(grid[z, y, x, :], grid_w[z, y, x, :],
            #       of[z, y, x, :], grid[z, y, x, :] + of[z, y, x, :])

new_cords = grid + of
voxel_val_w = ndimage.map_coordinates(voxel_val,
                                      [new_cords[:, :, :, 2],
                                       new_cords[:, :, :, 1],
                                       new_cords[:, :, :, 0]],
                                      order=1, mode="constant")

plot_slices(voxel_val_w, str="voxel_val_w", block=False)

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
