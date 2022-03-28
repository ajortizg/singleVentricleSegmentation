import numpy as np
import matplotlib.pyplot as plt
import sys
import cv2
from objects import *
import transforms as T
import torch.nn.functional as F
import torch
import os.path as osp

# utils_path = osp.abspath(osp.join(osp.dirname(__file__), '../utils'))
# sys.path.append(utils_path)
# from flow_viz import plotOpticalFlow3D


np.set_printoptions(precision=2, suppress=True)


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


def compute_optical_flow(grid, R, S, t):
    NZ, NY, NX = grid.shape[:3]
    # grid_t = np.zeros((NZ, NY, NX, 3), dtype=np.float64)
    grid_t = np.copy(grid)
    of = np.zeros((NZ, NY, NX, 3), dtype=np.float64)
    R_left = np.linalg.inv(R)
    noise = np.random.rand(NZ, NY, NX, 3)

    for z in range(NZ):
        for y in range(NY):
            for x in range(NX):
                grid_t[z, y, x, :] = S@(R@grid[z, y, x, :]) + \
                    t + noise[z, y, x, :]
                of[z, y, x, :] = grid_t[z, y, x, :] - grid[z, y, x, :]
                # print(grid[z, y, x, :], grid_t[z, y, x, :],
                #       of[z, y, x, :], grid[z, y, x, :] + of[z, y, x, :])
    return of


# Dimension of the volume
NX, NY, NZ = 32, 44, 14
grid = create_grid(NZ, NY, NX)

ellip_1 = Ellipsoid(NX//2, NY//2, NZ//2, 12, 16, 14)
ellip_1.create_voxels(grid)

mask = Ellipsoid(NX//2.2, NY//2.5, NZ//3, 3, 6, 14)
mask.create_voxels(grid, constant=True, value=0.9)


R = np.eye(3, 3)
S = T.scale(1, 1, 1)
t = np.array([3, 0, 2])

# of_mask = compute_optical_flow(grid, R, S, t)
# mask_gray_w = mask.warp(grid, of_mask)
# plot_slices(mask.gray_values, str="mask", block=False)
# plot_slices(mask_gray_w, str="mask_w", block=True)


plot_slices(ellip_1.gray_values, str="gray", block=False)
of = compute_optical_flow(grid, R, S, t)

gray_w = ellip_1.warp(grid, of)
plot_slices(gray_w, str="gray_w", block=True)


# # 3D plot
# Voxel color
# alpha = 0.3
# voxel_color = np.empty([NZ, NY, NX, 4])
# voxel_color[ellip_1.voxels] = [0, 1, 0, alpha]  # green
# voxel_color[mask.voxels] = [0, 0, 1, alpha*2]  # blue
# voxelarray = ellip_1.voxels | mask.voxels
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.voxels(np.transpose(voxelarray, (2, 1, 0)),
#           facecolors=np.transpose(voxel_color, (2, 1, 0, 3)),
#           edgecolor='k', linewidth=0.5)
# ax.set(xlabel='X', ylabel='Y', zlabel='Z')
