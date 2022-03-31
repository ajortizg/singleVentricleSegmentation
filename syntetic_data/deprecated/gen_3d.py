
import numpy as np
import matplotlib.pyplot as plt
import sys
import cv2
from objects import *
import transforms as T
import math
import torch.nn.functional as F
import torch
import utils
import os.path as osp

utils_path = osp.abspath(osp.join(osp.dirname(__file__), '../utils'))
sys.path.append(utils_path)
from flow_viz import plotOpticalFlow3D

np.set_printoptions(precision=2, suppress=True)


# Dimension of the volume
NX, NY, NZ = 32, 44, 14
grid = utils.create_grid(NZ, NY, NX)

ellip = Ellipsoid(NX//2, NY//2, NZ//2, 12, 20, 14)
ellip.create_voxels(grid, low=0.2, high=0.5)

mask = Ellipsoid(NX//2.2, round(NY*0.6), 0, 3, 6, 16)
mask.create_voxels(grid, constant=True, value=1.0)

for z in range(NZ):
    for y in range(NY):
        for x in range(NX):
            if mask.voxels[z, y, x]:
                ellip.gray_values[z, y, x] = 0.2 + \
                    np.random.normal(0, 0.05, 1)


np.save("output/base_object.npy", ellip.gray_values)
np.save("output/base_mask.npy", mask.gray_values)

R = np.eye(3, 3)
S = T.scale(1, 1, 1)
t = np.array([3, 0, 0])

of_mask = utils.compute_optical_flow(grid, mask.voxels, R, S, t)
plotOpticalFlow3D(of_mask, "of", "output", 1)
# mask_gray_w = mask.warp(grid, of_mask)
# plot_slices(mask.gray_values, str="mask", block=False)
# plot_slices(mask_gray_w, str="mask_w", block=True)


utils.plot_slices(ellip.gray_values, str="ellip", block=False)
utils.plot_slices(mask.gray_values, str="mask", block=False)
# of = utils.compute_optical_flow(grid, R, S, t)

# gray_w = mask.warp(grid, of)
# utils.plot_slices(gray_w, str="gray_w", block=False)


# 3D plot
# Voxel color
alpha = 0.3
voxel_color = np.empty([NZ, NY, NX, 4])
voxel_color[ellip.voxels] = [0, 1, 0, alpha]  # green
voxel_color[mask.voxels] = [0, 0, 1, alpha*2]  # blue
voxelarray = ellip.voxels | mask.voxels
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.voxels(np.transpose(voxelarray, (2, 1, 0)),
          facecolors=np.transpose(voxel_color, (2, 1, 0, 3)),
          edgecolor='k', linewidth=0.5)
ax.set(xlabel='X', ylabel='Y', zlabel='Z')
plt.show()
