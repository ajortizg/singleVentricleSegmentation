import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import ndimage
import sys
import torch
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

# # # Cube
# # # Create the x,y,z axis
# # X = 12
# # Y = 14
# # Z = 8
# # axes = [X,Y,Z]

# # # Create the data
# # data = np.zeros(axes)
# # data[4:8,3:5,:]=1

# # # Control transparency
# # alpha = 0.9

# # colors = np.empty(axes+[4])
# # colors[0] = [1, 0, 0, alpha]  # red
# # colors[1] = [0, 1, 0, alpha]  # green
# # colors[2] = [0, 0, 1, alpha]  # blue
# # colors[3] = [1, 1, 0, alpha]  # yellow
# # colors[4] = [1, 1, 1, alpha]  # grey

# # fig = plt.figure()
# # ax = fig.add_subplot(111, projection="3d")
# # ax.voxels(data)
# # ax.set_xlabel("X")
# # ax.set_ylabel("Y")
# # ax.set_zlabel("Z")
# # plt.show()


# NX = 32
# NY = 32
# NZ = 24
# D = 8
# vol = np.zeros([NZ, NY, NX])

# CX = NX//2
# CY = NY//2
# CZ = NZ//2

# # Sphere
# for x in range(NX):
#     for y in range(NY):
#         for z in range(NZ):
#             dist = math.sqrt((CX-x)**2 + (CY-y)**2 + (CZ-z)**2)
#             if dist < D:
#                 vol[z, y, x] = 1
# # Cube
# # vol[CZ:CX+5, CY:CY+5, CX:CX+5]=1

# Tr = T.SE3(T.roty(4), [0, 0, 0])
# Tr = T.SE3(np.eye(3,3), [2,0,0])
# T_left = np.linalg.inv(Tr)

# zz, yy, xx = np.meshgrid(np.arange(NZ), np.arange(NY),
#                          np.arange(NX), indexing="ij")
# grid = np.stack((xx, yy, zz, np.ones_like(vol)), axis=3)

# # print(grid.shape)
# # print(grid[4, 6, 2, :])

# grid_w = np.zeros_like(grid)
# optical_flow = np.zeros([NZ, NY, NX, 3])
# for x in range(NX):
#     for y in range(NY):
#         for z in range(NZ):
#             grid_w[z, y, x, :] = Tr@grid[z, y, x, :]
#             optical_flow[z, y, x, :] = grid_w[z, y, x, :3] - grid[z, y, x, :3]
#             # print(grid[z, y, x, :], grid_w[z, y, x, :],
#             #       optical_flow[z, y, x, :], grid[z, y, x, :3] + optical_flow[z, y, x, :])


# # # grid_w = grid_w[:, :, :, :3]
# # new_xx = grid_w[:, :, :, 0]
# # new_yy = grid_w[:, :, :, 1]
# # new_zz = grid_w[:, :, :, 2]
# # Optical flow is negative because Coordinate system in python is left-handed
# # new_xx = -optical_flow[:, :, :, 0] + xx
# # new_yy = -optical_flow[:, :, :, 1] + yy
# # new_zz = -optical_flow[:, :, :, 2] + zz
# # vol_w = ndimage.map_coordinates(
# #     vol, [new_zz, new_yy, new_xx], order=1, mode="constant")

# vol_w = warp(vol, grid, optical_flow)
# vol_w[vol_w < 0.01] = 0

# plot_slices(vol, str="vol", block=False)
# plot_slices(vol_w, str="vol_w", block=False)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# ax.voxels(np.transpose(vol, (2, 1, 0)))
# ax.set_title("vol")
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")

# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# ax.set_title("vol_w")
# ax.voxels(np.transpose(vol_w, (2, 1, 0)))
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")

# plt.show()


# # # set the colors of each object
# # colors = np.empty(V.shape, dtype=object)
# # colors[link] = 'red'
# # colors[cube1] = 'blue'
# # colors[cube2] = 'green'

# # # and plot everything
# # ax = plt.figure().add_subplot(projection='3d')
# # ax.voxels(voxelarray, facecolors=colors, edgecolor='k')


# # # draw cuboids in the top left and bottom right corners, and a link between
# # # them
# # cube1 = (x < 3) & (y < 3) & (z < 3)
# # cube2 = (x >= 5) & (y >= 5) & (z >= 5)
# # link = abs(x - y) + abs(y - z) + abs(z - x) <= 2

# # # combine the objects into a single boolean array
# # voxelarray = cube1 | cube2 | link
# # print(voxelarray)


def create_grid(NZ, NY, NX):
    zz, yy, xx = np.meshgrid(np.arange(NZ), np.arange(NY),
                             np.arange(NX), indexing="ij")
    return np.stack((xx, yy, zz), axis=3)


NX, NY, NZ = 32, 44, 16
grid = create_grid(NZ, NY, NX)

sphere = Sphere(NX//2, NY//2, NZ//2, 9)
sphere.create_voxels(grid)

cube = Cube(NX//3, NY//2, NZ//2, 12, 6, NZ)
cube.create_voxels(grid)

voxelarray = sphere.voxels | cube.voxels

# Voxel color
alpha = 0.3
voxel_color = np.empty([NZ, NY, NX, 4])
voxel_color[sphere.voxels] = [0, 1, 0, alpha]  # green
voxel_color[cube.voxels] = [0, 0, 1, alpha*2]  # green

print(voxelarray.shape)
print(voxel_color.shape)

# 3D plot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.voxels(np.transpose(voxelarray, (2, 1, 0)), 
          edgecolor='k', linewidth=0.5)
ax.set(xlabel='X', ylabel='Y', zlabel='Z')

# voxel value
voxel_val = np.where(cube.voxels == True, 0.5,
                     np.where(sphere.voxels == True, 1, 0))
plot_slices(voxel_val, str="voxelarray", block=False)


grid_w = np.zeros_like(grid)
of = np.zeros([NZ, NY, NX, 3])
# R = T.rotz(20) * T.rotx(16) * T.roty(20)
R = T.rotz(12)
t = np.array([NX//2, NY//2, 0])

R_left = np.linalg.inv(R)

for x in range(NX):
    for y in range(NY):
        for z in range(NZ):
            # grid_w[z, y, x, :] = R@grid[z, y, x, :]
            grid_w[z, y, x, :] = R@grid[z, y, x, :]
            of[z, y, x, :] = grid_w[z, y, x, :] - grid[z, y, x, :]
            # print(grid[z, y, x, :], grid_w[z, y, x, :],
            #       of[z, y, x, :], grid[z, y, x, :] + of[z, y, x, :])

new_cords = grid + of
voxel_val_w = ndimage.map_coordinates(voxel_val,
                                      [new_cords[:, :, :, 2],
                                       new_cords[:, :, :, 1],
                                       new_cords[:, :, :, 0]],
                                      order=3, mode="constant")
plot_slices(voxel_val_w, str="voxel_val_w", block=False)

voxelarray_w = ndimage.map_coordinates(voxelarray,
                                       [new_cords[:, :, :, 2],
                                        new_cords[:, :, :, 1],
                                        new_cords[:, :, :, 0]],
                                       order=3, mode="constant")
voxelarray_w[voxelarray_w < 0.5] = 0
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.voxels(np.transpose(voxelarray_w, (2, 1, 0)),
          edgecolor='k', linewidth=0.5)
ax.set(xlabel='X', ylabel='Y', zlabel='Z')
plt.show()
