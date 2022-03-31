from email.mime import base
import numpy as np
from objects import *
import matplotlib.pyplot as plt
import utils
import sys
from transforms import rotz, scale
import os.path as osp

utils_path = osp.abspath(osp.join(osp.dirname(__file__), '../utils'))
sys.path.append(utils_path)
from flow_viz import plotOpticalFlow3D

np.set_printoptions(precision=2, suppress=True)


# # Dimension of the volume
# NX, NY, NZ = 32, 44, 14
# grid = utils.create_grid(NZ, NY, NX)

# ellip = Ellipsoid(NX//2, NY//2, NZ//2, 12, 20, 14)
# ellip.create_voxels(grid, low=0.2, high=1.0)

# mask = Ellipsoid(NX//2.2, round(NY*0.6), 0, 3, 6, 16)
# mask.create_voxels(grid, constant=True, value=1.0)

# # prob = 0.02

# for z in range(NZ):
#     for y in range(NY):
#         for x in range(NX):
#             # # Add noise to ellipsoid
#             # d = np.random.rand(1)
#             # if d < prob:
#             #     s = np.random.randint(0, 1)
#             #     ellip.gray_values[z, y, x] = s
#             if mask.voxels[z, y, x]:
#                 ellip.gray_values[z, y, x] = 0.25 + \
#                     np.random.normal(0, 0.02, 1)


# np.save("output/base_object.npy", ellip.gray_values)
# np.save("output/base_mask.npy", mask.gray_values)


# utils.plot_slices(ellip.gray_values, str="ellip", block=False)
# utils.plot_slices(mask.gray_values, str="mask", block=False)

# # # 3D plot
# # # Voxel color
# alpha = 0.3
# voxel_color = np.empty([NZ, NY, NX, 4])
# voxel_color[ellip.voxels] = [0, 1, 0, alpha]  # green
# voxel_color[mask.voxels] = [0, 0, 1, alpha*2]  # blue
# voxelarray = ellip.voxels | mask.voxels
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.voxels(np.transpose(voxelarray, (2, 1, 0)),
#           facecolors=np.transpose(voxel_color, (2, 1, 0, 3)),
#           edgecolor='k', linewidth=0.5)
# ax.set(xlabel='X', ylabel='Y', zlabel='Z')
# plt.show()

def compute_new_grid(grid, S, R, t):
    NZ, NY, NX = grid.shape[:3]
    grid_t = np.zeros((NZ, NY, NX, 3), dtype=np.float64)
    # grid_t = np.copy(grid)
    of = np.zeros((NZ, NY, NX, 3), dtype=np.float64)
    R_left = np.linalg.inv(R)

    for z in range(NZ):
        for y in range(NY):
            for x in range(NX):
                # if voxels[z, y, x]:
                grid_t[z, y, x, :] = S@(R@grid[z, y, x, :]) + t
                # if voxels[z, y, x]:
                of[z, y, x, :] = grid_t[z, y, x, :] - grid[z, y, x, :]
                # print(grid[z, y, x, :], grid_t[z, y, x, :],
                #       of[z, y, x, :], grid[z, y, x, :] + of[z, y, x, :])
    return grid_t, of


NX, NY, NZ = 256, 332, 14
grid = utils.create_grid(NZ, NY, NX)

e1 = Ellipsoid(NX//2, NY//2, NZ//2, 96, 160, 14)
e1.create_voxels(grid, constant=True, value=0.5)
# utils.plot_slices(e1.gray_values, str="e1", block=False)

e2 = Ellipsoid(NX//2, NY//2, NZ//2, 48, 80, 14)
e2.create_voxels(grid, constant=True, value=0.9)
# utils.plot_slices(e2.gray_values, str="e2", block=False)

e3 = Ellipsoid(NX//2, NY//2, NZ//2, 24, 40, 14)
e3.create_voxels(grid, constant=True, value=0.2)
# utils.plot_slices(e3.gray_values, str="e3", block=False)

# Combine three ellipsoids
base_obj = np.zeros((NZ, NY, NX))

for z in range(NZ):
    for y in range(NY):
        for x in range(NX):
            if e3.voxels[z, y, x]:
                base_obj[z, y, x] = e3.gray_values[z, y, x]
            elif e2.voxels[z, y, x]:
                base_obj[z, y, x] = e2.gray_values[z, y, x]
            elif e1.voxels[z, y, x]:
                base_obj[z, y, x] = e1.gray_values[z, y, x]

# --------- Move e3 --------
S3 = scale(1, 1, 1)
R3 = rotz(9)
t3 = np.array([15, 0, 0])
grid_t_e3, of3 = compute_new_grid(grid, S3, R3, t3)

e3_w = Ellipsoid(0, 0, 0, 0, 0, 0)
e3_w.gray_values = utils.warp_grid(e3.gray_values, grid-of3)
e3_w.update_voxels()

# --------- Move e2 --------
S2 = scale(1, 1, 1)
R2 = rotz(-9)
t2 = np.array([-40, 20, 0])
grid_t_e2, of2 = compute_new_grid(grid, S2, R2, t2)

e2_w = Ellipsoid(0, 0, 0, 0, 0, 0)
e2_w.gray_values = utils.warp_grid(e2.gray_values, grid-of2)
e2_w.update_voxels()

# --------- Move e1 --------
S1 = scale(1, 1, 1)
R1 = np.eye(3, 3)
t1 = np.array([-5, 0, 0])
grid_t_e1, of1 = compute_new_grid(grid, S1, R1, t1)

e1_w = Ellipsoid(0, 0, 0, 0, 0, 0)
e1_w.gray_values = utils.warp_grid(e1.gray_values, grid-of1)
e1_w.update_voxels()

final_obj = np.zeros((NZ, NY, NX))

for z in range(NZ):
    for y in range(NY):
        for x in range(NX):
            if e3_w.voxels[z, y, x]:
                final_obj[z, y, x] = e3_w.gray_values[z, y, x]
            elif e2_w.voxels[z, y, x]:
                final_obj[z, y, x] = e2_w.gray_values[z, y, x]
            elif e1_w.voxels[z, y, x]:
                final_obj[z, y, x] = e1_w.gray_values[z, y, x]

# Create one optical flow
of_b_to_f = np.zeros((NZ, NY, NX, 3), dtype=np.float64)
for z in range(NZ):
    for y in range(NY):
        for x in range(NX):
            if e3.voxels[z, y, x]:
                of_b_to_f[z, y, x] = of3[z, y, x]
            elif e2.voxels[z, y, x]:
                of_b_to_f[z, y, x] = of2[z, y, x]
            elif e1.voxels[z, y, x]:
                of_b_to_f[z, y, x] = of1[z, y, x]


plotOpticalFlow3D(of_b_to_f, "flo", "output/flow", 1)

final_w = utils.warp_grid(base_obj, grid+of_b_to_f)

utils.plot_slices(base_obj, str="obj", block=False)
utils.plot_slices(final_w, str="final_w", block=False)
utils.plot_slices(final_obj, str="final1", block=True)
