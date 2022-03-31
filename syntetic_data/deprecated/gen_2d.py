# im1 = cv2.imread(
#     "/home/antonio/Documents/vot_ws/sequences/tiger/color/00000001.jpg", cv2.IMREAD_GRAYSCALE)
# im2 = cv2.imread(
#     "/home/antonio/Documents/vot_ws/sequences/tiger/color/00000002.jpg", cv2.IMREAD_GRAYSCALE)
# flo_uv = cv2.readOpticalFlow(
#     "/home/antonio/Documents/vot_ws/sequences/tiger/flow/fwd/fwd_0001.flo")
# print(flo_uv.shape)
# print(im1.shape)

# NY, NX = im1.shape
# grid = create_grid(NY, NX)

# gray_w = cv2.remap(im1, map1=(grid-flo_uv).astype(np.float32), map2=None,
#                    borderMode=cv2.BORDER_REFLECT101, interpolation=cv2.INTER_LINEAR)

# fig = plt.figure(0)
# plt.imshow(im2, cmap="gray", origin="upper")
# plt.title("im2")

# fig = plt.figure(1)
# plt.imshow(gray_w, cmap="gray", origin="upper")
# plt.title("warp2")

# plt.show()


# from flow_viz import plotOpticalFlow2D
import numpy as np
from pyparsing import original_text_for
import objects as obj
import matplotlib.pyplot as plt
import cv2
from transforms import rot2d
import utils
import sys
import os.path as osp

utils_path = osp.abspath(osp.join(osp.dirname(__file__), "../utils"))
sys.path.append(utils_path)

np.set_printoptions(precision=3, suppress=True)


def create_grid(NY, NX):
    yy, xx = np.meshgrid(np.arange(NY), np.arange(NX), indexing="ij")
    return np.stack((xx, yy), axis=2).astype(np.float64)


def compute_optical_flow(grid, mask, M):
    NY, NX = grid.shape[:2]
    # new_coords = np.zeros((NY, NX), dtype=np.float64)
    # new_coords = np.copy(grid)
    of = np.zeros((NY, NX, 2), dtype=np.float64)
    # noise = np.random.rand(NY, NX, 2)
    # Ri = np.linalg.inv(R)
    M = np.vstack((M, [0, 0, 1]))
    print(M)

    for y in range(NY):
        for x in range(NX):
            # if mask[y, x]:
            ch = np.hstack((grid[y, x, :], 1.0))
            nc = np.dot(M, ch)
            # new_coords = R @ grid[y, x, :] + t
            new_coords = nc[:2]
            of[y, x, :] = new_coords - grid[y, x, :]
            # else:
            # new_coords = R @ grid[y, x, :] + t / 2
            # of[y, x, :] = new_coords - grid[y, x, :]
            # nx = M[0, 0] * grid[y, x, 0] + M[0, 1] * grid[y, x, 1] + M[0, 2]
            # ny = M[1, 0] * grid[y, x, 0] + M[1, 1] * grid[y, x, 1] + M[1, 2]
            # new_coords = [nx, ny]
            # of[y, x, :] = new_coords - grid[y, x, :]
        # grid_t[y, x, :] = grid[y, x, :] + noise[y, x, :]
        # of[y, x, :] = grid_t[y, x, :] - grid[y, x, :]
        # print(grid[y, x, :], grid_t[y, x, :],
        #    of[y, x, :], grid[y, x, :] + of[y, x, :])
    return of


NY, NX = 40, 55
CY, CX = NY // 2, NX // 2

img = np.ones((NY, NX))
# gray = np.linspace(0.1, 0.9, NX)
# for x in range(NX):
#     img[:, x] = gray[x]

grid = create_grid(NY, NX)
e1 = obj.Ellipse(NX // 2, NY // 2, 5, 10)
e1.create_voxels(grid)

for y in range(NY):
    for x in range(NX):
        if e1.voxels[y, x]:
            img[y, x] = 0.0

# R = rot2d(10)
# t = np.array([5, 0])

M = cv2.getRotationMatrix2D((CX, CY), 45, 1.0)
print(M)

rot = cv2.warpAffine(img, M, (NX, NY))

flo = compute_optical_flow(grid, e1.voxels, M)

# plotOpticalFlow2D(flo, "flo", "output/flow", 1)

# gray_w = utils.warp_grid(e1.gray_values, grid - flo)
# gray_w = e1.warp(grid, flo)

gray_w = cv2.remap(img.astype(np.float32),
                   map1=(grid - flo).astype(np.float32),
                   map2=None,
                   borderMode=cv2.BORDER_CONSTANT,
                   interpolation=cv2.INTER_LINEAR)

fig = plt.figure()
plt.quiver(grid[:, :, 0],
           grid[:, :, 1],
           flo[:, :, 0],
           flo[:, :, 1],
           angles='xy',
           units='dots',
           scale_units='xy',
           color='g')
plt.imshow(img, cmap="gray", origin="upper")
plt.title("base")

fig = plt.figure()
plt.imshow(rot, cmap="gray", origin="upper")
plt.title("rot cv")

fig = plt.figure()
plt.imshow(gray_w, cmap="gray", origin="upper")
plt.title("warp")

plt.show()
