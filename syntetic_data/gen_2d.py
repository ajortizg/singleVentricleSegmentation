import numpy as np
import objects as obj
import matplotlib.pyplot as plt
import cv2
import sys

np.set_printoptions(precision=3, suppress=True)

def create_grid(NY, NX):
    yy, xx = np.meshgrid(np.arange(NY), np.arange(NX), indexing="ij")
    return np.stack((xx, yy), axis=2).astype(np.float64)


def compute_optical_flow(grid, voxels, t):
    NY, NX = grid.shape[:2]
    # grid_t = np.zeros((NZ, NY, NX, 3), dtype=np.float64)
    grid_t = np.copy(grid)
    of = np.zeros((NY, NX, 2), dtype=np.float64)
    noise = np.random.rand(NY, NX, 2)

    for y in range(NY):
        for x in range(NX):
            if voxels[y, x]:
                grid_t[y, x, :] = grid[y, x, :] + t
                of[y, x, :] = grid_t[y, x, :] - grid[y, x, :]
            else:
                #     grid_t[y, x, :] = grid[y, x, :] + noise[y, x, :]
                of[y, x, :] = 0.0
            #     of[y, x, :] = grid_t[y, x, :] - grid[y, x, :]
                # print(grid[y, x, :], grid_t[y, x, :],
                #       of[y, x, :], grid[y, x, :] + of[y, x, :])
    return of


im1 = cv2.imread(
    "/home/antonio/Documents/vot_ws/sequences/tiger/color/00000001.jpg", cv2.IMREAD_GRAYSCALE)
im2 = cv2.imread(
    "/home/antonio/Documents/vot_ws/sequences/tiger/color/00000002.jpg", cv2.IMREAD_GRAYSCALE)
flo_uv = cv2.readOpticalFlow(
    "/home/antonio/Documents/vot_ws/sequences/tiger/flow/fwd/fwd_0001.flo")
print(flo_uv.shape)
print(im1.shape)

NY, NX = im1.shape
grid = create_grid(NY, NX)

gray_w = cv2.remap(im1, map1=(grid-flo_uv).astype(np.float32), map2=None,
                   borderMode=cv2.BORDER_REFLECT101, interpolation=cv2.INTER_LINEAR)


fig = plt.figure(0)
plt.imshow(im2, cmap="gray", origin="upper")
plt.title("im2")

fig = plt.figure(1)
plt.imshow(gray_w, cmap="gray", origin="upper")
plt.title("warp2")

plt.show()
sys.exit()

NY, NX = 4, 6

grid = create_grid(NY, NX)
ellip = obj.Ellipse(NX//2, NY//2, 1, 1)
ellip.create_voxels(grid)

print(ellip.gray_values)
# print(grid.shape)

of = compute_optical_flow(grid, ellip.voxels, np.array([2, 0]))

gray_w = ellip.warp(grid, of)
# gray_w = cv2.remap(ellip.gray_values.astype(np.float32), map1=(grid-of).astype(np.float32), map2=None,
#                    borderMode=cv2.BORDER_REFLECT101, interpolation=cv2.INTER_LINEAR)
print(gray_w)

# new_coords = grid - of
# gray_w = np.zeros((NY,NX))


fig = plt.figure(0)
plt.imshow(ellip.gray_values, cmap="gray", origin="upper")
plt.title("gray")

fig = plt.figure(1)
plt.imshow(gray_w, cmap="gray", origin="upper")
plt.title("gray_w")

plt.show()
