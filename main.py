import nibabel as nib
from utils.config import *
import numpy as np
from utils.plots import *
from tvl1.tvl1 import TVL1
import os

np.set_printoptions(precision=2)

# Load 4D nifty [x,y,z,t]
vol = nib.load(os.path.sep.join([VOLUMES_PATH, "Adult_30.nii.gz"]))
data = vol.get_fdata()
print(f"Dims: {data.ndim}, shape: {data.shape}, type: {data.dtype}")

# Get two adjacent volumes
ti, tf = 8, 9
source = data[:, :, :, ti]
target = data[:, :, :, tf]
# plot_slices(source)

# Compute the optical flow
alg = TVL1()
alg.compute(source, target)

# u = np.random.randn(3, 4, 2)
# v = np.random.randn(3, 4, 2)
# w = np.random.randn(3, 4, 2)
# uvw = np.array([u, v, w])
# print(uvw.shape)
# # print(u)
# # print(uv[1,:,:]==v)

# rows, cols = u.shape
# # xx, yy = np.meshgrid(range(cols), range(rows), indexing='xy')

# # grid = np.array([xx, yy])

# # print("uv\n", uv)
# # print("grid\n", grid)

# # sum = uv + grid
# # print("sum\n", sum)
# # print(sum.shape)

# x = np.array([[1, 2, 3],
#               [4, 5, 6],
#               [7, 8, 9]])
# print(x**2)
