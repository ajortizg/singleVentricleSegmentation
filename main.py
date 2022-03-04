import nibabel as nib
from utils.config import *
import numpy as np
from utils.plots import *
from tvl1.tvl1 import TVL1
import os

np.set_printoptions(precision=2, suppress=True)

# Load 4D nifty [x,y,z,t]
vol = nib.load(os.path.sep.join([VOLUMES_PATH, "Adult_30.nii.gz"]))
data = vol.get_fdata()
print(f"Dims: {data.ndim}, shape: {data.shape}, type: {data.dtype}")

# Get two adjacent volumes and change axes to z,y,x
ti, tf = 8, 9
I0 = np.swapaxes(data[:, :, :12, ti], 0, 2)
I1 = np.swapaxes(data[:, :, :12, tf], 0, 2)

# Compute the optical flow
alg = TVL1()
alg.compute(I0, I1)
