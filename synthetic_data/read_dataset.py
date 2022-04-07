import numpy as np
import utils
from glob import glob
import time
import os.path as osp

BASE_PATH = "output"
vols_path = osp.join(BASE_PATH, "vol")
masks_path = osp.join(BASE_PATH, "mask")

vol_list = sorted(glob(osp.join(vols_path, "*.npy")))
mask_list = sorted(glob(osp.join(masks_path, "*.npy")))

for i,(v, m) in enumerate(zip(vol_list, mask_list)):
    vol = np.load(v)
    mask = np.load(m)
    utils.plot_slices(vol, str=f"vol_{i} with shape (NZ, NY, NX)", block=False)
    utils.plot_slices(mask, str=f"mask_{i} with shape (NZ, NY, NX)", block=True)


