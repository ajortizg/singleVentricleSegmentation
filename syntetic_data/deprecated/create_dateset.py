import numpy as np
import transforms as T
import matplotlib.pyplot as plt
import utils

# Load base object and segmentation mask
obj = np.load("output/base_object.npy")
mask = np.load("output/base_mask.npy")


# Number of transformations
N = 10
timestamps = np.arange(N)
rotations = np.random.rand(N, 3)
translations = np.random.rand(N, 3)

# Create some transformations
R = T.roty(0.4)
S = T.scale(1, 1, 1)
t = np.array([3, 0, 0])


utils.plot_slices(obj, str="ellip", block=False)
utils.plot_slices(mask, str="mask", block=True)

# for t in timestamps:
    
