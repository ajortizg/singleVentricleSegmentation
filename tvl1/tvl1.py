import numpy as np
from scipy import ndimage
from derivatives.derivatives import Nabla3D_Central as Nabla
from utils.plots import *
import torch
from utils.config import *
import sys
sys.path.append("../utils")
sys.path.append("../derivatives")


class TVL1:
    def __init__(self):
        # binomial filter for restriction operator
        self.binomial = (1.0 / 256.0) * np.array([[[1, 4, 6, 4, 1],
                                                 [4, 16, 24, 16, 4],
                                                 [6, 24, 36, 24, 6],
                                                 [4, 16, 24, 16, 4],
                                                 [1, 4, 6, 4, 1]]])
        self.binomial = np.pad(self.binomial, ((2, 2), (0, 0), (0, 0)))
        self.binomial = np.transpose(self.binomial, (1, 2, 0))

    def compute(self, src, tgt):
        # Normalize the data between 0 and 255 [0, 1]
        src = self.normalize(src)
        tgt = self.normalize(tgt)

        # Smooth volumes with a Gaussian filter
        src = ndimage.gaussian_filter(src, sigma=SIGMA)
        tgt = ndimage.gaussian_filter(tgt, sigma=SIGMA)

        # List for volumes pyramids
        srcs = [src]
        tgts = [tgt]

        # List for store x,y,z components of the optical flow at multiple scales
        u = v = w = np.zeros(src.shape)
        uvws = [np.array([u, v, w])]

        # List of grids for warping
        grids = [self.create_grid(src)]

        # Create the gaussian pyramid
        for s in range(1, NUM_SCALES):
            srcs.append(self.restriction(srcs[s-1]))
            tgts.append(self.restriction(tgts[s-1]))

            u = v = w = np.zeros(srcs[s].shape)
            uvws.append(np.array([u, v, w]))

            grids.append(self.create_grid(srcs[s]))

        # Compute the optical flow at scale s
        for s in range(NUM_SCALES-1, -1, -1):
            self.step(srcs[s], tgts[s], uvws[s], grids[s])

        for i, s in enumerate(srcs):
            print(f"scale: {i}", s.shape, uvws[i].shape)
            block = False
            if i == len(srcs) - 1:
                block = True
            plot_slices(s, block)

    def step(self, src, tgt, uvw, grid):
        # Compute gradients
        nabla = Nabla()
        tgt_torch = torch.from_numpy(np.swapaxes(tgt, 0, 2)).to(DEVICE)
        tgt_grad = nabla.forward(tgt_torch).cpu().detach().numpy()

        for w in range(WARPS):
            new_coords = uvw + grid
            tgt_warp = ndimage.map_coordinates(
                tgt, new_coords, order=3, mode="constant")
        # plot_slices(tgt, str="tgt", block=False)
        # plot_slices(tgt_warp, str="warp", block=True)

    def create_grid(self, a):
        x, y, z = a.shape
        xx, yy, zz = np.meshgrid(range(x), range(y), range(z), indexing="ij")
        grid = np.array([xx, yy, zz])
        return grid

    def restriction(self, x):
        c = ndimage.convolve(x, self.binomial, mode="reflect")
        return ndimage.zoom(c, (ZOOM_FACTOR, ZOOM_FACTOR, 1.0), order=3, mode="reflect")

    def normalize(self, x):
        min = np.amin(x)
        max = np.amax(x)
        return x * 255.0 / max
