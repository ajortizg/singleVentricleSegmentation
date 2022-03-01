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
        us = [np.zeros((3, src.shape[0], src.shape[1], src.shape[2]))]

        # List of grids for warping
        grids = [self.create_grid(src)]

        # Create the gaussian pyramid
        for s in range(1, NUM_SCALES):
            srcs.append(self.restriction(srcs[s-1]))
            tgts.append(self.restriction(tgts[s-1]))
            us.append(
                np.zeros((3, srcs[s].shape[0], srcs[s].shape[1], srcs[s].shape[2])))
            grids.append(self.create_grid(srcs[s]))

        # Compute the optical flow at scale s
        for s in range(NUM_SCALES-1, -1, -1):
            self.step(srcs[s], tgts[s], us[s], grids[s])

        # for i, s in enumerate(srcs):
        #     print(f"scale: {i}", s.shape, uvws[i].shape)
        #     block = False
        #     if i == len(srcs) - 1:
        #         block = True
        #     plot_slices(s, block)

    def step(self, src, tgt, u, grid):
        # Dual variables for every dimension
        p1 = p2 = p3 = np.zeros((3, src.shape[0], src.shape[1], src.shape[2]))

        # Compute gradients
        nabla = Nabla()
        tgt_torch = torch.from_numpy(np.swapaxes(tgt, 0, 2)).to(DEVICE)
        tgt_grad = nabla.forward(tgt_torch).cpu().detach().numpy()
        tgt_grad = np.swapaxes(tgt_grad, 0, 3)

        for w in range(WARPS):
            # Compute the warping of the target image and its derivatives
            new_coords = u + grid
            tgt_w = ndimage.map_coordinates(tgt, new_coords)

            tgt_grad_w = np.array([
                ndimage.map_coordinates(tgt_grad[0, :, :, :], new_coords),
                ndimage.map_coordinates(tgt_grad[1, :, :, :], new_coords),
                ndimage.map_coordinates(tgt_grad[2, :, :, :], new_coords)])

            # |Grad1|^2
            tgt_grad_w2 = np.square(tgt_grad_w).sum(axis=0)

            # Constant part of the rho function
            rho_c = tgt_w \
                - u[0, :, :, :] * tgt_grad_w[0, :, :, :] \
                - u[1, :, :, :] * tgt_grad_w[1, :, :, :] \
                - u[2, :, :, :] * tgt_grad_w[2, :, :, :] \
                - src

            for n in range(MAX_ITERATIONS):
                # Compute the fidelity data term \rho(u)
                rho = u[0, :, :, :] * tgt_grad_w[0, :, :, :] \
                    + u[1, :, :, :] * tgt_grad_w[1, :, :, :] \
                    + u[2, :, :, :] * tgt_grad_w[2, :, :, :] \
                    + rho_c

                # Proposition 3. Thresholding step to estimate v
                v = self.proposition3(u, rho, tgt_grad_w, tgt_grad_w2)

                # Divergence of dual variables
                p1_torch = torch.from_numpy(
                    p1.transpose(3, 2, 1, 0)).to(DEVICE)
                p2_torch = torch.from_numpy(
                    p2.transpose(3, 2, 1, 0)).to(DEVICE)
                p3_torch = torch.from_numpy(
                    p3.transpose(3, 2, 1, 0)).to(DEVICE)

                # Divergence shape [z,y,x]
                p1_div = nabla.backward(p1_torch).cpu().detach().numpy()
                p2_div = nabla.backward(p2_torch).cpu().detach().numpy()
                p3_div = nabla.backward(p3_torch).cpu().detach().numpy()

                # Divergence shape [3,x,y,z]
                p_div = np.array([np.swapaxes(p1_div, 0, 2),
                                  np.swapaxes(p2_div, 0, 2),
                                  np.swapaxes(p3_div, 0, 2)])

                # Compute the 3D optical flow Eq. 14
                u = v + THETA * p_div

                # Proposition 1
                # Compute the gradient of the optical flow
                u_torch = torch.from_numpy(np.swapaxes(u, 1, 3)).to(DEVICE)
                # shape [z,y,x,3]
                Jx = nabla.forward(u_torch[0, :, :, :])  # gradient of u_x
                Jy = nabla.forward(u_torch[1, :, :, :])  # gradient of u_y
                Jz = nabla.forward(u_torch[2, :, :, :])  # gradient of u_z
                # shape [3,x,y,z]
                Jx = np.transpose(Jx.cpu().detach().numpy(), (3, 2, 1, 0))
                Jy = np.transpose(Jy.cpu().detach().numpy(), (3, 2, 1, 0))
                Jz = np.transpose(Jz.cpu().detach().numpy(), (3, 2, 1, 0))

                p1 = p1 + (TAU / THETA) * Jx

        # plot_slices(tgt, str="tgt", block=False)
        # plot_slices(tgt_warp, str="warp", block=True)

    def proposition3(self, u, rho, grad, grad2):
        """
        Solution of the minimization task in Eq. 15
        """
        # v2 = np.where(rho < -LT * tgt_grad_w2,
        #               LT * tgt_grad_w,
        #               np.where(rho > LT * tgt_grad_w2,
        #                        -LT * tgt_grad_w,
        #                        (-rho / tgt_grad_w2) * tgt_grad_w if tgt_grad_w2.all() != 0. else np.zeros(3))
        #               )
        v = np.zeros(u.shape)
        w, h, d = rho.shape[:3]

        for x in range(w):
            for y in range(h):
                for z in range(d):
                    r = rho[x, y, z]
                    g2 = grad2[x, y, z]

                    if (r < - LT * g2):
                        delta = LT * grad[:, x, y, z]
                    elif (r > LT * g2):
                        delta = -LT * grad[:, x, y, z]
                    else:
                        if g2 < 1e-10:
                            delta = np.zeros(3)
                        else:
                            delta = (- r / g2) * grad[:, x, y, z]

                    v[:, x, y, z] = u[:, x, y, z] + delta
        return v

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
