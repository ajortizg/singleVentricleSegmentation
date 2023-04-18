#==================================
import sys
# sys.path.append('core')

import numpy as np
from scipy import ndimage
import torch
import time
from tqdm import tqdm


sys.path.append("../utils")
from utilities.plots import *
from utilities.config import *
from utilities.flow_viz import *

# sys.path.append("../pythonOps/")
# from pythonOps.mesh import *
# from pythonOps.differentialOps import *
pythonOps_lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../pythonOps'))
sys.path.append(pythonOps_lib_path)
import mesh
import differentialOps

from opticalFlow_cuda_ext import opticalFlow


class TVL1Scipy:
    def __init__(self,saveDir):
        self.saveDir = saveDir
        # binomial filter for restriction operator
        # self.binomial = (1.0 / 256.0) * np.array([[[1, 4, 6, 4, 1],
        #                                          [4, 16, 24, 16, 4],
        #                                          [6, 24, 36, 24, 6],
        #                                          [4, 16, 24, 16, 4],
        #                                          [1, 4, 6, 4, 1]]])
        # self.binomial = np.pad(self.binomial, ((2, 2), (0, 0), (0, 0)))
        # self.binomial = np.transpose(self.binomial, (1, 2, 0))
        x = 1

    def compute(self, I0, I1):
        # Normalize the data between 0 and 1
        # I0 = self.normalize(I0)
        # I1 = self.normalize(I1)

        # Smooth volumes with a Gaussian filter
        # I0 = ndimage.gaussian_filter(I0, sigma=SIGMA)
        # I1 = ndimage.gaussian_filter(I1, sigma=SIGMA)

        plot_slices(I0,str="I0", block=False)
        plot_slices(I1,str="I1",block=True)

        # List for volumes pyramids
        I0s = [I0]
        I1s = [I1]

        # List for store x,y,z components of the optical flow at multiple scales
        us = [np.zeros(I0.shape + (3,))]

        # Lists to store de dual variables for every dimension
        p1s = [np.zeros(I0.shape + (3,))]
        p2s = [np.zeros(I0.shape + (3,))]
        p3s = [np.zeros(I0.shape + (3,))]

        # List of grids for warping
        grids = [self.create_grid(I0)]

        # Create the gaussian pyramid
        for s in range(1, NUM_SCALES):
            I0s.append(self.restriction(I0s[s-1]))
            I1s.append(self.restriction(I1s[s-1]))
            us.append(np.zeros(I0s[s].shape + (3,)))
            p1s.append(np.zeros(I0s[s].shape + (3,)))
            p2s.append(np.zeros(I0s[s].shape + (3,)))
            p3s.append(np.zeros(I0s[s].shape + (3,)))
            grids.append(self.create_grid(I0s[s]))

        # Compute the optical flow at scale s
        pbar = tqdm(total=NUM_SCALES*MAX_WARPS *
                    MAX_OUTER_ITERATIONS * MAX_INNER_ITERATIOS)
        for s in range(NUM_SCALES-1, -1, -1):
            us[s], p1s[s], p2s[s], p3s[s] = self.step(
                I0s[s], I1s[s], us[s], grids[s], p1s[s], p2s[s], p3s[s], pbar)

            #save step
            plotOpticalFlow(us[s], "u", self.saveDir, s)
            save_slices(torch.from_numpy(I0s[s]).to(DEVICE),f"I0_it{s}.png", self.saveDir)
            save_slices(torch.from_numpy(I1s[s]).to(DEVICE),f"I1_it{s}.png", self.saveDir)
            xx, yy, zz = grids[s]
            new_xx = us[s][:, :, :, 0] + xx
            new_yy = us[s][:, :, :, 1] + yy
            new_zz = us[s][:, :, :, 2] + zz
            I1_warped = ndimage.map_coordinates(I1s[s], [new_zz, new_yy, new_xx], order=3, mode="reflect")
            save_slices(torch.from_numpy(I1_warped).to(DEVICE), f"I1_warped_it{s}.png", self.saveDir)

            if s == 0:
                break

            # Prolongate the optical flow and dual variables for the next pyramid level
            us[s-1] = self.prolongation(us[s])
            us[s-1] = INV_ZOOM_FACTOR * us[s-1]

            p1s[s] = self.dirichlet(p1s[s])
            p2s[s] = self.dirichlet(p2s[s])
            p3s[s] = self.dirichlet(p3s[s])

            p1s[s-1] = self.prolongation(p1s[s])
            p2s[s-1] = self.prolongation(p2s[s])
            p3s[s-1] = self.prolongation(p3s[s])

        pbar.close()

    def step(self, I0, I1, u, grid, p1, p2, p3, pbar):
        NZ = I0.shape[0]
        NY = I0.shape[1]
        NX = I0.shape[2]
        LZ = NZ-1
        LY = NY-1
        LX = NX-1
        meshInfo3D_python = mesh.MeshInfo3D(NZ,NY,NX,LZ,LY,LX)

        # Compute target image gradients
        nabla_ctrl = differentialOps.Nabla3D_Central(meshInfo3D_python)
        I1_grad = nabla_ctrl.forward(torch.from_numpy(I1).to(DEVICE)).cpu().detach().numpy()

        for w in range(MAX_WARPS):
            # Compute the warping of the target image and its derivatives
            xx, yy, zz = grid
            new_xx = u[:, :, :, 0] + xx
            new_yy = u[:, :, :, 1] + yy
            new_zz = u[:, :, :, 2] + zz
            I1w = ndimage.map_coordinates(
                I1, [new_zz, new_yy, new_xx], order=3, mode="reflect")

            I1w_gradx = ndimage.map_coordinates(
                I1_grad[:, :, :, 0], [new_zz, new_yy, new_xx], order=3, mode="reflect")
            I1w_grady = ndimage.map_coordinates(
                I1_grad[:, :, :, 1], [new_zz, new_yy, new_xx], order=3, mode="reflect")
            I1w_gradz = ndimage.map_coordinates(
                I1_grad[:, :, :, 2], [new_zz, new_yy, new_xx], order=3, mode="reflect")

            # |Grad|^2
            # tgt_grad_w2 = np.square(tgt_grad_w).sum(axis=3)
            I1w_grad2 = I1w_gradx**2 + I1w_grady**2 + I1w_gradz**2

            # Constant part of the rho function
            rho_c = I1w \
                - u[:, :, :, 0] * I1w_gradx \
                - u[:, :, :, 1] * I1w_grady \
                - u[:, :, :, 2] * I1w_gradz \
                - I0

            for n in range(MAX_OUTER_ITERATIONS):
                # Compute the fidelity data term \rho(u)
                rho = u[:, :, :, 0] * I1w_gradx \
                    + u[:, :, :, 1] * I1w_grady \
                    + u[:, :, :, 2] * I1w_gradz \
                    + rho_c

                # Proposition 3. Thresholding step to estimate v
                I1w_grad = np.stack( (I1w_gradx, I1w_grady, I1w_gradz), axis=3)
                v = self.thresholding(u, rho, I1w_grad, I1w_grad2)

                #self.updateDualVariable(u,v,p1,p2,p3)

                for m in range(MAX_INNER_ITERATIOS):
                    # Divergence of dual variables
                    p1_div = nabla_ctrl.backward(torch.from_numpy(p1).to(DEVICE)).cpu().detach().numpy()
                    p2_div = nabla_ctrl.backward(torch.from_numpy(p2).to(DEVICE)).cpu().detach().numpy()
                    #OLD: ERROR 
                    # p3_div = nabla_ctrl.backward(torch.from_numpy(p2).to(DEVICE)).cpu().detach().numpy()
                    p3_div = nabla_ctrl.backward(torch.from_numpy(p3).to(DEVICE)).cpu().detach().numpy()
                    p_div = np.stack((p1_div, p2_div, p3_div), axis=3)

                    # Compute the 3D optical flow Eq. 14
                    u = v - THETA * p_div  # TODO! check sign

                    # Proposition 1
                    # Compute the gradient of the optical flow using forward differences
                    # nabla_fwd = NablaForward()
                    u_torch = torch.from_numpy(u).to(DEVICE)
                    u_gradx = nabla_ctrl.forward(u_torch[:, :, :, 0]).cpu().detach().numpy()  # gradient of u_x
                    u_grady = nabla_ctrl.forward(u_torch[:, :, :, 1]).cpu().detach().numpy()  # gradient of u_y
                    u_gradz = nabla_ctrl.forward(u_torch[:, :, :, 2]).cpu().detach().numpy()  # gradient of u_z

                    p1_tilde = p1 + (TAU / THETA) * u_gradx
                    p2_tilde = p2 + (TAU / THETA) * u_grady
                    p3_tilde = p3 + (TAU / THETA) * u_gradz

                    p1_tilde_norm = np.sqrt(p1_tilde[:, :, :, 0]**2 +
                                            p1_tilde[:, :, :, 1]**2 + p1_tilde[:, :, :, 2]**2)
                    p2_tilde_norm = np.sqrt(p2_tilde[:, :, :, 0]**2 +
                                            p2_tilde[:, :, :, 1]**2 + p2_tilde[:, :, :, 2]**2)
                    p3_tilde_norm = np.sqrt(p3_tilde[:, :, :, 0]**2 +
                                            p3_tilde[:, :, :, 1]**2 + p3_tilde[:, :, :, 2]**2)

                    den1 = np.maximum(1, p1_tilde_norm)
                    den2 = np.maximum(1, p2_tilde_norm)
                    den3 = np.maximum(1, p3_tilde_norm)

                    p1[:, :, :, 0] = p1_tilde[:, :, :, 0] / den1
                    p1[:, :, :, 1] = p1_tilde[:, :, :, 1] / den1
                    p1[:, :, :, 2] = p1_tilde[:, :, :, 2] / den1

                    p2[:, :, :, 0] = p2_tilde[:, :, :, 0] / den2
                    p2[:, :, :, 1] = p2_tilde[:, :, :, 1] / den2
                    p2[:, :, :, 2] = p2_tilde[:, :, :, 2] / den2

                    p3[:, :, :, 0] = p3_tilde[:, :, :, 0] / den3
                    p3[:, :, :, 1] = p3_tilde[:, :, :, 1] / den3
                    p3[:, :, :, 2] = p3_tilde[:, :, :, 2] / den3

                    pbar.update(1)

        # plot_slices(I1, str="tgt", block=False)
        # plot_slices(I1w, str="tgt_w", block=True)

        return u, p1, p2, p3

    def dirichlet(self, x):
        x1 = x[:, :, :, 0]
        x2 = x[:, :, :, 1]
        x3 = x[:, :, :, 2]

        x1 = self.zero_border(x1)
        x2 = self.zero_border(x2)
        x3 = self.zero_border(x3)

        return np.stack((x1, x2, x3), axis=3)

    def zero_border(self, x):
        x[:, :, 0] = 0
        x[:, :, -1] = 0
        x[:, 0, :] = 0
        x[:, -1, :] = 0
        x[0, :, :] = 0
        x[-1, :, :] = 0
        return x

    def thresholding(self, u, rho, I1w_grad, I1w_grad2):
        """
        Solution of the minimization task in Eq. 15
        """
        v = np.zeros(u.shape)
        d, h, w = rho.shape[:3]

        for x in range(w):
            for y in range(h):
                for z in range(d):
                    r = rho[z, y, x]
                    g2 = I1w_grad2[z, y, x]

                    if (r < - LT * g2):
                        delta = LT * I1w_grad[z, y, x, :]
                    elif (r > LT * g2):
                        delta = -LT * I1w_grad[z, y, x, :]
                    else:
                        if g2 < 1e-10:
                            delta = np.zeros(3)
                        else:
                            delta = - r * I1w_grad[z, y, x, :] / g2

                    v[z, y, x, :] = u[z, y, x, :] + delta

        return v

    def create_grid(self, a):
        """
        Generate grid of x,y,z coordinates for each voxel
        """
        z, y, x = a.shape
        zz, yy, xx = np.meshgrid(range(z), range(y), range(x), indexing="ij")
        return (xx, yy, zz)

    def restriction(self, x):
        return ndimage.zoom(x, DOWN_FACTOR, order=3, prefilter=True, mode="reflect")

    def prolongation(self, x):
        x1 = x[:, :, :, 0]
        x2 = x[:, :, :, 1]
        x3 = x[:, :, :, 2]

        x1i = ndimage.zoom(x1, UP_FACTOR, order=3, mode="constant")
        x2i = ndimage.zoom(x2, UP_FACTOR, order=3, mode="constant")
        x3i = ndimage.zoom(x3, UP_FACTOR, order=3, mode="constant")

        xi = np.stack((x1i, x2i, x3i), axis=3)
        return xi

    def normalize(self, x):
        min = np.amin(x)
        max = np.amax(x)
        return x * 1.0 / max  # return 2.0 * x / max - 1.0


    def updateDualVariable(self,u,v,p1,p2,p3,meshInfo3D_python):
        nabla_ctrl = differentialOps.Nabla3D_Central(meshInfo3D_python)
        for m in range(MAX_INNER_ITERATIOS):
            # Divergence of dual variables
            p1_div = nabla_ctrl.backward(torch.from_numpy(p1).to(DEVICE)).cpu().detach().numpy()
            p2_div = nabla_ctrl.backward(torch.from_numpy(p2).to(DEVICE)).cpu().detach().numpy()
            p3_div = nabla_ctrl.backward(torch.from_numpy(p3).to(DEVICE)).cpu().detach().numpy()
            p_div = np.stack((p1_div, p2_div, p3_div), axis=3)

            # Compute the 3D optical flow Eq. 14
            u = v - THETA * p_div  # TODO! check sign

            # Proposition 1
            # Compute the gradient of the optical flow using forward differences
            # nabla_fwd = NablaForward()
            u_torch = torch.from_numpy(u).to(DEVICE)
            u_gradx = nabla_ctrl.forward(u_torch[:, :, :, 0]).cpu().detach().numpy()  # gradient of u_x
            u_grady = nabla_ctrl.forward(u_torch[:, :, :, 1]).cpu().detach().numpy()  # gradient of u_y
            u_gradz = nabla_ctrl.forward(u_torch[:, :, :, 2]).cpu().detach().numpy()  # gradient of u_z

            p1_tilde = p1 + (TAU / THETA) * u_gradx
            p2_tilde = p2 + (TAU / THETA) * u_grady
            p3_tilde = p3 + (TAU / THETA) * u_gradz

            p1_tilde_norm = np.sqrt(p1_tilde[:, :, :, 0]**2 + p1_tilde[:, :, :, 1]**2 + p1_tilde[:, :, :, 2]**2)
            p2_tilde_norm = np.sqrt(p2_tilde[:, :, :, 0]**2 + p2_tilde[:, :, :, 1]**2 + p2_tilde[:, :, :, 2]**2)
            p3_tilde_norm = np.sqrt(p3_tilde[:, :, :, 0]**2 + p3_tilde[:, :, :, 1]**2 + p3_tilde[:, :, :, 2]**2)

            den1 = np.maximum(1, p1_tilde_norm)
            den2 = np.maximum(1, p2_tilde_norm)
            den3 = np.maximum(1, p3_tilde_norm)

            p1[:, :, :, 0] = p1_tilde[:, :, :, 0] / den1
            p1[:, :, :, 1] = p1_tilde[:, :, :, 1] / den1
            p1[:, :, :, 2] = p1_tilde[:, :, :, 2] / den1

            p2[:, :, :, 0] = p2_tilde[:, :, :, 0] / den2
            p2[:, :, :, 1] = p2_tilde[:, :, :, 1] / den2
            p2[:, :, :, 2] = p2_tilde[:, :, :, 2] / den2

            p3[:, :, :, 0] = p3_tilde[:, :, :, 0] / den3
            p3[:, :, :, 1] = p3_tilde[:, :, :, 1] / den3
            p3[:, :, :, 2] = p3_tilde[:, :, :, 2] / den3

        return u