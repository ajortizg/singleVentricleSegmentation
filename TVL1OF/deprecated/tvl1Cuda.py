#==================================
import sys

import math
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


#InterpolationTypeCuda = opticalFlow.InterpolationType.INTERPOLATE_LINEAR
InterpolationTypeCuda = opticalFlow.InterpolationType.INTERPOLATE_CUBIC_HERMITESPLINE


class TVL1OpticalFlowCuda:
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
        # x = 1

    def computeOnPyramid(self, I0, I1, u, p):
        # TODO should be done directly for input values
        # Normalize the data between 0 and 1
        # I0 = self.normalize(I0)
        # I1 = self.normalize(I1)

        # TODO necessary?
        # Smooth volumes with a Gaussian filter
        # I0 = ndimage.gaussian_filter(I0, sigma=SIGMA)
        # I1 = ndimage.gaussian_filter(I1, sigma=SIGMA)

        # plot_slices(I0,str="I0", block=False)
        # plot_slices(I1,str="I1",block=True)

        # List for volumes pyramids
        NZ = I0.shape[0]
        NY = I0.shape[1]
        NX = I0.shape[2]
        #TODO possibly change lenght scale
        LZ = NZ-1
        LY = NY-1
        LX = NX-1
        #meshInfo3D_python = MeshInfo3D(NZ,NY,NX,LZ,LY,LX)
        meshInfo3D_cuda = opticalFlow.MeshInfo3D(NZ,NY,NX,LZ,LY,LX)

        #meshes for pyramid
        meshInfos = [meshInfo3D_cuda]
        NZ_restr = NZ
        NY_restr = NY
        NX_restr = NX
        for s in range(1, NUM_SCALES):
            NZ_restr = math.ceil(0.5*NZ_restr)
            NY_restr = math.ceil(0.5*NY_restr)
            NX_restr = math.ceil(0.5*NX_restr)
            LZ_restr = NZ_restr-1
            LY_restr = NY_restr-1
            LX_restr = NX_restr-1
            meshInfos.append(opticalFlow.MeshInfo3D(NZ_restr,NY_restr,NX_restr,LZ_restr,LY_restr,LX_restr))

        #lists for pyramid
        I0s = [I0]
        I1s = [I1]
        us = [u]
        ps = [p]

        # Create the pyramid
        for s in range(1, NUM_SCALES):
            prolongationOp_cuda = opticalFlow.Prolongation3D(meshInfos[s-1],meshInfos[s])
            I0s.append(prolongationOp_cuda.forward(I0s[s-1],InterpolationTypeCuda))
            I1s.append(prolongationOp_cuda.forward(I1s[s-1],InterpolationTypeCuda))
            us.append(torch.zeros([meshInfos[s].getNZ(),meshInfos[s].getNY(),meshInfos[s].getNX(),3]).float().to(DEVICE))
            ps.append(torch.zeros([meshInfos[s].getNZ(),meshInfos[s].getNY(),meshInfos[s].getNX(),3,3]).float().to(DEVICE))

        # Compute the optical flow at scale s
        print("start to compute optical flow for pyramid")
        progress_bar = tqdm(total=NUM_SCALES*MAX_WARPS * MAX_OUTER_ITERATIONS * MAX_INNER_ITERATIOS)
        for s in range(NUM_SCALES-1, -1, -1):
            print("step = ", s)
            us[s], ps[s] = self.computeOnSingleStep(I0s[s], I1s[s], us[s], ps[s], meshInfos[s], progress_bar)

            #save step
            plotOpticalFlow(us[s].cpu().detach().numpy(), "u", self.saveDir, s)
            save_slices(I0s[s],f"I0_it{s}.png", self.saveDir)
            save_slices(I1s[s],f"I1_it{s}.png", self.saveDir)
            warpingOp = opticalFlow.Warping3D(meshInfos[s])
            I1_warped = warpingOp.forward(I1s[s],us[s],InterpolationTypeCuda)
            save_slices(I1_warped, f"I1_warped_it{s}.png", self.saveDir)

            if s == 0:
                break

            # Prolongate the optical flow and dual variables for the next pyramid level
            prolongationOp_cuda = opticalFlow.Prolongation3D(meshInfos[s],meshInfos[s-1])
            us[s-1][:,:,:,0] = prolongationOp_cuda.forward(us[s][:,:,:,0].contiguous(),InterpolationTypeCuda)
            us[s-1][:,:,:,1] = prolongationOp_cuda.forward(us[s][:,:,:,1].contiguous(),InterpolationTypeCuda)
            us[s-1][:,:,:,2] = prolongationOp_cuda.forward(us[s][:,:,:,2].contiguous(),InterpolationTypeCuda)
            us[s-1] *= INV_ZOOM_FACTOR

            #TODO Dirichlet boundary
            # p1s[s] = self.dirichlet(p1s[s])
            # p2s[s] = self.dirichlet(p2s[s])
            # p3s[s] = self.dirichlet(p3s[s])

            ps[s-1][:,:,:,0,0] = prolongationOp_cuda.forward(ps[s][:,:,:,0,0].contiguous(),InterpolationTypeCuda)
            ps[s-1][:,:,:,0,1] = prolongationOp_cuda.forward(ps[s][:,:,:,0,1].contiguous(),InterpolationTypeCuda)
            ps[s-1][:,:,:,0,2] = prolongationOp_cuda.forward(ps[s][:,:,:,0,2].contiguous(),InterpolationTypeCuda)

            ps[s-1][:,:,:,1,0] = prolongationOp_cuda.forward(ps[s][:,:,:,1,0].contiguous(),InterpolationTypeCuda)
            ps[s-1][:,:,:,1,1] = prolongationOp_cuda.forward(ps[s][:,:,:,1,1].contiguous(),InterpolationTypeCuda)
            ps[s-1][:,:,:,1,2] = prolongationOp_cuda.forward(ps[s][:,:,:,1,2].contiguous(),InterpolationTypeCuda)

            ps[s-1][:,:,:,2,0] = prolongationOp_cuda.forward(ps[s][:,:,:,2,0].contiguous(),InterpolationTypeCuda)
            ps[s-1][:,:,:,2,1] = prolongationOp_cuda.forward(ps[s][:,:,:,2,1].contiguous(),InterpolationTypeCuda)
            ps[s-1][:,:,:,2,2] = prolongationOp_cuda.forward(ps[s][:,:,:,2,2].contiguous(),InterpolationTypeCuda)

        progress_bar.close()


    def computeOnSingleStep(self, I0, I1, u, p, meshInfo, progress_bar):

        print("start to compute optical flow for single step")

        # Compute target image gradients
        #nablaOp = Nabla3D_Central(meshInfo)
        nablaOp = opticalFlow.Nabla3D_CD(meshInfo)
        warpingOp = opticalFlow.Warping3D(meshInfo)
        I1_grad = nablaOp.forward(I1)

        for w in range(MAX_WARPS):
            # Compute the warping of the target image and its derivatives
            I1_warped = warpingOp.forward(I1,u,InterpolationTypeCuda)
            I1_warped_gradx = warpingOp.forward(I1_grad[:,:,:,0].contiguous(),u,InterpolationTypeCuda)
            I1_warped_grady = warpingOp.forward(I1_grad[:,:,:,1].contiguous(),u,InterpolationTypeCuda)
            I1_warped_gradz = warpingOp.forward(I1_grad[:,:,:,2].contiguous(),u,InterpolationTypeCuda)

            # Constant part of the rho function
            rho_c = I1_warped \
                - u[:, :, :, 0] * I1_warped_gradx \
                - u[:, :, :, 1] * I1_warped_grady \
                - u[:, :, :, 2] * I1_warped_gradz \
                - I0

            for n in range(MAX_OUTER_ITERATIONS):
                # Compute the fidelity data term \rho(u)
                rho = u[:, :, :, 0] * I1_warped_gradx \
                    + u[:, :, :, 1] * I1_warped_grady \
                    + u[:, :, :, 2] * I1_warped_gradz \
                    + rho_c

                # Proposition 3. Thresholding step to estimate v
                # vPython = self.thresholding(u, rho, I1_warped_gradx, I1_warped_grady, I1_warped_gradz, meshInfo)
                # ts = time.time()
                # vCuda = opticalFlow.TVL1OF_threshold(u, rho, I1_warped_gradx, I1_warped_grady, I1_warped_gradz, LT, meshInfo )
                # print('finished threshold - elapsed time cuda: ', (time.time()-ts))
                # print("diff = ", torch.max(torch.abs(vPython - vCuda)).item())
                v = opticalFlow.TVL1OF_threshold(u, rho, I1_warped_gradx, I1_warped_grady, I1_warped_gradz, LT, meshInfo )
      
                self.updateDualVariable(u,v,p,meshInfo)

                progress_bar.update(1)

        return u, p

    def updateDualVariable(self,u,v,p,meshInfo):
        nablaOp = opticalFlow.Nabla3D_CD(meshInfo)
        for m in range(MAX_INNER_ITERATIOS):
            # Divergence of dual variables
            p_div_x = nablaOp.backward(p[:,:,:,:,0].contiguous())
            p_div_y = nablaOp.backward(p[:,:,:,:,1].contiguous())
            p_div_z = nablaOp.backward(p[:,:,:,:,2].contiguous())

            # Compute the 3D optical flow Eq. 14 # TODO! check sign
            u[:,:,:,0] = v[:,:,:,0] - THETA * p_div_x 
            u[:,:,:,1] = v[:,:,:,1] - THETA * p_div_y
            u[:,:,:,2] = v[:,:,:,2] - THETA * p_div_z 

            # Proposition 1
            # Compute the gradient of the optical flow using forward differences
            # nabla_fwd = NablaForward()
            u_gradx = nablaOp.forward(u[:, :, :, 0].contiguous())
            u_grady = nablaOp.forward(u[:, :, :, 1].contiguous())
            u_gradz = nablaOp.forward(u[:, :, :, 2].contiguous())

            p_tilde_x = p[:,:,:,:,0] + (TAU / THETA) * u_gradx
            p_tilde_y = p[:,:,:,:,1] + (TAU / THETA) * u_grady
            p_tilde_z = p[:,:,:,:,2] + (TAU / THETA) * u_gradz

            # p_tilde_x_norm = np.sqrt(p_tilde_x[:, :, :, 0]**2 + p_tilde_x[:, :, :, 1]**2 + p_tilde_x[:, :, :, 2]**2)
            # p_tilde_y_norm = np.sqrt(p_tilde_y[:, :, :, 0]**2 + p_tilde_y[:, :, :, 1]**2 + p_tilde_y[:, :, :, 2]**2)
            # p_tilde_z_norm = np.sqrt(p_tilde_z[:, :, :, 0]**2 + p_tilde_z[:, :, :, 1]**2 + p_tilde_z[:, :, :, 2]**2)

            p_tilde_x_norm = torch.norm(p_tilde_x, dim=3)
            p_tilde_y_norm = torch.norm(p_tilde_y, dim=3)
            p_tilde_z_norm = torch.norm(p_tilde_z, dim=3)

            den_x = torch.clamp(p_tilde_x_norm, min=1.)
            den_y = torch.clamp(p_tilde_y_norm, min=1.)
            den_z = torch.clamp(p_tilde_z_norm, min=1.)

            p[:, :, :, 0,0] = p_tilde_x[:, :, :, 0] / den_x
            p[:, :, :, 1,0] = p_tilde_x[:, :, :, 1] / den_x
            p[:, :, :, 2,0] = p_tilde_x[:, :, :, 2] / den_x

            p[:, :, :, 0,1] = p_tilde_y[:, :, :, 0] / den_y
            p[:, :, :, 1,1] = p_tilde_y[:, :, :, 1] / den_y
            p[:, :, :, 2,1] = p_tilde_y[:, :, :, 2] / den_y

            p[:, :, :, 0,2] = p_tilde_z[:, :, :, 0] / den_z
            p[:, :, :, 1,2] = p_tilde_z[:, :, :, 1] / den_z
            p[:, :, :, 2,2] = p_tilde_z[:, :, :, 2] / den_z


    # def dirichlet(self, x):
    #     x1 = x[:, :, :, 0]
    #     x2 = x[:, :, :, 1]
    #     x3 = x[:, :, :, 2]

    #     x1 = self.zero_border(x1)
    #     x2 = self.zero_border(x2)
    #     x3 = self.zero_border(x3)

    #     return np.stack((x1, x2, x3), axis=3)

    # def zero_border(self, x):
    #     x[:, :, 0] = 0
    #     x[:, :, -1] = 0
    #     x[:, 0, :] = 0
    #     x[:, -1, :] = 0
    #     x[0, :, :] = 0
    #     x[-1, :, :] = 0
    #     return x

    # def thresholding(self, u, rho, I1_warped_gradx, I1_warped_grady, I1_warped_gradz, meshInfo):
    #     """
    #     Solution of the minimization task in Eq. 15
    #     """
    #     print("start threshold")
    #     ts = time.time()

    #     v = torch.zeros(u.shape).float().to(DEVICE)

    #     for x in range(meshInfo.getNX()):
    #         for y in range(meshInfo.getNY()):
    #             for z in range(meshInfo.getNZ()):
    #                 r = rho[z, y, x]
    #                 g2 = I1_warped_gradx[z,y,x].item()**2 + I1_warped_grady[z,y,x].item()**2 + I1_warped_gradz[z,y,x].item()**2

    #                 delta_x, delta_y, delta_z = 0.,0.,0.
    #                 if (r < - LT * g2):
    #                     delta_x = LT * I1_warped_gradx[z, y, x]
    #                     delta_y = LT * I1_warped_grady[z, y, x]
    #                     delta_z = LT * I1_warped_gradz[z, y, x]
    #                 elif (r > LT * g2):
    #                     delta_x = -LT * I1_warped_gradx[z, y, x]
    #                     delta_y = -LT * I1_warped_grady[z, y, x]
    #                     delta_z = -LT * I1_warped_gradz[z, y, x]
    #                 elif (g2 > 1e-10):
    #                     delta_x = - r * I1_warped_gradx[z, y, x] / g2
    #                     delta_y = - r * I1_warped_grady[z, y, x] / g2
    #                     delta_z = - r * I1_warped_gradz[z, y, x] / g2

    #                 v[z, y, x, 0] = u[z, y, x, 0] + delta_x
    #                 v[z, y, x, 1] = u[z, y, x, 1] + delta_y
    #                 v[z, y, x, 2] = u[z, y, x, 2] + delta_z

    #     print('finished threshold - elapsed time: ', (time.time()-ts))
    #     return v

    # def create_grid(self, a):
    #     """
    #     Generate grid of x,y,z coordinates for each voxel
    #     """
    #     z, y, x = a.shape
    #     zz, yy, xx = np.meshgrid(range(z), range(y), range(x), indexing="ij")
    #     return (xx, yy, zz)

    # def restriction(self, x):
    #     return ndimage.zoom(x, DOWN_FACTOR, order=3, prefilter=True, mode="reflect")

    # def prolongation(self, x):
    #     x1 = x[:, :, :, 0]
    #     x2 = x[:, :, :, 1]
    #     x3 = x[:, :, :, 2]

    #     x1i = ndimage.zoom(x1, UP_FACTOR, order=3, mode="constant")
    #     x2i = ndimage.zoom(x2, UP_FACTOR, order=3, mode="constant")
    #     x3i = ndimage.zoom(x3, UP_FACTOR, order=3, mode="constant")

    #     xi = np.stack((x1i, x2i, x3i), axis=3)
    #     return xi

    # def normalize(self, x):
    #     min = np.amin(x)
    #     max = np.amax(x)
    #     return x * 1.0 / max  # return 2.0 * x / max - 1.0
