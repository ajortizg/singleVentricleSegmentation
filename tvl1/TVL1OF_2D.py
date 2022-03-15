#==================================
import sys
import os
import math
import numpy as np
import torch
import time
from tqdm import tqdm


sys.path.append("../utils")
from utils.plots import *
from utils.config import *
from utils.flow_viz import *
# utils_lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils'))
# sys.path.append(utils_lib_path)

# sys.path.append("../pythonOps/")
# from pythonOps.mesh import *
# from pythonOps.differentialOps import *
pythonOps_lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../pythonOps'))
sys.path.append(pythonOps_lib_path)
# import mesh
# import differentialOps



from opticalFlow_cuda_ext import opticalFlow


InterpolationTypeCuda = opticalFlow.InterpolationType.INTERPOLATE_LINEAR
#InterpolationTypeCuda = opticalFlow.InterpolationType.INTERPOLATE_CUBIC_HERMITESPLINE


class TVL1OpticalFlow2D:
    def __init__(self,saveDir):
        self.saveDir = saveDir

    def generatePyramid(self, I0, I1, u, p):

        # TODO should be done directly for input values
        # Normalize the data between 0 and 1
        # I0 = self.normalize(I0)
        # I1 = self.normalize(I1)

        # TODO necessary?
        # Smooth inputs with a Gaussian filter
        # I0 = ndimage.gaussian_filter(I0, sigma=SIGMA)
        # I1 = ndimage.gaussian_filter(I1, sigma=SIGMA)

        # List for volumes pyramids
        NY, NX = I0.shape[0], I0.shape[1]
        #TODO possibly change lenght scale
        LY, LX = NY-1, NX-1
        #meshInfo2D_python = MeshInfo2D(NY,NX,LY,LX)
        meshInfo2D_cuda = opticalFlow.MeshInfo2D(NY,NX,LY,LX)

        #meshes for pyramid
        meshInfos = [meshInfo2D_cuda]
        NY_restr, NX_restr = NY, NX
        for s in range(1, NUM_SCALES):
            NY_restr, NX_restr = math.ceil(0.5*NY_restr), math.ceil(0.5*NX_restr)
            LY_restr, LX_restr = NY_restr-1, NX_restr-1
            meshInfos.append(opticalFlow.MeshInfo2D(NY_restr,NX_restr,LY_restr,LX_restr))

        #lists for pyramid
        I0s = [I0]
        I1s = [I1]
        us = [u]
        ps = [p]

        # Create the pyramid
        for s in range(1, NUM_SCALES):
            prolongationOp_cuda = opticalFlow.Prolongation2D(meshInfos[s-1],meshInfos[s])
            I0s.append(prolongationOp_cuda.forward(I0s[s-1].contiguous(),InterpolationTypeCuda))
            I1s.append(prolongationOp_cuda.forward(I1s[s-1].contiguous(),InterpolationTypeCuda))
            us.append(torch.zeros([meshInfos[s].getNY(),meshInfos[s].getNX(),2]).float().to(DEVICE))
            ps.append(torch.zeros([meshInfos[s].getNY(),meshInfos[s].getNX(),2,2]).float().to(DEVICE))

        return I0s, I1s, us, ps, meshInfos



    def computeOnPyramid(self, I0, I1, u, p):

        I0s, I1s, us, ps, meshInfos = self.generatePyramid(I0,I1,u,p)

        # Compute the optical flow at scale s
        print("start to compute optical flow for pyramid")
        for s in range(NUM_SCALES-1, -1, -1):
            print("step = ", s)
            us[s], ps[s] = self.computeOnSingleStep(I0s[s], I1s[s], us[s], ps[s], meshInfos[s])

            #save step
            self.saveSingleStepToFile(s,I0s[s],I1s[s],us[s],ps[s],meshInfos[s])

            if s == 0:
                break

            # Prolongate the optical flow and dual variables for the next pyramid level
            prolongationOp_cuda = opticalFlow.Prolongation2D(meshInfos[s],meshInfos[s-1])
            us[s-1] = prolongationOp_cuda.forwardVectorField(us[s],InterpolationTypeCuda)
            #TODO factor LX/LXNew, ...
            us[s-1] *= INV_ZOOM_FACTOR

            #TODO Dirichlet boundary condition for p?
            # ps[s] = self.dirichlet(ps[s])

            ps[s-1] = prolongationOp_cuda.forwardMatrixField(ps[s],InterpolationTypeCuda)

            #TODO prolongation factor for p?


    def computeOnSingleStep(self, I0, I1, u, p, meshInfo):

        primalFctWeight_Matching = 1.
        dualFctWeight_TV = 25.
        weightNorm = 0.01

        print("start to compute optical flow for single step")
        progress_bar = tqdm(total=MAX_WARPS * MAX_OUTER_ITERATIONS)

        # Compute target image gradients
        #nablaOp = Nabla2D_Central(meshInfo)
        nablaOp = opticalFlow.Nabla2D_CD(meshInfo)
        warpingOp = opticalFlow.Warping2D(meshInfo)
        I1_grad = nablaOp.forward(I1.contiguous())

        z = u
        sigma = 0.5
        tau = 0.5

        for w in range(MAX_WARPS):
            # Compute the warping of the target image and its derivatives
            I1_warped = warpingOp.forward(I1.contiguous(),u,InterpolationTypeCuda)
            I1_warped_grad = warpingOp.forwardVectorField(I1_grad,u,InterpolationTypeCuda)
            # Constant part of the rho function
            rho_c = I1_warped - u[:, :, 0] * I1_warped_grad[:, :, 0] - u[:, :, 1] * I1_warped_grad[:, :, 1] - I0

            for n in range(MAX_OUTER_ITERATIONS):

                # update of dual variable
                pold = p
                u_grad = nablaOp.forwardVectorField(z)
                dualVariable = p + sigma * u_grad
                p = opticalFlow.TVL1OF2D_proxDual( dualVariable, sigma, dualFctWeight_TV, meshInfo)
                #print("|p-pold| = ", torch.norm(p-pold).item() )

                # update of primal variable
                # Compute the fidelity data term \rho(u)
                rho = rho_c + u[:, :, 0] * I1_warped_grad[:, :, 0] + u[:, :, 1] * I1_warped_grad[:, :,1]
                #print("rho.norm = ", torch.norm(rho).item() )
                p_div = nablaOp.backwardVectorField(p)
                primalVariable = u - tau * p_div
                u = opticalFlow.TVL1OF2D_proxPrimal(primalVariable, tau, primalFctWeight_Matching, rho, I1_warped_grad, weightNorm, meshInfo )

                # overrelaxation
                zold = z
                z = u
                #print("|z-zold| = ", torch.norm(z-zold).item() )

                progress_bar.update(1)

        return u, p


    def saveSingleStepToFile(self,step,I0,I1,u,p,meshInfo):
        
        saveDirStep = os.path.sep.join([self.saveDir, f"it{step}"])
        if not os.path.exists(saveDirStep):
            os.makedirs(saveDirStep)

        # plotOpticalFlow(u.cpu().detach().numpy(), "u", saveDirStep, step)
        # save_slices(I0,f"I0_it{step}.png", saveDirStep)
        # save_slices(I1,f"I1_it{step}.png", saveDirStep)
        warpingOp = opticalFlow.Warping2D(meshInfo)
        I1_warped = warpingOp.forward(I1.contiguous(),u,InterpolationTypeCuda)
        # save_slices(I1_warped, f"I1_warped_it{step}.png", saveDirStep)
        # save_slices(torch.abs(I1_warped-I0), f"Diff_I1warped_to_I0_it{step}.png", saveDirStep)

        # save_single_zslices(I0, saveDirStep, "I0Slices", 1., 0)
        # save_single_zslices(I1_warped, saveDirStep, "I1WarpedSlices", 1., 0)
        
        flowName = f"flow_it{step}.pt"
        fileNameFlow = os.path.join(saveDirStep, flowName) 
        torch.save(u, fileNameFlow)

        dualName = f"dual_it{step}.pt"
        fileNameDual = os.path.join(saveDirStep, dualName) 
        torch.save(p, fileNameDual)