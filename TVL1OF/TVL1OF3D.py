#==================================
import sys
import os
import math
import numpy as np
# from scipy import ndimage
import torch
import time
from tqdm import tqdm
import configparser


sys.path.append("../utils")
from utils.plots import *
# from utils.config import *
from utils.flow_viz import *

# sys.path.append("../pythonOps/")
# from pythonOps.mesh import *
# from pythonOps.differentialOps import *
pythonOps_lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../pythonOps'))
sys.path.append(pythonOps_lib_path)
# import mesh
# import differentialOps



from opticalFlow_cuda_ext import opticalFlow


#InterpolationTypeCuda = opticalFlow.InterpolationType.INTERPOLATE_LINEAR
#InterpolationTypeCuda = opticalFlow.InterpolationType.INTERPOLATE_CUBIC_HERMITESPLINE


class TVL1OpticalFlow3D:
    def __init__(self,saveDir,config):
        self.saveDir = saveDir
        self.NUM_SCALES = config.getint('PARAMETERS', 'NUM_SCALES')
        self.MAX_WARPS = config.getint('PARAMETERS', 'MAX_WARPS')
        self.MAX_OUTER_ITERATIONS = config.getint('PARAMETERS', 'MAX_OUTER_ITERATIONS')
        self.primalFctWeight_Matching = config.getfloat('PARAMETERS', 'primalFctWeight_Matching')
        self.dualFctWeight_TV = config.getfloat('PARAMETERS', 'dualFctWeight_TV')
        self.PRIMALDUAL_ALGO_TYPE = config.getint('PARAMETERS', 'PRIMALDUAL_ALGO_TYPE')
        self.sigma = config.getfloat('PARAMETERS', 'sigma')
        self.tau = config.getfloat('PARAMETERS', 'tau')
        self.theta = config.getfloat('PARAMETERS', 'theta')
        self.gamma = config.getfloat('PARAMETERS', 'gamma')
        #interpolation
        interType = config.get('PARAMETERS', 'InterpolationType')
        self.InterpolationTypeCuda = opticalFlow.InterpolationType.INTERPOLATE_LINEAR
        if interType == "LINEAR":
            self.InterpolationTypeCuda = opticalFlow.InterpolationType.INTERPOLATE_LINEAR
        elif interType == "CUBIC_HERMITESPLINE":
            self.InterpolationTypeCuda = opticalFlow.InterpolationType.INTERPOLATE_CUBIC_HERMITESPLINE
        #boundary
        boundaryType = config.get('PARAMETERS', 'BoundaryType')
        self.BoundaryTypeCuda = opticalFlow.BoundaryType.BOUNDARY_ZERO
        if boundaryType == "ZERO":
            self.BoundaryTypeCuda = opticalFlow.BoundaryType.BOUNDARY_ZERO
        elif boundaryType == "NEAREST":
            self.BoundaryTypeCuda = opticalFlow.BoundaryType.BOUNDARY_NEAREST
        elif boundaryType == "MIRROR":
            self.BoundaryTypeCuda = opticalFlow.BoundaryType.BOUNDARY_MIRROR
        elif boundaryType == "REFLECT":
            self.BoundaryTypeCuda = opticalFlow.BoundaryType.BOUNDARY_REFLECT
        #cuda
        cuda_availabe = config.get('DEVICE', 'cuda_availabe')
        self.DEVICE = "cuda" if cuda_availabe else "cpu"
        self.saveDirDebug = os.path.sep.join([self.saveDir, "debug"])
        self.useDebugOutput = config.getboolean("DEBUG","useDebugOutput")
        if self.useDebugOutput:
            os.makedirs(self.saveDirDebug)

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
        NZ, NY, NX = I0.shape[0], I0.shape[1], I0.shape[2]
        #TODO possibly change lenght scale
        LZ, LY, LX = NZ-1, NY-1, NX-1
        #meshInfo3D_python = MeshInfo3D(NZ,NY,NX,LZ,LY,LX)
        meshInfo3D_cuda = opticalFlow.MeshInfo3D(NZ,NY,NX,LZ,LY,LX)

        #meshes for pyramid
        meshInfos = [meshInfo3D_cuda]
        NZ_restr, NY_restr, NX_restr = NZ, NY, NX
        for s in range(1, self.NUM_SCALES):
            NZ_restr, NY_restr, NX_restr = math.ceil(0.5*NZ_restr), math.ceil(0.5*NY_restr), math.ceil(0.5*NX_restr)
            LZ_restr, LY_restr, LX_restr = NZ_restr-1, NY_restr-1, NX_restr-1
            meshInfos.append(opticalFlow.MeshInfo3D(NZ_restr,NY_restr,NX_restr,LZ_restr,LY_restr,LX_restr))

        #lists for pyramid
        I0s = [I0]
        I1s = [I1]
        us = [u]
        ps = [p]

        # Create the pyramid
        for s in range(1, self.NUM_SCALES):
            prolongationOp_cuda = opticalFlow.Prolongation3D(meshInfos[s-1],meshInfos[s])
            I0s.append(prolongationOp_cuda.forward(I0s[s-1],self.InterpolationTypeCuda))
            I1s.append(prolongationOp_cuda.forward(I1s[s-1],self.InterpolationTypeCuda))
            us.append(torch.zeros([meshInfos[s].getNZ(),meshInfos[s].getNY(),meshInfos[s].getNX(),3]).float().to(self.DEVICE))
            ps.append(torch.zeros([meshInfos[s].getNZ(),meshInfos[s].getNY(),meshInfos[s].getNX(),3,3]).float().to(self.DEVICE))

        return I0s, I1s, us, ps, meshInfos



    def computeOnPyramid(self, I0, I1, u, p):

        I0s, I1s, us, ps, meshInfos = self.generatePyramid(I0,I1,u,p)

        print("\n")
        print("============================================")
        print("start to compute optical flow for pyramid")
        print("============================================")
        print("\n")

        for s in range(self.NUM_SCALES-1, -1, -1):

            # Compute the optical flow at scale s
            us[s], ps[s] = self.computeOnSingleStep(s, I0s[s], I1s[s], us[s], ps[s], meshInfos[s])

            #save step
            self.saveSingleStepToFile(s,I0s[s],I1s[s],us[s],ps[s],meshInfos[s])

            if s == 0:
                break

            # Prolongate the optical flow and dual variables to the next pyramid level
            prolongationOp_cuda = opticalFlow.Prolongation3D(meshInfos[s],meshInfos[s-1])
            us[s-1] = prolongationOp_cuda.forwardVectorField(us[s],self.InterpolationTypeCuda)
            #factor LXNew/LXOld, ...
            us[s-1][:,:,:,0] *= meshInfos[s-1].getLX() / meshInfos[s].getLX()
            us[s-1][:,:,:,1] *= meshInfos[s-1].getLY() / meshInfos[s].getLY()
            us[s-1][:,:,:,2] *= meshInfos[s-1].getLZ() / meshInfos[s].getLZ()

            #TODO Dirichlet boundary condition for p?
            # ps[s] = self.dirichlet(ps[s])

            ps[s-1] = prolongationOp_cuda.forwardMatrixField(ps[s],self.InterpolationTypeCuda)

            #TODO prolongation factor for p?


    def computeOnSingleStep(self, s, I0, I1, u, p, meshInfo):

        print("start to compute optical flow for single step = ", s)
        progress_bar = tqdm(total = self.MAX_WARPS * self.MAX_OUTER_ITERATIONS)

        # Compute target image gradients
        #nablaOp = Nabla3D_Central(meshInfo)
        nablaOp = opticalFlow.Nabla3D_CD(meshInfo,self.BoundaryTypeCuda)
        warpingOp = opticalFlow.Warping3D(meshInfo)
        I1_grad = nablaOp.forward(I1)

        z = u

        for w in range(self.MAX_WARPS):
            # Compute the warping of the target image and its derivatives
            I1_warped = warpingOp.forward(I1,u,self.InterpolationTypeCuda)
            I1_warped_grad = warpingOp.forwardVectorField(I1_grad,u,self.InterpolationTypeCuda)
            # Constant part of the rho function
            #rho_c = I1_warped - u[:, :, :, 0] * I1_warped_grad[:, :, :, 0] - u[:, :, :, 1] * I1_warped_grad[:, :, :, 1] - u[:, :, :, 2] * I1_warped_grad[:, :, :, 2] - I0
            rho_c = I1_warped - torch.sum(u * I1_warped_grad, dim=3) - I0

            breakConditionVecPrimal = torch.zeros([self.MAX_OUTER_ITERATIONS]).float().to(self.DEVICE)
            breakConditionVecDual = torch.zeros([self.MAX_OUTER_ITERATIONS]).float().to(self.DEVICE)
            breakConditionVecUpdate = torch.zeros([self.MAX_OUTER_ITERATIONS]).float().to(self.DEVICE)

            # TODO possible use results for new warp
            sigma = self.sigma 
            tau = self.tau
            theta = self.theta 
            gamma = self.gamma

            for n in range(self.MAX_OUTER_ITERATIONS):

                # update of dual variable
                pold = p
                z_grad = nablaOp.forwardVectorField(z)
                dualVariable = p + sigma * z_grad
                p = opticalFlow.TVL1OF3D_proxDual( dualVariable, sigma, self.dualFctWeight_TV, meshInfo)
                breakConditionVecDual[n] = torch.norm(p-pold).item()
                if self.dualFctWeight_TV == 0.:
                   p = torch.zeros([meshInfo.getNZ(),meshInfo.getNY(),meshInfo.getNX(),3,3]).float().to(self.DEVICE)

                # update of primal variable
                uold = u
                # Compute the fidelity data term \rho(u)
                #rho = rho_c + u[:, :, :, 0] * I1_warped_grad[:, :, :, 0] + u[:, :, :, 1] * I1_warped_grad[:, :, :, 1] + u[:, :, :, 2] * I1_warped_grad[:, :, :, 2]
                rho = rho_c + torch.sum(u * I1_warped_grad, dim=3)
                #print("rho.norm = ", torch.norm(rho).item() )
                p_div = nablaOp.backwardVectorField(p)
                primalVariable = u - tau * p_div
                u = opticalFlow.TVL1OF3D_proxPrimal(primalVariable, tau, self.primalFctWeight_Matching, rho, I1_warped_grad, meshInfo )
                breakConditionVecPrimal[n] = torch.norm(u-uold).item()
                # if self.primalFctWeight_Matching == 0.:
                #     u = torch.zeros([meshInfo.getNZ(),meshInfo.getNY(),meshInfo.getNX(),3]).float().to(self.DEVICE)

                #update of stepsizes
                if self.PRIMALDUAL_ALGO_TYPE == 1:
                    # dot nothing 
                    print("apply CP1")
                elif self.PRIMALDUAL_ALGO_TYPE == 2:
                    theta = 1. / math.sqrt( 1. + 2. * gamma * tau )
                    tau *= theta
                    sigma /= theta
                else:
                    print("wrong method for Chambolle-Pock-Algorithm")

                # overrelaxation
                zold = z
                z = u
                breakConditionVecUpdate[n] = torch.norm(z-zold).item()

                progress_bar.update(1)

            if self.useDebugOutput:
                #saveCurve1D(primalFctVec, MAX_OUTER_ITERATIONS, self.saveDirDebug, f"PrimalFct_it{s}_warp{w}")
                saveCurve1D(breakConditionVecPrimal, self.MAX_OUTER_ITERATIONS, self.saveDirDebug, f"CPErrorPrimal_it{s}_warp{w}", "loglog")
                saveCurve1D(breakConditionVecDual, self.MAX_OUTER_ITERATIONS, self.saveDirDebug, f"CPErrorDual_it{s}_warp{w}", "loglog")
                saveCurve1D(breakConditionVecUpdate, self.MAX_OUTER_ITERATIONS, self.saveDirDebug, f"CPErrorUpdate_it{s}_warp{w}", "loglog")

        return u, p


    def saveSingleStepToFile(self,step,I0,I1,u,p,meshInfo):
        
        saveDirStep = os.path.sep.join([self.saveDir, f"it{step}"])
        if not os.path.exists(saveDirStep):
            os.makedirs(saveDirStep)

        plotOpticalFlow3D(u.cpu().detach().numpy(), "u", saveDirStep, step)
        save_slices(I0,f"I0_it{step}.png", saveDirStep)
        save_slices(I1,f"I1_it{step}.png", saveDirStep)
        warpingOp = opticalFlow.Warping3D(meshInfo)
        I1_warped = warpingOp.forward(I1,u,self.InterpolationTypeCuda)
        save_slices(I1_warped, f"I1_warped_it{step}.png", saveDirStep)
        save_slices(torch.abs(I1_warped-I0), f"Diff_I1warped_to_I0_it{step}.png", saveDirStep)

        save_single_zslices(I0, saveDirStep, "I0Slices", 1., 0)
        save_single_zslices(I1_warped, saveDirStep, "I1WarpedSlices", 1., 0)
        
        flowName = f"flow_it{step}.pt"
        fileNameFlow = os.path.join(saveDirStep, flowName) 
        torch.save(u, fileNameFlow)

        dualName = f"dual_it{step}.pt"
        fileNameDual = os.path.join(saveDirStep, dualName) 
        torch.save(p, fileNameDual)

    # def updateDualVariable(self,u,v,p,meshInfo):
    #     nablaOp = opticalFlow.Nabla3D_CD(meshInfo)
    #     for m in range(MAX_INNER_ITERATIOS):
    #         # Divergence of dual variables
    #         p_div_x = nablaOp.backward(p[:,:,:,:,0].contiguous())
    #         p_div_y = nablaOp.backward(p[:,:,:,:,1].contiguous())
    #         p_div_z = nablaOp.backward(p[:,:,:,:,2].contiguous())

    #         # Compute the 3D optical flow Eq. 14 # TODO! check sign
    #         u[:,:,:,0] = v[:,:,:,0] - THETA * p_div_x 
    #         u[:,:,:,1] = v[:,:,:,1] - THETA * p_div_y
    #         u[:,:,:,2] = v[:,:,:,2] - THETA * p_div_z 

    #         # Proposition 1
    #         # Compute the gradient of the optical flow using forward differences
    #         # nabla_fwd = NablaForward()
    #         u_gradx = nablaOp.forward(u[:, :, :, 0].contiguous())
    #         u_grady = nablaOp.forward(u[:, :, :, 1].contiguous())
    #         u_gradz = nablaOp.forward(u[:, :, :, 2].contiguous())

    #         p_tilde_x = p[:,:,:,:,0] + (TAU / THETA) * u_gradx
    #         p_tilde_y = p[:,:,:,:,1] + (TAU / THETA) * u_grady
    #         p_tilde_z = p[:,:,:,:,2] + (TAU / THETA) * u_gradz

    #         # p_tilde_x_norm = np.sqrt(p_tilde_x[:, :, :, 0]**2 + p_tilde_x[:, :, :, 1]**2 + p_tilde_x[:, :, :, 2]**2)
    #         # p_tilde_y_norm = np.sqrt(p_tilde_y[:, :, :, 0]**2 + p_tilde_y[:, :, :, 1]**2 + p_tilde_y[:, :, :, 2]**2)
    #         # p_tilde_z_norm = np.sqrt(p_tilde_z[:, :, :, 0]**2 + p_tilde_z[:, :, :, 1]**2 + p_tilde_z[:, :, :, 2]**2)

    #         p_tilde_x_norm = torch.norm(p_tilde_x, dim=3)
    #         p_tilde_y_norm = torch.norm(p_tilde_y, dim=3)
    #         p_tilde_z_norm = torch.norm(p_tilde_z, dim=3)

    #         den_x = torch.clamp(p_tilde_x_norm, min=1.)
    #         den_y = torch.clamp(p_tilde_y_norm, min=1.)
    #         den_z = torch.clamp(p_tilde_z_norm, min=1.)

    #         p[:, :, :, 0,0] = p_tilde_x[:, :, :, 0] / den_x
    #         p[:, :, :, 1,0] = p_tilde_x[:, :, :, 1] / den_x
    #         p[:, :, :, 2,0] = p_tilde_x[:, :, :, 2] / den_x

    #         p[:, :, :, 0,1] = p_tilde_y[:, :, :, 0] / den_y
    #         p[:, :, :, 1,1] = p_tilde_y[:, :, :, 1] / den_y
    #         p[:, :, :, 2,1] = p_tilde_y[:, :, :, 2] / den_y

    #         p[:, :, :, 0,2] = p_tilde_z[:, :, :, 0] / den_z
    #         p[:, :, :, 1,2] = p_tilde_z[:, :, :, 1] / den_z
    #         p[:, :, :, 2,2] = p_tilde_z[:, :, :, 2] / den_z


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
