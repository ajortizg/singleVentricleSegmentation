#==================================
import sys
import os
import math
import numpy as np
import torch
import time
import configparser
from tqdm import tqdm


sys.path.append("../utils")
from utilities.plots import *
# from utils.config import *
from utilities.flow_viz import *
# utils_lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils'))
# sys.path.append(utils_lib_path)

pythonOps_lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../pythonOps'))
sys.path.append(pythonOps_lib_path)
# import mesh
# import differentialOps

from opticalFlow_cuda_ext import opticalFlow


import torch.nn.functional as F

def generateGaussianKernel2D(sigma):
    kernelSize = int(sigma * 5)
    if kernelSize % 2 == 0:
        kernelSize += 1
    ts = torch.linspace(-kernelSize // 2, kernelSize // 2 + 1, kernelSize)
    gauss = torch.exp((-(ts / sigma)**2 / 2))
    kernel = gauss / gauss.sum()

    return kernel

def applyGaussianBlur2D(img, sigma):
    img_in = img.reshape(1, 1, *img.shape)
    k = generateGaussianKernel2D(sigma)
    k2d = torch.einsum('i,j->ij', k, k).cuda()
    k2d = k2d / k2d.sum()
    img_2d = F.conv2d(img_in, k2d.reshape(1, 1, *k2d.shape), stride=1, padding=len(k) // 2)
    img_out = img_2d.reshape(*img.shape)
    return img_out


class TVL1OpticalFlow2D:
    def __init__(self,saveDir,config):
        self.config = config
        self.saveDir = saveDir
        self.NUM_SCALES = config.getint('PARAMETERS', 'NUM_SCALES')
        self.MAX_WARPS = config.getint('PARAMETERS', 'MAX_WARPS')
        self.MAX_OUTER_ITERATIONS = config.getint('PARAMETERS', 'MAX_OUTER_ITERATIONS')
        self.weight_Matching = config.getfloat('PARAMETERS', 'weight_Matching')
        self.weight_TV = config.getfloat('PARAMETERS', 'weight_TV')
        self.PRIMALDUAL_ALGO_TYPE = config.getint('PARAMETERS', 'PRIMALDUAL_ALGO_TYPE')
        self.sigma = config.getfloat('PARAMETERS', 'sigma')
        self.tau = config.getfloat('PARAMETERS', 'tau')
        self.theta = config.getfloat('PARAMETERS', 'theta')
        self.gamma = config.getfloat('PARAMETERS', 'gamma')
        # anisotropic differential op
        self.useAnisotropicDifferentialOp = config.getboolean("PARAMETERS","useAnisotropicDifferentialOp")
        self.anisotropicDifferentialOp_alpha, self.anisotropicDifferentialOp_beta = None, None
        if self.useAnisotropicDifferentialOp:
            self.anisotropicDifferentialOp_alpha = config.getfloat("PARAMETERS","anisotropicDifferentialOp_alpha")
            self.anisotropicDifferentialOp_beta = config.getfloat("PARAMETERS","anisotropicDifferentialOp_beta")
        # blurring
        self.useGaussianBlur = config.getboolean("PARAMETERS","useGaussianBlur")
        self.GaussianBlurSigma = config.getfloat('PARAMETERS', 'GaussianBlurSigma')
        #interpolation
        interType = config.get('PARAMETERS', 'InterpolationType')
        self.InterpolationTypeCuda = None
        if interType == "LINEAR":
            self.InterpolationTypeCuda = opticalFlow.InterpolationType.INTERPOLATE_LINEAR
        elif interType == "CUBIC_HERMITESPLINE":
            self.InterpolationTypeCuda = opticalFlow.InterpolationType.INTERPOLATE_CUBIC_HERMITESPLINE
        else:
            raise Exception("wrong InterpolationType in configParser")
        #boundary
        boundaryType = config.get('PARAMETERS', 'BoundaryType')
        self.BoundaryTypeCuda = None
        if boundaryType == "NEAREST":
            self.BoundaryTypeCuda = opticalFlow.BoundaryType.BOUNDARY_NEAREST
        elif boundaryType == "MIRROR":
            self.BoundaryTypeCuda = opticalFlow.BoundaryType.BOUNDARY_MIRROR
        elif boundaryType == "REFLECT":
            self.BoundaryTypeCuda = opticalFlow.BoundaryType.BOUNDARY_REFLECT
        else:
            raise Exception("wrong BoundaryType in configParser")
        #cuda
        cuda_availabe = config.get('DEVICE', 'cuda_availabe')
        self.DEVICE = "cuda" if cuda_availabe else "cpu"
        #debug
        self.saveDirDebug = os.path.sep.join([self.saveDir, "debug"])
        self.useDebugOutput = config.getboolean("DEBUG","useDebugOutput")
        if self.useDebugOutput:
            os.makedirs(self.saveDirDebug)

    def getMeshLength(self,config,NY,NX):
        LenghtType = config.get('PARAMETERS', 'LenghtType')
        if LenghtType == "numDofs":
            LY = NY-1
            LX = NX-1
            return LY, LX
        elif LenghtType == "fixed":
            LY = config.getfloat('PARAMETERS', "LenghtY")
            LX = config.getfloat('PARAMETERS', "LenghtX")
            return LY, LX

    def generatePyramid(self, I0, I1, u, p):

        # Smooth inputs with a Gaussian filter
        if self.useGaussianBlur:
            I0 = applyGaussianBlur2D(I0,self.GaussianBlurSigma)
            I1 = applyGaussianBlur2D(I1,self.GaussianBlurSigma)

        # List for volumes pyramids
        NY, NX = I0.shape[0], I0.shape[1]
        LY, LX = self.getMeshLength(self.config, NY, NX)
        #meshInfo2D_python = MeshInfo2D(NY,NX,LY,LX)
        meshInfo2D_cuda = opticalFlow.MeshInfo2D(NY,NX,LY,LX)

        #meshes for pyramid
        meshInfos = [meshInfo2D_cuda]
        NY_restr, NX_restr = NY, NX
        for s in range(1, self.NUM_SCALES):
            NY_restr, NX_restr = math.ceil(0.5*NY_restr), math.ceil(0.5*NX_restr)
            LY_restr, LX_restr = self.getMeshLength(self.config, NY_restr, NX_restr)
            meshInfos.append(opticalFlow.MeshInfo2D(NY_restr,NX_restr,LY_restr,LX_restr))

        #lists for pyramid
        I0s = [I0]
        I1s = [I1]
        us = [u]
        ps = [p]

        # Create the pyramid
        for s in range(1, self.NUM_SCALES):
            prolongationOp_cuda = opticalFlow.Prolongation2D(meshInfos[s-1],meshInfos[s],self.InterpolationTypeCuda,self.BoundaryTypeCuda)
            I0s.append(prolongationOp_cuda.forward(I0s[s-1]))
            I1s.append(prolongationOp_cuda.forward(I1s[s-1]))
            us.append(torch.zeros([meshInfos[s].getNY(),meshInfos[s].getNX(),2]).float().to(self.DEVICE))
            ps.append(torch.zeros([meshInfos[s].getNY(),meshInfos[s].getNX(),2,2]).float().to(self.DEVICE))

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
            prolongationOp_cuda = opticalFlow.Prolongation2D(meshInfos[s],meshInfos[s-1],self.InterpolationTypeCuda,self.BoundaryTypeCuda)
            us[s-1] = prolongationOp_cuda.forwardVectorField(us[s])
            #factor LXNew/LXOld, ...
            us[s-1][:,:,0] *= meshInfos[s-1].getLX() / meshInfos[s].getLX()
            us[s-1][:,:,1] *= meshInfos[s-1].getLY() / meshInfos[s].getLY()

            #TODO Dirichlet boundary condition for p?
            # ps[s] = self.dirichlet(ps[s])

            ps[s-1] = prolongationOp_cuda.forwardMatrixField(ps[s])

            #TODO prolongation factor for p?


    def computeOnSingleStep(self, s, I0, I1, u, p, meshInfo):

        print("\n")
        print("start to compute optical flow for single step = ", s)
        progress_bar = tqdm(total = self.MAX_WARPS * self.MAX_OUTER_ITERATIONS)

        # Compute target image gradients
        #nablaOp = Nabla2D_Central(meshInfo)
        nablaOp = opticalFlow.Nabla2D_CD(meshInfo,self.BoundaryTypeCuda)
        warpingOp = opticalFlow.Warping2D(meshInfo,self.InterpolationTypeCuda,self.BoundaryTypeCuda)
        I1_grad = nablaOp.forward(I1)

        #optionally apply anisotropic differential op 
        scalars = torch.zeros([meshInfo.getNY(),meshInfo.getNX()]).float().to(self.DEVICE)
        normals = torch.zeros([meshInfo.getNY(),meshInfo.getNX(),2]).float().to(self.DEVICE)
        tangents = torch.zeros([meshInfo.getNY(),meshInfo.getNX(),2]).float().to(self.DEVICE)
        if self.useAnisotropicDifferentialOp:
            anistropicNablaOp = opticalFlow.AnisotropicNabla2D(meshInfo,self.anisotropicDifferentialOp_alpha,self.anisotropicDifferentialOp_beta)
            scalars,normals,tangents = anistropicNablaOp.computeTangentVecs(I1_grad)

        z = u

        for w in range(self.MAX_WARPS):
            # Compute the warping of the target image and its derivatives
            I1_warped = warpingOp.forward(I1,u)
            I1_warped_grad = warpingOp.forwardVectorField(I1_grad,u)
            # Constant part of the rho function
            #rho_c = I1_warped - u[:, :, 0] * I1_warped_grad[:, :, 0] - u[:, :, 1] * I1_warped_grad[:, :, 1] - I0
            rho_c = I1_warped - torch.sum(u * I1_warped_grad, dim=2) - I0

            primalFctVec = torch.zeros([self.MAX_OUTER_ITERATIONS])
            dualFctVec = torch.zeros([self.MAX_OUTER_ITERATIONS])
            totalFctVec = torch.zeros([self.MAX_OUTER_ITERATIONS])
            breakConditionVecPrimal = torch.zeros([self.MAX_OUTER_ITERATIONS])
            breakConditionVecDual = torch.zeros([self.MAX_OUTER_ITERATIONS])
            breakConditionVecUpdate = torch.zeros([self.MAX_OUTER_ITERATIONS])

            # TODO possible use results for new warp
            sigma = self.sigma 
            tau = self.tau
            theta = self.theta 
            gamma = self.gamma

            for n in range(self.MAX_OUTER_ITERATIONS):

                # update of dual variable
                pold = p
                z_grad = nablaOp.forwardVectorField(z)
                Dz_grad = z_grad
                if self.useAnisotropicDifferentialOp:
                    Dz_grad = anistropicNablaOp.forwardVectorField(z_grad,scalars,normals,tangents)
                dualVariable = p + sigma * Dz_grad
                # #start test
                # z_grad_compx = nablaOp.forward(z[:,:,0].contiguous())
                # z_grad_compy = nablaOp.forward(z[:,:,1].contiguous())
                # print("z.norm=",torch.norm(z).item())
                # print("p.norm=",torch.norm(p).item())
                # print("z_grad.norm=",torch.norm(z_grad).item())
                # print("z_grad_x.norm=",torch.norm(z_grad_compx).item())
                # print("z_grad_y.norm=",torch.norm(z_grad_compy).item())
                # print("diff z_grad_x.norm=",torch.norm(z_grad_compx - z_grad[:,:,0,:]).item())
                # print("diff z_grad_y.norm=",torch.norm(z_grad_compy - z_grad[:,:,1,:]).item())
                # print("dualVariable.norm=",torch.norm(dualVariable).item())
                # #end test
                p = opticalFlow.TVL1OF2D_proxDual( dualVariable, sigma, self.weight_TV, meshInfo)
                if self.weight_TV == 0.:
                   p = torch.zeros([meshInfo.getNY(),meshInfo.getNX(),2,2]).float().to(self.DEVICE)
                #debug
                breakConditionVecDual[n] = torch.norm(p-pold).item()
                dualFctVec[n] = self.weight_TV * torch.sum(torch.norm(p, dim=3))


                # update of primal variable
                uold = u
                # Compute the fidelity data term \rho(u)
                #rho = rho_c + u[:, :, 0] * I1_warped_grad[:, :, 0] + u[:, :, 1] * I1_warped_grad[:, :,1]
                rho = rho_c + torch.sum(u * I1_warped_grad, dim=2)
                # # start test 
                # testVec1 = u[:, :, 0] * I1_warped_grad[:, :, 0] + u[:, :, 1] * I1_warped_grad[:, :,1]
                # testVec2 = torch.sum(u * I1_warped_grad, dim=2)
                # print("testVec1.norm=",torch.norm(testVec1).item())
                # print("testVec2.norm=",torch.norm(testVec2).item())
                # print("diff=",torch.norm(testVec1 - testVec2).item())
                # #print("rho.norm = ", torch.norm(rho).item() )
                #end test
                Dp = p
                if self.useAnisotropicDifferentialOp:
                    Dp = anistropicNablaOp.backwardVectorField(p,scalars,normals,tangents)
                Dp_div = nablaOp.backwardVectorField(Dp)
                primalVariable = u - tau * Dp_div
                u = opticalFlow.TVL1OF2D_proxPrimal(primalVariable, tau, self.weight_Matching, rho, I1_warped_grad, meshInfo )
                # if self.weight_Matching == 0.:
                #     u = torch.zeros([meshInfo.getNY(),meshInfo.getNX(),2]).float().to(self.DEVICE)
                # debug
                breakConditionVecPrimal[n] = torch.norm(u-uold).item()
                #primalFctVec[n] = opticalFlow.TVL1OF2D_PrimalFct(weight_Matching, rho, meshInfo )
                rho_new = rho_c + torch.sum(u * I1_warped_grad, dim=2)
                primalFctVec[n] = self.weight_Matching * torch.sum(torch.abs(rho_new))

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
                z = (1. + theta) * u - theta * uold
                #debug
                breakConditionVecUpdate[n] = torch.norm(z-zold).item()
                totalFctVec[n]=primalFctVec[n]+dualFctVec[n]

                progress_bar.update(1)
 
            if self.useDebugOutput:
                save_curve_1d(primalFctVec, self.MAX_OUTER_ITERATIONS, self.saveDirDebug, f"PrimalFct_it{s}_warp{w}")
                save_curve_1d(dualFctVec, self.MAX_OUTER_ITERATIONS, self.saveDirDebug, f"DualFct_it{s}_warp{w}")
                save_curve_1d(totalFctVec, self.MAX_OUTER_ITERATIONS, self.saveDirDebug, f"TotalFct_it{s}_warp{w}")
                save_curve_1d(breakConditionVecPrimal, self.MAX_OUTER_ITERATIONS, self.saveDirDebug, f"CPErrorPrimal_it{s}_warp{w}", "loglog")
                save_curve_1d(breakConditionVecDual, self.MAX_OUTER_ITERATIONS, self.saveDirDebug, f"CPErrorDual_it{s}_warp{w}", "loglog")
                save_curve_1d(breakConditionVecUpdate, self.MAX_OUTER_ITERATIONS, self.saveDirDebug, f"CPErrorUpdate_it{s}_warp{w}", "loglog")

        return u, p


    def saveSingleStepToFile(self,step,I0,I1,u,p,meshInfo):
        
        saveDirStep = os.path.sep.join([self.saveDir, f"it{step}"])
        if not os.path.exists(saveDirStep):
            os.makedirs(saveDirStep)

        plotOpticalFlow2D(u.cpu().detach().numpy(), "u", saveDirStep, step)
        save_image(I0,saveDirStep,f"I0_it{step}.png")
        save_image(I1,saveDirStep,f"I1_it{step}.png")
        warpingOp = opticalFlow.Warping2D(meshInfo,self.InterpolationTypeCuda,self.BoundaryTypeCuda)
        I1_warped = warpingOp.forward(I1,u)
        save_image(I1_warped,saveDirStep,f"I1_warped_it{step}.png")
        save_image(torch.abs(I1_warped-I0),saveDirStep,f"Diff_I1warped_to_I0_it{step}.png")
        
        flowName = f"flow_it{step}.pt"
        fileNameFlow = os.path.join(saveDirStep, flowName) 
        torch.save(u, fileNameFlow)

        dualName = f"dual_it{step}.pt"
        fileNameDual = os.path.join(saveDirStep, dualName) 
        torch.save(p, fileNameDual)


