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
from utilities.flow_viz import *
# utils_lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils'))
# sys.path.append(utils_lib_path)

pythonOps_lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../pythonOps'))
sys.path.append(pythonOps_lib_path)
# import mesh
# import differentialOps

from opticalFlow_cuda_ext import opticalFlow

class ROF2D:
    def __init__(self,saveDir,config):
        self.config = config
        self.saveDir = saveDir
        self.NUM_SCALES = config.getint('PARAMETERS', 'NUM_SCALES')
        self.MAX_OUTER_ITERATIONS = config.getint('PARAMETERS', 'MAX_OUTER_ITERATIONS')
        self.weight_MatchingFct = config.getfloat('PARAMETERS', 'weight_MatchingFct')
        self.weight_TVFct = config.getfloat('PARAMETERS', 'weight_TVFct')
        self.PRIMALDUAL_ALGO_TYPE = config.getint('PARAMETERS', 'PRIMALDUAL_ALGO_TYPE')
        self.sigma = config.getfloat('PARAMETERS', 'sigma')
        self.tau = config.getfloat('PARAMETERS', 'tau')
        self.theta = config.getfloat('PARAMETERS', 'theta')
        self.gamma = config.getfloat('PARAMETERS', 'gamma')
        # interpolation
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
            LX = config.getfloat('PARAMETERS', "LenghtY")
            return LY, LX

    def generatePyramid(self, I0, I, p):

        # List for pyramids
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
        Is = [I]
        ps = [p]

        # Create the pyramid
        for s in range(1, self.NUM_SCALES):
            prolongationOp_cuda = opticalFlow.Prolongation2D(meshInfos[s-1],meshInfos[s],self.InterpolationTypeCuda,self.BoundaryTypeCuda)
            I0s.append(prolongationOp_cuda.forward(I0s[s-1]))
            Is.append(prolongationOp_cuda.forward(Is[s-1]))
            ps.append(torch.zeros([meshInfos[s].getNY(),meshInfos[s].getNX(),2]).float().to(self.DEVICE))

        return I0s, Is, ps, meshInfos


    def computeOnPyramid(self, I0, I, p):

        I0s, Is, ps, meshInfos = self.generatePyramid(I0,I,p)

        print("\n")
        print("============================================")
        print("start to compute optical flow for pyramid")
        print("============================================")
        print("\n")

        for s in range(self.NUM_SCALES-1, -1, -1):

            # Compute the optical flow at scale s
            Is[s], ps[s] = self.computeOnSingleStep(s, I0s[s], Is[s], ps[s], meshInfos[s])

            #save step
            self.saveSingleStepToFile(s,I0s[s],Is[s],ps[s],meshInfos[s])

            if s == 0:
                break

            # Prolongate the optical flow and dual variables to the next pyramid level
            prolongationOp_cuda = opticalFlow.Prolongation2D(meshInfos[s],meshInfos[s-1],self.InterpolationTypeCuda,self.BoundaryTypeCuda)
            Is[s-1] = prolongationOp_cuda.forward(Is[s])
            ps[s-1] = prolongationOp_cuda.forwardVectorField(ps[s])
            #TODO prolongation factor for p?


    def computeOnSingleStep(self, s, I0, I, p, meshInfo):

        print("\n")
        print("start to compute ROF for single step = ", s)
        progress_bar = tqdm(total = self.MAX_OUTER_ITERATIONS)

        # Compute target image gradients
        #nablaOp = Nabla2D_Central(meshInfo)
        nablaOp = opticalFlow.Nabla2D_CD(meshInfo,self.BoundaryTypeCuda)
        # ROFOp = opticalFlow.ROF2D(meshInfo,I0,self.weight_TVFct,self.weight_MatchingFct)
        ROFOp = opticalFlow.ROF2D(meshInfo,self.weight_TVFct,self.weight_MatchingFct)
        
        z = I
        sigma = self.sigma 
        tau = self.tau
        theta = self.theta 
        gamma = self.gamma

        #debug
        primalFctVec_F = torch.zeros([self.MAX_OUTER_ITERATIONS])
        primalFctVec_G = torch.zeros([self.MAX_OUTER_ITERATIONS])
        primalFctVec_Total = torch.zeros([self.MAX_OUTER_ITERATIONS])
        dualFctVec_F = torch.zeros([self.MAX_OUTER_ITERATIONS])
        dualFctVec_G = torch.zeros([self.MAX_OUTER_ITERATIONS])
        dualFctVec_Total = torch.zeros([self.MAX_OUTER_ITERATIONS])
        primalDualGabVec = torch.zeros([self.MAX_OUTER_ITERATIONS])
        breakConditionVecPrimal = torch.zeros([self.MAX_OUTER_ITERATIONS])
        breakConditionVecDual = torch.zeros([self.MAX_OUTER_ITERATIONS])
        breakConditionVecUpdate = torch.zeros([self.MAX_OUTER_ITERATIONS])

        for n in range(self.MAX_OUTER_ITERATIONS):

            # update of dual variable
            pold = p
            z_grad = nablaOp.forward(z)
            dualVariable = p + sigma * z_grad
            p = ROFOp.proxDual( dualVariable, sigma )
            if self.weight_TVFct == 0.:
                p = torch.zeros([meshInfo.getNY(),meshInfo.getNX(),2]).float().to(self.DEVICE)

            # update of primal variable
            Iold = I
            p_div = nablaOp.backward(p)
            primalVariable = I - tau * p_div
            I = ROFOp.proxPrimal( primalVariable, I0, tau )
            if self.weight_MatchingFct == 0.:
                I = torch.zeros([meshInfo.getNY(),meshInfo.getNX()]).float().to(self.DEVICE)

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
            z = (1. + theta) * I - theta * Iold


            #debug
            breakConditionVecDual[n] = torch.norm(p-pold).item()
            breakConditionVecPrimal[n] = torch.norm(I-Iold).item()
            breakConditionVecUpdate[n] = torch.norm(z-zold).item()
            # vol 
            volElement = meshInfo.gethX() * meshInfo.gethY()
            #primal(I) = F(LI) + G(I)
            primalFctVec_G[n] = self.weight_MatchingFct * volElement * torch.norm(I-I0).item()**2
            I_grad = nablaOp.forward(I)
            primalFctVec_F[n] = self.weight_TVFct * volElement * torch.sum(torch.norm(I_grad, dim=2))
            primalFctVec_Total[n] = primalFctVec_F[n] + primalFctVec_G[n]
            #dual(p) = -F*(p)-G*(-L*p)
            p_norm = torch.norm(p, dim=2)
            max_p_norm = torch.max(p_norm).item()
            tol = 0.000001
            if max_p_norm > self.weight_TVFct + tol:
                print("proj failed:", max_p_norm)
                dualFctVec_F[n] = -1000000.
            else:
                dualFctVec_F[n] = 0.
            #dualFctVec_G[n] = volElement * ( 0.5/self.weight_MatchingFct * torch.norm(p_div)**2 + p_div.reshape(-1).dot(I0.reshape(-1)) )
            dualFctVec_G[n] = -1. * volElement * ( 0.5/self.weight_MatchingFct * torch.norm(p_div)**2 - p_div.reshape(-1).dot(I0.reshape(-1)) )
            dualFctVec_Total[n] = dualFctVec_F[n] + dualFctVec_G[n]
            #primal-dual-gab
            primalDualGabVec[n] = primalFctVec_Total[n] - dualFctVec_Total[n]

            progress_bar.update(1)
 
        if self.useDebugOutput:
                save_curve_1d(primalFctVec_F, self.MAX_OUTER_ITERATIONS, self.saveDirDebug, f"PrimalFct_F_it{s}")
                save_curve_1d(primalFctVec_G, self.MAX_OUTER_ITERATIONS, self.saveDirDebug, f"PrimalFct_G_it{s}")
                save_curve_1d(primalFctVec_Total, self.MAX_OUTER_ITERATIONS, self.saveDirDebug, f"PrimalFct_Total_it{s}")
                save_curve_1d(dualFctVec_F, self.MAX_OUTER_ITERATIONS, self.saveDirDebug, f"DualFct_F_it{s}")
                save_curve_1d(dualFctVec_G, self.MAX_OUTER_ITERATIONS, self.saveDirDebug, f"DualFct_G_it{s}")
                save_curve_1d(dualFctVec_Total, self.MAX_OUTER_ITERATIONS, self.saveDirDebug, f"DualFct_Total_it{s}")
                save_curve_1d(primalDualGabVec, self.MAX_OUTER_ITERATIONS, self.saveDirDebug, f"primalDualGabVec_it{s}", "loglog")
                save_curve_1d(breakConditionVecPrimal, self.MAX_OUTER_ITERATIONS, self.saveDirDebug, f"CPErrorPrimal_it{s}", "loglog")
                save_curve_1d(breakConditionVecDual, self.MAX_OUTER_ITERATIONS, self.saveDirDebug, f"CPErrorDual_it{s}", "loglog")
                save_curve_1d(breakConditionVecUpdate, self.MAX_OUTER_ITERATIONS, self.saveDirDebug, f"CPErrorUpdate_it{s}", "loglog")

        return I, p


    def saveSingleStepToFile(self,step,I0,I,p,meshInfo):
        saveDirStep = os.path.sep.join([self.saveDir, f"it{step}"])
        if not os.path.exists(saveDirStep):
            os.makedirs(saveDirStep)

        save_image(I0,saveDirStep,f"I0_it{step}.png")
        save_image(I,saveDirStep,f"I_it{step}.png")


