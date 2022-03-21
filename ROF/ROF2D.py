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
from utils.plots import *
from utils.flow_viz import *
# utils_lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils'))
# sys.path.append(utils_lib_path)

pythonOps_lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../pythonOps'))
sys.path.append(pythonOps_lib_path)
# import mesh
# import differentialOps

from opticalFlow_cuda_ext import opticalFlow

class ROF2D:
    def __init__(self,saveDir,config):
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
        interType = config.get('PARAMETERS', 'InterpolationType')
        self.InterpolationTypeCuda = opticalFlow.InterpolationType.INTERPOLATE_LINEAR
        if interType == "LINEAR":
            self.InterpolationTypeCuda = opticalFlow.InterpolationType.INTERPOLATE_LINEAR
        elif interType == "CUBIC_HERMITESPLINE":
            self.InterpolationTypeCuda = opticalFlow.InterpolationType.INTERPOLATE_CUBIC_HERMITESPLINE
        cuda_availabe = config.get('DEVICE', 'cuda_availabe')
        self.DEVICE = "cuda" if cuda_availabe else "cpu"
        self.saveDirDebug = os.path.sep.join([self.saveDir, "debug"])
        self.useDebugOutput = config.getboolean("DEBUG","useDebugOutput")
        if self.useDebugOutput:
            os.makedirs(self.saveDirDebug)

    def generatePyramid(self, I0, I, p):

        # List for pyramids
        NY, NX = I0.shape[0], I0.shape[1]
        #TODO possibly change lenght scale
        LY, LX = NY-1, NX-1
        #meshInfo2D_python = MeshInfo2D(NY,NX,LY,LX)
        meshInfo2D_cuda = opticalFlow.MeshInfo2D(NY,NX,LY,LX)

        #meshes for pyramid
        meshInfos = [meshInfo2D_cuda]
        NY_restr, NX_restr = NY, NX
        for s in range(1, self.NUM_SCALES):
            NY_restr, NX_restr = math.ceil(0.5*NY_restr), math.ceil(0.5*NX_restr)
            LY_restr, LX_restr = NY_restr-1, NX_restr-1
            meshInfos.append(opticalFlow.MeshInfo2D(NY_restr,NX_restr,LY_restr,LX_restr))

        #lists for pyramid
        I0s = [I0]
        Is = [I]
        ps = [p]

        # Create the pyramid
        for s in range(1, self.NUM_SCALES):
            prolongationOp_cuda = opticalFlow.Prolongation2D(meshInfos[s-1],meshInfos[s])
            I0s.append(prolongationOp_cuda.forward(I0s[s-1].contiguous(),self.InterpolationTypeCuda))
            Is.append(prolongationOp_cuda.forward(Is[s-1].contiguous(),self.InterpolationTypeCuda))
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
            prolongationOp_cuda = opticalFlow.Prolongation2D(meshInfos[s],meshInfos[s-1])
            #I0s[s-1] = prolongationOp_cuda.forward(I0s[s],self.InterpolationTypeCuda)
            Is[s-1] = prolongationOp_cuda.forward(Is[s],self.InterpolationTypeCuda)
            ps[s-1] = prolongationOp_cuda.forwardVectorField(ps[s],self.InterpolationTypeCuda)
            #TODO prolongation factor for p?


    def computeOnSingleStep(self, s, I0, I, p, meshInfo):

        print("start to compute ROF for single step = ", s)
        progress_bar = tqdm(total = self.MAX_OUTER_ITERATIONS)

        # Compute target image gradients
        #nablaOp = Nabla2D_Central(meshInfo)
        nablaOp = opticalFlow.Nabla2D_CD(meshInfo)
        
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
            p = opticalFlow.ROF2D_proxDual( dualVariable, sigma, self.weight_TVFct, meshInfo)
            if self.weight_TVFct == 0.:
                p = torch.zeros([meshInfo.getNY(),meshInfo.getNX(),2]).float().to(self.DEVICE)

            # update of primal variable
            Iold = I
            p_div = nablaOp.backward(p)
            primalVariable = I - tau * p_div
            I = opticalFlow.ROF2D_proxPrimal(primalVariable, I0, tau, self.weight_MatchingFct, meshInfo )
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
            #primal(I) = F(LI) + G(I)
            primalFctVec_G[n] = self.weight_MatchingFct * torch.norm(I-I0).item()**2
            I_grad = nablaOp.forward(I)
            primalFctVec_F[n] = self.weight_TVFct * torch.sum(torch.norm(I_grad, dim=2))
            primalFctVec_Total[n] = primalFctVec_F[n] + primalFctVec_G[n]
            #dual(p) = -F*(p)-G*(-L*p)
            p_norm = torch.norm(p, dim=2)
            max_p_norm = torch.max(p_norm).item()
            if max_p_norm > self.weight_TVFct + 0.0001:
                print("proj failed:", max_p_norm)
                dualFctVec_F[n] = -1000000.
            else:
                #print("proj true")
                dualFctVec_G[n] = 0.
            dualFctVec_G[n] = 1./self.weight_TVFct * torch.norm(p_div)**2 + p_div.reshape(-1).dot(I0.reshape(-1))
            dualFctVec_Total[n] = dualFctVec_F[n] + dualFctVec_G[n]
            #primal-dual-gab
            primalDualGabVec[n] = primalFctVec_Total[n] - dualFctVec_Total[n]

            progress_bar.update(1)
 
        if self.useDebugOutput:
                saveCurve1D(primalFctVec_F, self.MAX_OUTER_ITERATIONS, self.saveDirDebug, f"PrimalFct_F_it{s}")
                saveCurve1D(primalFctVec_G, self.MAX_OUTER_ITERATIONS, self.saveDirDebug, f"PrimalFct_G_it{s}")
                saveCurve1D(primalFctVec_Total, self.MAX_OUTER_ITERATIONS, self.saveDirDebug, f"PrimalFct_Total_it{s}")
                saveCurve1D(dualFctVec_F, self.MAX_OUTER_ITERATIONS, self.saveDirDebug, f"DualFct_F_it{s}")
                saveCurve1D(dualFctVec_G, self.MAX_OUTER_ITERATIONS, self.saveDirDebug, f"DualFct_G_it{s}")
                saveCurve1D(dualFctVec_Total, self.MAX_OUTER_ITERATIONS, self.saveDirDebug, f"DualFct_Total_it{s}")
                saveCurve1D(primalDualGabVec, self.MAX_OUTER_ITERATIONS, self.saveDirDebug, f"primalDualGabVec_it{s}")
                saveCurve1D(breakConditionVecPrimal, self.MAX_OUTER_ITERATIONS, self.saveDirDebug, f"CPErrorPrimal_it{s}", "loglog")
                saveCurve1D(breakConditionVecDual, self.MAX_OUTER_ITERATIONS, self.saveDirDebug, f"CPErrorDual_it{s}", "loglog")
                saveCurve1D(breakConditionVecUpdate, self.MAX_OUTER_ITERATIONS, self.saveDirDebug, f"CPErrorUpdate_it{s}", "loglog")

        return I, p


    def saveSingleStepToFile(self,step,I0,I,p,meshInfo):
        
        print("save step ", step)

        saveDirStep = os.path.sep.join([self.saveDir, f"it{step}"])
        if not os.path.exists(saveDirStep):
            os.makedirs(saveDirStep)

        saveImage(I0,saveDirStep,f"I0_it{step}.png")
        saveImage(I,saveDirStep,f"I_it{step}.png")


