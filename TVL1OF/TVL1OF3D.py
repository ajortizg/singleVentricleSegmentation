# ==================================
import sys
import os
import configparser
import math
from scipy import ndimage
import numpy as np
import torch.functional as F
import nibabel as nib
import torch
import time
from tqdm import tqdm

utils_lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils'))
sys.path.append(utils_lib_path)
# sys.path.append("../utils")
#from utils.plots import *
# import torch_utils
# import flow_viz

# sys.path.append("../pythonOps/")
# from pythonOps.mesh import *
# from pythonOps.differentialOps import *
pythonOps_lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../pythonOps'))
sys.path.append(pythonOps_lib_path)
#import utils.torch_utils as tu
# import mesh
# import differentialOps


from opticalFlow_cuda_ext import opticalFlow


import torch.nn.functional as F


def generateGaussianKernel3D(sigma):
    kernelSize = int(sigma * 5)
    if kernelSize % 2 == 0:
        kernelSize += 1
    ts = torch.linspace(-kernelSize // 2, kernelSize // 2 + 1, kernelSize)
    gauss = torch.exp((-(ts / sigma)**2 / 2))
    kernel = gauss / gauss.sum()

    return kernel


def applyGaussianBlur3D(vol, sigma):
    # 3D convolution
    vol_in = vol.reshape(1, 1, *vol.shape)
    k = generateGaussianKernel3D(sigma)
    k3d = torch.einsum('i,j,k->ijk', k, k, k).cuda()
    k3d = k3d / k3d.sum()
    vol_3d = F.conv3d(vol_in, k3d.reshape(1, 1, *k3d.shape), stride=1, padding=len(k) // 2)
    vol_out = vol_3d.reshape(*vol.shape)

    return vol_out


class TVL1OpticalFlow3D:
    def __init__(self, config):
        self.config = config
        # self.saveDir = saveDir
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

        self.USE_MEDIAN_FILTER = config.getboolean('PARAMETERS', 'USE_MEDIAN_FILTER')
        self.KERNEL_MF = config.getint('PARAMETERS', 'KERNEL_MF')

        # anisotropic differential op
        self.useAnisotropicDifferentialOp = config.getboolean("PARAMETERS", "useAnisotropicDifferentialOp")
        self.anisotropicDifferentialOp_alpha, self.anisotropicDifferentialOp_beta = None, None
        if self.useAnisotropicDifferentialOp:
            self.anisotropicDifferentialOp_alpha = config.getfloat("PARAMETERS", "anisotropicDifferentialOp_alpha")
            self.anisotropicDifferentialOp_beta = config.getfloat("PARAMETERS", "anisotropicDifferentialOp_beta")
        # blurring
        self.useGaussianBlur = config.getboolean("PARAMETERS", "useGaussianBlur")
        self.GaussianBlurSigma = config.getfloat('PARAMETERS', 'GaussianBlurSigma')
        # interpolation
        interType = config.get('PARAMETERS', 'InterpolationType')
        self.InterpolationTypeCuda = None
        if interType == "NEAREST":
            self.InterpolationTypeCuda = opticalFlow.InterpolationType.INTERPOLATE_NEAREST
        elif interType == "LINEAR":
            self.InterpolationTypeCuda = opticalFlow.InterpolationType.INTERPOLATE_LINEAR
        elif interType == "CUBIC_HERMITESPLINE":
            self.InterpolationTypeCuda = opticalFlow.InterpolationType.INTERPOLATE_CUBIC_HERMITESPLINE
        else:
            raise Exception("wrong InterpolationType in configParser")
        # boundary
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

        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

        # debug
        # self.saveDirDebug = os.path.sep.join([self.saveDir, "debug"])
        self.useDebugOutput = config.getboolean("DEBUG", "useDebugOutput")
        # if self.useDebugOutput:
        # os.makedirs(self.saveDirDebug)

    def set_save_dir(self, save_dir):
        self.saveDir = save_dir
        if self.useDebugOutput:
            self.saveDirDebug = os.path.join(self.saveDir, "debug")
            os.makedirs(self.saveDirDebug)

    def getMeshLength(self, config, NZ, NY, NX):
        LenghtType = config.get('PARAMETERS', 'LenghtType')
        if LenghtType == "numDofs":
            LZ = NZ - 1
            LY = NY - 1
            LX = NX - 1
            return LZ, LY, LX
        elif LenghtType == "fixed":
            LZ = config.getfloat('PARAMETERS', "LenghtZ")
            LY = config.getfloat('PARAMETERS', "LenghtY")
            LX = config.getfloat('PARAMETERS', "LenghtX")
            return LZ, LY, LX

    def generatePyramid(self, I0, I1, u, p):
        # Smooth inputs with a Gaussian filter
        if self.useGaussianBlur:
            I0 = applyGaussianBlur3D(I0, self.GaussianBlurSigma)
            I1 = applyGaussianBlur3D(I1, self.GaussianBlurSigma)

        # List for volumes pyramids
        NZ, NY, NX = I0.shape[0], I0.shape[1], I0.shape[2]
        LZ, LY, LX = self.getMeshLength(self.config, NZ, NY, NX)
        # meshInfo3D_python = MeshInfo3D(NZ,NY,NX,LZ,LY,LX)
        meshInfo3D_cuda = opticalFlow.MeshInfo3D(NZ, NY, NX, LZ, LY, LX)

        # meshes for pyramid
        meshInfos = [meshInfo3D_cuda]
        NZ_restr, NY_restr, NX_restr = NZ, NY, NX
        for s in range(1, self.NUM_SCALES):
            NZ_restr, NY_restr, NX_restr = math.ceil(0.5 * NZ_restr), math.ceil(0.5 * NY_restr), math.ceil(0.5 * NX_restr)
            LZ_restr, LY_restr, LX_restr = self.getMeshLength(self.config, NZ_restr, NY_restr, NX_restr)
            meshInfos.append(opticalFlow.MeshInfo3D(NZ_restr, NY_restr, NX_restr, LZ_restr, LY_restr, LX_restr))

        # lists for pyramid
        I0s = [I0]
        I1s = [I1]
        us = [u]
        ps = [p]

        # Create the pyramid
        for s in range(1, self.NUM_SCALES):
            prolongationOp_cuda = opticalFlow.Prolongation3D(
                meshInfos[s - 1],
                meshInfos[s],
                self.InterpolationTypeCuda, self.BoundaryTypeCuda)
            I0s.append(prolongationOp_cuda.forward(I0s[s - 1]))
            I1s.append(prolongationOp_cuda.forward(I1s[s - 1]))
            us.append(torch.zeros([meshInfos[s].getNZ(), meshInfos[s].getNY(), meshInfos[s].getNX(), 3], dtype=torch.float32, device=self.DEVICE))
            ps.append(torch.zeros([meshInfos[s].getNZ(), meshInfos[s].getNY(), meshInfos[s].getNX(), 3, 3], dtype=torch.float32, device=self.DEVICE))

        return I0s, I1s, us, ps, meshInfos

    def computeOnPyramid(self, I0, I1, u, p):
        I0s, I1s, us, ps, meshInfos = self.generatePyramid(I0, I1, u, p)

        # print("\n")
        # print("============================================")
        # print("start to compute optical flow for pyramid")
        # print("============================================")
        # print("\n")

        for s in range(self.NUM_SCALES - 1, -1, -1):

            # Compute the optical flow at scale s
            us[s], ps[s] = self.computeOnSingleStep(s, I0s[s], I1s[s], us[s], ps[s], meshInfos[s])

            # save step
            # self.saveSingleStepToFile(s, I0s[s], I1s[s], us[s], ps[s], meshInfos[s])

            if s == 0:
                self.saveSingleStepToFile(s, I0s[s], I1s[s], us[s], ps[s], meshInfos[s])
                break

            # Prolongate the optical flow and dual variables to the next pyramid level
            prolongationOp_cuda = opticalFlow.Prolongation3D(
                meshInfos[s],
                meshInfos[s - 1],
                self.InterpolationTypeCuda, self.BoundaryTypeCuda)
            us[s - 1] = prolongationOp_cuda.forwardVectorField(us[s])
            # factor LXNew/LXOld, ...
            us[s - 1][:, :, :, 0] *= meshInfos[s - 1].getLX() / meshInfos[s].getLX()
            us[s - 1][:, :, :, 1] *= meshInfos[s - 1].getLY() / meshInfos[s].getLY()
            us[s - 1][:, :, :, 2] *= meshInfos[s - 1].getLZ() / meshInfos[s].getLZ()

            # TODO Dirichlet boundary condition for p?
            # ps[s] = self.dirichlet(ps[s])

            ps[s - 1] = prolongationOp_cuda.forwardMatrixField(ps[s])

            # TODO prolongation factor for p?

        return us[0], ps[0]

    def computeOnSingleStep(self, s, I0, I1, u, p, meshInfo):
        # print("\n")
        # print("start to compute optical flow for single step = ", s)
        # progress_bar = tqdm(total=self.MAX_WARPS * self.MAX_OUTER_ITERATIONS)

        # Compute target image gradients
        # nablaOp = Nabla3D_Central(meshInfo)
        nablaOp = opticalFlow.Nabla3D_CD(meshInfo, self.BoundaryTypeCuda)
        warpingOp = opticalFlow.Warping3D(meshInfo, self.InterpolationTypeCuda, self.BoundaryTypeCuda)
        I1_grad = nablaOp.forward(I1)

        # optionally apply anisotropic differential op
        scalars = torch.zeros([meshInfo.getNZ(), meshInfo.getNY(), meshInfo.getNX()], dtype=torch.float32, device=self.DEVICE)
        normals = torch.zeros([meshInfo.getNZ(), meshInfo.getNY(), meshInfo.getNX(), 3], dtype=torch.float32, device=self.DEVICE)
        tangents1 = torch.zeros([meshInfo.getNZ(), meshInfo.getNY(), meshInfo.getNX(), 3], dtype=torch.float32, device=self.DEVICE)
        tangents2 = torch.zeros([meshInfo.getNZ(), meshInfo.getNY(), meshInfo.getNX(), 3], dtype=torch.float32, device=self.DEVICE)
        if self.useAnisotropicDifferentialOp:
            anistropicNablaOp = opticalFlow.AnisotropicNabla3D(meshInfo, self.anisotropicDifferentialOp_alpha, self.anisotropicDifferentialOp_beta)
            scalars, normals, tangents1, tangents2 = anistropicNablaOp.computeTangentVecs(I1_grad)

        z = u

        for w in range(self.MAX_WARPS):
            # Compute the warping of the target image and its derivatives
            I1_warped = warpingOp.forward(I1, u)
            I1_warped_grad = warpingOp.forwardVectorField(I1_grad, u)
            # Constant part of the rho function
            # rho_c = I1_warped - u[:, :, :, 0] * I1_warped_grad[:, :, :, 0] - u[:, :, :, 1] * I1_warped_grad[:, :, :, 1] - u[:, :, :, 2] * I1_warped_grad[:, :, :, 2] - I0
            rho_c = I1_warped - torch.sum(u * I1_warped_grad, dim=3) - I0

            breakConditionVecPrimal = torch.zeros([self.MAX_OUTER_ITERATIONS], dtype=torch.float32, device=self.DEVICE)
            breakConditionVecDual = torch.zeros([self.MAX_OUTER_ITERATIONS], dtype=torch.float32, device=self.DEVICE)
            breakConditionVecUpdate = torch.zeros([self.MAX_OUTER_ITERATIONS], dtype=torch.float32, device=self.DEVICE)

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
                    Dz_grad = anistropicNablaOp.forwardVectorField(z_grad, scalars, normals, tangents1, tangents2)
                dualVariable = p + sigma * Dz_grad
                p = opticalFlow.TVL1OF3D_proxDual(dualVariable, sigma, self.weight_TV, meshInfo)
                breakConditionVecDual[n] = torch.norm(p - pold).item()
                if self.weight_TV == 0.:
                    p = torch.zeros([meshInfo.getNZ(), meshInfo.getNY(), meshInfo.getNX(), 3, 3], dtype=torch.float32, device=self.DEVICE)

                # update of primal variable
                uold = u
                # Compute the fidelity data term \rho(u)
                # rho = rho_c + u[:, :, :, 0] * I1_warped_grad[:, :, :, 0] + u[:, :, :, 1] * I1_warped_grad[:, :, :, 1] + u[:, :, :, 2] * I1_warped_grad[:, :, :, 2]
                rho = rho_c + torch.sum(u * I1_warped_grad, dim=3)
                # print("rho.norm = ", torch.norm(rho).item() )
                Dp = p
                if self.useAnisotropicDifferentialOp:
                    Dp = anistropicNablaOp.backwardVectorField(p, scalars, normals, tangents1, tangents2)
                Dp_div = nablaOp.backwardVectorField(Dp)
                primalVariable = u - tau * Dp_div
                u = opticalFlow.TVL1OF3D_proxPrimal(primalVariable, tau, self.weight_Matching, rho, I1_warped_grad, meshInfo)

                breakConditionVecPrimal[n] = torch.norm(u - uold).item()
                # if self.weight_Matching == 0.:
                #     u = torch.zeros([meshInfo.getNZ(),meshInfo.getNY(),meshInfo.getNX(),3]).float().to(self.DEVICE)

                # update of stepsizes
                if self.PRIMALDUAL_ALGO_TYPE == 1:
                    # dot nothing
                    print("apply CP1")
                elif self.PRIMALDUAL_ALGO_TYPE == 2:
                    theta = 1. / math.sqrt(1. + 2. * gamma * tau)
                    tau *= theta
                    sigma /= theta
                else:
                    print("wrong method for Chambolle-Pock-Algorithm")

                # overrelaxation
                zold = z
                z = u
                breakConditionVecUpdate[n] = torch.norm(z - zold).item()

                # progress_bar.update(1)

            if self.useDebugOutput:
                # saveCurve1D(primalFctVec, MAX_OUTER_ITERATIONS, self.saveDirDebug, f"PrimalFct_it{s}_warp{w}")
                saveCurve1D(breakConditionVecPrimal, self.MAX_OUTER_ITERATIONS,
                            self.saveDirDebug, f"CPErrorPrimal_it{s}_warp{w}", "loglog")
                saveCurve1D(breakConditionVecDual, self.MAX_OUTER_ITERATIONS,
                            self.saveDirDebug, f"CPErrorDual_it{s}_warp{w}", "loglog")
                saveCurve1D(breakConditionVecUpdate, self.MAX_OUTER_ITERATIONS,
                            self.saveDirDebug, f"CPErrorUpdate_it{s}_warp{w}", "loglog")

        return u, p

    def apply_median_filter(self, u):
        if self.USE_MEDIAN_FILTER:
            ks = self.KERNEL_MF
            uf = ndimage.median_filter(u.cpu().detach().numpy(), size=(ks, ks, ks, 1))
            return torch.from_numpy(uf).float().to(self.DEVICE)
        else:
            return u

    def saveSingleStepToFile(self, step, I0, I1, u, p, meshInfo):
        saveDirStep = os.path.sep.join([self.saveDir, f"it{step}"])
        if not os.path.exists(saveDirStep):
            os.makedirs(saveDirStep)

        # plotOpticalFlow3D(u.cpu().detach().numpy(), "u", saveDirStep, step)

        # Median filter pag 12. paper
        if self.USE_MEDIAN_FILTER:
            ks = self.KERNEL_MF
            uf = ndimage.median_filter(u.cpu().detach().numpy(), size=(ks, ks, ks, 1))
            # plotOpticalFlow3D(uf, "uf", saveDirStep, step)
            # flowName = f"flow_m_it{step}.pt"
            # fileNameFlow = os.path.join(saveDirStep, flowName)
            # torch.save(torch.from_numpy(uf), fileNameFlow)
            flowName = f"flow_m_it{step}.npy"
            fileNameFlow = os.path.join(saveDirStep, flowName)
            np.save(fileNameFlow, uf)

        # save3D_torch_to_nifty(I0, saveDirStep, f"I0.nii")
        # save_slices(I0, f"I0_it{step}.png", saveDirStep)
        # save3D_torch_to_nifty(I1, saveDirStep, f"I1.nii")
        # save_slices(I1, f"I1_it{step}.png", saveDirStep)
        # warpingOp = opticalFlow.Warping3D(meshInfo, self.InterpolationTypeCuda, self.BoundaryTypeCuda)
        # I1_warped = warpingOp.forward(I1, u)
        # save_slices(I1_warped, f"I1_warped_it{step}.png", saveDirStep)

        # save_single_zslices(I0, saveDirStep, "I0Slices", 1., 0)
        # save_single_zslices(I1_warped, saveDirStep, "I1WarpedSlices", 1., 0)

        # diff = torch.abs(I1_warped - I0)
        # n_diff = tu.normalize(diff)
        # save_slices(diff, f"Diff_I1warped_to_I0_it{step}.png", saveDirStep)
        # save_colorbar_slices(n_diff, f"Diff_I1warped_to_I0_it{step}.png", saveDirStep)
        # save_single_zslices(n_diff, saveDirStep, "Diff_I1warped_to_I0_Slices", 1., 0)
        # print("norm of diff = ", diff.norm().item())
        # numZSlices = diff.shape[0]
        # for z in range(numZSlices):
            # print(" norm of diff(z=", z, ") = ", diff[z, :, :].norm().item())

        # flowName = f"flow_it{step}.pt"
        # fileNameFlow = os.path.join(saveDirStep, flowName)
        # torch.save(u.cpu().detach(), fileNameFlow)

        flowName = f'flow_it{step}.npy'
        fileNameFlow = os.path.join(saveDirStep, flowName)
        np.save(fileNameFlow, u.cpu().detach().numpy())

        # dualName = f"dual_it{step}.pt"
        # fileNameDual = os.path.join(saveDirStep, dualName)
        # torch.save(p, fileNameDual)

    def warpMask(self, mask, u, I, t0, saveDir):
        print("warp with optical flow for time step: ", t0)
        print("norm of u = ", u.norm().item())

        # flowName = "flow_it0.pt"
        # fileNameFlow = os.path.join(saveDirStep, flowName)
        # u = torch.load(fileNameFlow, map_location=torch.device(DEVICE))

        NZ, NY, NX = mask.shape[0], mask.shape[1], mask.shape[2]
        LZ, LY, LX = self.getMeshLength(self.config, NZ, NY, NX)
        meshInfo = opticalFlow.MeshInfo3D(NZ, NY, NX, LZ, LY, LX)

        warpingOp = opticalFlow.Warping3D(meshInfo, self.InterpolationTypeCuda, self.BoundaryTypeCuda)
        mask_warped = warpingOp.forward(mask, u)

        save3D_torch_to_nifty(mask_warped, saveDir, f"mask_warped_time{t0}.nii")
        save_slices(mask_warped, f"mask_warped_time{t0}.png", saveDir)
        save_single_zslices(mask_warped, saveDir, "mask_warped_slices", 1., 2)

        save_img_mask_single_zslices(I, mask, saveDir, 'color_mask')
        save_img_mask_slices(I, mask, 'img_mask.png', saveDir)

        maskName = f"masked_warped_time{t0}.pt"
        fileNameMask = os.path.join(saveDir, maskName)
        torch.save(mask_warped, fileNameMask)

        return mask_warped
