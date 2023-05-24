# ==================================
import sys
import os
import configparser
import math
import numpy as np
from scipy import ndimage
import nibabel as nib
import torch
import time
from tqdm import tqdm

sys.path.append("../utils")
from utilities.plots import *
from utilities.flow_viz import *

pythonOps_lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../pythonOps'))
sys.path.append(pythonOps_lib_path)
import utilities.deprecated.torch_utils as tu

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


class TVL1SymOpticalFlow3D:
    def __init__(self, saveDir, config):
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
        # cuda
        cuda_availabe = config.get('DEVICE', 'cuda_availabe')
        self.DEVICE = "cuda" if cuda_availabe else "cpu"
        # debug
        self.saveDirDebug = os.path.sep.join([self.saveDir, "debug"])
        self.useDebugOutput = config.getboolean("DEBUG", "useDebugOutput")
        if self.useDebugOutput:
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

    def generatePyramid(self, Il, Ic, Ir, u, p):

        # Smooth inputs with a Gaussian filter
        if self.useGaussianBlur:
            Il = applyGaussianBlur3D(Il, self.GaussianBlurSigma)
            Ic = applyGaussianBlur3D(Ic, self.GaussianBlurSigma)
            Ir = applyGaussianBlur3D(Ir, self.GaussianBlurSigma)

        # List for volumes pyramids
        NZ, NY, NX = Ic.shape[0], Ic.shape[1], Ic.shape[2]
        LZ, LY, LX = self.getMeshLength(self.config, NZ, NY, NX)
        meshInfo3D_cuda = opticalFlow.MeshInfo3D(NZ, NY, NX, LZ, LY, LX)

        # meshes for pyramid
        meshInfo_pyramid = [meshInfo3D_cuda]
        NZ_restr, NY_restr, NX_restr = NZ, NY, NX
        for s in range(1, self.NUM_SCALES):
            NZ_restr, NY_restr, NX_restr = math.ceil(
                0.5 * NZ_restr), math.ceil(0.5 * NY_restr), math.ceil(0.5 * NX_restr)
            LZ_restr, LY_restr, LX_restr = self.getMeshLength(self.config, NZ_restr, NY_restr, NX_restr)
            meshInfo_pyramid.append(opticalFlow.MeshInfo3D(NZ_restr, NY_restr, NX_restr, LZ_restr, LY_restr, LX_restr))

        # lists for pyramid
        Il_pyramid = [Il]
        Ic_pyramid = [Ic]
        Ir_pyramid = [Ir]
        u_pyramid = [u]
        p_pyramid = [p]

        # Create the pyramid
        for s in range(1, self.NUM_SCALES):
            prolongationOp_cuda = opticalFlow.Prolongation3D(
                meshInfo_pyramid[s - 1],
                meshInfo_pyramid[s],
                self.InterpolationTypeCuda, self.BoundaryTypeCuda)
            Il_pyramid.append(prolongationOp_cuda.forward(Il_pyramid[s - 1]))
            Ic_pyramid.append(prolongationOp_cuda.forward(Ic_pyramid[s - 1]))
            Ir_pyramid.append(prolongationOp_cuda.forward(Ir_pyramid[s - 1]))
            u_pyramid.append(torch.zeros([meshInfo_pyramid[s].getNZ(), meshInfo_pyramid[s].getNY(),
                                          meshInfo_pyramid[s].getNX(), 3]).float().to(self.DEVICE))
            p_pyramid.append(torch.zeros([meshInfo_pyramid[s].getNZ(), meshInfo_pyramid[s].getNY(),
                                          meshInfo_pyramid[s].getNX(), 3, 3]).float().to(self.DEVICE))

        return Il_pyramid, Ic_pyramid, Ir_pyramid, u_pyramid, p_pyramid, meshInfo_pyramid

    def computeOnPyramid(self, Il, Ic, Ir, u, p):

        Il_pyramid, Ic_pyramid, Ir_pyramid, u_pyramid, p_pyramid, meshInfo_pyramid = self.generatePyramid(Il, Ic, Ir, u, p)

        print("\n")
        print("============================================")
        print("start to compute optical flow for pyramid")
        print("============================================")
        print("\n")

        for s in range(self.NUM_SCALES - 1, -1, -1):

            # Compute the optical flow at scale s
            u_pyramid[s], p_pyramid[s] = self.computeOnSingleStep(
                s, Il_pyramid[s],
                Ic_pyramid[s],
                Ir_pyramid[s],
                u_pyramid[s],
                p_pyramid[s],
                meshInfo_pyramid[s])

            # save step
            self.saveSingleStepToFile(s, Il_pyramid[s], Ic_pyramid[s], Ir_pyramid[s], u_pyramid[s], p_pyramid[s], meshInfo_pyramid[s])

            if s == 0:
                break

            # Prolongate the optical flow and dual variables to the next pyramid level
            prolongationOp_cuda = opticalFlow.Prolongation3D(
                meshInfo_pyramid[s],
                meshInfo_pyramid[s - 1],
                self.InterpolationTypeCuda, self.BoundaryTypeCuda)
            u_pyramid[s - 1] = prolongationOp_cuda.forwardVectorField(u_pyramid[s])
            # factor LXNew/LXOld, ...
            u_pyramid[s - 1][:, :, :, 0] *= meshInfo_pyramid[s - 1].getLX() / meshInfo_pyramid[s].getLX()
            u_pyramid[s - 1][:, :, :, 1] *= meshInfo_pyramid[s - 1].getLY() / meshInfo_pyramid[s].getLY()
            u_pyramid[s - 1][:, :, :, 2] *= meshInfo_pyramid[s - 1].getLZ() / meshInfo_pyramid[s].getLZ()

            # TODO Dirichlet boundary condition for p?
            # p_pyramid[s] = self.dirichlet(p_pyramid[s])

            p_pyramid[s - 1] = prolongationOp_cuda.forwardMatrixField(p_pyramid[s])

            # TODO prolongation factor for p?

        return u_pyramid[0], p_pyramid[0]

    def computeOnSingleStep(self, s, Il, Ic, Ir, u, p, meshInfo):

        print("\n")
        print("start to compute optical flow for single step = ", s)
        progress_bar = tqdm(total=self.MAX_WARPS * self.MAX_OUTER_ITERATIONS)

        # Compute target image gradients
        nablaOp = opticalFlow.Nabla3D_CD(meshInfo, self.BoundaryTypeCuda)
        warpingOp = opticalFlow.Warping3D(meshInfo, self.InterpolationTypeCuda, self.BoundaryTypeCuda)
        Il_grad = nablaOp.forward(Il)
        Ic_grad = nablaOp.forward(Ic)
        Ir_grad = nablaOp.forward(Ir)

        # optionally apply anisotropic differential op
        scalars = torch.zeros([meshInfo.getNZ(), meshInfo.getNY(), meshInfo.getNX()]).float().to(self.DEVICE)
        normals = torch.zeros([meshInfo.getNZ(), meshInfo.getNY(), meshInfo.getNX(), 3]).float().to(self.DEVICE)
        tangents1 = torch.zeros([meshInfo.getNZ(), meshInfo.getNY(), meshInfo.getNX(), 3]).float().to(self.DEVICE)
        tangents2 = torch.zeros([meshInfo.getNZ(), meshInfo.getNY(), meshInfo.getNX(), 3]).float().to(self.DEVICE)
        if self.useAnisotropicDifferentialOp:
            anistropicNablaOp = opticalFlow.AnisotropicNabla3D(meshInfo, self.anisotropicDifferentialOp_alpha, self.anisotropicDifferentialOp_beta)
            # TODO for Il or Ir?
            scalars, normals, tangents1, tangents2 = anistropicNablaOp.computeTangentVecs(Ic_grad)

        z = u

        for w in range(self.MAX_WARPS):
            # Compute the warping of the target image and its derivatives
            Il_warped = warpingOp.forward(Il, -u)
            Il_warped_grad = warpingOp.forwardVectorField(Il_grad, -u)
            Ir_warped = warpingOp.forward(Ir, u)
            Ir_warped_grad = warpingOp.forwardVectorField(Ir_grad, u)
            # Constant part of the rho function
            rho_const_l = Il_warped + torch.sum(u * Il_warped_grad, dim=3) - Ic
            rho_const_r = Ir_warped - torch.sum(u * Ir_warped_grad, dim=3) - Ic

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
                Dz_grad = z_grad
                if self.useAnisotropicDifferentialOp:
                    Dz_grad = anistropicNablaOp.forwardVectorField(z_grad, scalars, normals, tangents1, tangents2)
                dualVariable = p + sigma * Dz_grad
                p = opticalFlow.TVL1OF3D_proxDual(dualVariable, sigma, self.weight_TV, meshInfo)
                breakConditionVecDual[n] = torch.norm(p - pold).item()
                if self.weight_TV == 0.:
                    p = torch.zeros(
                        [meshInfo.getNZ(),
                         meshInfo.getNY(),
                         meshInfo.getNX(),
                         3, 3]).float().to(self.DEVICE)

                # update of primal variable
                uold = u
                # Compute the fidelity data term \rho(u)
                # rho_vec_l = - torch.sum(u * Il_warped_grad, dim=3)
                # rho_vec_r = torch.sum(u * Ir_warped_grad, dim=3)
                rho_vec_l = -Il_warped_grad
                rho_vec_r = Ir_warped_grad
                Dp = p
                if self.useAnisotropicDifferentialOp:
                    Dp = anistropicNablaOp.backwardVectorField(p, scalars, normals, tangents1, tangents2)
                Dp_div = nablaOp.backwardVectorField(Dp)
                primalVariable = u - tau * Dp_div
                u = opticalFlow.TVL1SymOF3D_proxPrimal(
                    primalVariable, tau, self.weight_Matching, rho_const_l, rho_vec_l, rho_const_r, rho_vec_r, meshInfo)
                breakConditionVecPrimal[n] = torch.norm(u - uold).item()

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

                progress_bar.update(1)

            if self.useDebugOutput:
                #saveCurve1D(primalFctVec, MAX_OUTER_ITERATIONS, self.saveDirDebug, f"PrimalFct_it{s}_warp{w}")
                save_curve_1d(breakConditionVecPrimal, self.MAX_OUTER_ITERATIONS,
                            self.saveDirDebug, f"CPErrorPrimal_it{s}_warp{w}", "loglog")
                save_curve_1d(breakConditionVecDual, self.MAX_OUTER_ITERATIONS,
                            self.saveDirDebug, f"CPErrorDual_it{s}_warp{w}", "loglog")
                save_curve_1d(breakConditionVecUpdate, self.MAX_OUTER_ITERATIONS,
                            self.saveDirDebug, f"CPErrorUpdate_it{s}_warp{w}", "loglog")

        return u, p

    def saveSingleStepToFile(self, step, Il, Ic, Ir, u, p, meshInfo):
        saveDirStep = os.path.sep.join([self.saveDir, f"it{step}"])
        if not os.path.exists(saveDirStep):
            os.makedirs(saveDirStep)

        # save original images
        save3D_torch_to_nifty(Il, saveDirStep, f"Il.nii")
        save_slices(Il, f"Il_it{step}.png", saveDirStep)
        save3D_torch_to_nifty(Ic, saveDirStep, f"Ic.nii")
        save_slices(Ic, f"Ic_it{step}.png", saveDirStep)
        save3D_torch_to_nifty(Ir, saveDirStep, f"Ir.nii")
        save_slices(Ir, f"Ir_it{step}.png", saveDirStep)

        # save warped images
        warpingOp = opticalFlow.Warping3D(meshInfo, self.InterpolationTypeCuda, self.BoundaryTypeCuda)
        Il_warped = warpingOp.forward(Il, -u)
        save_slices(Il_warped, f"Il_warped_it{step}.png", saveDirStep)
        Ir_warped = warpingOp.forward(Ir, u)
        save_slices(Ir_warped, f"Ir_warped_it{step}.png", saveDirStep)

        # save warping error
        save_single_zslices(Ic, saveDirStep, "IcSlices", 1., 0)
        save_single_zslices(Ir_warped, saveDirStep, "IrWarpedSlices", 1., 0)
        diff = torch.abs(Ir_warped - Ic)
        save_slices(diff, f"Diff_Irwarped_to_Ic_it{step}.png", saveDirStep)
        save_single_zslices(diff, saveDirStep, "Diff_Irwarped_to_Ic_Slices", 1., 0)
        # test
        # print("norm of diff = ", diff.norm().item())
        # numZSlices = diff.shape[0]
        # for z in range(numZSlices):
        #     print(" norm of diff(z=", z, ") = ", diff[z, :, :].norm().item())

        # save flow
        plotOpticalFlow3D(u.cpu().detach().numpy(), "u", saveDirStep, step)
        flowName = f"flow_it{step}.pt"
        fileNameFlow = os.path.join(saveDirStep, flowName)
        torch.save(u, fileNameFlow)

        flowName = f'flow_it{step}.npy'
        fileNameFlow = os.path.join(saveDirStep, flowName)
        np.save(fileNameFlow, u.cpu().detach().numpy())

        # Median filter pag 12. paper
        if self.USE_MEDIAN_FILTER:
            ks = self.KERNEL_MF
            uf = ndimage.median_filter(u.cpu().detach().numpy(), size=(ks, ks, ks, 1))
            plotOpticalFlow3D(uf, "uf", saveDirStep, step)
            flowName = f"flow_m_it{step}.pt"
            fileNameFlow = os.path.join(saveDirStep, flowName)
            torch.save(torch.from_numpy(uf).to(self.DEVICE), fileNameFlow)

            flowName = f"flow_m_it{step}.npy"
            fileNameFlow = os.path.join(saveDirStep, flowName)
            np.save(fileNameFlow, uf)

        # save dual variable
        dualName = f"dual_it{step}.pt"
        fileNameDual = os.path.join(saveDirStep, dualName)
        torch.save(p, fileNameDual)

    def warpMask(self, mask, u, Ic, tc, saveDir):

        print("warp with optical flow for time step: ", tc)
        print("norm of u = ", u.norm().item())

        # flowName = "flow_it0.pt"
        # fileNameFlow = os.path.join(saveDirStep, flowName)
        # u = torch.load(fileNameFlow, map_location=torch.device(DEVICE))

        NZ, NY, NX = mask.shape[0], mask.shape[1], mask.shape[2]
        LZ, LY, LX = self.getMeshLength(self.config, NZ, NY, NX)
        meshInfo = opticalFlow.MeshInfo3D(NZ, NY, NX, LZ, LY, LX)

        warpingOp = opticalFlow.Warping3D(meshInfo, self.InterpolationTypeCuda, self.BoundaryTypeCuda)
        mask_warped = warpingOp.forward(mask, u)
        # mask_warped = tu.normalize(mask_warped)
        # mask_warped = torch.where(mask_warped > 0.1, 1.0, 0.0)

        save3D_torch_to_nifty(mask_warped, saveDir, f"mask_warped_time{tc}.nii")
        save_slices(mask_warped, f"mask_warped_time{tc}.png", saveDir)
        save_single_zslices(mask_warped, saveDir, "mask_warped_slices", 1., 2)

        # save_color_slices(Ic, mask, saveDir, 'color_mask')
        save_img_mask_single_zslices(Ic, mask, saveDir, 'color_mask')
        save_img_mask_slices(Ic, mask, 'img_mask.png', saveDir)

        maskName = f"masked_warped_time{tc}.pt"
        fileNameMask = os.path.join(saveDir, maskName)
        torch.save(mask_warped, fileNameMask)

        # TODO
        # add mask to mri images
        # saveDirMRISlices = os.path.join(saveDir, "it0/I0Slices")
        # saveDirMaskSlices = os.path.join(saveDir, "mask_slices")
        # saveDirSumSlices = os.path.join(saveDir, "sum_slices")
        # if not os.path.exists(saveDirSumSlices):
        #     os.makedirs(saveDirSumSlices)
        # for z in range(0, NZ):
        #     fileNameMRISlice = os.path.join(saveDirMRISlices, f"colorimg_z{z}.png")
        #     img_mri = cv2.imread(fileNameMRISlice)
        #     fileNameMaskSlice = os.path.join(saveDirMaskSlices, f"colorimg_z{z}.png")
        #     img_mask = cv2.imread(fileNameMaskSlice)
        #     fileNameSumSlice = os.path.join(saveDirSumSlices, f"sumimg_z{z}.png")
        #     img_sum = img_mri + img_mask
        #     cv2.imwrite(fileNameSumSlice, img_sum)
        #     fileNameSumSliceInvert = os.path.join(saveDirSumSlices, f"invertsumimg_z{z}.png")
        #     img_sum_invert = 255. - img_sum
        #     cv2.imwrite(fileNameSumSliceInvert, img_sum_invert)

        return mask_warped
