import torch.nn.functional as F
from opticalFlow_cuda_ext import opticalFlow
import sys
import os
import math
import numpy as np
import torch
from typing import Literal, Tuple

from svs.utils.flow_utils import get_interpolation_type, get_mesh_length, apply_gaussian_blur3d


class TVL13DOpticalFlow:
    def __init__(
        self,
        num_scales: int,
        max_warps: int,
        max_outer_iterations: int,
        weight_matching: float,
        weight_tv: float,
        primaldual_algorithm_type: Literal[1, 2],
        primaldual_algorithm_params: Tuple[float, float, float, float],
        use_anisotropic_diff: bool,
        anisotropic_diff_params: Tuple[float, float],
        use_gaussian_blur: bool,
        gaussian_blur_sigma: float,
        interpolation_type: Literal['NEAREST', 'LINEAR', 'CUBIC_HERMITESPLINE'],
        boundary_type: Literal['NEAREST', 'MIRROR', 'REFLECT'],
        length_type: Literal['numDofs', 'fixed'],
        length: Tuple[int, int, int],
        device: torch.device
    ):
        """
        Initializes TV-L1 3D Optical Flow with provided parameters.

        Args:
            num_scales (int): Number of scales to use in the multi-scale approach. Higher values can capture finer details but require more computational resources.
            max_warps (int): Maximum number of warping iterations per scale. More iterations may lead to better accuracy but increase computation time.
            max_outer_iterations (int):  Maximum number of iterations for the outer loop optimization process. Higher values can lead to better convergence but increase computation time.
            weight_matching (float): Weight of the data term in the TV-L1 energy functional. It controls the influence of matching the target image to the warped source image.
            weight_tv (float): Weight of the TV (Total Variation) regularization term in the energy functional. It controls the smoothness of the computed optical flow field.
            primaldual_algorithm_type (Literal[1, 2]): Type of primal-dual algorithm (1 or 2).
            primaldual_algorithm_params (List[float, float, float, float]): sigma, tau, theta and gamma.
            use_anisotropic_diff (bool): Indicates whether to apply anisotropic diffusion regularization to the gradient field. Anisotropic diffusion can enhance edge preservation in the computed optical flow.
            anisotropic_diff_params (List[float, float]): Alpha, beta parameters for anistropic diffusion.
            use_gaussian_blur (bool): Indicates whether to apply Gaussian blurring to the input images before computing optical flow. Gaussian blurring can help in reducing noise and artifacts in the images.
            gaussian_blur_sigma (float): Sigma parameter for Gaussian blur.
            interpolation_type (Literal['NEAREST', 'LINEAR', 'CUBIC_HERMITESPLINE']): Interpolation type.
            boundary_type (Literal['NEAREST', 'MIRROR', 'REFLECT']): Boundary type.
            length_type (Literal['numDofs', 'fixed']): Length type.
            device (torch.device): Torch device.
        """
        self.num_scales = num_scales
        self.max_warps = max_warps
        self.max_outer_iterations = max_outer_iterations
        self.weight_matching = weight_matching
        self.weight_tv = weight_tv
        self.primaldual_algorithm_type = primaldual_algorithm_type
        self.sigma, self.tau, self.theta, self.gamma = primaldual_algorithm_params
        self.use_anisotropic_diff = use_anisotropic_diff
        self.anisotropic_diff_alpha, self.anistropic_diff_beta = anisotropic_diff_params
        self.use_gaussian_blur = use_gaussian_blur
        self.gaussian_blur_sigma = gaussian_blur_sigma
        self.length_type = length_type
        self.length_x, self.length_y, self.length_z = length
        self.interpolation_type, self.boundary_type = get_interpolation_type(interpolation_type, boundary_type)
        self.device = device

    def generate_pyramid(self, I0: torch.Tensor, I1: torch.Tensor, u: torch.Tensor, p: torch.Tensor):
        """
        Generates pyramid for the input images and variables.

        Args:
            I0 (torch.Tensor): Input image 0.
            I1 (torch.Tensor): Input image 1.
            u (torch.Tensor): Optical flow variable.
            p (torch.Tensor): Dual variable.

        Returns:
            Tuple: Tuple containing the lists of images, optical flow variables, dual variables, and mesh infos.
        """
        # Smooth inputs with a Gaussian filter
        if self.use_gaussian_blur:
            I0 = apply_gaussian_blur3d(I0, self.gaussian_blur_sigma)
            I1 = apply_gaussian_blur3d(I1, self.gaussian_blur_sigma)

        # List for volumes pyramids
        NZ, NY, NX = I0.shape[0], I0.shape[1], I0.shape[2]
        LZ, LY, LX = get_mesh_length(self.length_type, NZ, NY, NX, self.length_z, self.length_y, self.length_x)
        mesh_info3d = opticalFlow.MeshInfo3D(NZ, NY, NX, LZ, LY, LX)

        # Meshes for pyramid
        mesh_infos = [mesh_info3d]
        NZ_restr, NY_restr, NX_restr = NZ, NY, NX
        for s in range(1, self.num_scales):
            NZ_restr, NY_restr, NX_restr = math.ceil(0.5 * NZ_restr), math.ceil(0.5 * NY_restr), math.ceil(0.5 * NX_restr)
            LZ_restr, LY_restr, LX_restr = get_mesh_length(self.length_type, NZ_restr, NY_restr, NX_restr,
                                                           self.length_z, self.length_y, self.length_x)
            mesh_infos.append(opticalFlow.MeshInfo3D(NZ_restr, NY_restr, NX_restr, LZ_restr, LY_restr, LX_restr))

        # lists for pyramid
        I0s = [I0]
        I1s = [I1]
        us = [u]
        ps = [p]

        # Create the pyramid
        for s in range(1, self.num_scales):
            prolongation_op = opticalFlow.Prolongation3D(
                mesh_infos[s - 1],
                mesh_infos[s],
                self.interpolation_type, self.boundary_type
            )
            I0s.append(prolongation_op.forward(I0s[s - 1]))
            I1s.append(prolongation_op.forward(I1s[s - 1]))
            us.append(torch.zeros([mesh_infos[s].getNZ(), mesh_infos[s].getNY(), mesh_infos[s].getNX(), 3], dtype=torch.float32, device=self.device))
            ps.append(torch.zeros([mesh_infos[s].getNZ(), mesh_infos[s].getNY(),
                      mesh_infos[s].getNX(), 3, 3], dtype=torch.float32, device=self.device))

        return I0s, I1s, us, ps, mesh_infos

    def compute(self, I0: torch.Tensor, I1: torch.Tensor, u: torch.Tensor, p: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes TV-L1 Optical Flow.

        Args:
            I0 (torch.Tensor): Initial image.
            I1 (torch.Tensor): Target image.
            u (torch.Tensor): Optical flow field.
            p (torch.Tensor): Dual variable.

        Returns:
            Tuple of computed optical flow and dual variable.
        """
        I0s, I1s, us, ps, mesh_infos = self.generate_pyramid(I0, I1, u, p)

        for s in range(self.num_scales - 1, -1, -1):
            # Compute the optical flow at scale s
            us[s], ps[s] = self.step(s, I0s[s], I1s[s], us[s], ps[s], mesh_infos[s])

            if s == 0:
                break

            # Prolongate the optical flow and dual variables to the next pyramid level
            prolongation_op = opticalFlow.Prolongation3D(
                mesh_infos[s],
                mesh_infos[s - 1],
                self.interpolation_type, self.boundary_type
            )
            us[s - 1] = prolongation_op.forwardVectorField(us[s])
            us[s - 1][..., 0] *= mesh_infos[s - 1].getLX() / mesh_infos[s].getLX()
            us[s - 1][..., 1] *= mesh_infos[s - 1].getLY() / mesh_infos[s].getLY()
            us[s - 1][..., 2] *= mesh_infos[s - 1].getLZ() / mesh_infos[s].getLZ()

            # TODO! Dirichlet boundary condition for p?
            # ps[s] = self.dirichlet(ps[s])

            ps[s - 1] = prolongation_op.forwardMatrixField(ps[s])

            # TODO! Prolongation factor for p?

        return us[0], ps[0]

    def step(self, s: int, I0: torch.Tensor, I1: torch.Tensor, u: torch.Tensor, p: torch.Tensor, meshInfo):
        """
        Performs a single step in the TV-L1 Optical Flow computation.

        Args:
            s (int): Scale level.
            I0 (torch.Tensor): Initial image.
            I1 (torch.Tensor): Target image.
            u (torch.Tensor): Optical flow field.
            p (torch.Tensor): Dual variable.
            meshInfo (opticalFlow.MeshInfo3D): Mesh information.

        Returns:
            Tuple of updated optical flow field and dual variable.
        """
        # Compute target image gradients
        nabla_op = opticalFlow.Nabla3D_CD(meshInfo, self.boundary_type)
        warping_op = opticalFlow.Warping3D(meshInfo, self.interpolation_type, self.boundary_type)
        I1_grad = nabla_op.forward(I1)

        # optionally apply anisotropic differential op
        scalars = torch.zeros([meshInfo.getNZ(), meshInfo.getNY(), meshInfo.getNX()], dtype=torch.float32, device=self.device)
        normals = torch.zeros([meshInfo.getNZ(), meshInfo.getNY(), meshInfo.getNX(), 3], dtype=torch.float32, device=self.device)
        tangents1 = torch.zeros([meshInfo.getNZ(), meshInfo.getNY(), meshInfo.getNX(), 3], dtype=torch.float32, device=self.device)
        tangents2 = torch.zeros([meshInfo.getNZ(), meshInfo.getNY(), meshInfo.getNX(), 3], dtype=torch.float32, device=self.device)

        if self.use_anisotropic_diff:
            anistropic_nabla_op = opticalFlow.AnisotropicNabla3D(meshInfo, self.anisotropic_diff_alpha, self.anistropic_diff_beta)
            scalars, normals, tangents1, tangents2 = anistropic_nabla_op.computeTangentVecs(I1_grad)

        z = u

        for w in range(self.max_warps):
            # Compute the warping of the target image and its derivatives
            I1_warped = warping_op.forward(I1, u)
            I1_warped_grad = warping_op.forwardVectorField(I1_grad, u)
            # Constant part of the rho function
            rho_c = I1_warped - torch.sum(u * I1_warped_grad, dim=3) - I0

            break_cond_primal = torch.zeros([self.max_outer_iterations], dtype=torch.float32, device=self.device)
            break_cond_Dual = torch.zeros([self.max_outer_iterations], dtype=torch.float32, device=self.device)
            break_cond_update = torch.zeros([self.max_outer_iterations], dtype=torch.float32, device=self.device)

            # TODO! possible use results for new warp
            sigma = self.sigma
            tau = self.tau
            theta = self.theta
            gamma = self.gamma

            for n in range(self.max_outer_iterations):
                # update of dual variable
                pold = p
                z_grad = nabla_op.forwardVectorField(z)
                Dz_grad = z_grad
                if self.use_anisotropic_diff:
                    Dz_grad = anistropic_nabla_op.forwardVectorField(z_grad, scalars, normals, tangents1, tangents2)
                dualVariable = p + sigma * Dz_grad
                p = opticalFlow.TVL1OF3D_proxDual(dualVariable, sigma, self.weight_tv, meshInfo)
                break_cond_Dual[n] = torch.norm(p - pold).item()
                if self.weight_tv == 0.:
                    p = torch.zeros([meshInfo.getNZ(), meshInfo.getNY(), meshInfo.getNX(), 3, 3], dtype=torch.float32, device=self.device)

                # update of primal variable
                uold = u
                # Compute the fidelity data term rho(u)
                rho = rho_c + torch.sum(u * I1_warped_grad, dim=3)
                Dp = p
                if self.use_anisotropic_diff:
                    Dp = anistropic_nabla_op.backwardVectorField(p, scalars, normals, tangents1, tangents2)
                Dp_div = nabla_op.backwardVectorField(Dp)
                primalVariable = u - tau * Dp_div
                u = opticalFlow.TVL1OF3D_proxPrimal(primalVariable, tau, self.weight_matching, rho, I1_warped_grad, meshInfo)

                break_cond_primal[n] = torch.norm(u - uold).item()

                # Update of step sizes
                if self.primaldual_algorithm_type == 1:
                    # Do nothing
                    print("Apply CP1")
                elif self.primaldual_algorithm_type == 2:
                    theta = 1. / math.sqrt(1. + 2. * gamma * tau)
                    tau *= theta
                    sigma /= theta
                else:
                    print("wrong method for Chambolle-Pock-Algorithm")

                # Overrelaxation
                zold = z
                z = u
                break_cond_update[n] = torch.norm(z - zold).item()

        return u, p
