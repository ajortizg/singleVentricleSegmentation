import numpy as np
import math
import torch
# from colorama import init
from termcolor import colored

import coreDefines
import mesh
# from .coreDefines import *
# from .mesh import *

# forward difference quotients in 1d


class Nabla1D_Forward:

    def __init__(self, meshInfo1D):
        self.meshInfo = meshInfo1D

    def forward(self, u):
        device = u.device
        NX = u.size(dim=0)
        p = torch.zeros((NX, 1), device=device)
        p[:-1, 0] += (u[1:] - u[:-1]) / self.meshInfo.hX
        return p

    def backward(self, p):
        device = p.device
        NX, _ = p.shape
        u = torch.zeros((NX), device=device)
        u[1:] += p[:-1, 0] / self.meshInfo.hX
        u[:-1] -= p[:-1, 0] / self.meshInfo.hX
        return u

    def check_adjointness(self, size_in, size_out):
        print("check adjointness of Nabla1D_Forward:", end=" ")
        u = torch.randn(*size_in).cuda()
        p = torch.randn(*size_out).cuda()
        lhs = self.forward(u).reshape(-1).dot(p.reshape(-1))
        rhs = self.backward(p).reshape(-1).dot(u.reshape(-1))
        diff = torch.max(torch.abs(lhs - rhs)).item()
        coreDefines.printColoredError(diff)


# forward difference quotients in 2d
class Nabla2D_Forward:

    def __init__(self, meshInfo2D):
        self.meshInfo = meshInfo2D

    def forward(self, u):
        device = u.device
        NY = u.size(dim=0)
        NX = u.size(dim=1)
        p = torch.zeros((NY, NX, 2), device=device)
        p[:, :-1, 0] += (u[:, 1:] - u[:, :-1]) / self.meshInfo.hX
        p[:-1, :, 1] += (u[1:, :] - u[:-1, :]) / self.meshInfo.hY
        return p

    def backward(self, p):
        device = p.device
        NY, NX, _ = p.shape
        u = torch.zeros((NY, NX), device=device)

        u[:, 1:] += p[:, :-1, 0] / self.meshInfo.hX
        u[:, :-1] -= p[:, :-1, 0] / self.meshInfo.hX

        u[1:, :] += p[:-1, :, 1] / self.meshInfo.hY
        u[:-1, :] -= p[:-1, :, 1] / self.meshInfo.hY

        return u

    def check_adjointness(self, size_in, size_out):
        print("check adjointness of Nabla2D_Forward", end=" ")
        u = torch.randn(*size_in).cuda()
        p = torch.randn(*size_out).cuda()
        lhs = self.forward(u).reshape(-1).dot(p.reshape(-1))
        rhs = self.backward(p).reshape(-1).dot(u.reshape(-1))
        diff = torch.max(torch.abs(lhs - rhs)).item()
        coreDefines.printColoredError(diff)


# forward difference quotients in 3D
class Nabla3D_Forward:

    def __init__(self, meshInfo3D):
        self.meshInfo = meshInfo3D

    def forward(self, u):
        device = u.device
        NZ = u.size(dim=0)
        NY = u.size(dim=1)
        NX = u.size(dim=2)
        p = torch.zeros((NZ, NY, NX, 3), device=device)
        p[:, :, :-1, 0] += (u[:, :, 1:] - u[:, :, :-1]) / self.meshInfo.hX
        p[:, :-1, :, 1] += (u[:, 1:, :] - u[:, :-1, :]) / self.meshInfo.hY
        p[:-1, :, :, 2] += (u[1:, :, :] - u[:-1, :, :]) / self.meshInfo.hZ
        return p

    def backward(self, p):
        device = p.device
        NZ, NY, NX, _ = p.shape
        u = torch.zeros((NZ, NY, NX), device=device)

        u[:, :, 1:] += p[:, :, :-1, 0] / self.meshInfo.hX
        u[:, :, :-1] -= p[:, :, :-1, 0] / self.meshInfo.hX

        u[:, 1:, :] += p[:, :-1, :, 1] / self.meshInfo.hY
        u[:, :-1, :] -= p[:, :-1, :, 1] / self.meshInfo.hY

        u[1:, :, :] += p[:-1, :, :, 2] / self.meshInfo.hZ
        u[:-1, :, :] -= p[:-1, :, :, 2] / self.meshInfo.hZ

        return u

    def check_adjointness(self, size_in, size_out):
        print("check adjointness of Nabla3D_Forward:", end=" ")
        u = torch.randn(*size_in).cuda()
        p = torch.randn(*size_out).cuda()
        lhs = self.forward(u).reshape(-1).dot(p.reshape(-1))
        rhs = self.backward(p).reshape(-1).dot(u.reshape(-1))
        diff = torch.max(torch.abs(lhs - rhs)).item()
        coreDefines.printColoredError(diff)


# central difference quotients in 1D
class Nabla1D_Central:

    def __init__(self, meshInfo1D):
        self.meshInfo = meshInfo1D

    def forward(self, u):
        device = u.device
        NX = u.size(dim=0)
        p = torch.zeros((NX, 1), device=device)
        p[0, 0] = 0.5 * (u[1] - u[0]) / self.meshInfo.hX
        p[1:-1, 0] = 0.5 * (u[2:] - u[:-2]) / self.meshInfo.hX
        p[-1, 0] = 0.5 * (u[-1] - u[-2]) / self.meshInfo.hX
        return p

    def backward(self, p):
        device = p.device
        NX, _ = p.shape
        u = torch.zeros((NX), device=device)
        u[0] -= 0.5 * p[0, 0] / self.meshInfo.hX
        u[1:] += 0.5 * p[:-1, 0] / self.meshInfo.hX
        u[:-1] -= 0.5 * p[1:, 0] / self.meshInfo.hX
        u[-1] += 0.5 * p[-1, 0] / self.meshInfo.hX
        return u

    def check_adjointness(self, size_in, size_out):
        print("check adjointness of Nabla1D_Central:", end=" ")
        u = torch.randn(*size_in).cuda()
        p = torch.randn(*size_out).cuda()
        lhs = self.forward(u).reshape(-1).dot(p.reshape(-1))
        rhs = self.backward(p).reshape(-1).dot(u.reshape(-1))
        diff = torch.max(torch.abs(lhs - rhs)).item()
        coreDefines.printColoredError(diff)


# central difference quotients in 2D
class Nabla2D_Central:

    def __init__(self, meshInfo2D):
        self.meshInfo = meshInfo2D

    def forward(self, u):
        device = u.device
        NY = u.size(dim=0)
        NX = u.size(dim=1)
        p = torch.zeros((NY, NX, 2), device=device)

        p[:, 0, 0] = 0.5 * (u[:, 1] - u[:, 0]) / self.meshInfo.hX
        p[:, 1:-1, 0] = 0.5 * (u[:, 2:] - u[:, :-2]) / self.meshInfo.hX
        p[:, -1, 0] = 0.5 * (u[:, -1] - u[:, -2]) / self.meshInfo.hX

        p[0, :, 1] = 0.5 * (u[1, :] - u[0, :]) / self.meshInfo.hY
        p[1:-1, :, 1] = 0.5 * (u[2:, :] - u[:-2, :]) / self.meshInfo.hY
        p[-1, :, 1] = 0.5 * (u[-1, :] - u[-2, :]) / self.meshInfo.hY

        return p

    def backward(self, p):
        device = p.device
        NY, NX, _ = p.shape
        u = torch.zeros((NY, NX), device=device)

        u[:, 0] -= 0.5 * p[:, 0, 0] / self.meshInfo.hX
        u[:, 1:] += 0.5 * p[:, :-1, 0] / self.meshInfo.hX
        u[:, :-1] -= 0.5 * p[:, 1:, 0] / self.meshInfo.hX
        u[:, -1] += 0.5 * p[:, -1, 0] / self.meshInfo.hX

        u[0, :] -= 0.5 * p[0, :, 1] / self.meshInfo.hY
        u[1:, :] += 0.5 * p[:-1, :, 1] / self.meshInfo.hY
        u[:-1, :] -= 0.5 * p[1:, :, 1] / self.meshInfo.hY
        u[-1, :] += 0.5 * p[-1, :, 1] / self.meshInfo.hY

        return u

    def check_adjointness(self, size_in, size_out):
        print("check adjointness of Nabla2D_Central:", end=" ")
        u = torch.randn(*size_in).cuda()
        p = torch.randn(*size_out).cuda()
        lhs = self.forward(u).reshape(-1).dot(p.reshape(-1))
        rhs = self.backward(p).reshape(-1).dot(u.reshape(-1))
        diff = torch.max(torch.abs(lhs - rhs)).item()
        coreDefines.printColoredError(diff)


# central difference quotients in 3D
class Nabla3D_Central:

    def __init__(self, meshInfo3D):
        self.meshInfo = meshInfo3D

    def forward(self, u):
        device = u.device
        NZ = u.size(dim=0)
        NY = u.size(dim=1)
        NX = u.size(dim=2)
        p = torch.zeros((NZ, NY, NX, 3), device=device)

        p[:, :, 0, 0] += 0.5 * (u[:, :, 1] - u[:, :, 0]) / self.meshInfo.hX
        p[:, :, 1:-1, 0] += 0.5 * (u[:, :, 2:] - u[:, :, :-2]) / self.meshInfo.hX
        p[:, :, -1, 0] += 0.5 * (u[:, :, -1] - u[:, :, -2]) / self.meshInfo.hX

        p[:, 0, :, 1] += 0.5 * (u[:, 1, :] - u[:, 0, :]) / self.meshInfo.hY
        p[:, 1:-1, :, 1] += 0.5 * (u[:, 2:, :] - u[:, :-2, :]) / self.meshInfo.hY
        p[:, -1, :, 1] += 0.5 * (u[:, -1, :] - u[:, -2, :]) / self.meshInfo.hY

        p[0, :, :, 2] += 0.5 * (u[1, :, :] - u[0, :, :]) / self.meshInfo.hZ
        p[1:-1, :, :, 2] += 0.5 * (u[2:, :, :] - u[:-2, :, :]) / self.meshInfo.hZ
        p[-1, :, :, 2] += 0.5 * (u[-1, :, :] - u[-2, :, :]) / self.meshInfo.hZ

        return p

    def backward(self, p):
        device = p.device
        NZ, NY, NX, _ = p.shape
        hZ = NZ / math.sqrt(0.5 * NX * NX + 0.5 * NY * NY)
        u = torch.zeros((NZ, NY, NX), device=device)

        u[:, :, 0] -= 0.5 * p[:, :, 0, 0] / self.meshInfo.hX
        u[:, :, 1:] += 0.5 * p[:, :, :-1, 0] / self.meshInfo.hX
        u[:, :, :-1] -= 0.5 * p[:, :, 1:, 0] / self.meshInfo.hX
        u[:, :, -1] += 0.5 * p[:, :, -1, 0] / self.meshInfo.hX

        u[:, 0, :] -= 0.5 * p[:, 0, :, 1] / self.meshInfo.hY
        u[:, 1:, :] += 0.5 * p[:, :-1, :, 1] / self.meshInfo.hY
        u[:, :-1, :] -= 0.5 * p[:, 1:, :, 1] / self.meshInfo.hY
        u[:, -1, :] += 0.5 * p[:, -1, :, 1] / self.meshInfo.hY

        u[0, :, :] -= 0.5 * p[0, :, :, 2] / self.meshInfo.hZ
        u[1:, :, :] += 0.5 * p[:-1, :, :, 2] / self.meshInfo.hZ
        u[:-1, :, :] -= 0.5 * p[1:, :, :, 2] / self.meshInfo.hZ
        u[-1, :, :] += 0.5 * p[-1, :, :, 2] / self.meshInfo.hZ

        return u

    def check_adjointness(self, size_in, size_out):
        print("check adjointness of Nabla3D_Central:", end=" ")
        u = torch.randn(*size_in).cuda()
        p = torch.randn(*size_out).cuda()
        lhs = self.forward(u).reshape(-1).dot(p.reshape(-1))
        rhs = self.backward(p).reshape(-1).dot(u.reshape(-1))
        diff = torch.max(torch.abs(lhs - rhs)).item()
        coreDefines.printColoredError(diff)
