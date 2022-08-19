import torch
import configparser
from opticalFlow_cuda_ext import opticalFlow

__all__ = ['Warp', 'WarpCNN']


class Warp:
    def __init__(self, config, NZ, NY, NX):
        # Interpolation
        interp_str = config.get('WARPING', 'InterpolationType')
        self.interp = None
        if interp_str == "NEAREST":
            self.interp = opticalFlow.InterpolationType.INTERPOLATE_NEAREST
        elif interp_str == "LINEAR":
            self.interp = opticalFlow.InterpolationType.INTERPOLATE_LINEAR
        elif interp_str == "CUBIC_HERMITESPLINE":
            self.interp = opticalFlow.InterpolationType.INTERPOLATE_CUBIC_HERMITESPLINE
        else:
            raise Exception("wrong InterpolationType in configParser")

        # Boundary
        boundary_str = config.get('WARPING', 'BoundaryType')
        self.boundary = None
        if boundary_str == "NEAREST":
            self.boundary = opticalFlow.BoundaryType.BOUNDARY_NEAREST
        elif boundary_str == "MIRROR":
            self.boundary = opticalFlow.BoundaryType.BOUNDARY_MIRROR
        elif boundary_str == "REFLECT":
            self.boundary = opticalFlow.BoundaryType.BOUNDARY_REFLECT
        else:
            raise Exception("wrong BoundaryType in configParser")

        self.LZ, self.LY, self.LX = self.getMeshLength(config, NZ, NY, NX)
        self.NZ, self.NY, self.NX = NZ, NY, NX

    def __call__(self, m, u):
        meshInfo = opticalFlow.MeshInfo3D(self.NZ, self.NY, self.NX, self.LZ, self.LY, self.LX)
        warpingOp = opticalFlow.Warping3D(meshInfo, self.interp, self.boundary)
        mw = warpingOp.forward(m, u)
        return mw

    def getMeshLength(self, config, NZ, NY, NX):
        LenghtType = config.get('WARPING', 'LenghtType')
        if LenghtType == "numDofs":
            LZ = NZ - 1
            LY = NY - 1
            LX = NX - 1
            return LZ, LY, LX
        elif LenghtType == "fixed":
            LZ = config.getfloat('WARPING', "LenghtZ")
            LY = config.getfloat('WARPING', "LenghtY")
            LX = config.getfloat('WARPING', "LenghtX")
            return LZ, LY, LX


class WarpCNN:
    def __init__(self, config, NZ, NY, NX):
        # Interpolation
        interp_str = config.get('WARPING', 'InterpolationType')
        self.interp = None
        if interp_str == "NEAREST":
            self.interp = opticalFlow.InterpolationType.INTERPOLATE_NEAREST
        elif interp_str == "LINEAR":
            self.interp = opticalFlow.InterpolationType.INTERPOLATE_LINEAR
        elif interp_str == "CUBIC_HERMITESPLINE":
            self.interp = opticalFlow.InterpolationType.INTERPOLATE_CUBIC_HERMITESPLINE
        else:
            raise Exception("wrong InterpolationType in configParser")

        # Boundary
        boundary_str = config.get('WARPING', 'BoundaryType')
        self.boundary = None
        if boundary_str == "NEAREST":
            self.boundary = opticalFlow.BoundaryType.BOUNDARY_NEAREST
        elif boundary_str == "MIRROR":
            self.boundary = opticalFlow.BoundaryType.BOUNDARY_MIRROR
        elif boundary_str == "REFLECT":
            self.boundary = opticalFlow.BoundaryType.BOUNDARY_REFLECT
        else:
            raise Exception("wrong BoundaryType in configParser")

        self.LZ, self.LY, self.LX = self.getMeshLength(config, NZ, NY, NX)
        self.NZ, self.NY, self.NX = NZ, NY, NX

    def __call__(self, m, u):
        meshInfo = opticalFlow.MeshInfo3D(self.NZ, self.NY, self.NX, self.LZ, self.LY, self.LX)
        warpingOp = opticalFlow.WarpingCNN3D(meshInfo, self.interp, self.boundary)
        mw = warpingOp.forward(m.contiguous(), u.contiguous())
        return mw

    def getMeshLength(self, config, NZ, NY, NX):
        LenghtType = config.get('WARPING', 'LenghtType')
        if LenghtType == "numDofs":
            LZ = NZ - 1
            LY = NY - 1
            LX = NX - 1
            return LZ, LY, LX
        elif LenghtType == "fixed":
            LZ = config.getfloat('WARPING', "LenghtZ")
            LY = config.getfloat('WARPING', "LenghtY")
            LX = config.getfloat('WARPING', "LenghtX")
            return LZ, LY, LX
