#==================================
import sys
# sys.path.append('core')

#==================================
import argparse
import os
import cv2
import nibabel as nib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tikzplotlib #to save from matplotlib to tikz
import math
import torch
import time
import torch.nn.functional as F
from termcolor import colored

#==================================
# sys.path.append("../pythonOps")
# # import differentialOps
# # import mesh
# from pythonOps.mesh import *
# from pythonOps.differentialOps import *
# # from pythonOps import differentialOps

pythonOps_lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../pythonOps'))
sys.path.append(pythonOps_lib_path)
import mesh
import differentialOps

from opticalFlow_cuda_ext import opticalFlow


#==================================
DEVICE = 'cuda'


def printColoredError( diff, tol=1.e-5, accTol=1.e-2 ):
    if( diff < tol):
        print(colored(diff, 'green'))
    elif( diff < accTol ):
        print(colored(diff, 'yellow'))
    else:
        print(colored(diff, 'red'))

#####################################################################
def checkDiffOpPytorchVsCuda( nameDiffOp,  dimVec, diffOpPython, diffOpCuda ):
    
    print("====================================")
    print("check DiffOp Pytorch Vs Cuda:",  nameDiffOp)
    print("====================================")

    #############
    # check nabla
    #############
    testVecForward = torch.randn(dimVec).cuda()
    #dim = testVecForward.dim()
    # pytorch result
    ts = time.time()
    D_testVecForward_torch = diffOpPython.forward(testVecForward)
    print('nabla 1d - elapsed time pytorch: ', (time.time()-ts))
    # cuda kernel result
    ts = time.time()
    D_testVecForward_cuda = diffOpCuda.forward(testVecForward)
    print('nabla 1d - elapsed time cuda kernel: ', (time.time()-ts))
    # difference pytorch to cuda
    print("nabla 1d - Check diff with infty norm: ", end=" ")
    error1dFD = torch.max(torch.abs(D_testVecForward_cuda - D_testVecForward_torch)).item()
    printColoredError(error1dFD)

    #############
    # check divergence
    #############
    testVecBackward = torch.randn(D_testVecForward_torch.shape).cuda()
    # pytorch result
    ts = time.time()
    div_testVecBackward_torch = torch.zeros((dimVec)).cuda()
    div_testVecBackward_torch = diffOpPython.backward(testVecBackward)
    print('divergence 1d - elapsed time pytorch: ', (time.time()-ts))
    # cuda kernel result
    ts = time.time()
    div_testVecBackward_cuda = diffOpCuda.backward(testVecBackward)
    print('divergence 1d - elapsed time cuda kernel: ', (time.time()-ts))
    # difference pytorch to cuda
    print("divergence 1d - Check diff with infty norm: ", end=" ")
    error1dFDBack = torch.max(torch.abs(div_testVecBackward_cuda - div_testVecBackward_torch)).item()
    printColoredError(error1dFDBack)

    #############
    # check adjointness
    #############
    diffOpPython.check_adjointness(testVecForward.shape,testVecBackward.shape)
    print("check adjointness in cuda ", end=" ")
    lhs_cuda =  D_testVecForward_cuda.reshape(-1).dot(testVecBackward.reshape(-1))
    rhs_cuda =  div_testVecBackward_cuda.reshape(-1).dot(testVecForward.reshape(-1))
    diff_cuda = torch.max(torch.abs(lhs_cuda-rhs_cuda)).item()
    printColoredError(diff_cuda)

print("""
==================================
==================================
   check differential operators
==================================
==================================
""")


####################################
# test sizes for meshes
####################################
NX1D = 257
LX1D = 2.
meshInfo1D_python = mesh.MeshInfo1D(NX1D,LX1D)
meshInfo1D_cuda = opticalFlow.MeshInfo1D(NX1D,LX1D)
dimVec1D = torch.Size([NX1D])

NY2D = 5
NX2D = 6
# NY2D = 129
# NX2D = 257
LY2D = 8.
LX2D = 2.
meshInfo2D_python = mesh.MeshInfo2D(NY2D,NX2D,LY2D,LX2D)
meshInfo2D_cuda = opticalFlow.MeshInfo2D(NY2D,NX2D,LY2D,LX2D)
dimVec2D = torch.Size([NY2D,NX2D])

NZ3D = 5
NY3D = 4
NX3D = 5
# NZ3D = 17
# NY3D = 129
# NX3D = 257
LZ3D = 1.
LY3D = 8.
LX3D = 2.
meshInfo3D_python = mesh.MeshInfo3D(NZ3D,NY3D,NX3D,LZ3D,LY3D,LX3D)
meshInfo3D_cuda = opticalFlow.MeshInfo3D(NZ3D,NY3D,NX3D,LZ3D,LY3D,LX3D)
dimVec3D = torch.Size([NZ3D,NY3D,NX3D])

####################################
# forward difference quotients
####################################
# nabla1DFDOp = differentialOps.Nabla1D_Forward(meshInfo1D_python)
# nabla1DFDOpCuda = opticalFlow.Nabla1D_FD(meshInfo1D_cuda)
# checkDiffOpPytorchVsCuda( "1D forward difference quotients", dimVec1D, nabla1DFDOp, nabla1DFDOpCuda )

# nabla2DFDOp = differentialOps.Nabla2D_Forward(meshInfo2D_python)
# nabla2DFDOpCuda = opticalFlow.Nabla2D_FD(meshInfo2D_cuda)
# checkDiffOpPytorchVsCuda( "2D forward difference quotients", dimVec2D, nabla2DFDOp, nabla2DFDOpCuda )

# nabla3DFDOp = differentialOps.Nabla3D_Forward(meshInfo3D_python)
# nabla3DFDOpCuda = opticalFlow.Nabla3D_FD(meshInfo3D_cuda)
# checkDiffOpPytorchVsCuda( "3D forward difference quotients", dimVec3D, nabla3DFDOp, nabla3DFDOpCuda )


####################################
# central difference quotients
####################################
nabla1DCDOp = differentialOps.Nabla1D_Central(meshInfo1D_python)
nabla1DCDOpCuda = opticalFlow.Nabla1D_CD(meshInfo1D_cuda,opticalFlow.BoundaryType.BOUNDARY_NEAREST)
#nabla1DCDOpCuda = opticalFlow.Nabla1D_CD(meshInfo1D_cuda,opticalFlow.BoundaryType.BOUNDARY_ZERO)
checkDiffOpPytorchVsCuda( "1D central difference quotients", dimVec1D, nabla1DCDOp, nabla1DCDOpCuda )

nabla2DCDOp = differentialOps.Nabla2D_Central(meshInfo2D_python)
nabla2DCDOpCuda = opticalFlow.Nabla2D_CD(meshInfo2D_cuda,opticalFlow.BoundaryType.BOUNDARY_NEAREST)
checkDiffOpPytorchVsCuda( "2D central difference quotients", dimVec2D, nabla2DCDOp, nabla2DCDOpCuda )

nabla3DCDOp = differentialOps.Nabla3D_Central(meshInfo3D_python)
nabla3DCDOpCuda = opticalFlow.Nabla3D_CD(meshInfo3D_cuda,opticalFlow.BoundaryType.BOUNDARY_NEAREST)
checkDiffOpPytorchVsCuda( "3D central difference quotients", dimVec3D, nabla3DCDOp, nabla3DCDOpCuda )


####################################
# central difference quotients for vector valued functions
####################################
print("\n\n")
print("==========================================")
print("check diff ops for vector fields in cuda:")
print("==========================================")
print("\n")

boundaryList = [opticalFlow.BoundaryType.BOUNDARY_NEAREST,opticalFlow.BoundaryType.BOUNDARY_MIRROR,opticalFlow.BoundaryType.BOUNDARY_REFLECT]


for boundary in boundaryList:
    print("check adjointness in cuda for 1d cd with", boundary, ": ", end=" ")
    nabla1DCDOpCuda_bdry = opticalFlow.Nabla1D_CD(meshInfo1D_cuda,boundary)
    testVecForwardVector = torch.randn([NX1D]).cuda()
    D_testVecForwardVector_cuda = nabla1DCDOpCuda_bdry.forward(testVecForwardVector)
    testVecBackwardVector = torch.randn(D_testVecForwardVector_cuda.shape).cuda()
    div_testVecBackwardVector_cuda = nabla1DCDOpCuda_bdry.backward(testVecBackwardVector)
    lhs_cuda =  D_testVecForwardVector_cuda.reshape(-1).dot(testVecBackwardVector.reshape(-1))
    rhs_cuda =  div_testVecBackwardVector_cuda.reshape(-1).dot(testVecForwardVector.reshape(-1))
    diff_cuda = torch.max(torch.abs(lhs_cuda-rhs_cuda)).item()
    printColoredError(diff_cuda)

for boundary in boundaryList:
    print("check adjointness in cuda for 2d cd with", boundary, ": ", end=" ")
    nabla2DCDOpCuda_bdry = opticalFlow.Nabla2D_CD(meshInfo2D_cuda,boundary)
    testVecForwardVector = torch.randn([NY2D,NX2D]).cuda()
    D_testVecForwardVector_cuda = nabla2DCDOpCuda_bdry.forward(testVecForwardVector)
    testVecBackwardVector = torch.randn(D_testVecForwardVector_cuda.shape).cuda()
    div_testVecBackwardVector_cuda = nabla2DCDOpCuda_bdry.backward(testVecBackwardVector)
    lhs_cuda =  D_testVecForwardVector_cuda.reshape(-1).dot(testVecBackwardVector.reshape(-1))
    rhs_cuda =  div_testVecBackwardVector_cuda.reshape(-1).dot(testVecForwardVector.reshape(-1))
    diff_cuda = torch.max(torch.abs(lhs_cuda-rhs_cuda)).item()
    printColoredError(diff_cuda)

for boundary in boundaryList:
    print("check adjointness in cuda for 3d cd with", boundary, ": ", end=" ")
    nabla3DCDOpCuda_bdry = opticalFlow.Nabla3D_CD(meshInfo3D_cuda,boundary)
    testVecForwardVector = torch.randn([NZ3D,NY3D,NX3D]).cuda()
    D_testVecForwardVector_cuda = nabla3DCDOpCuda_bdry.forward(testVecForwardVector)
    testVecBackwardVector = torch.randn(D_testVecForwardVector_cuda.shape).cuda()
    div_testVecBackwardVector_cuda = nabla3DCDOpCuda_bdry.backward(testVecBackwardVector)
    lhs_cuda =  D_testVecForwardVector_cuda.reshape(-1).dot(testVecBackwardVector.reshape(-1))
    rhs_cuda =  div_testVecBackwardVector_cuda.reshape(-1).dot(testVecForwardVector.reshape(-1))
    diff_cuda = torch.max(torch.abs(lhs_cuda-rhs_cuda)).item()
    printColoredError(diff_cuda)

for boundary in boundaryList:
    print("check adjointness in cuda for 2d vector field cd with", boundary, ": ", end=" ")
    nabla2DCDOpCuda_bdry = opticalFlow.Nabla2D_CD(meshInfo2D_cuda,boundary)
    testVecForwardVectorField = torch.randn([NY2D,NX2D,2]).cuda()
    D_testVecForwardVectorField_cuda = nabla2DCDOpCuda_bdry.forwardVectorField(testVecForwardVectorField)
    testVecBackwardVectorField = torch.randn(D_testVecForwardVectorField_cuda.shape).cuda()
    div_testVecBackwardVectorField_cuda = nabla2DCDOpCuda_bdry.backwardVectorField(testVecBackwardVectorField)
    lhs_cuda =  D_testVecForwardVectorField_cuda.reshape(-1).dot(testVecBackwardVectorField.reshape(-1))
    rhs_cuda =  div_testVecBackwardVectorField_cuda.reshape(-1).dot(testVecForwardVectorField.reshape(-1))
    diff_cuda = torch.max(torch.abs(lhs_cuda-rhs_cuda)).item()
    printColoredError(diff_cuda)

for boundary in boundaryList:
    print("check adjointness in cuda for 3d vector field cd with", boundary, ": ", end=" ")
    nabla3DCDOpCuda_bdry = opticalFlow.Nabla3D_CD(meshInfo3D_cuda,boundary)
    testVecForwardVectorField3D = torch.randn([NZ3D,NY3D,NX3D,3]).cuda()
    D_testVecForwardVectorField3D_cuda = nabla3DCDOpCuda_bdry .forwardVectorField(testVecForwardVectorField3D)
    testVecBackwardVectorField3D = torch.randn(D_testVecForwardVectorField3D_cuda.shape).cuda()
    div_testVecBackwardVectorField3D_cuda = nabla3DCDOpCuda_bdry .backwardVectorField(testVecBackwardVectorField3D)
    lhs3D_cuda = D_testVecForwardVectorField3D_cuda.reshape(-1).dot(testVecBackwardVectorField3D.reshape(-1))
    rhs3D_cuda = div_testVecBackwardVectorField3D_cuda.reshape(-1).dot(testVecForwardVectorField3D.reshape(-1))
    diff3D_cuda = torch.max(torch.abs(lhs3D_cuda-rhs3D_cuda)).item()
    printColoredError(diff3D_cuda)


# print("test:")
# testVecDim0 = torch.tensor(0.)
# print(testVecDim0)
# print(testVecDim0.shape)
# testVecDim1 = torch.zeros([1])
# print(testVecDim1)
# print(testVecDim1.shape)