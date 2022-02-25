#==================================
import sys
sys.path.append('core')

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
    print("checkDiffOpPytorchVsCuda:",  nameDiffOp)
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


print("""
==================================
==================================
   check differential operators
==================================
==================================
""")


####################################
# forward difference quotients
####################################
dimVec1DFD = torch.Size([2000])
nabla1DFDOp = differentialOps.Nabla1D_Forward()
nabla1DFDOpCuda = opticalFlow.Nabla1D_FD()
checkDiffOpPytorchVsCuda( "1D forward difference quotients", dimVec1DFD, nabla1DFDOp, nabla1DFDOpCuda )

dimVec2DFD = torch.Size([157,200])
nabla2DFDOp = differentialOps.Nabla2D_Forward()
nabla2DFDOpCuda = opticalFlow.Nabla2D_FD()
checkDiffOpPytorchVsCuda( "2D forward difference quotients", dimVec2DFD, nabla2DFDOp, nabla2DFDOpCuda )

dimVec3DFD = torch.Size([27,157,200])
nabla3DFDOp = differentialOps.Nabla3D_Forward()
nabla3DFDOpCuda = opticalFlow.Nabla3D_FD()
checkDiffOpPytorchVsCuda( "3D forward difference quotients", dimVec3DFD, nabla3DFDOp, nabla3DFDOpCuda )


####################################
# central difference quotients
####################################
dimVec1DCD = torch.Size([2000])
nabla1DCDOp = differentialOps.Nabla1D_Central()
nabla1DCDOpCuda = opticalFlow.Nabla1D_CD()
checkDiffOpPytorchVsCuda( "1D central difference quotients", dimVec1DCD, nabla1DCDOp, nabla1DCDOpCuda )

dimVec2DCD = torch.Size([157,200])
nabla2DCDOp = differentialOps.Nabla2D_Central()
nabla2DCDOpCuda = opticalFlow.Nabla2D_CD()
checkDiffOpPytorchVsCuda( "2D central difference quotients", dimVec2DCD, nabla2DCDOp, nabla2DCDOpCuda )

dimVec3DCD = torch.Size([27,157,200])
nabla3DCDOp = differentialOps.Nabla3D_Central()
nabla3DCDOpCuda = opticalFlow.Nabla3D_CD()
checkDiffOpPytorchVsCuda( "3D central difference quotients", dimVec3DCD, nabla3DCDOp, nabla3DCDOpCuda )