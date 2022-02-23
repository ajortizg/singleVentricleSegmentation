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

from tvl1OF3d_cuda_ext import tvl1OF3d


#==================================
DEVICE = 'cuda'


def printColoredError( diff, tol=1.e-5, accTol=1.e-2 ):
    if( diff < tol):
        print(colored(diff, 'green'))
    elif( diff < accTol ):
        print(colored(diff, 'yellow'))
    else:
        print(colored(diff, 'red'))


print("""
==================================
==================================
   check differential operators
==================================
==================================
""")


#####################################################################
print("====================================")
print("test forward finite difference in 1d")
print("====================================")

NX1DFD = 2000
nabla1DFDOp = differentialOps.Nabla1D_Forward()

#############
# check nabla
#############
testVec1dFD = torch.randn(NX1DFD).cuda()
# pytorch result
ts = time.time()
D_testVec1dFD_torch = torch.zeros((NX1DFD, 1)).cuda()
D_testVec1dFD_torch = nabla1DFDOp.forward(testVec1dFD)
print('nabla 1d - elapsed time pytorch: ', (time.time()-ts))
# cuda kernel result
ts = time.time()
D_testVec1dFD_cuda = tvl1OF3d.nabla1d_fd_forward(testVec1dFD)
print('nabla 1d - elapsed time cuda kernel: ', (time.time()-ts))
# difference pytorch to cuda
print("nabla 1d - Check diff with infty norm: ", end=" ")
error1dFD = torch.max(torch.abs(D_testVec1dFD_cuda - D_testVec1dFD_torch)).item()
printColoredError(error1dFD)

#############
# check divergence
#############
testVec1dFDBack = torch.randn(NX1DFD,1).cuda()
# pytorch result
ts = time.time()
div_testVec1dFD_torch = torch.zeros((NX1DFD)).cuda()
div_testVec1dFD_torch = nabla1DFDOp.backward(testVec1dFDBack)
print('divergence 1d - elapsed time pytorch: ', (time.time()-ts))
# cuda kernel result
ts = time.time()
div_testVec1dFD_cuda = tvl1OF3d.divergence1d_fd_backward(testVec1dFDBack)
print('divergence 1d - elapsed time cuda kernel: ', (time.time()-ts))
# difference pytorch to cuda
print("divergence 1d - Check diff with infty norm: ", end=" ")
error1dFDBack = torch.max(torch.abs(div_testVec1dFD_cuda - div_testVec1dFD_torch)).item()
printColoredError(error1dFDBack)

#############
# check adjointness
#############
nabla1DFDOp.check_adjointness(testVec1dFD.shape,D_testVec1dFD_torch.shape)


#####################################################################
print("====================================")
print("test central finite difference in 1d")
print("====================================")

NX1DCD = 2000
nabla1DCDOp = differentialOps.Nabla1D_Central()

#############
# check nabla
#############
testVec1dCD = torch.randn(NX1DCD).cuda()
# pytorch result
ts = time.time()
D_testVec1dCD_torch = torch.zeros((NX1DCD, 1)).cuda()
D_testVec1dCD_torch = nabla1DCDOp.forward(testVec1dCD)
print('nabla 1d - elapsed time pytorch: ', (time.time()-ts))
# cuda kernel result
ts = time.time()
D_testVec1dCD_cuda = tvl1OF3d.nabla1d_cd_forward(testVec1dCD)
print('nabla 1d - elapsed time cuda kernel: ', (time.time()-ts))
# difference pytorch to cuda
print("nabla 1d - Check diff with infty norm:", end=" " )
error1dCD = torch.max(torch.abs(D_testVec1dCD_cuda - D_testVec1dCD_torch)).item()
printColoredError(error1dCD)

#############
# check divergence
#############
testVec1dCDBack = torch.randn(NX1DCD,1).cuda()
# pytorch result
ts = time.time()
div_testVec1dCD_torch = torch.zeros((NX1DCD)).cuda()
div_testVec1dCD_torch = nabla1DCDOp.backward(testVec1dCDBack)
print('divergence 1d - elapsed time pytorch: ', (time.time()-ts))
# cuda kernel result
ts = time.time()
div_testVec1dCD_cuda = tvl1OF3d.divergence1d_cd_backward(testVec1dCDBack)
print('divergence 1d- elapsed time cuda kernel: ', (time.time()-ts))
# difference pytorch to cuda
print("divergence 1d - Check diff with infty norm:", end=" " )
error1dCDBack = torch.max(torch.abs(div_testVec1dCD_cuda - div_testVec1dCD_torch)).item()
printColoredError(error1dCDBack)

#############
#check adjoint 
#############
nabla1DCDOp.check_adjointness(testVec1dCD.shape,D_testVec1dCD_torch.shape)






# ####################################################################
print("====================================")
print("test forward finite difference in 2d")
print("====================================")

NY2DFD=157
NX2DFD=200
nabla2DFDOp = differentialOps.Nabla2D_Forward()

#############
# check nabla
#############
testVec2dFD = torch.randn(NY2DFD,NX2DFD).cuda()
# pytorch result
ts = time.time()
D_testVec2dFD_torch = torch.zeros((NY2DFD,NX2DFD, 2)).cuda()
D_testVec2dFD_torch = nabla2DFDOp.forward(testVec2dFD)
print('nabla 2d - elapsed time pytorch: ', (time.time()-ts))
# cuda kernel result
ts = time.time()
D_testVec2dFD_cuda = tvl1OF3d.nabla2d_fd_forward(testVec2dFD)
print('nabla 2d - elapsed time cuda kernel: ', (time.time()-ts))
# difference pytorch to cuda
print("nabla 2d - Check diff with infty norm:", end=" " )
error2dFD = torch.max(torch.abs(D_testVec2dFD_cuda - D_testVec2dFD_torch)).item()
printColoredError(error2dFD)

#############
# check divergence
#############
testVec2dFDBack = torch.randn(NY2DFD,NX2DFD,2).cuda()
# pytorch result
ts = time.time()
div_testVec2dFD_torch = torch.zeros((NY2DFD,NX2DFD)).cuda()
div_testVec2dFD_torch = nabla2DFDOp.backward(testVec2dFDBack)
print('divergence 2d - elapsed time pytorch: ', (time.time()-ts))
# cuda kernel result
ts = time.time()
div_testVec2dFD_cuda = tvl1OF3d.divergence2d_fd_backward(testVec2dFDBack)
print('divergence 2d - elapsed time cuda kernel: ', (time.time()-ts))
# difference pytorch to cuda
print("divergence 2d - Check diff with infty norm:", end=" " )
error2dFDBack = torch.max(torch.abs(div_testVec2dFD_cuda - div_testVec2dFD_torch)).item()
printColoredError(error2dFDBack)

#############
# check adjointness
#############
nabla2DFDOp.check_adjointness(testVec2dFD.shape,D_testVec2dFD_torch.shape)


#####################################################################
print("====================================")
print("test central finite difference in 2d")
print("====================================")

NY2DCD=157
NX2DCD=200
nabla2dCDOp = differentialOps.Nabla2D_Central()

#############
# check nabla
#############
testVec2dCD = torch.randn(NY2DCD,NX2DCD).cuda()
# pytorch result
ts = time.time()
D_testVec2dCD_torch = torch.zeros((NY2DCD,NX2DCD, 2)).cuda()
D_testVec2dCD_torch = nabla2dCDOp.forward(testVec2dCD)
print('nabla 2d - elapsed time pytorch: ', (time.time()-ts))
# cuda kernel result
ts = time.time()
D_testVec2dCD_cuda = tvl1OF3d.nabla2d_cd_forward(testVec2dCD)
print('nabla 2d - elapsed time cuda kernel: ', (time.time()-ts))
# difference pytorch to cuda
print("nabla 2d - Check diff with infty norm:", end=" " )
error2dCD = torch.max(torch.abs(D_testVec2dCD_cuda - D_testVec2dCD_torch)).item()
printColoredError(error2dCD)

#############
# check divergence
#############
testVec2dCDBack = torch.randn(NY2DCD,NX2DCD,2).cuda()
# pytorch result
ts = time.time()
div_testVec2dCD_torch = torch.zeros((NY2DCD,NX2DCD)).cuda()
div_testVec2dCD_torch = nabla2dCDOp.backward(testVec2dCDBack)
print('divergence 2d - elapsed time pytorch: ', (time.time()-ts))
# cuda kernel result
ts = time.time()
div_testVec2dCD_cuda = tvl1OF3d.divergence2d_cd_backward(testVec2dCDBack)
print('divergence 2d - elapsed time cuda kernel: ', (time.time()-ts))
# difference pytorch to cuda
print("divergence 2d - Check diff with infty norm:", end=" " )
error2dCDBack = torch.max(torch.abs(div_testVec2dCD_cuda - div_testVec2dCD_torch)).item()
printColoredError(error2dCDBack)

#############
#check adjoint 
#############
nabla2dCDOp.check_adjointness(testVec2dCD.shape,D_testVec2dCD_torch.shape)






# ####################################################################
print("====================================")
print("test forward finite difference in 3d")
print("====================================")

NZ3DFD=16
NY3DFD=157
NX3DFD=213
nabla3dFDOp = differentialOps.Nabla3D_Forward()

#############
# check nabla
#############
testVec3dFD = torch.randn(NZ3DFD,NY3DFD,NX3DFD).cuda()
# pytorch result
ts = time.time()
D_testVec3dFD_torch = torch.zeros((NZ3DFD,NY3DFD,NX3DFD, 3)).cuda()
D_testVec3dFD_torch = nabla3dFDOp.forward(testVec3dFD)
print('nabla 3d - elapsed time pytorch: ', (time.time()-ts))
# cuda kernel result
ts = time.time()
D_testVec3dFD_cuda = tvl1OF3d.nabla3d_fd_forward(testVec3dFD)
print('nabla 3d - elapsed time cuda kernel: ', (time.time()-ts))
# difference pytorch to cuda
print("nabla 3d - Check diff with infty norm:", end=" " )
error3dFD = torch.max(torch.abs(D_testVec3dFD_cuda - D_testVec3dFD_torch)).item()
printColoredError(error3dFD)

#############
# check divergence
#############
testVec3dFDBack = torch.randn(NZ3DFD,NY3DFD,NX3DFD,3).cuda()
# pytorch result
ts = time.time()
div_testVec3dFD_torch = torch.zeros((NZ3DFD,NY3DFD,NX3DFD)).cuda()
div_testVec3dFD_torch = nabla3dFDOp.backward(testVec3dFDBack)
print('divergence 3d - elapsed time pytorch: ', (time.time()-ts))
# cuda kernel result
ts = time.time()
div_testVec3dFD_cuda = tvl1OF3d.divergence3d_fd_backward(testVec3dFDBack)
print('divergence 3d - elapsed time cuda kernel: ', (time.time()-ts))
# difference pytorch to cuda
print("divergence 3d - Check diff with infty norm:", end=" " )
error3dFDBack = torch.max(torch.abs(div_testVec3dFD_cuda - div_testVec3dFD_torch)).item()
printColoredError(error3dFDBack)

#############
# check adjointness
#############
nabla3dFDOp.check_adjointness(testVec3dFD.shape,D_testVec3dFD_torch.shape)


#####################################################################
print("====================================")
print("test central finite difference in 3d")
print("====================================")

NZ3DCD=16
NY3DCD=157
NX3DCD=213
nabla3dCDOp = differentialOps.Nabla3D_Central()

#############
# check nabla
#############
testVec3dCD = torch.randn(NZ3DCD,NY3DCD,NX3DCD).cuda()
# pytorch result
ts = time.time()
D_testVec3dCD_torch = torch.zeros((NZ3DCD,NY3DCD,NX3DCD,3)).cuda()
D_testVec3dCD_torch = nabla3dCDOp.forward(testVec3dCD)
print('nabla 3d - elapsed time pytorch: ', (time.time()-ts))
# cuda kernel result
ts = time.time()
D_testVec3dCD_cuda = tvl1OF3d.nabla3d_cd_forward(testVec3dCD)
print('nabla 3d - elapsed time cuda kernel: ', (time.time()-ts))
# difference pytorch to cuda
print("nabla 3d - Check diff with infty norm:", end=" " )
error3dCD = torch.max(torch.abs(D_testVec3dCD_cuda - D_testVec3dCD_torch)).item()
printColoredError(error3dCD)

#############
# check divergence
#############
testVec3dCDBack = torch.randn(NZ3DCD,NY3DCD,NX3DCD,3).cuda()
# pytorch result
ts = time.time()
div_testVec3dCD_torch = torch.zeros((NZ3DCD,NY3DCD,NX3DCD)).cuda()
div_testVec3dCD_torch = nabla3dCDOp.backward(testVec3dCDBack)
print('divergence 3d - elapsed time pytorch: ', (time.time()-ts))
# cuda kernel result
ts = time.time()
div_testVec3dCD_cuda = tvl1OF3d.divergence3d_cd_backward(testVec3dCDBack)
print('divergence 3d - elapsed time cuda kernel: ', (time.time()-ts))
# difference pytorch to cuda
print("divergence 3d - Check diff with infty norm:", end=" " )
error3dCDBack = torch.max(torch.abs(div_testVec3dCD_cuda - div_testVec3dCD_torch)).item()
printColoredError(error3dCDBack)

#############
#check adjoint 
#############
nabla3dCDOp.check_adjointness(testVec3dCD.shape,D_testVec3dCD_torch.shape)




# ####################################################################
# # # test nabla3d_cd 
# print('\ntest nabla3d_cd with (X,Y,Z): ')

# b = torch.randn(NX, NY, NZ).cuda()
# #b = imageA

# ts = time.time()
# # pytorch result
# Db_torch = torch.zeros((NX, NY, NZ, 3)).cuda()
# Db_torch = forward_torch(b)
# print('nabla3d_cd - elapsed time pytorch: ', (time.time()-ts))

# ts = time.time()
# # cuda kernel result
# Db_cuda = tvl1OF3d.alternative_nabla3d_cd_forward(b)
# print('nabla3d_cd - elapsed time cuda kernel: ', (time.time()-ts))
# print(Db_cuda.shape)

# # infinity norm to measure difference
# print('nabla3d_cd - Check diff with infty norm: ', torch.max(torch.abs(Db_cuda - Db_torch)).item())
# print('pytorch norm: ', torch.max(torch.abs(Db_torch)).item())
# print('cuda norm: ', torch.max(torch.abs(Db_cuda)).item())
# print('nabla3d_cd - Check diff with infty norm 0: ', torch.max(torch.abs(Db_cuda[:,:,:,0] - Db_torch[:,:,:,0])).item())
# print('nabla3d_cd - Check diff with infty norm 1: ', torch.max(torch.abs(Db_cuda[:,:,:,1] - Db_torch[:,:,:,1])).item())
# print('nabla3d_cd - Check diff with infty norm 2: ', torch.max(torch.abs(Db_cuda[:,:,:,2] - Db_torch[:,:,:,2])).item())
# print('nabla3d_cd - Check diff with norm 0: ', torch.max(torch.norm(Db_cuda[:,:,:,0] - Db_torch[:,:,:,0])).item())
# print('nabla3d_cd - Check diff with norm 1: ', torch.max(torch.norm(Db_cuda[:,:,:,1] - Db_torch[:,:,:,1])).item())
# print('nabla3d_cd - Check diff with norm 2: ', torch.max(torch.norm(Db_cuda[:,:,:,2] - Db_torch[:,:,:,2])).item())