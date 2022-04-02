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



#==================================
pythonOps_lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../pythonOps'))
sys.path.append(pythonOps_lib_path)
import mesh
import prolongationOps

from opticalFlow_cuda_ext import opticalFlow


#==================================
DEVICE = 'cuda'


interpolationList = [opticalFlow.InterpolationType.INTERPOLATE_LINEAR,opticalFlow.InterpolationType.INTERPOLATE_CUBIC_HERMITESPLINE]
boundaryList = [opticalFlow.BoundaryType.BOUNDARY_NEAREST,opticalFlow.BoundaryType.BOUNDARY_MIRROR,opticalFlow.BoundaryType.BOUNDARY_REFLECT]



print("""
==================================
   testProlongationOps
==================================
""")



print("""
===========
   1D
===========
""")

NX1D = 39
LX1D = 2.
meshInfo1D_old = opticalFlow.MeshInfo1D(NX1D,LX1D)

NX1D_new = 147
LX1D_new = 3.
meshInfo1D_new = opticalFlow.MeshInfo1D(NX1D_new,LX1D_new)

for boundary in boundaryList:
    print("check prolongation for 1d with", boundary, ":")
    prolongationOp1D_lin_nearest = opticalFlow.Prolongation1D(meshInfo1D_old,meshInfo1D_new,interpolationList[0],boundary)
    prolongationOp1D_cubic_nearest = opticalFlow.Prolongation1D(meshInfo1D_old,meshInfo1D_new,interpolationList[1],boundary)

    testVec1D = torch.randn([NX1D]).cuda()
    testVec1D_prolong_lin = prolongationOp1D_lin_nearest.forward(testVec1D)
    testVec1D_prolong_cubicHS = prolongationOp1D_cubic_nearest.forward(testVec1D)
    print("diff linear vs cubic = ", torch.norm(testVec1D_prolong_lin - testVec1D_prolong_cubicHS).item() )

    # testMatrixField = torch.randn([NY2D,NX2D,2,2]).cuda()
    # #
    # testMatrixField_prolong_lin_compare = torch.zeros([NY2D_new,NX2D_new,2,2]).cuda()
    # ts_sep = time.time()
    # for comp_i in range(0,2):
    #     for comp_j in range(0,2):
    #         testMatrixField_prolong_lin_compare[:,:,comp_i,comp_j] = prolongationOp2D_lin_nearest.forward( testMatrixField[:,:,comp_i,comp_j].contiguous() )
    # print('matrix field seperately in cuda: ', (time.time()-ts_sep))
    # #
    # ts_full = time.time()
    # testMatrixField_prolong_lin = prolongationOp2D_lin_nearest.forwardMatrixField(testMatrixField)
    # print('matrix field full in cuda: ', (time.time()-ts_full))
    # #compare
    # print("diff matrix field = ", torch.norm(testMatrixField_prolong_lin - testMatrixField_prolong_lin_compare).item() )

print("""
===========
   2D
===========
""")

NY2D = 27
NX2D = 39
LY2D = 8.
LX2D = 2.
meshInfo_old = opticalFlow.MeshInfo2D(NY2D,NX2D,LY2D,LX2D)

NY2D_new = 69
NX2D_new = 147
LY2D_new = 8.
LX2D_new = 3.
meshInfo_new = opticalFlow.MeshInfo2D(NY2D_new,NX2D_new,LY2D_new,LX2D_new)

for boundary in boundaryList:
    print("check prolongation for 2d with", boundary, ":")
    prolongationOp2D_lin_nearest = opticalFlow.Prolongation2D(meshInfo_old,meshInfo_new,interpolationList[0],boundary)
    prolongationOp2D_cubic_nearest = opticalFlow.Prolongation2D(meshInfo_old,meshInfo_new,interpolationList[1],boundary)

    testVec = torch.randn([NY2D,NX2D]).cuda()
    testVec_prolong_lin = prolongationOp2D_lin_nearest.forward(testVec)
    testVec_prolong_cubicHS = prolongationOp2D_cubic_nearest.forward(testVec)
    print("diff linear vs cubic = ", torch.norm(testVec_prolong_lin - testVec_prolong_cubicHS).item() )

    testMatrixField = torch.randn([NY2D,NX2D,2,2]).cuda()
    #
    testMatrixField_prolong_lin_compare = torch.zeros([NY2D_new,NX2D_new,2,2]).cuda()
    ts_sep = time.time()
    for comp_i in range(0,2):
        for comp_j in range(0,2):
            testMatrixField_prolong_lin_compare[:,:,comp_i,comp_j] = prolongationOp2D_lin_nearest.forward( testMatrixField[:,:,comp_i,comp_j].contiguous() )
    print('matrix field seperately in cuda: ', (time.time()-ts_sep))
    #
    ts_full = time.time()
    testMatrixField_prolong_lin = prolongationOp2D_lin_nearest.forwardMatrixField(testMatrixField)
    print('matrix field full in cuda: ', (time.time()-ts_full))
    #compare
    print("diff matrix field = ", torch.norm(testMatrixField_prolong_lin - testMatrixField_prolong_lin_compare).item() )