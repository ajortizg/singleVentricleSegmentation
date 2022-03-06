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
from scipy import ndimage

#==================================
pythonOps_lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../pythonOps'))
sys.path.append(pythonOps_lib_path)
import mesh
import warpingOps

sys.path.append("../utils/")
import plots

from opticalFlow_cuda_ext import opticalFlow


#==================================
DEVICE = 'cuda'


def generateTestVec2D( meshInfo2D_python ):
  size_testVec2D = torch.Size([meshInfo2D_python.NY,meshInfo2D_python.NX])  
  testVec2D = torch.zeros(size_testVec2D).cuda()

  center_x = 0.5 * meshInfo2D_python.LX
  center_y = 0.5 * meshInfo2D_python.LY

  for ix in range(0,meshInfo2D_python.NX):
    for iy in range(0,meshInfo2D_python.NY):
        x = ix * meshInfo2D_python.hX
        y = iy * meshInfo2D_python.hY
        testVec2D[iy][ix] = (x - center_x) ** 2 * (y - center_y) ** 2

  maxValue = torch.max(testVec2D).item()
  testVec2D *= -255. / maxValue
  testVec2D += 255.

  return testVec2D

def generateTestDisp2D( meshInfo2D_python ):
  size_testDisp2D = torch.Size([meshInfo2D_python.NY,meshInfo2D_python.NX,2])  
  testDisp2D = torch.zeros(size_testDisp2D).cuda()

  eps=0.3

  for ix in range(0,meshInfo2D_python.NX):
    for iy in range(0,meshInfo2D_python.NY):
        x = 2. * ix / (meshInfo2D_python.NX-1.) - 1
        y = 2. * iy / (meshInfo2D_python.NY-1.) - 1
        f = eps/4 * (math.cos(math.pi * x) + 1) * (math.cos(math.pi * y) + 1)
        g = eps * math.cos(0.5 * math.pi * x) * math.cos(0.5 * math.pi * y)
        h = 0.5 * ( f + g )
        testDisp2D[iy][ix][0] = (-y * h + 1) * 0.5 * meshInfo2D_python.LX
        testDisp2D[iy][ix][1] = (x * h) * 0.5 * meshInfo2D_python.LY 

  return testDisp2D


def testWarpingOps(args):

    print("""
    ==================================
        testWarpingOps
    ==================================
    """)

    #==================================
    # save directory
    saveDir = os.path.dirname("results") 
    if args.saveDir is not None:
        saveDir = os.path.dirname(args.saveDir) 
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    ####################################
    # test sizes for meshes
    ####################################
    NX1D = 2013
    LX1D = 2. * math.pi
    #LX1D = 2.
    meshInfo1D_python = mesh.MeshInfo1D(NX1D,LX1D)
    meshInfo1D_cuda = opticalFlow.MeshInfo1D(NX1D,LX1D)
    dimVec1D = torch.Size([NX1D])
    warpingOp1D_python = warpingOps.Warping1D(meshInfo1D_python)
    warpingOp1D_cuda = opticalFlow.Warping1D(meshInfo1D_cuda)

    NY2D = 129
    NX2D = 257
    LY2D = 8.
    LX2D = 2.
    meshInfo2D_python = mesh.MeshInfo2D(NY2D,NX2D,LY2D,LX2D)
    meshInfo2D_cuda = opticalFlow.MeshInfo2D(NY2D,NX2D,LY2D,LX2D)
    dimVec2D = torch.Size([NY2D,NX2D])
    # warpingOp2D_python = warpingOps.Warping2D(meshInfo2D_python)
    warpingOp2D_cuda = opticalFlow.Warping2D(meshInfo2D_cuda)

    NZ3D = 17
    NY3D = 129
    NX3D = 257
    LZ3D = 0.5
    LY3D = 8.
    LX3D = 2.
    meshInfo3D_python = mesh.MeshInfo3D(NZ3D,NY3D,NX3D,LZ3D,LY3D,LX3D)
    meshInfo3D_cuda = opticalFlow.MeshInfo3D(NZ3D,NY3D,NX3D,LZ3D,LY3D,LX3D)
    dimVec3D = torch.Size([NZ3D,NY3D,NX3D])


    
    ####################################
    # test in 1d
    ####################################

    # input(x) = sin(x), x in (0,2pi)
    # disp(x) = sin(x/2), x in (0,2pi)
    gridVec1D = torch.linspace(0, LX1D, steps=NX1D).cuda()
    testVec1D = torch.sin(gridVec1D)
    testDisp1D_tmp = torch.linspace(0,0.5*LX1D,steps=NX1D).cuda()
    testDisp1D_tmp = torch.sin(testDisp1D_tmp)
    size_testDisp1D = torch.Size([NX1D,1])
    testDisp1D = torch.reshape(testDisp1D_tmp, size_testDisp1D)
    #output output(x) = input(x+disp(x)) = sin(x+sin(x/2))
    outputVec1D = torch.linspace(0, LX1D, steps=NX1D).cuda()
    outputVec1D = torch.sin(gridVec1D + testDisp1D_tmp)
    plots.saveCurve1D(outputVec1D, LX1D, saveDir, "warping1D_exact")
 
    ###############  linear ############################
    ts = time.time()
    testVec1D_warped_linear_cuda = warpingOp1D_cuda.forward(testVec1D,testDisp1D,opticalFlow.InterpolationType.INTERPOLATE_LINEAR)
    print('warping 1d linear - elapsed time cuda: ', (time.time()-ts))
    plots.saveCurve1D(testVec1D_warped_linear_cuda, LX1D, saveDir, "warping1D_linear_cuda")
    print("diff linear cuda vs exact in linf = ", torch.max(torch.abs(testVec1D_warped_linear_cuda - outputVec1D)).item())
    print("diff linear cuda vs exact in l2   = ", torch.norm(testVec1D_warped_linear_cuda - outputVec1D).item())

    ts = time.time()
    testVec1D_warped_linear_python = warpingOp1D_python.forward(testVec1D,testDisp1D,"linear")
    print('warping 1d linear - elapsed time python: ', (time.time()-ts))
    plots.saveCurve1D(testVec1D_warped_linear_python, LX1D, saveDir, "warping1D_linear_python")
    print("diff linear python vs exact in linf = ", torch.max(torch.abs(testVec1D_warped_linear_python - outputVec1D)).item())
    print("diff linear python vs exact in l2   = ", torch.norm(testVec1D_warped_linear_python - outputVec1D).item())

    diff1DLinearPythonVsCuda = torch.max(torch.abs(testVec1D_warped_linear_cuda - testVec1D_warped_linear_python)).item()
    print("diff linear python vs cuda = ", diff1DLinearPythonVsCuda)

    ###############  cubic ############################
    ts = time.time()
    testVec1D_warped_cubicHS_cuda = warpingOp1D_cuda.forward(testVec1D,testDisp1D,opticalFlow.InterpolationType.INTERPOLATE_CUBIC_HERMITESPLINE)
    print('warping 1d cubic HS - elapsed time cuda: ', (time.time()-ts))
    plots.saveCurve1D(testVec1D_warped_cubicHS_cuda, LX1D, saveDir, "warping1D_cubicHS_cuda")
    print("diff cubic HS cuda vs exact in linf = ", torch.max(torch.abs(testVec1D_warped_cubicHS_cuda - outputVec1D)).item())
    print("diff cubic HS cuda vs exact in l2   = ", torch.norm(testVec1D_warped_cubicHS_cuda - outputVec1D).item())

    ts = time.time()
    testVec1D_warped_cubicHS_python = warpingOp1D_python.forward(testVec1D,testDisp1D,"cubic")
    print('warping 1d cubic - elapsed time python: ', (time.time()-ts))
    plots.saveCurve1D(testVec1D_warped_cubicHS_python, LX1D, saveDir, "warping1D_cubicHS_python")
    print("diff cubic HS python vs exact in linf = ", torch.max(torch.abs(testVec1D_warped_cubicHS_python - outputVec1D)).item())
    print("diff cubic HS python vs exact in l2   = ", torch.norm(testVec1D_warped_cubicHS_python - outputVec1D).item())

    diff1DCubicHSPythonVsCuda = torch.max(torch.abs(testVec1D_warped_cubicHS_cuda - testVec1D_warped_cubicHS_python)).item()
    print("diff cubic HS python vs cuda = ", diff1DCubicHSPythonVsCuda)

    # diff1DCubicPythonVsCuda = torch.max(torch.abs(testVec1D_warped_cubic_cuda - testVec1D_warped_cubic_python)).item()
    # print("diff cubic python vs cuda = ", diff1DCubicPythonVsCuda)

    # ts = time.time()
    # testVec1D_warped_cubicSpline = warpingOp_cuda.forward(testVec1D,testDisp1D,opticalFlow.InterpolationType.INTERPOLATE_CUBIC_SPLINE)
    # print('warping 1d cubic spline - elapsed time cuda: ', (time.time()-ts))
    # plots.saveCurve1D(testVec1D_warped_cubicSpline, LX1D, saveDir, "warping1D_cubicSpline")


    # diff1DLinearVsCubic = torch.max(torch.abs(testVec1D_warped_linear - testVec1D_warped_cubic)).item()
    # print("diff linear vs cubic = ", diff1DLinearVsCubic)
    # diff1DLinearVsCubicSpline = torch.max(torch.abs(testVec1D_warped_linear - testVec1D_warped_cubicSpline)).item()
    # print("diff linear vs cubic spline = ", diff1DLinearVsCubicSpline)
    # diff1DCubicVsCubicSpline = torch.max(torch.abs(testVec1D_warped_cubic - testVec1D_warped_cubicSpline)).item()
    # print("diff cubic vs cubic spline = ", diff1DCubicVsCubicSpline)


    ####################################
    # test in 2d
    ####################################

    testVec2D = generateTestVec2D(meshInfo2D_python)
    plots.saveImage(testVec2D, saveDir, "warping2D_input.png")
    testDisp2D = generateTestDisp2D(meshInfo2D_python)

    # gridVec1D = torch.linspace(0, LX1D, steps=NX1D).cuda()
    # testVec1D = torch.sin(gridVec1D)
    # testDisp1D_tmp = torch.linspace(0,0.5*LX1D,steps=NX1D).cuda()
    # testDisp1D_tmp = torch.sin(testDisp1D_tmp)
    # size_testDisp1D = torch.Size([NX1D,1])
    # testDisp1D = torch.reshape(testDisp1D_tmp, size_testDisp1D)
    # #output output(x) = input(x+disp(x)) = sin(x+sin(x/2))
    # outputVec1D = torch.linspace(0, LX1D, steps=NX1D).cuda()
    # outputVec1D = torch.sin(gridVec1D + testDisp1D_tmp)
    # plots.saveCurve1D(outputVec1D, LX1D, saveDir, "warping1D_exact")
 
    ###############  linear ############################
    ts = time.time()
    testVec2D_warped_linear_cuda = warpingOp2D_cuda.forward(testVec2D,testDisp2D,opticalFlow.InterpolationType.INTERPOLATE_LINEAR)
    print('warping 2d linear - elapsed time cuda: ', (time.time()-ts))
    plots.saveImage(testVec2D_warped_linear_cuda, saveDir, "warping2D_linear_cuda.png")
    # print("diff linear cuda vs exact in linf = ", torch.max(torch.abs(testVec1D_warped_linear_cuda - outputVec1D)).item())
    # print("diff linear cuda vs exact in l2   = ", torch.norm(testVec1D_warped_linear_cuda - outputVec1D).item())

    # ts = time.time()
    # testVec1D_warped_linear_python = warpingOp1D_python.forward(testVec1D,testDisp1D,"linear")
    # print('warping 1d linear - elapsed time python: ', (time.time()-ts))
    # plots.saveCurve1D(testVec1D_warped_linear_python, LX1D, saveDir, "warping1D_linear_python")
    # print("diff linear python vs exact in linf = ", torch.max(torch.abs(testVec1D_warped_linear_python - outputVec1D)).item())
    # print("diff linear python vs exact in l2   = ", torch.norm(testVec1D_warped_linear_python - outputVec1D).item())

    # diff1DLinearPythonVsCuda = torch.max(torch.abs(testVec1D_warped_linear_cuda - testVec1D_warped_linear_python)).item()
    # print("diff linear python vs cuda = ", diff1DLinearPythonVsCuda)


    ###############  cubic ############################
    ts = time.time()
    testVec2D_warped_cubicHS_cuda = warpingOp2D_cuda.forward(testVec2D,testDisp2D,opticalFlow.InterpolationType.INTERPOLATE_CUBIC_HERMITESPLINE)
    print('warping 2d cubic HS - elapsed time cuda: ', (time.time()-ts))
    plots.saveImage(testVec2D_warped_cubicHS_cuda, saveDir, "warping2D_cubicHS_cuda.png")
    # print("diff cubic cuda vs exact in linf = ", torch.max(torch.abs(testVec1D_warped_cubic_cuda - outputVec1D)).item())
    # print("diff cubic cuda vs exact in l2   = ", torch.norm(testVec1D_warped_cubic_cuda - outputVec1D).item())

    # ts = time.time()
    # testVec1D_warped_cubic_python = warpingOp1D_python.forward(testVec1D,testDisp1D,"cubic")
    # print('warping 1d cubic - elapsed time python: ', (time.time()-ts))
    # plots.saveCurve1D(testVec1D_warped_cubic_python, LX1D, saveDir, "warping1D_cubic_python")
    # print("diff cubic python vs exact in linf = ", torch.max(torch.abs(testVec1D_warped_cubic_python - outputVec1D)).item())
    # print("diff cubic python vs exact in l2   = ", torch.norm(testVec1D_warped_cubic_python - outputVec1D).item())

    # diff1DCubicPythonVsCuda = torch.max(torch.abs(testVec1D_warped_cubic_cuda - testVec1D_warped_cubic_python)).item()
    # print("diff cubic python vs cuda = ", diff1DCubicPythonVsCuda)




    # #==================================
    # # check in 2D

    # #==================================
    # # Load 4D nifty [x,y,z,t]
    # print("\n=================\nLoad Nifty File\n=================")
    # if not (args.fileName):
    #    parser.error('add -fileName')
    # fileName = args.fileName

    # nii_img = nib.load(fileName)
    # nii_data_xyzt = nii_img.get_fdata()

    # NX = nii_data_xyzt.shape[0]
    # NY = nii_data_xyzt.shape[1]
    # NZ = nii_data_xyzt.shape[2]
    # NT = nii_data_xyzt.shape[3]
    # print( f"dimension of input: (X,Y,Z,T) = {NX,NY,NZ,NT}")


    # ## swap from nibabel (X,Y,Z) to cuda-compatible (Z,Y,X):
    # print("swap axes (X,Y,Z,T) to (Z,Y,X,T)")
    # nii_data = np.swapaxes(nii_data_xyzt, 0, 2)
    # print( f"dimension after swap: (Z,Y,X,T) = {nii_data.shape}")

    # #==================================
    # #scaling of data 
    # totalMinValue = np.amin(nii_data)
    # totalMaxValue = np.amax(nii_data)
    # print(f"input (min,max) = {totalMinValue,totalMaxValue}")
    # scaleMaxValue = 255.
    # print("scaling of data to max value", scaleMaxValue)
    # nii_data *= scaleMaxValue / totalMaxValue

    # #
    # tA = 8
    # z=8
    # imageA = torch.from_numpy(nii_data[z,:,:,tA]).float().to(DEVICE)


    # ## example warping 

    # #test random
    # phi_0 = np.random.uniform(-0.4, 0.4, (NY, NX, 2))
    # mask = np.zeros_like(phi_0)
    # mask[1:-1,1:-1, :] = 1
    # phi_0 *= mask
    # #phi_0 = _np.zeros_like(phi_0)

    # #test 2
    # # phi_0 = np.zeros((NY, NX, 2))

    # phi = torch.from_numpy(phi_0).float().to(DEVICE)
    # imageA_warped_linear = opticalFlow.warp2d(imageA,phi,opticalFlow.InterpolationType.INTERPOLATE_LINEAR)
    # imageA_warped_cubic = opticalFlow.warp2d(imageA,phi,opticalFlow.InterpolationType.INTERPOLATE_CUBIC)

    # imageA_scipy = imageA.cpu().detach().numpy()
    # phi_scipy = phi.cpu().detach().numpy()
    # imageA_warped_scipy = np.zeros(imageA_scipy.shape)
    # warpingOps.warp2d(imageA_scipy,phi_scipy,imageA_warped_scipy)


    # #TODO swap result (Z,Y,X) back to (X,Y,Z):
    # #result_backSwap = np.swapaxes(result, 0, 2)

    # #save 
    # imageA_np = imageA.cpu().detach().numpy()
    # plots.saveImage(imageA_np,saveDir,"imgA2D.png")
    # imageA_warped_linear_np = imageA_warped_linear.cpu().detach().numpy()
    # plots.saveImage(imageA_warped_linear_np,saveDir,"imgA2D_warped_linear.png")
    # imageA_warped_cubic_np = imageA_warped_cubic.cpu().detach().numpy()
    # plots.saveImage(imageA_warped_cubic_np,saveDir,"imgA2D_warped_cubic.png")
    # plots.saveImage(imageA_warped_scipy,saveDir,"imgA2D_warped_scipy.png")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--fileName', help="file name of 4d nifty")
    parser.add_argument('--saveDir', help="directory for saving")

    args = parser.parse_args()

    testWarpingOps(args)