import torch
import sys
import os
from torch.autograd import gradcheck

utils_lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils'))
sys.path.append(utils_lib_path)
import plots
from termcolor import colored

from opticalFlow_cuda_ext import opticalFlow


src_lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
sys.path.append(src_lib_path)
import warpingOps
from warpingOps import WarpingOpFunction

interpolationList = [opticalFlow.InterpolationType.INTERPOLATE_CUBIC_HERMITESPLINE]
# interpolationList = [opticalFlow.InterpolationType.INTERPOLATE_LINEAR,opticalFlow.InterpolationType.INTERPOLATE_CUBIC_HERMITESPLINE,opticalFlow.InterpolationType.INTERPOLATE_NEAREST]
boundaryList = [opticalFlow.BoundaryType.BOUNDARY_NEAREST,opticalFlow.BoundaryType.BOUNDARY_MIRROR,opticalFlow.BoundaryType.BOUNDARY_REFLECT]
derivateDirList = [0,1]
floatType = torch.float64
# floatType = torch.float

# check gradient for E(img,flow) = 1/2 sum_x Warp(img,flow)(x)^2
def checkGradientInRandomDir(dim,meshInfo,warpingOp,numTests=10,dtype=torch.float64,derivativeDir=0):

    if dim==1:
        imageShape = [meshInfo.getNX()]
        flowShape = [meshInfo.getNX(),1]
    elif dim==2:
        imageShape = [meshInfo.getNY(),meshInfo.getNX()]
        flowShape = [meshInfo.getNY(),meshInfo.getNX(),2]
    elif dim==3:
        imageShape = [meshInfo.getNZ(),meshInfo.getNY(),meshInfo.getNX()]
        flowShape = [meshInfo.getNZ(),meshInfo.getNY(),meshInfo.getNX(),3]

    alpha = 0.5

    if derivativeDir == 0:
        forward = lambda image, flow, alpha, d: warpingOp.forward(image+alpha*d, flow)
        backward = lambda image, flow, alpha, d: torch.sum(d*warpingOp.backward(image+alpha*d, flow, forward(image, flow, alpha, d))[0])
    else:
        forward = lambda image, flow, alpha, d: warpingOp.forward(image, flow+alpha*d)
        backward = lambda image, flow, alpha, d: torch.sum(d*warpingOp.backward(image, flow+alpha*d, forward(image, flow, alpha, d))[1])

    loss = lambda x, flow, alpha, d: torch.sum(forward(x, flow, alpha, d)**2)/2

    for i in range(numTests):
        image = torch.rand(imageShape, dtype=dtype).cuda()
        flow = torch.randn(flowShape, dtype=dtype).cuda()

        d = torch.randn_like(image) if derivativeDir == 0 else torch.randn_like(flow)

        grad = backward(image, flow, alpha, d).item()
        eps = 1e-5
        approx_grad = (loss(image,flow,alpha+eps,d).item() - loss(image,flow,alpha-eps,d).item()) / (2*eps)

        print(f'{i:03d}: \t {grad=:.5e} \t {approx_grad=:.5e} \t diff=', end="") 
        plots.printColoredError(abs(grad-approx_grad),tol=1.e-8)
        # if abs(grad) > 1.e-8: 
        #     relError = abs( 1. - approx_grad/grad )
        #     print("rel error=", relError)


# check gradient with autograd
def checkGradientWithAutograd(dim,meshInfo,warpingOp,dtype=torch.float64,derivativeDir=0):

    if dim==1:
        imageShape = [meshInfo.getNX()]
        flowShape = [meshInfo.getNX(),1]
    elif dim==2:
        imageShape = [meshInfo.getNY(),meshInfo.getNX()]
        flowShape = [meshInfo.getNY(),meshInfo.getNX(),2]
    elif dim==3:
        imageShape = [meshInfo.getNZ(),meshInfo.getNY(),meshInfo.getNX()]
        flowShape = [meshInfo.getNZ(),meshInfo.getNY(),meshInfo.getNX(),3]

    # kwargs = {'dtype': torch.float64,'device': "cuda", 'requires_grad': True}
    # image = torch.rand((NX1D), **kwargs)
    # flow = torch.randn((NX1D,1), **kwargs)
    if derivativeDir == 0:
        image = torch.rand(imageShape, dtype=dtype, device="cuda", requires_grad=True)
        flow = torch.randn(flowShape, dtype=dtype, device="cuda", requires_grad=False)
    if derivativeDir == 1:
        image = torch.rand(imageShape, dtype=dtype, device="cuda", requires_grad=False)
        flow = torch.randn(flowShape, dtype=dtype, device="cuda", requires_grad=True)
    variables = [image,flow, warpingOp]
    # if gradcheck(WarpingOp1DFunction.apply, variables,  eps=1e-05):
    #     print("gradcheck in 1d ok")
    # else:
    #     print("gradcheck in 1d wrong")
    try:
        if gradcheck(WarpingOpFunction.apply, variables, eps=1e-05, raise_exception=False):
            print(colored("gradcheck true", 'green'))
        else:
            print(colored("gradcheck wrong", 'red'))
    except :
        print(colored("gradcheck throws exception", 'red'))


####################################
# test in 1D
####################################
NX1D = 9
LX1D = 2.
# LX1D = NX1D - 1
meshInfo1D = opticalFlow.MeshInfo1D(NX1D,LX1D)

print("\n")
print("====================================")
print("check in 1d:")
print("====================================")
print("\n")
for interpolation in interpolationList:
    for boundary in boundaryList:
        for derivativeDir in derivateDirList:
            
            print("check gradient for \t interpolation = ", interpolation, "\t boundary = ", boundary, "\t derivDir = ", derivativeDir, ": ")
            warpingOp1D = opticalFlow.Warping1D(meshInfo1D,interpolation,boundary)
            checkGradientInRandomDir(1,meshInfo1D, warpingOp1D, derivativeDir=derivativeDir, dtype=floatType)
            
            print("check with autograd:")
            checkGradientWithAutograd(1,meshInfo1D, warpingOp1D, derivativeDir=derivativeDir)


####################################
# test in 2D
####################################
NX2D = 9
NY2D = 17
LX2D = 2.
LY2D = 8.
# LX2D = NX2D - 1
# LY2D = NY2D - 1
meshInfo2D = opticalFlow.MeshInfo2D(NY2D,NX2D,LY2D,LX2D)

print("\n")
print("====================================")
print("check in 2d:")
print("====================================")
print("\n")
for interpolation in interpolationList:
    for boundary in boundaryList:
        for derivativeDir in derivateDirList:
            print("check gradient for \t interpolation = ", interpolation, "\t boundary = ", boundary, "\t derivDir = ", derivativeDir, ": ")
            warpingOp2D = opticalFlow.Warping2D(meshInfo2D,interpolation,boundary)
            checkGradientInRandomDir(2, meshInfo2D, warpingOp2D, derivativeDir=derivativeDir, dtype=floatType)

            print("check with autograd:")
            checkGradientWithAutograd(2,meshInfo2D, warpingOp2D, derivativeDir=derivativeDir)



####################################
# test  in 3D
####################################
NX3D = 9
NY3D = 9
NZ3D = 7
LX3D = 2.
LY3D = 4.
LZ3D = 6.
# LX3D = NX3D - 1
# LY3D = NY3D - 1
# LZ3D = NZ3D - 1
meshInfo3D = opticalFlow.MeshInfo3D(NZ3D,NY3D,NX3D,LZ3D,LY3D,LX3D)

print("\n")
print("====================================")
print("check in 3d:")
print("====================================")
print("\n")
for interpolation in interpolationList:
    for boundary in boundaryList:
        for derivativeDir in derivateDirList:
            print("check gradient for \t interpolation = ", interpolation, "\t boundary = ", boundary, "\t derivDir = ", derivativeDir, ": ")
            warpingOp3D = opticalFlow.Warping3D(meshInfo3D,interpolation,boundary)
            checkGradientInRandomDir(3, meshInfo3D, warpingOp3D, derivativeDir=derivativeDir, dtype=floatType)

            print("check with autograd:")
            checkGradientWithAutograd(3,meshInfo3D, warpingOp3D, derivativeDir=derivativeDir)



####################################
#test interface
####################################

# NX1D = 17
# LX1D = 2.
# meshInfo1D = opticalFlow.MeshInfo1D(NX1D,LX1D)

# warpingOp1D = opticalFlow.Warping1D(meshInfo1D,opticalFlow.InterpolationType.INTERPOLATE_LINEAR,opticalFlow.BoundaryType.BOUNDARY_MIRROR)

# image = torch.randn(NX1D).cuda()
# flow = torch.ones(NX1D,1).cuda()

# image.requires_grad_(True)
# flow.requires_grad_(True)

# out = warp1D(image,flow,warpingOp1D)
# print(out)

# loss = torch.sum(out**2)
# loss.backward()

# print(image.grad)
# print(flow.grad)