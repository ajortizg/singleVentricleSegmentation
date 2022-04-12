import torch
import sys
import os

utils_lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils'))
sys.path.append(utils_lib_path)
import plots

from opticalFlow_cuda_ext import opticalFlow


interpolationList = [opticalFlow.InterpolationType.INTERPOLATE_NEAREST,opticalFlow.InterpolationType.INTERPOLATE_LINEAR,opticalFlow.InterpolationType.INTERPOLATE_CUBIC_HERMITESPLINE]
boundaryList = [opticalFlow.BoundaryType.BOUNDARY_NEAREST,opticalFlow.BoundaryType.BOUNDARY_MIRROR,opticalFlow.BoundaryType.BOUNDARY_REFLECT]
derivateDirList = [0,1]


def checkGradient1D(warpingOp1D,numTests=10,dtype=torch.float64,derivativeDir=0):

    alpha = 0.5

    if derivativeDir == 0:
        forward = lambda image, flow, alpha, d: warpingOp1D.forward(image+alpha*d, flow)
        backward = lambda image, flow, alpha, d: torch.sum(d*warpingOp1D.backward(image+alpha*d, flow, forward(image, flow, alpha, d))[0])
    else:
        forward = lambda image, flow, alpha, d: warpingOp1D.forward(image, flow+alpha*d)
        backward = lambda image, flow, alpha, d: torch.sum(d*warpingOp1D.backward(image, flow+alpha*d, forward(image, flow, alpha, d))[1])

    loss = lambda x, flow, alpha, d: torch.sum(forward(x, flow, alpha, d)**2)/2

    for i in range(numTests):
        image = torch.rand((NX1D), dtype=dtype).cuda()
        flow = torch.randn((NX1D,1), dtype=dtype).cuda()

        d = torch.randn_like(image) if derivativeDir == 0 else torch.randn_like(flow)

        grad = backward(image, flow, alpha, d).item()
        eps = 1e-5
        num_grad = (loss(image,flow,alpha+eps,d).item() - loss(image,flow,alpha-eps,d).item()) / (2*eps)

        # print(f'{i:03d}: \t {grad=:.5e} \t {num_grad=:.5e} \t diff={grad-num_grad:.5e}')  
        print(f'{i:03d}: \t {grad=:.5e} \t {num_grad=:.5e} \t diff=', end="") 
        plots.printColoredError(abs(grad-num_grad),tol=1.e-8)




####################################
# test sizes for meshes
####################################
NX1D = 17
LX1D = 2.
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
            checkGradient1D(warpingOp1D, derivativeDir=derivativeDir)






def checkGradient2D(warpingOp2D,numTests=10,dtype=torch.float64,derivativeDir=0):

    alpha = 0.5

    if derivativeDir == 0:
        forward = lambda image, flow, alpha, d: warpingOp2D.forward(image+alpha*d, flow)
        backward = lambda image, flow, alpha, d: torch.sum(d*warpingOp2D.backward(image+alpha*d, flow, forward(image, flow, alpha, d))[0])
    else:
        forward = lambda image, flow, alpha, d: warpingOp2D.forward(image, flow+alpha*d)
        backward = lambda image, flow, alpha, d: torch.sum(d*warpingOp2D.backward(image, flow+alpha*d, forward(image, flow, alpha, d))[1])

    loss = lambda x, flow, alpha, d: torch.sum(forward(x, flow, alpha, d)**2)/2

    for i in range(numTests):
        image = torch.rand((NY2D,NX2D), dtype=dtype).cuda()
        flow = torch.randn((NY2D,NX2D,2), dtype=dtype).cuda()

        d = torch.randn_like(image) if derivativeDir == 0 else torch.randn_like(flow)

        grad = backward(image, flow, alpha, d).item()
        eps = 1e-5
        num_grad = (loss(image,flow,alpha+eps,d).item() - loss(image,flow,alpha-eps,d).item()) / (2*eps)

        # print(f'{i:03d}: \t {grad=:.5e} \t {num_grad=:.5e} \t diff={grad-num_grad:.5e}')  
        print(f'{i:03d}: \t {grad=:.5e} \t {num_grad=:.5e} \t diff=', end="") 
        plots.printColoredError(abs(grad-num_grad),tol=1.e-8)


####################################
# test sizes for meshes
####################################
NX2D = 9
NY2D = 17
LX2D = 2.
LY2D = 4.
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
            checkGradient2D(warpingOp2D, derivativeDir=derivativeDir)