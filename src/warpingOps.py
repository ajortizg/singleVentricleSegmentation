import torch
# from torch.autograd.function import once_differentiable

from opticalFlow_cuda_ext import opticalFlow


# interpolationList = [opticalFlow.InterpolationType.INTERPOLATE_NEAREST,opticalFlow.InterpolationType.INTERPOLATE_LINEAR,opticalFlow.InterpolationType.INTERPOLATE_CUBIC_HERMITESPLINE]
# boundaryList = [opticalFlow.BoundaryType.BOUNDARY_NEAREST,opticalFlow.BoundaryType.BOUNDARY_MIRROR,opticalFlow.BoundaryType.BOUNDARY_REFLECT]

# warpingOp1D = opticalFlow.Warping1D(meshInfo1D,interpolation,boundary)

class WarpingOp1D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, image, flow, warpingOp):
        ctx.save_for_backward(image, flow)
        ctx.warpingOp = warpingOp
        return warpingOp.forward(image, flow)

    @staticmethod
    def backward(ctx, grad_out):
        image, flow = ctx.saved_tensors
        grad_image, grad_flow = ctx.warpingOp.backward(image, flow, grad_out)
        return grad_image, grad_flow, None


def warp1D(image: torch.Tensor, flow: torch.Tensor, warpingOp) -> torch.Tensor:
    return WarpingOp1D().apply(image,flow,warpingOp)



#test interface
NX1D = 17
LX1D = 2.
meshInfo1D = opticalFlow.MeshInfo1D(NX1D,LX1D)

warpingOp1D = opticalFlow.Warping1D(meshInfo1D,opticalFlow.InterpolationType.INTERPOLATE_LINEAR,opticalFlow.BoundaryType.BOUNDARY_MIRROR)

image = torch.randn(NX1D).cuda()
flow = torch.ones(NX1D,1).cuda()

image.requires_grad_(True)
flow.requires_grad_(True)

out = warp1D(image,flow,warpingOp1D)
print(out)

loss = torch.sum(out**2)
loss.backward()

print(image.grad)
print(flow.grad)