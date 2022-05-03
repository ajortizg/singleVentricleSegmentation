import torch
# from torch.autograd.function import once_differentiable

from opticalFlow_cuda_ext import opticalFlow


# interpolationList = [opticalFlow.InterpolationType.INTERPOLATE_NEAREST,opticalFlow.InterpolationType.INTERPOLATE_LINEAR,opticalFlow.InterpolationType.INTERPOLATE_CUBIC_HERMITESPLINE]
# boundaryList = [opticalFlow.BoundaryType.BOUNDARY_NEAREST,opticalFlow.BoundaryType.BOUNDARY_MIRROR,opticalFlow.BoundaryType.BOUNDARY_REFLECT]

# warpingOp1D = opticalFlow.Warping1D(meshInfo1D,interpolation,boundary)


class WarpingOpFunction(torch.autograd.Function):
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
        # return grad_image, None, None


def warp(image: torch.Tensor, flow: torch.Tensor, warpingOp) -> torch.Tensor:
    return WarpingOpFunction().apply(image,flow,warpingOp)

# class WarpingOp1DFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, image, flow, warpingOp):
#         ctx.save_for_backward(image, flow)
#         ctx.warpingOp = warpingOp
#         return warpingOp.forward(image, flow)

#     @staticmethod
#     def backward(ctx, grad_out):
#         image, flow = ctx.saved_tensors
#         grad_image, grad_flow = ctx.warpingOp.backward(image, flow, grad_out)
#         return grad_image, grad_flow, None
#         # return grad_image, None, None


# def warp1D(image: torch.Tensor, flow: torch.Tensor, warpingOp) -> torch.Tensor:
#     return WarpingOp1DFunction().apply(image,flow,warpingOp)




# class WarpingOp2DFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, image, flow, warpingOp):
#         ctx.save_for_backward(image, flow)
#         ctx.warpingOp = warpingOp
#         return warpingOp.forward(image, flow)

#     @staticmethod
#     def backward(ctx, grad_out):
#         image, flow = ctx.saved_tensors
#         grad_image, grad_flow = ctx.warpingOp.backward(image, flow, grad_out)
#         return grad_image, grad_flow, None
#         # return grad_image, None, None


# def warp2D(image: torch.Tensor, flow: torch.Tensor, warpingOp) -> torch.Tensor:
#     return WarpingOp2DFunction().apply(image,flow,warpingOp)



# class WarpingOp3DFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, image, flow, warpingOp):
#         ctx.save_for_backward(image, flow)
#         ctx.warpingOp = warpingOp
#         return warpingOp.forward(image, flow)

#     @staticmethod
#     def backward(ctx, grad_out):
#         image, flow = ctx.saved_tensors
#         grad_image, grad_flow = ctx.warpingOp.backward(image, flow, grad_out)
#         return grad_image, grad_flow, None
#         # return grad_image, None, None


# def warp3D(image: torch.Tensor, flow: torch.Tensor, warpingOp) -> torch.Tensor:
#     return WarpingOp3DFunction().apply(image,flow,warpingOp)