import os
import torch
import torch.nn.functional as F


def create_grid(NZ, NY, NX):
    zz, yy, xx = torch.meshgrid(torch.arange(NZ), torch.arange(NY),
                                torch.arange(NX), indexing="ij")
    return torch.stack((xx, yy, zz), dim=3)


def scale_grid(grid):
    # scale grid to [-1,1]
    _, NZ, NY, NX, _ = grid.shape
    grid[:, :, :, :, 0] = 2.0 * grid[:, :, :, :, 0] / max(NX - 1, 1) - 1.0
    grid[:, :, :, :, 1] = 2.0 * grid[:, :, :, :, 1] / max(NY - 1, 1) - 1.0
    grid[:, :, :, :, 2] = 2.0 * grid[:, :, :, :, 2] / max(NZ - 1, 1) - 1.0
    return grid


def warp(vol, grid, mode="bilinear"):
    """
    mode = {bilinear, nearest}
    """
    grid = scale_grid(grid)
    return F.grid_sample(vol, grid, align_corners=True,
                         mode=mode, padding_mode="zeros")


def normalize(x):
    # Normalize between 0 and 1
    min = torch.amin(x)
    max = torch.amax(x)
    return (x - min) / (max - min)


# def getTorchDevice(gpuNum: int):
#     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#     #os.environ['CUDA_VISIBLE_DEVICES'] = str(gpuNum)

#     if torch.cuda.is_available():
#         DEVICE = torch.device('cuda:' + str(gpuNum))
#         #DEVICE = torch.device('cuda:')
#     else:
#         DEVICE = 'cpu'
    
#     return DEVICE

def getTorchDevice(gpuNum: str):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = gpuNum

    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
    else:
        DEVICE = 'cpu'
    
    return DEVICE