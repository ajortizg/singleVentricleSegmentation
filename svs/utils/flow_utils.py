import numpy as np
from svs.utils.enums import FlowDirection
from scipy import ndimage
import torch.nn.functional as F
import torch
from typing import Literal, Union, Tuple

from opticalFlow_cuda_ext import opticalFlow


def get_interpolation_type(inter_type_str: Literal['NEAREST', 'LINEAR', 'CUBIC_HERMITESPLINE'], bound_type_str: Literal['NEAREST', 'MIRROR', 'REFLECT']):
    """
    Determines the interpolation and boundary type for optical flow operations.

    Args:
        inter_type_str (str): A string specifying the interpolation type. Valid options are "NEAREST", "LINEAR", and "CUBIC_HERMITESPLINE".
        bound_type_str (str): A string specifying the boundary type. Valid options are "NEAREST", "MIRROR", and "REFLECT".

    Returns:
        A tuple containing the interpolation type and boundary type from the opticalFlow module.

    Raises:
        Exception: If an invalid interpolation type or boundary type is provided.
    """
    interpolation = None
    if inter_type_str == "NEAREST":
        interpolation = opticalFlow.InterpolationType.INTERPOLATE_NEAREST
    elif inter_type_str == "LINEAR":
        interpolation = opticalFlow.InterpolationType.INTERPOLATE_LINEAR
    elif inter_type_str == "CUBIC_HERMITESPLINE":
        interpolation = opticalFlow.InterpolationType.INTERPOLATE_CUBIC_HERMITESPLINE
    else:
        raise Exception("Wrong interpolation type")

    boundary = None
    if bound_type_str == "NEAREST":
        boundary = opticalFlow.BoundaryType.BOUNDARY_NEAREST
    elif bound_type_str == "MIRROR":
        boundary = opticalFlow.BoundaryType.BOUNDARY_MIRROR
    elif bound_type_str == "REFLECT":
        boundary = opticalFlow.BoundaryType.BOUNDARY_REFLECT
    else:
        raise Exception("wrong boundary type")

    return interpolation, boundary


def get_mesh_length(length_type: Literal['numDofs', 'fixed'], nz: int, ny: int, nx: int, lz: int, ly: int, lx: int):
    """
    Computes the mesh length based on the length type.

    Args:
        length_type (str): A string specifying the length type. Valid options are "numDofs" and "fixed".
        nz, ny, nx (int): The number of elements in the z,y,x-dimension.
        lz, ly, lx (int): The length in the z,y,x-dimension for the "fixed" length type.

    Returns:
        A tuple containing the lengths (LZ, LY, LX) based on the length type.

    Raises:
        Exception: If an invalid length type is provided.
    """
    if length_type == "numDofs":
        lz = nz - 1
        ly = ny - 1
        lx = nx - 1
        return lz, ly, lx
    elif length_type == "fixed":
        return lz, ly, lx
    else:
        raise Exception("Wrong length type")


def _generate_gaussian_kernel3d(sigma: float) -> torch.Tensor:
    """
    Generates a 3D Gaussian kernel given the standard deviation.

    Args:
        sigma (float): The standard deviation of the Gaussian kernel.

    Returns:
        torch.Tensor: A 3D Gaussian kernel as a 1D tensor.
    """
    kernelSize = int(sigma * 5)
    if kernelSize % 2 == 0:
        kernelSize += 1
    ts = torch.linspace(-kernelSize // 2, kernelSize // 2 + 1, kernelSize)
    gauss = torch.exp((-(ts / sigma)**2 / 2))
    kernel = gauss / gauss.sum()

    return kernel


def apply_gaussian_blur3d(vol: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Applies a 3D Gaussian blur to a 3D volume.

    Args:
        vol (torch.Tensor): The input 3D volume as a tensor.
        sigma (float): The standard deviation of the Gaussian kernel.

    Returns:
        torch.Tensor: The blurred 3D volume.
    """
    # 3D convolution
    vol_in = vol.reshape(1, 1, *vol.shape)
    k = _generate_gaussian_kernel3d(sigma)
    k3d = torch.einsum('i,j,k->ijk', k, k, k).cuda()
    k3d = k3d / k3d.sum()
    vol_3d = F.conv3d(vol_in, k3d.reshape(1, 1, *k3d.shape), stride=1, padding=len(k) // 2)
    vol_out = vol_3d.reshape(*vol.shape)
    return vol_out


def apply_median_filter(u: torch.Tensor, ks: int, device: torch.device) -> torch.Tensor:
    """
    Applies a median filter to a 3D tensor using SciPy's ndimage median_filter function.

    Args:
        u (torch.Tensor): Input 3D tensor to be filtered.
        ks (int): Kernel size for the median filter.
        device (torch.device): The device on which the output tensor should reside.

    Returns:
        torch.Tensor: Filtered tensor after applying median filter.

    Note:
        This function converts the input tensor to a NumPy array to apply the median filter using SciPy's
        ndimage.median_filter function and then converts the filtered result back to a PyTorch tensor.
    """
    uf = ndimage.median_filter(u.cpu().detach().numpy(), size=(ks, ks, ks, 1))
    return torch.from_numpy(uf).float().to(device)


def compute_timepoints(
    dia_ts: int,
    sys_ts: int,
    seg_dia: Union[np.ndarray, torch.Tensor],
    seg_sys: Union[np.ndarray, torch.Tensor],
    mode: Union[str, FlowDirection]
) -> Tuple[Tuple[int, int], Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]], list]:
    """
    Determines the end timepoints, corresponding segmentations, and times for optical flow computation.

    Args:
        dia_ts (int): Timestamp for diastolic phase.
        sys_ts (int): Timestamp for systolic phase.
        seg_dia (np.ndarray or torch.Tensor): Segmentation for diastolic phase.
        seg_sys (np.ndarray or torch.Tensor): Segmentation for systolic phase.
        mode (str or FlowDirection): Direction of optical flow computation. Can be 'forward' or 'backward'.

    Returns:
        tuple: 
            timepoints (tuple): A tuple containing the initial and final timepoints.
            maskpoints (tuple): A tuple containing the corresponding segmentations for the timepoints.
            times (list): A list of indices from initial to final timepoint, optionally reversed if mode is 'backward'.
    """
    if sys_ts < dia_ts:
        timepoints = (sys_ts, dia_ts)       # init, final timepoints
        segmentations = (seg_sys, seg_dia)
    else:
        timepoints = (dia_ts, sys_ts)
        segmentations = (seg_dia, seg_sys)

    times = np.arange(timepoints[0], timepoints[1] + 1, 1)
    if mode == FlowDirection.BACKWARD:
        times = np.flip(times)

    return (timepoints, segmentations, times.tolist())
