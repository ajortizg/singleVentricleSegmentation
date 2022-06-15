import math
import numpy as np
import torch
from scipy.ndimage import affine_transform


def rotx(deg):
    rad = np.deg2rad(deg)
    return np.array([
        [1, 0, 0],
        [0, np.cos(rad), -np.sin(rad)],
        [0, np.sin(rad), np.cos(rad)],
    ])


def roty(deg):
    rad = np.deg2rad(deg)
    return np.array([
        [np.cos(rad), 0, np.sin(rad)],
        [0, 1, 0],
        [-np.sin(rad), 0, np.cos(rad)]
    ])


def rotz(deg):
    rad = np.deg2rad(deg)
    return np.array([
        [np.cos(rad), -np.sin(rad), 0],
        [np.sin(rad), np.cos(rad), 0],
        [0, 0, 1],
    ])


def scale(sx, sy, sz):
    return np.array([[sx, 0, 0], [0, sy, 0], [0, 0, sz]])


def rot2d(deg):
    rad = np.deg2rad(deg)
    return np.array([[np.cos(rad), -np.sin(rad)],
                     [np.sin(rad), np.cos(rad)]])


def rotate(angx: float, angy: float, angz: float, data: torch.Tensor) -> torch.Tensor:
    NZ, NY, NX = data.shape
    CZ, CY, CX = NZ // 2, NY // 2, NX // 2

    # Rotation about the image center
    Rx = rotx(angx)
    Ry = roty(angy)
    Rz = rotz(angz)
    Rot = Rz @ Ry @ Rx
    Rot = np.linalg.inv(Rot)

    tx = CX - Rot[0, 0] * CX - Rot[0, 1] * CY - Rot[0, 2] * CZ
    ty = CY - Rot[1, 0] * CX - Rot[1, 1] * CY - Rot[1, 2] * CZ
    tz = CZ - Rot[2, 0] * CX - Rot[2, 1] * CY - Rot[2, 2] * CZ
    offset = np.array([tx, ty, tz])

    data_t = affine_transform(data.numpy(), matrix=Rot, offset=offset, order=3, mode='reflect')
    return torch.from_numpy(data_t)


def flip(axis: tuple, data: torch.Tensor) -> torch.Tensor:
    return torch.from_numpy(np.flip(data.numpy(), axis).copy())
