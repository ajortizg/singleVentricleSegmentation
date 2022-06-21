import numpy as np
import torch
from scipy.ndimage import affine_transform
import torch.nn.functional as F
import elasticdeform as ed
import cv2
import utils.torch_utils as torch_utils


class ComposeTernary:
    def __init__(self, transforms):
        self.transforms = transforms

    # vol is 4D tensor with shape(NZ, NY, NX, NT)
    # ms and md are the systole and diastole mask with shape(NZ, NY, NX)
    def __call__(self, vol: torch.Tensor, ms: torch.Tensor, md: torch.Tensor):
        for t in self.transforms:
            vol, ms, md = t(vol, ms, md)
        return (vol, ms, md)

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


class ComposeUnary:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, vol: torch.Tensor):
        for t in self.transforms:
            vol = t(vol)
        return vol

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


class Normalize:
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, x: torch.Tensor):
        x = torch_utils.normalize(x)  # normalize between 0 and 1
        if self.mean is not None and self.std is not None:
            return (x - self.mean) / self.std
        else:
            return x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class PadTime:
    def __init__(self, maxt=40):
        self.maxt = maxt

    def __call__(self, vol: torch.Tensor) -> torch.Tensor:
        *_, NT = vol.shape
        diff_t = self.maxt - NT
        return F.pad(vol, [0, diff_t,
                           0, 0,
                           0, 0,
                           0, 0])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Pad:
    def __init__(self, max_z, max_y, max_x, max_t):
        self.max_z = max_z
        self.max_y = max_y
        self.max_x = max_x
        self.max_t = max_t

    def __call__(self, x: list) -> list:
        BS = len(x)
        x_t = []
        for b in range(BS):
            # Masks
            if len(x[b].shape) == 3:
                NZ, NY, NX = x[b].shape
                diff_z = self.max_z - NZ
                diff_y = self.max_y - NY
                diff_x = self.max_x - NX
                y = F.pad(x[b],
                          [diff_x // 2, diff_x - diff_x // 2,
                          diff_y // 2, diff_y - diff_y // 2,
                          diff_z // 2, diff_z - diff_z // 2])
                x_t.append(y)
            # volumes
            elif len(x[b].shape) == 4:
                NZ, NY, NX, NT = x[b].shape
                diff_z = self.max_z - NZ
                diff_y = self.max_y - NY
                diff_x = self.max_x - NX
                diff_t = self.max_t - NT
                y = F.pad(x[b],
                          [0, diff_t,
                           diff_x // 2, diff_x - diff_x // 2,
                           diff_y // 2, diff_y - diff_y // 2,
                           diff_z // 2, diff_z - diff_z // 2])
                x_t.append(y)
            # Optical flow
            elif len(x[b].shape) == 5:
                NT, NZ, NY, NX, C3 = x[b].shape
                diff_z = self.max_z - NZ
                diff_y = self.max_y - NY
                diff_x = self.max_x - NX
                y = F.pad(x[b],
                          [0, 0,
                          diff_x // 2, diff_x - diff_x // 2,
                          diff_y // 2, diff_y - diff_y // 2,
                          diff_z // 2, diff_z - diff_z // 2,
                          0, 0])
                x_t.append(y)
            else:
                x_t.append(x[b])

        return x_t

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class ListToTensor:
    def __init__(self):
        pass

    def __call__(self, x: list) -> torch.tensor:
        xt = torch.stack([t.float() for t in x], dim=0)
        return xt

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class FlipBase:
    def __init__(self, p=0.5):
        self.p = p

    # vol is 4D tensor with shape(NZ, NY, NX, NT)
    # ms and md are the systole and diastole mask with shape(NZ, NY, NX)
    def flip(self, vol: torch.Tensor, ms: torch.Tensor, md: torch.Tensor, axis: int):
        vol_n = np.zeros(vol.shape)
        *_, NT = vol.shape
        for t in range(NT):
            vol_n[:, :, :, t] = np.flip(vol[:, :, :, t].numpy(), axis).copy()
        ms_n = np.flip(ms.numpy(), axis).copy()
        md_n = np.flip(md.numpy(), axis).copy()
        return (torch.from_numpy(vol_n), torch.from_numpy(ms_n), torch.from_numpy(md_n))


class RandomFlipZ(FlipBase):
    def __init__(self, p=0.5):
        super().__init__(p)

    def __call__(self, vol: torch.Tensor, ms: torch.Tensor, md: torch.Tensor):
        return super().flip(vol, ms, md, 0) if np.random.rand() < self.p else (vol, ms, md)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class RandomFlipY(FlipBase):
    def __init__(self, p=0.5):
        super().__init__(p)

    def __call__(self, vol: torch.Tensor, ms: torch.Tensor, md: torch.Tensor):
        return super().flip(vol, ms, md, 2) if np.random.rand() < self.p else (vol, ms, md)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class RandomFlipX(FlipBase):
    def __init__(self, p=0.5):
        super().__init__(p)

    def __call__(self, vol: torch.Tensor, ms: torch.Tensor, md: torch.Tensor):
        return super().flip(vol, ms, md, 1) if np.random.rand() < self.p else (vol, ms, md)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class RandomRotate:
    def __init__(self, p=0.5, range_x: tuple = (0, 0), range_y: tuple = (0, 0), range_z: tuple = (0, 0)):
        self.p = p
        self.range_x = range_x
        self.range_y = range_y
        self.range_z = range_z

    def __call__(self, vol: torch.Tensor, ms: torch.Tensor, md: torch.Tensor):
        if np.random.rand() < self.p:
            angx = np.random.uniform(self.range_x[0], self.range_x[1])
            angy = np.random.uniform(self.range_y[0], self.range_y[1])
            angz = np.random.uniform(self.range_z[0], self.range_z[1])
            NZ, NY, NX, NT = vol.shape
            CZ, CY, CX = NZ // 2, NY // 2, NX // 2

            # Rotation about the image center
            Rx = rotx(angx)
            Ry = roty(angy)
            Rz = rotz(angz)
            Rot = Rz @ Ry @ Rx
            Rot = Rot.T
            tx = CX - Rot[0, 0] * CX - Rot[0, 1] * CY - Rot[0, 2] * CZ
            ty = CY - Rot[1, 0] * CX - Rot[1, 1] * CY - Rot[1, 2] * CZ
            tz = CZ - Rot[2, 0] * CX - Rot[2, 1] * CY - Rot[2, 2] * CZ
            offset = np.array([tx, ty, tz])

            vol_n = np.zeros(shape=(NX, NY, NZ, NT))
            for t in range(NT):
                vol_n[:, :, :, t] = affine_transform(vol[:, :, :, t].swapaxes(0, 2).numpy(), matrix=Rot, offset=offset, order=3, mode='nearest')
            ms_n = affine_transform(ms.swapaxes(0, 2).numpy(), matrix=Rot, offset=offset, order=3, mode='nearest')
            md_n = affine_transform(md.swapaxes(0, 2).numpy(), matrix=Rot, offset=offset, order=3, mode='nearest')
            return (torch.from_numpy(vol_n).swapaxes(0, 2), torch.from_numpy(ms_n).swapaxes(0, 2), torch.from_numpy(md_n).swapaxes(0, 2))
        else:
            return (vol, ms, md)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class ElasticDeformation:
    def __init__(self, p=0.5, sigma_range: tuple = (1, 4), points_range: tuple = (3, 9)):
        self.p = p
        self.sigma_range = sigma_range
        self.points_range = points_range

    def __call__(self, vol: torch.Tensor, ms: torch.Tensor, md: torch.Tensor):
        if np.random.rand() < self.p:
            sigma = np.random.uniform(self.sigma_range[0], self.sigma_range[1])
            points = np.random.uniform(self.points_range[0], self.points_range[1])
            [vol_d, ms_d, md_d] = (ed.deform_random_grid([vol.numpy(), ms.numpy(), md.numpy()], sigma=round(sigma),
                                   points=round(points), mode='reflect', axis=[(1, 2), (1, 2), (1, 2)]))
            return (torch.from_numpy(vol_d), torch.from_numpy(ms_d), torch.from_numpy(md_d))
        else:
            return (vol, ms, md)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Round:
    def __init__(self, th=0.5):
        self.th = th

    def __call__(self, x: torch.Tensor):
        y = torch.where(x > self.th, 1.0, 0.0)
        return y

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Erode:
    def __init__(self, th=0.5):
        self.th = th

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.where(x > self.th, 1.0, 0.0)
        NZ = x.shape[0]
        borders = np.zeros(x.shape)
        for z in range(NZ):
            mask = x[z, :, :].numpy()
            borders[z, :, :] = (mask - cv2.erode(mask, kernel=None, borderValue=0))
        return torch.from_numpy(borders)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Resize:
    def __init__(self, size: tuple[int, int, int]):
        self.size = size

    def __call__(self, vol: torch.Tensor, ms: torch.Tensor, md: torch.Tensor):
        NZ, NY, NX, NT = vol.shape
        nsize = (1, 1, NZ, NY, NX)
        ms_i = F.interpolate(ms.reshape(nsize), size=self.size, align_corners=True, mode='trilinear').squeeze()
        md_i = F.interpolate(md.reshape(nsize), size=self.size, align_corners=True, mode='trilinear').squeeze()
        vol_i = torch.zeros(size=(*self.size, NT), dtype=vol.dtype, device=vol.device)
        for t in range(NT):
            vol_i[:, :, :, t] = F.interpolate(vol[:, :, :, t].reshape(nsize), size=self.size, align_corners=True, mode='trilinear').squeeze()
        return (vol_i, ms_i, md_i)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class ResizeFlow3d:
    def __init__(self, size: tuple[int, int, int]):
        self.size = size

    def __call__(self, uvw: torch.Tensor):
        NZ, NY, NX, C = uvw.shape
        nsize = (1, 1, NZ, NY, NX)
        uvw_i = torch.zeros(size=(*self.size, C), dtype=uvw.dtype, device=uvw.device)
        for c in range(C):
            uvw_i[:, :, :, c] = F.interpolate(uvw[:, :, :, c].reshape(nsize), size=self.size, align_corners=True, mode='trilinear').squeeze()
        return uvw_i

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


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
