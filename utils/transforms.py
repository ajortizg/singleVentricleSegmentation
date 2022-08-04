import numpy as np
import torch
import torch.nn.functional as F
import elasticdeform as ed
import cv2
from scipy import ndimage


class ComposeTernary:
    def __init__(self, transforms):
        self.transforms = transforms

    # vol is 4D tensor with shape(NZ, NY, NX, NT)
    # ms and md are the systole and diastole mask with shape(NZ, NY, NX)
    def __call__(self, img4d, ms, md):
        for t in self.transforms:
            img4d, ms, md = t(img4d, ms, md)
        return (img4d, ms, md)

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

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


class ToArray:
    def __init__(self):
        pass

    def __call__(self, x: torch.Tensor) -> np.array:
        return x.detach().cpu().numpy()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class ToTensor:
    def __init__(self):
        pass

    def __call__(self, x: np.array) -> torch.Tensor:
        return torch.from_numpy(x).float()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Normalize:
    def __init__(self, min=None, max=None):
        self.min = min
        self.max = max
        if self.min is None or self.max is None:
            self.local_norm = True
        else:
            self.local_norm = False

    def __call__(self, x: np.array) -> np.array:
        # Normalize between 0 and 1
        if self.local_norm:
            self.min = np.amin(x)
            self.max = np.amax(x)

        # print(self.minv, self.maxv)
        return (x - self.min) / (self.max - self.min)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class PercentileClip:
    def __init__(self, q1, q2):
        self.q1 = q1
        self.q2 = q2

    def __call__(self, x: np.array) -> np.array:
        low, high = np.percentile(x, (self.q1, self.q2))
        return np.clip(x, low, high)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class BinaryClosing:
    def __init__(self):
        pass

    def __call__(self, x: np.array) -> np.array:
        return ndimage.binary_closing(x)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class BinaryOpening:
    def __init__(self):
        pass

    def __call__(self, x: np.array) -> np.array:
        return ndimage.binary_opening(x)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Standarize:
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std

    def __call__(self, x: np.array) -> np.array:
        return (x - self.mean) / self.std

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class PadTime:
    def __init__(self, maxt=40):
        self.maxt = maxt

    def __call__(self, img4d: torch.Tensor) -> torch.Tensor:
        *_, NT = img4d.shape
        diff_t = self.maxt - NT
        return F.pad(img4d, [0, diff_t,
                             0, 0,
                             0, 0,
                             0, 0])

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
    def flip(self, img4d: np.array, ms: np.array, md: np.array, axis: int):
        vol_n = np.zeros(img4d.shape)
        *_, NT = img4d.shape
        for t in range(NT):
            vol_n[:, :, :, t] = np.flip(img4d[:, :, :, t], axis).copy()
        ms_n = np.flip(ms, axis).copy()
        md_n = np.flip(md, axis).copy()
        return (vol_n, ms_n, md_n)


class RandomFlipZ(FlipBase):
    def __init__(self, p=0.5):
        super().__init__(p)

    def __call__(self, img4d: np.array, ms: np.array, md: np.array):
        return super().flip(img4d, ms, md, 0) if np.random.rand() < self.p else (img4d, ms, md)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class RandomFlipY(FlipBase):
    def __init__(self, p=0.5):
        super().__init__(p)

    def __call__(self, img4d: np.array, ms: np.array, md: np.array):
        return super().flip(img4d, ms, md, 2) if np.random.rand() < self.p else (img4d, ms, md)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class RandomFlipX(FlipBase):
    def __init__(self, p=0.5):
        super().__init__(p)

    def __call__(self, img4d: np.array, ms: np.array, md: np.array):
        return super().flip(img4d, ms, md, 1) if np.random.rand() < self.p else (img4d, ms, md)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class OneOf:
    def __init__(self, transforms):
        self.transforms = transforms
        self.n = len(self.transforms)

    def __call__(self, img4d: np.array, ms: np.array, md: np.array):
        idx = np.random.randint(self.n)
        return self.transforms[idx](img4d, ms, md)


class RandomRotate:
    def __init__(self, p=0.5, range_x: tuple = (0, 0), range_y: tuple = (0, 0), range_z: tuple = (0, 0), total: int = None, boundary='nearest'):
        self.p = p
        self.range_x = range_x
        self.range_y = range_y
        self.range_z = range_z
        self.total = total
        self.boundary = boundary

        self.i = 1
        if self.total:
            step_x = abs(self.range_x[1] - self.range_x[0]) / self.total
            step_y = abs(self.range_y[1] - self.range_y[0]) / self.total
            step_z = abs(self.range_z[1] - self.range_z[0]) / self.total
            self.angles_x = np.linspace(self.range_x[0] + step_x, self.range_x[1] - step_x, self.total)
            self.angles_y = np.linspace(self.range_y[0] + step_y, self.range_y[1] - step_y, self.total)
            self.angles_z = np.linspace(self.range_z[0] + step_z, self.range_z[1] - step_z, self.total)
            np.random.shuffle(self.angles_x)
            np.random.shuffle(self.angles_y)
            np.random.shuffle(self.angles_z)
            print(self.angles_x)
            print(self.angles_y)
            print(self.angles_z)
            self.random_angles = False
        else:
            self.angles_x = self.angles_y = self.angles_z = None
            self.random_angles = True

    def __call__(self, img4d: np.array, ms: np.array, md: np.array):
        angx = angy = angz = 0
        if np.random.rand() < self.p:
            if self.random_angles:
                angx = np.random.uniform(self.range_x[0], self.range_x[1])
                angy = np.random.uniform(self.range_y[0], self.range_y[1])
                angz = np.random.uniform(self.range_z[0], self.range_z[1])
            else:
                angx = self.angles_x[self.i]
                angy = self.angles_y[self.i]
                angz = self.angles_z[self.i]
                self.i += 1
                if self.i >= self.total:
                    self.i = 0
                    np.random.shuffle(self.angles_x)
                    np.random.shuffle(self.angles_y)
                    np.random.shuffle(self.angles_z)

            NZ, NY, NX, NT = img4d.shape
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

            img4d_rot = np.zeros(shape=(NX, NY, NZ, NT))
            for t in range(NT):
                img4d_rot[:, :, :, t] = ndimage.affine_transform(np.swapaxes(img4d[:, :, :, t], 0, 2),
                                                                 matrix=Rot, offset=offset, order=3, mode=self.boundary)
            ms_rot = ndimage.affine_transform(np.swapaxes(ms, 0, 2), matrix=Rot, offset=offset, order=3, mode=self.boundary)
            md_rot = ndimage.affine_transform(np.swapaxes(md, 0, 2), matrix=Rot, offset=offset, order=3, mode=self.boundary)
            return (np.swapaxes(img4d_rot, 0, 2), np.swapaxes(ms_rot, 0, 2), np.swapaxes(md_rot, 0, 2))
        else:
            return (img4d, ms, md)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class ElasticDeformation:
    def __init__(self, p, sigma_range, points, boundaryMode, usePrefilter):
        self.p = p
        self.sigma_range = sigma_range
        self.points = points
        self.boundaryMode = boundaryMode
        self.usePrefilter = usePrefilter

    def __call__(self, img4d: np.array, ms: np.array, md: np.array):
        if np.random.rand() < self.p:
            sigma = np.random.uniform(self.sigma_range[0], self.sigma_range[1])
            #points = np.random.uniform(self.points_range[0], self.points_range[1])
            [img4d_d, ms_d, md_d] = (ed.deform_random_grid([img4d, ms, md], sigma,
                                                           points=self.points, mode=self.boundaryMode,
                                                           prefilter=self.usePrefilter,
                                                           axis=[(0, 1, 2), (0, 1, 2), (0, 1, 2)]))
            return (img4d_d, ms_d, md_d)
        else:
            return (img4d, ms, md)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class AdditiveGaussianNoise:
    def __init__(self, p, mu, sigma):
        self.p = p
        self.mu = mu
        self.sigma = sigma

    def __call__(self, img4d: np.array, ms: np.array, md: np.array):
        if np.random.rand() < self.p:
            noise = np.random.normal(self.mu, self.sigma, size=img4d.shape)
            img4d_n = img4d + noise
            return (img4d_n, ms, md)
        else:
            return (img4d, ms, md)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class IntensityScaling:
    def __init__(self, p, scale_range: tuple):
        self.p = p
        self.scale_range = scale_range

    def __call__(self, img4d: np.array, ms: np.array, md: np.array):
        if np.random.rand() < self.p:
            sigma = np.random.uniform(self.scale_range[0], self.scale_range[1])
            img4d_t = sigma * img4d
            return (img4d_t, ms, md)
        else:
            return (img4d, ms, md)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Clip:
    def __init__(self, p, interval: tuple):
        self.p = p
        self.interval = interval

    def __call__(self, img4d: np.array, ms: np.array, md: np.array):
        if np.random.rand() < self.p:
            img4d_t = np.clip(img4d, a_min=self.interval[0], a_max=self.interval[1])
            return (img4d_t, ms, md)
        else:
            return (img4d, ms, md)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Round:
    def __init__(self, th=0.5):
        self.th = th

    def __call__(self, x: np.array) -> np.array:
        y = np.where(x > self.th, 1.0, 0.0)
        return y

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Erode:
    def __init__(self, th=0.5):
        self.th = th

    def __call__(self, x: np.array) -> np.array:
        # x = np.where(x > self.th, 1.0, 0.0)
        NZ = x.shape[0]
        borders = np.zeros(x.shape)
        for z in range(NZ):
            mask = x[z, :, :]
            borders[z, :, :] = (mask - cv2.erode(mask, kernel=None, borderValue=0))
        return borders

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# class Resize:
#     def __init__(self, size: tuple[int, int, int]):
#         self.size = size

#     def __call__(self, vol: torch.Tensor, ms: torch.Tensor, md: torch.Tensor):
#         NZ, NY, NX, NT = vol.shape
#         nsize = (1, 1, NZ, NY, NX)
#         ms_i = F.interpolate(ms.reshape(nsize), size=self.size, align_corners=True, mode='trilinear').squeeze()
#         md_i = F.interpolate(md.reshape(nsize), size=self.size, align_corners=True, mode='trilinear').squeeze()
#         vol_i = torch.zeros(size=(*self.size, NT), dtype=vol.dtype, device=vol.device)
#         for t in range(NT):
#             vol_i[:, :, :, t] = F.interpolate(vol[:, :, :, t].reshape(nsize), size=self.size, align_corners=True, mode='trilinear').squeeze()
#         return (vol_i, ms_i, md_i)

#     def __repr__(self) -> str:
#         return f"{self.__class__.__name__}()"


class Resize:
    def __init__(self, size: tuple[int, int, int], boundary='mirror'):
        self.size = size
        self.boundary = boundary

    def __call__(self, x: np.array) -> np.array:
        if len(x.shape) == 3:
            NZ, NY, NX = x.shape
            zz = self.size[0] / NZ
            zy = self.size[1] / NY
            zx = self.size[2] / NX
            return ndimage.zoom(x, zoom=(zz, zy, zx), order=3, mode=self.boundary)
        elif len(x.shape) == 4:
            NZ, NY, NX, NT = x.shape
            zz = self.size[0] / NZ
            zy = self.size[1] / NY
            zx = self.size[2] / NX
            return ndimage.zoom(x, zoom=(zz, zy, zx, 1), order=3, mode=self.boundary)
        else:
            return x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# class ResizeFlow3d:
#     def __init__(self, size: tuple[int, int, int]):
#         self.size = size

#     def __call__(self, uvw: torch.Tensor):
#         NZ, NY, NX, C = uvw.shape
#         nsize = (1, 1, NZ, NY, NX)
#         uvw_i = torch.zeros(size=(*self.size, C), dtype=uvw.dtype, device=uvw.device)
#         for c in range(C):
#             uvw_i[:, :, :, c] = F.interpolate(uvw[:, :, :, c].reshape(nsize), size=self.size, align_corners=True, mode='trilinear').squeeze()
#         return uvw_i

#     def __repr__(self) -> str:
#         return f"{self.__class__.__name__}()"


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


# class Pad:
#     def __init__(self, max_z, max_y, max_x, max_t):
#         self.max_z = max_z
#         self.max_y = max_y
#         self.max_x = max_x
#         self.max_t = max_t

#     def __call__(self, x: list) -> list:
#         BS = len(x)
#         x_t = []
#         for b in range(BS):
#             # Masks
#             if len(x[b].shape) == 3:
#                 NZ, NY, NX = x[b].shape
#                 diff_z = self.max_z - NZ
#                 diff_y = self.max_y - NY
#                 diff_x = self.max_x - NX
#                 y = F.pad(x[b],
#                           [diff_x // 2, diff_x - diff_x // 2,
#                           diff_y // 2, diff_y - diff_y // 2,
#                           diff_z // 2, diff_z - diff_z // 2])
#                 x_t.append(y)
#             # volumes
#             elif len(x[b].shape) == 4:
#                 NZ, NY, NX, NT = x[b].shape
#                 diff_z = self.max_z - NZ
#                 diff_y = self.max_y - NY
#                 diff_x = self.max_x - NX
#                 diff_t = self.max_t - NT
#                 y = F.pad(x[b],
#                           [0, diff_t,
#                            diff_x // 2, diff_x - diff_x // 2,
#                            diff_y // 2, diff_y - diff_y // 2,
#                            diff_z // 2, diff_z - diff_z // 2])
#                 x_t.append(y)
#             # Optical flow
#             elif len(x[b].shape) == 5:
#                 NT, NZ, NY, NX, C3 = x[b].shape
#                 diff_z = self.max_z - NZ
#                 diff_y = self.max_y - NY
#                 diff_x = self.max_x - NX
#                 y = F.pad(x[b],
#                           [0, 0,
#                           diff_x // 2, diff_x - diff_x // 2,
#                           diff_y // 2, diff_y - diff_y // 2,
#                           diff_z // 2, diff_z - diff_z // 2,
#                           0, 0])
#                 x_t.append(y)
#             else:
#                 x_t.append(x[b])

#         return x_t

#     def __repr__(self) -> str:
#         return f"{self.__class__.__name__}()"
