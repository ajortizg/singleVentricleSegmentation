import numpy as np
import torch
import os.path as osp
import sys
from scipy.ndimage import affine_transform
import elasticdeform as ed

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
from utils import torch_utils
from utils.transforms import rotx, roty, rotz


class Compose:
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


class Normalize:
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, vol: torch.Tensor) -> torch.Tensor:
        vol = torch_utils.normalize(vol)  # normalize between 0 and 1
        if self.mean is not None and self.std is not None:
            return (vol - self.mean) / self.std
        else:
            return vol

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
    def __init__(self, p=0.5, range_x: tuple = (-10, 10), range_y: tuple = (0, 0), range_z: tuple = (0, 0)):
        self.p = p
        self.range_x = range_x
        self.range_y = range_y
        self.range_z = range_z

    def __call__(self, vol: torch.Tensor, ms: torch.Tensor, md: torch.Tensor):
        if np.random.rand() < self.p:
            angx = np.random.uniform(self.range_x[0], self.range_x[1] + 1)
            angy = np.random.uniform(self.range_y[0], self.range_y[1] + 1)
            angz = np.random.uniform(self.range_z[0], self.range_z[1] + 1)
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

            vol_n = np.zeros(vol.shape)
            for t in range(NT):
                vol_n[:, :, :, t] = affine_transform(vol[:, :, :, t].numpy(), matrix=Rot, offset=offset, order=3, mode='reflect')
            ms_n = affine_transform(ms.numpy(), matrix=Rot, offset=offset, order=3, mode='reflect')
            md_n = affine_transform(md.numpy(), matrix=Rot, offset=offset, order=3, mode='reflect')
            return (torch.from_numpy(vol_n), torch.from_numpy(ms_n), torch.from_numpy(md_n))
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
