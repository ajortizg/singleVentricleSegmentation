from typing import Any
import torch
import numpy as np
import cv2
import torch.nn.functional as F


class Compose:
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

    def __call__(self, x):
        return x.detach().cpu().numpy()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class ToTensor:
    def __init__(self):
        pass

    def __call__(self, x):
        return torch.from_numpy(x).float()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Normalize:
    def __init__(self, min=None, max=None):
        self.min = min
        self.max = max
        self.local_norm = True if self.min is None or self.max is None else False

    def __call__(self, x):
        if self.local_norm:
            if isinstance(x, torch.Tensor):
                self.min = torch.amin(x)
                self.max = torch.amax(x)
            else:
                self.min = np.amin(x)
                self.max = np.amax(x)

        # print(self.minv, self.maxv)
        return (x - self.min) / (self.max - self.min)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Standarize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return (x - self.mean) / self.std

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Round:
    def __init__(self, th=0.5):
        self.th = th

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            x = torch.where(x > self.th, 1.0, 0.0)
        else:
            x = np.where(x > self.th, 1.0, 0.0)
        return x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Round_V2:
    def __init__(self):
        pass

    def __call__(self, x):
        return x.round()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


from scipy.ndimage import binary_erosion, generate_binary_structure


class Erode:
    def __init__(self, its=1):
        self.its = its

    def __call__(self, x):
        is_tensor = isinstance(x, torch.Tensor)
        if is_tensor:
            x = x.cpu().numpy()

        NZ = x.shape[0]
        borders = np.zeros(x.shape)
        for z in range(NZ):
            mask = x[z, :, :]
            borders[z, :, :] = (mask - cv2.erode(mask, kernel=None, borderValue=0, iterations=self.its))

        return torch.from_numpy(borders).float() if is_tensor else borders

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class ErodeOneHot:
    def __init__(self, its=1):
        self.its = its

    def __call__(self, x):
        is_tensor = isinstance(x, torch.Tensor)
        if is_tensor:
            x = x.cpu().numpy()

        nc, nz = x.shape[:2]
        borders = np.zeros(x.shape)
        for z in range(nz):
            for c in range(1, nc):
                mask = x[c, z, :, :]
                borders[c, z, :, :] = (mask - cv2.erode(mask, kernel=None, borderValue=0, iterations=self.its))

        return torch.from_numpy(borders).float() if is_tensor else borders


class Erode_V2:
    def __init__(self, connectivity=1, iterations=1):
        self.connectivity = connectivity
        self.iterations = iterations

    def __call__(self, x):
        is_tensor = isinstance(x, torch.Tensor)

        if is_tensor:
            x = x.cpu().numpy()
        x = x > 0

        footprint = generate_binary_structure(3, self.connectivity)
        border = x ^ binary_erosion(x, structure=footprint, iterations=self.iterations)
        return torch.from_numpy(border).float() if is_tensor else border

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Resize:
    def __init__(self, size: tuple[int, int, int]):
        self.size = size

    def __call__(self, x):
        is_array = isinstance(x, np.ndarray)
        if is_array:
            x = torch.from_numpy(x).float()

        if len(x.shape) == 3:
            # NZ, NY, NX = x.shape
            x = x.unsqueeze(0).unsqueeze(0)
            x = F.interpolate(x, size=self.size, align_corners=True, mode='trilinear').squeeze()
        elif len(x.shape) == 4:
            # NZ, NY, NX, NT = x.shape
            x = x.unsqueeze(0).permute(4, 0, 1, 2, 3)   # TCDHW
            x = F.interpolate(x, size=self.size, align_corners=True, mode='trilinear').squeeze()
            x = x.permute(1, 2, 3, 0)                   # DHWT

        return x.numpy() if is_array else x

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
