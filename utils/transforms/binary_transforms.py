import torch
import elasticdeform as ed
import numpy as np
import torch.nn.functional as F
import math
from basic_transforms import rotx, roty, rotz


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


class ToTensor:
    def __init__(self):
        pass

    def __call__(self, img, mask):
        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()
        return img, mask

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Resize:
    def __init__(self, size: tuple[int, int, int]):
        self.size = size

    def __call__(self, img, mask):
        img = F.interpolate(img.unsqueeze(0), size=self.size, align_corners=True, mode='trilinear').squeeze(0)
        mask = F.interpolate(mask.unsqueeze(0), size=self.size, align_corners=True, mode='trilinear').squeeze(0)
        return img, mask

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class BinarizeMasks:
    def __init__(self, th=0.5):
        self.th = th

    def __call__(self, img, mask):
        mask = torch.where(mask > 0.5, 1.0, 0.0)
        return img, mask

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class RandomVerticalFlip:
    def __init__(self, p):
        self.p = p

    def __call__(self, img, mask):
        if np.random.rand() < self.p:
            img = torch.flip(img, dims=(2,))
            mask = torch.flip(mask, dims=(2,))
        return img, mask

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class RandomHorizontalFlip:
    def __init__(self, p):
        self.p = p

    def __call__(self, img, mask):
        if np.random.rand() < self.p:
            img = torch.flip(img, dims=(3,))
            mask = torch.flip(mask, dims=(3,))
        return img, mask

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class RandomDepthFlip:
    def __init__(self, p):
        self.p = p

    def __call__(self, img, mask):
        if np.random.rand() < self.p:
            img = torch.flip(img, dims=(0,))
            mask = torch.flip(mask, dims=(0,))
        return img, mask

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Cut:
    def __init__(self, tol=10):
        self.tol = tol

    def __call__(self, img, mask):
        CH, NZ, NY, NX = img.shape
        zmin_dia, zmax_dia, ymin_dia, ymax_dia, xmin_dia, xmax_dia = self.mask_range(mask)
        xmin_total = max(0, xmin_dia - self.tol)
        xmax_total = min(NX - 1, xmax_dia + self.tol)
        ymin_total = max(0, ymin_dia - self.tol)
        ymax_total = min(NY - 1, ymax_dia + self.tol)
        zmin_total = max(0, zmin_dia - self.tol)
        zmax_total = min(NZ - 1, zmax_dia + self.tol)

        img = img[:, zmin_total:zmax_total + 1, ymin_total:ymax_total + 1, xmin_total:xmax_total + 1]
        mask = mask[:, zmin_total:zmax_total + 1, ymin_total:ymax_total + 1, xmin_total:xmax_total + 1]
        return img, mask

    def mask_range(self, mask):
        z, y, x = torch.nonzero(mask[0, ...], as_tuple=True)
        xmin = torch.min(x)
        xmax = torch.max(x)
        ymin = torch.min(y)
        ymax = torch.max(y)
        zmin = torch.min(z)
        zmax = torch.max(z)
        return zmin, zmax, ymin, ymax, xmin, xmax

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

    def __call__(self, img, mask):
        # Normalize between 0 and 1
        if self.local_norm:
            self.min = torch.amin(img)
            self.max = torch.amax(img)
        img = (img - self.min) / (self.max - self.min)
        return img, mask

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class InstanceQuadraticNormalization:
    def __init__(self):
        pass

    def __call__(self, img, mask):
        img = img.numpy()
        mask = mask.numpy()

        per95 = np.percentile(img, 95)
        img = np.clip(img, 0, per95)
        avg = np.mean(img, where=mask.astype('bool'))

        # normalization n(I) = a I/sqrt(1+beta I**2)
        norm_a = math.sqrt(per95 * per95 - avg * avg) / (math.sqrt(3) * per95 * avg)
        norm_b = (per95 * per95 - 4. * avg * avg) / (3. * per95 * per95 * avg * avg)
        img_norm = norm_a * img / np.sqrt(1 + norm_b * img**2)

        return torch.from_numpy(img_norm).float(), torch.from_numpy(mask).float()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class RandomRotate:
    def __init__(self, p=0.5, range_x: tuple = (0, 0), range_y: tuple = (0, 0), range_z: tuple = (0, 0),
                 boundary='zeros', clip_interval: tuple = (0.0, 1.0)):
        self.p = p
        self.range_x = range_x
        self.range_y = range_y
        self.range_z = range_z
        self.boundary = boundary
        self.clip_interval = clip_interval

    def __call__(self, img, mask):
        if np.random.rand() < self.p:
            CH, NZ, NY, NX = img.shape
            R, offset = self.create_rot_mat(NZ, NY, NX)
            grid_t = self.scale_grid(self.generate_rotation_grid(R, offset, NZ, NY, NX).unsqueeze(0))

            # Rotate mask
            mask_rot = F.grid_sample(mask.unsqueeze(0), grid_t, mode='bilinear', padding_mode=self.boundary, align_corners=True).squeeze(0)

            # Rotate image
            img_rot = F.grid_sample(img.unsqueeze(0), grid_t, mode='bilinear', padding_mode=self.boundary, align_corners=True).squeeze(0)
            # img_rot = torch.clip(img_rot, min=self.clip_interval[0], max=self.clip_interval[1])

            return img_rot, mask_rot
        else:
            return img, mask

    def scale_grid(self, grid):
        # scale grid to [-1,1]
        _, NZ, NY, NX, _ = grid.shape
        grid[..., 0] = 2.0 * grid[..., 0] / max(NX - 1, 1) - 1.0
        grid[..., 1] = 2.0 * grid[..., 1] / max(NY - 1, 1) - 1.0
        grid[..., 2] = 2.0 * grid[..., 2] / max(NZ - 1, 1) - 1.0
        return grid

    def generate_rotation_grid(self, rot, offset, NZ, NY, NX):
        zz, yy, xx = torch.meshgrid(torch.arange(NZ), torch.arange(NY), torch.arange(NX), indexing="ij")
        xx_t = (rot[0, 0] * xx + rot[0, 1] * yy + rot[0, 2] * zz) + offset[0]
        yy_t = (rot[1, 0] * xx + rot[1, 1] * yy + rot[1, 2] * zz) + offset[1]
        zz_t = (rot[2, 0] * xx + rot[2, 1] * yy + rot[2, 2] * zz) + offset[2]
        return torch.stack((xx_t, yy_t, zz_t), dim=3).float()

    def create_rot_mat(self, NZ, NY, NX):
        angx = np.random.uniform(self.range_x[0], self.range_x[1])
        angy = np.random.uniform(self.range_y[0], self.range_y[1])
        angz = np.random.uniform(self.range_z[0], self.range_z[1])

        # Rotation about the image center
        CZ, CY, CX = NZ // 2, NY // 2, NX // 2
        Rx = rotx(angx)
        Ry = roty(angy)
        Rz = rotz(angz)
        R = Rz @ Ry @ Rx
        R = R.T
        tx = CX - R[0, 0] * CX - R[0, 1] * CY - R[0, 2] * CZ
        ty = CY - R[1, 0] * CX - R[1, 1] * CY - R[1, 2] * CZ
        tz = CZ - R[2, 0] * CX - R[2, 1] * CY - R[2, 2] * CZ
        offset = np.array([tx, ty, tz])

        R = torch.from_numpy(R).float()
        offset = torch.from_numpy(offset).float()
        return R, offset

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class MutiplicativeScaling:
    def __init__(self, p, scale_range: tuple, clip_interval: tuple):
        self.p = p
        self.scale_range = scale_range
        self.clip_interval = clip_interval

    def __call__(self, img, mask):
        if np.random.rand() < self.p:
            sigma = np.random.uniform(self.scale_range[0], self.scale_range[1])
            img = torch.clip(sigma * img, min=self.clip_interval[0], max=self.clip_interval[1])
        return img, mask

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class AdditiveScaling:
    def __init__(self, p, mean, std, clip_interval):
        self.p = p
        self.mean = mean
        self.std = std
        self.clip_interval = clip_interval

    def __call__(self, img, mask):
        if np.random.rand() < self.p:
            sigma = np.random.normal(self.mean, self.std)
            img = torch.clip(sigma + img, min=self.clip_interval[0], max=self.clip_interval[1])
        return img, mask

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class GammaScaling:
    def __init__(self, p, gamma_range: tuple, clip_interval: tuple):
        self.p = p
        self.gamma_range = gamma_range
        self.clip_interval = clip_interval

    def __call__(self, img, mask):
        if np.random.rand() < self.p:
            gamma = np.random.uniform(self.gamma_range[0], self.gamma_range[1])
            img = torch.clip(img**gamma, min=self.clip_interval[0], max=self.clip_interval[1])
        return img, mask

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class AdditiveGaussianNoise:
    def __init__(self, p, mu, sigma, clip_interval):
        self.p = p
        self.mu = mu
        self.sigma = sigma
        self.clip_interval = clip_interval

    def __call__(self, img, mask):
        if np.random.rand() < self.p:
            noise = torch.normal(mean=self.mu, std=self.sigma, size=img.shape)
            img = torch.clip(noise + img, min=self.clip_interval[0], max=self.clip_interval[1])
        return img, mask

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Standarize:
    def __init__(self):
        pass

    def __call__(self, img, mask):
        std, mean = torch.std_mean(img)
        img_n = (img - mean) / std
        return img_n, mask

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class ElasticDeformation:
    def __init__(self, p, sigma_range, points, boundary, prefilter, axis, clip_interval):
        self.p = p
        self.sigma_range = sigma_range
        self.points = points
        self.boundary = boundary
        self.prefilter = prefilter
        self.axis_str = axis
        self.clip_interval = clip_interval

    def __call__(self, img, mask):
        if np.random.rand() < self.p:
            sigma = np.random.uniform(self.sigma_range[0], self.sigma_range[1])
            if self.axis_str == 'zyx':
                axis = [(1, 2, 3)] * 2
            else:
                axis = [(2, 3)] * 2
            [img_d, mask_d] = ed.deform_random_grid([img.numpy(), mask.numpy()], sigma,
                                                    points=self.points, mode=self.boundary,
                                                    prefilter=self.prefilter,
                                                    axis=axis)
            img_d = np.clip(img_d, self.clip_interval[0], self.clip_interval[1])
            return torch.from_numpy(img_d), torch.from_numpy(mask_d)
        else:
            return img, mask

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
