import numpy as np
import torch
import torch.nn.functional as F
import elasticdeform as ed
import os
import os.path as osp
import sys

import monai
import monai.transforms
import monai.data

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
from utils.transforms.basic_transforms import rotx, roty, rotz

flow_keys = ['forward_flow', 'backward_flow']


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


class OneOf:
    def __init__(self, transforms):
        self.transforms = transforms
        self.n = len(self.transforms)

    def __call__(self, data):
        idx = np.random.randint(self.n)
        return self.transforms[idx](data)

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


class MinMaxNormalization:
    def __init__(self, q1=0, q2=100):
        self.q1 = q1
        self.q2 = q2

    def __call__(self, data):
        img = data['img']
        per1 = np.percentile(img, self.q1)
        per2 = np.percentile(img, self.q2)
        img = np.clip(img, per1, per2)
        amin = np.amin(img)
        amax = np.amax(img)
        img = (img - amin) / (amax - amin)
        data['img'] = img
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class ZScoreNormalization:
    def __init__(self):
        pass

    def __call__(self, data):
        img = data['img']
        sigma = np.std(img)
        mu = np.mean(img)
        img_n = (img - mu) / sigma
        data['img'] = img_n
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class QuadraticNormalization:
    def __init__(self, mean_inside_mask):
        self.mean_inside_mask = mean_inside_mask

    def __call__(self, data):
        img = data['img']
        mask = data['mask']

        per95 = np.percentile(img, 95)
        img = np.clip(img, 0, per95)
        avg = np.mean(img, where=mask.astype('bool')) if self.mean_inside_mask else np.mean(img)
        # avg = np.mean(img, where=mask.astype('bool'))

        # normalization n(I) = a I/sqrt(1+beta I**2)
        norm_a = np.sqrt(per95 * per95 - avg * avg) / (np.sqrt(3) * per95 * avg)
        norm_b = (per95 * per95 - 4. * avg * avg) / (3. * per95 * per95 * avg * avg)
        img_norm = norm_a * img / np.sqrt(1 + norm_b * img**2)
        data['img'] = img_norm
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class RandomVerticalFlip:
    def __init__(self, p):
        self.p = p

    def __call__(self, data):
        if np.random.rand() < self.p:
            img = data['img']
            mask = data['mask']
            img_flip = np.flip(img, axis=1).copy()
            mask_flip = np.flip(mask, axis=1).copy()
            data['img'] = img_flip
            data['mask'] = mask_flip
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class RandomHorizontalFlip:
    def __init__(self, p):
        self.p = p

    def __call__(self, data):
        if np.random.rand() < self.p:
            img = data['img']
            mask = data['mask']
            img_flip = np.flip(img, axis=2).copy()
            mask_flip = np.flip(mask, axis=2).copy()
            data['img'] = img_flip
            data['mask'] = mask_flip
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class RandomDepthFlip:
    def __init__(self, p):
        self.p = p

    def __call__(self, data):
        if np.random.rand() < self.p:
            img = data['img']
            mask = data['mask']
            img_flip = np.flip(img, axis=0).copy()
            mask_flip = np.flip(mask, axis=0).copy()
            data['img'] = img_flip
            data['mask'] = mask_flip
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class MutiplicativeScaling:
    def __init__(self, p, scale_range):
        self.p = p
        self.scale_range = scale_range

    def __call__(self, data):
        if np.random.rand() < self.p:
            img = data['img']
            sigma = np.random.uniform(self.scale_range[0], self.scale_range[1])
            img = sigma * img
            data['img'] = img
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class AdditiveScaling:
    def __init__(self, p, mean, std):
        self.p = p
        self.mean = mean
        self.std = std

    def __call__(self, data):
        if np.random.rand() < self.p:
            img = data['img']
            sigma = np.random.normal(self.mean, self.std)
            img = sigma + img
            data['img'] = img
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class GammaScaling:
    def __init__(self, p, gamma_range):
        self.p = p
        self.gamma_range = gamma_range

    def __call__(self, data):
        if np.random.rand() < self.p:
            img = data['img']
            gamma = np.random.uniform(self.gamma_range[0], self.gamma_range[1])
            img = np.float_power(img, gamma)
            data['img'] = img
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class AdditiveGaussianNoise:
    def __init__(self, p, mu, sigma):
        self.p = p
        self.mu = mu
        self.sigma = sigma

    def __call__(self, data):
        if np.random.rand() < self.p:
            img = data['img']
            noise = np.random.normal(self.mu, self.sigma, size=img.shape)
            img = noise + img
            data['img'] = img
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class BinarizeMasks:
    def __init__(self, th):
        self.th = th

    def __call__(self, data):
        mask = data['mask']
        mask = np.where(mask > self.th, 1.0, 0.0)
        data['mask'] = mask
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class ToArray:
    def __init__(self):
        pass

    def __call__(self, data):
        img = data['img']
        mask = data['mask']
        img = img.cpu().detach().numpy()
        mask = mask.cpu().detach().numpy()
        data['img'] = img
        data['mask'] = mask
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class AddChannelDim:
    def __init__(self, axis=0):
        self.axis = axis

    def __call__(self, data):
        img = data['img']
        mask = data['mask']
        img = np.expand_dims(img, axis=self.axis)
        mask = np.expand_dims(mask, axis=self.axis)
        data['img'] = img
        data['mask'] = mask
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class ToTensor:
    def __init__(self):
        pass

    def __call__(self, data):
        img = data['img']
        mask = data['mask']
        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()
        data['img'] = img
        data['mask'] = mask

        for k in flow_keys:
            if k in data:
                flow = data[k]
                flow = torch.from_numpy(flow).float()
                data[k] = flow
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class ToRAS:
    def __init__(self):
        self.ortr = monai.transforms.Orientation(axcodes='RAS')

    def __call__(self, data):
        img_zyxt = data['img']
        mask_zyxt = data['mask']
        img_affine = data['img_meta']['affine']
        mask_affine = data['mask_meta']['affine']

        ts = img_zyxt.shape[-1]
        for t in range(ts):
            img = img_zyxt[..., t]
            img = np.expand_dims(img, axis=0)
            img = self.ortr(monai.data.MetaTensor(img, affine=img_affine))
            try:
                img_zyxt[..., t] = img.squeeze(0).cpu().detach().numpy()
            except ValueError:
                # TODO! I don't know why this happens :(
                img_zyxt[..., t] = img.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()

        ts = mask_zyxt.shape[-1]
        for t in range(ts):
            mask = mask_zyxt[..., t]
            mask = np.expand_dims(mask, axis=0)
            mask = self.ortr(monai.data.MetaTensor(mask, affine=mask_affine))
            try:
                mask_zyxt[..., t] = mask.squeeze(0).cpu().detach().numpy()
            except ValueError:
                mask_zyxt[..., t] = mask.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()

        data['img'] = img_zyxt
        data['mask'] = mask_zyxt
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Spacing:
    def __init__(self, pixdim):
        self.tr = monai.transforms.Spacing(pixdim=pixdim)

    def __call__(self, data):
        img_zyxt = data['img']
        img_affine = data['img_meta']['affine']
        mask_zyxt = data['mask']
        mask_affine = data['mask_meta']['affine']

        # 1.025, 5.75
        ts = img_zyxt.shape[-1]
        for t in range(ts):
            img = img_zyxt[..., t]
            img = np.expand_dims(img, axis=0)
            img = self.tr(monai.data.MetaTensor(img, affine=img_affine), mode='bilinear')
            img_zyxt[..., t] = img.squeeze(0).cpu().detach().numpy()

        ts = mask_zyxt.shape[-1]
        for t in range(ts):
            mask = mask_zyxt[..., t]
            mask = np.expand_dims(mask, axis=0)
            mask = self.tr(monai.data.MetaTensor(mask, affine=mask_affine), mode='nearest')
            mask_zyxt[..., t] = mask.squeeze(0).cpu().detach().numpy()

        data['img'] = img_zyxt
        data['mask'] = mask_zyxt
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Resize:
    def __init__(self, p: float, size: tuple[int, int, int]):
        self.p = p
        self.size = size

    def __call__(self, data):
        if np.random.rand() < self.p:
            img = data['img']
            img = torch.from_numpy(img).float()  # NZ, NY, NX, NT
            img = img.unsqueeze(0).permute(4, 0, 1, 2, 3)  # NT, CH, NZ, NY, NX
            img = F.interpolate(img, size=self.size, align_corners=True, mode='trilinear').squeeze()
            img = img.permute(1, 2, 3, 0).numpy()

            mask = data['mask']
            mask = torch.from_numpy(mask).float()  # NZ, NY, NX, NT
            mask = mask.unsqueeze(0).permute(4, 0, 1, 2, 3)  # NT, CH, NZ, NY, NX
            mask = F.interpolate(mask, size=self.size, align_corners=True, mode='trilinear').squeeze()
            mask = mask.permute(1, 2, 3, 0).numpy()

            data['img'] = img
            data['mask'] = mask
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class CropForeground:
    def __init__(self, p: float, tol: int = 10):
        self.p = p
        self.tol = tol

    def __call__(self, data):
        if np.random.rand() < self.p:
            img = data['img']
            mask = data['mask']

            NZ, NY, NX, NT = mask.shape
            zmin, zmax, ymin, ymax, xmin, xmax = self.mask_range(mask)
            zmin = max(0, zmin - self.tol)
            zmax = min(NZ - 1, zmax + self.tol)
            ymin = max(0, ymin - self.tol)
            ymax = min(NY - 1, ymax + self.tol)
            xmin = max(0, xmin - self.tol)
            xmax = min(NX - 1, xmax + self.tol)

            img = img[zmin:zmax + 1, ymin:ymax + 1, xmin:xmax + 1]
            mask = mask[zmin:zmax + 1, ymin:ymax + 1, xmin:xmax + 1]
            data['img'] = img
            data['mask'] = mask
        return data

    def mask_range(self, masks):
        z, y, x, _ = np.nonzero(masks)
        xmin = np.min(x)
        xmax = np.max(x)
        ymin = np.min(y)
        ymax = np.max(y)
        zmin = np.min(z)
        zmax = np.max(z)
        return zmin, zmax, ymin, ymax, xmin, xmax

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class RandomRotate:
    def __init__(self, p=0.5, range_x: tuple = (0, 0), range_y: tuple = (0, 0), range_z: tuple = (0, 0),
                 boundary='zeros'):
        self.p = p
        self.range_x = range_x
        self.range_y = range_y
        self.range_z = range_z
        self.boundary = boundary

    def __call__(self, data):
        if np.random.rand() < self.p:
            img = data['img']
            mask = data['mask']
            NZ, NY, NX, NT = img.shape
            R, offset = self.create_rot_mat(NZ, NY, NX)
            grid_t = self.scale_grid(self.generate_rotation_grid(R, offset, NZ, NY, NX).unsqueeze(0))
            grid_img_t = grid_t.repeat(NT, 1, 1, 1, 1)

            # Rotate images
            img4d_rot = self.rotate(img, grid_img_t)

            # Rotate masks
            NZ, NY, NX, NT = mask.shape
            grid_mask_t = grid_t.repeat(NT, 1, 1, 1, 1)
            mask_rot = self.rotate(mask, grid_mask_t)

            data['img'] = img4d_rot.numpy()
            data['mask'] = mask_rot.numpy()
        return data

    def rotate(self, img4d, grid_t):
        NT = img4d.shape[-1]
        img4d = np.transpose(img4d, (3, 0, 1, 2))
        img4d = torch.from_numpy(img4d).float().unsqueeze(1)
        img4d_rot = F.grid_sample(img4d, grid_t, mode='bilinear', padding_mode=self.boundary, align_corners=True).squeeze()
        img4d_rot = torch.permute(img4d_rot, (1, 2, 3, 0))
        return img4d_rot

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


class ElasticDeformation:
    def __init__(self, p, sigma_range, points, boundary, prefilter, axis, order=3):
        self.p = p
        self.sigma_range = sigma_range
        self.points = points
        self.boundary = boundary
        self.prefilter = prefilter
        self.axis_str = axis
        self.order = order

    def __call__(self, data):
        if np.random.rand() < self.p:
            sigma = np.random.uniform(self.sigma_range[0], self.sigma_range[1])
            if self.axis_str == 'zyx':
                axis = [(0, 1, 2)] * 2
            else:
                axis = [(1, 2)] * 2

            img = data['img']
            mask = data['mask']
            [img_d, mask_d] = ed.deform_random_grid([img, mask], sigma,
                                                    points=self.points, mode=self.boundary,
                                                    prefilter=self.prefilter,
                                                    axis=axis, order=self.order)
            data['img'] = img_d
            data['mask'] = mask_d
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
