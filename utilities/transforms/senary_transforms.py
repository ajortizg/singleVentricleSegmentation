import numpy as np
import torch
import torch.nn.functional as F
import elasticdeform as ed
from scipy import ndimage
from .basic_transforms import rotx, roty, rotz


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img4d, ms, md, ff, bf, masks):
        for t in self.transforms:
            img4d, ms, md, ff, bf, masks = t(img4d, ms, md, ff, bf, masks)
        return img4d, ms, md, ff, bf, masks

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

    def __call__(self, img4d, ms, md, ff, bf, masks):
        idx = np.random.randint(self.n)
        return self.transforms[idx](img4d, ms, md, ff, bf, masks)


class RandomVerticalFlip:
    def __init__(self, p):
        self.p = p

    def __call__(self, img4d, ms, md, ff, bf, masks):
        if np.random.rand() < self.p:
            img4d_flip = np.flip(img4d, axis=1).copy()
            ms_flip = np.flip(ms, axis=1).copy()
            md_flip = np.flip(md, axis=1).copy()
            ff_flip = np.flip(ff, axis=1).copy()
            bf_flip = np.flip(bf, axis=1).copy()

            ff_flip[..., 1, :] *= -1
            bf_flip[..., 1, :] *= -1
            return img4d_flip, ms_flip, md_flip, ff_flip, bf_flip, masks
        else:
            return img4d, ms, md, ff, bf, masks

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class RandomHorizontalFlip:
    def __init__(self, p):
        self.p = p

    def __call__(self, img4d, ms, md, ff, bf, masks):
        if np.random.rand() < self.p:
            img4d_flip = np.flip(img4d, axis=2).copy()
            ms_flip = np.flip(ms, axis=2).copy()
            md_flip = np.flip(md, axis=2).copy()
            ff_flip = np.flip(ff, axis=2).copy()
            bf_flip = np.flip(bf, axis=2).copy()

            ff_flip[..., 0, :] *= -1
            bf_flip[..., 0, :] *= -1
            return img4d_flip, ms_flip, md_flip, ff_flip, bf_flip, masks
        else:
            return img4d, ms, md, ff, bf, masks

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class RandomDepthFlip:
    def __init__(self, p):
        self.p = p

    def __call__(self, img4d, ms, md, ff, bf, masks):
        if np.random.rand() < self.p:
            img4d_flip = np.flip(img4d, axis=0).copy()
            ms_flip = np.flip(ms, axis=0).copy()
            md_flip = np.flip(md, axis=0).copy()
            ff_flip = np.flip(ff, axis=0).copy()
            bf_flip = np.flip(bf, axis=0).copy()

            ff_flip[..., 2, :] *= -1
            bf_flip[..., 2, :] *= -1
            return img4d_flip, ms_flip, md_flip, ff_flip, bf_flip, masks
        else:
            return img4d, ms, md, ff, bf, masks

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class MutiplicativeScaling:
    def __init__(self, p, scale_range):
        self.p = p
        self.scale_range = scale_range

    def __call__(self, img4d, ms, md, ff, bf, masks):
        if np.random.rand() < self.p:
            sigma = np.random.uniform(self.scale_range[0], self.scale_range[1])
            img4d = sigma * img4d
        return img4d, ms, md, ff, bf, masks

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class AdditiveScaling:
    def __init__(self, p, mean, std):
        self.p = p
        self.mean = mean
        self.std = std

    def __call__(self, img4d, ms, md, ff, bf, masks):
        if np.random.rand() < self.p:
            sigma = np.random.normal(self.mean, self.std)
            img4d = sigma + img4d
        return img4d, ms, md, ff, bf, masks

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class GammaScaling:
    def __init__(self, p, gamma_range):
        self.p = p
        self.gamma_range = gamma_range

    def __call__(self, img4d, ms, md, ff, bf, masks):
        if np.random.rand() < self.p:
            gamma = np.random.uniform(self.gamma_range[0], self.gamma_range[1])
            # print(gamma, np.min(img4d), np.max(img4d), np.isnan(img4d).any())
            img4d = np.float_power(img4d, gamma)
            # img4d = img4d**gamma

        return img4d, ms, md, ff, bf, masks

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class GammaScaling_V2:
    def __init__(self, p, gamma_range, invert_image=False, retain_stats=False):
        """
        Augments by changing 'gamma' of the image (same as gamma correction in photos or computer monitors

        :param gamma_range: range to sample gamma from. If one value is smaller than 1 and the other one is
        larger then half the samples will have gamma <1 and the other >1 (in the inverval that was specified).
        Tuple of float. If one value is < 1 and the other > 1 then half the images will be augmented with gamma values
        smaller than 1 and the other half with > 1
        :param invert_image: whether to invert the image before applying gamma augmentation
        :param per_channel:
        :param data_key:
        :param retain_stats: Gamma transformation will alter the mean and std of the data in the patch. If retain_stats=True,
        the data will be transformed to match the mean and standard deviation before gamma augmentation.
        """
        self.p = p
        self.retain_stats = retain_stats
        self.gamma_range = gamma_range
        self.invert_image = invert_image
        self.epsilon = 1e-7

    def __call__(self, img4d, ms, md, ff, bf, masks):
        if np.random.uniform() < self.p:
            if self.invert_image:
                img4d = - img4d
            if self.retain_stats:
                mn = img4d.mean()
                sd = img4d.std()

            if np.random.random() < 0.5 and self.gamma_range[0] < 1:
                gamma = np.random.uniform(self.gamma_range[0], 1)
            else:
                gamma = np.random.uniform(max(self.gamma_range[0], 1), self.gamma_range[1])

            minm = img4d.min()
            rnge = img4d.max() - minm
            img4d = np.power(((img4d - minm) / float(rnge + self.epsilon)), gamma) * float(rnge + self.epsilon) + minm

            if self.retain_stats:
                img4d = img4d - img4d.mean()
                img4d = img4d / (img4d.std() + 1e-8) * sd
                img4d = img4d + mn
            if self.invert_image:
                img4d = - img4d

        return img4d, ms, md, ff, bf, masks

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class ContrastAugmentation:
    def __init__(self, p, contrast_range, preserve_range=True):
        self.p = p
        self.contrast_range = contrast_range
        self.preserve_range = preserve_range

    def __call__(self, img4d, ms, md, ff, bf, masks):
        if np.random.uniform() < self.p:
            if np.random.random() < 0.5 and self.contrast_range[0] < 1:
                factor = np.random.uniform(self.contrast_range[0], 1)
            else:
                factor = np.random.uniform(max(self.contrast_range[0], 1), self.contrast_range[1])

            mn = img4d.mean()
            if self.preserve_range:
                minm = img4d.min()
                maxm = img4d.max()

            img4d = (img4d - mn) * factor + mn

            if self.preserve_range:
                img4d[img4d < minm] = minm
                img4d[img4d > maxm] = maxm

        return img4d, ms, md, ff, bf, masks

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class GaussianBlur:
    def __init__(self, p, sigma_range):
        self.p = p
        self.sigma_range = sigma_range

    def __call__(self, img4d, ms, md, ff, bf, masks):
        if np.random.uniform() < self.p:
            sigma = np.random.uniform(self.sigma_range[0], self.sigma_range[1])
            img4d = ndimage.gaussian_filter(img4d, sigma, order=0)
        return img4d, ms, md, ff, bf, masks

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class AdditiveGaussianNoise:
    def __init__(self, p, mu, sigma_range):
        self.p = p
        self.mu = mu
        self.sigma_range = sigma_range

    def __call__(self, img4d, ms, md, ff, bf, masks):
        if np.random.uniform() < self.p:
            sigma = np.random.uniform(self.sigma_range[0], self.sigma_range[1])
            noise = np.random.normal(self.mu, sigma, size=img4d.shape)
            img4d = noise + img4d
        return img4d, ms, md, ff, bf, masks

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class BinarizeMasks:
    def __init__(self, th):
        self.th = th

    def __call__(self, img4d, ms, md, ff, bf, masks):
        ms = np.where(ms > self.th, 1.0, 0.0)
        md = np.where(md > self.th, 1.0, 0.0)
        if masks is not None:
            masks = np.where(masks > self.th, 1.0, 0.0)
        return img4d, ms, md, ff, bf, masks

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class RoundMasks:
    def __init__(self):
        pass

    def __call__(self, img4d, ms, md, ff, bf, masks):
        ms = np.round(ms)
        md = np.round(md)
        if masks is not None:
            masks = np.round(masks)
        return img4d, ms, md, ff, bf, masks

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class ToTensor:
    def __init__(self):
        pass

    def __call__(self, img4d, ms, md, ff, bf, masks):
        img4d = torch.from_numpy(img4d).float()
        ms = torch.from_numpy(ms).float()
        md = torch.from_numpy(md).float()
        ff = torch.from_numpy(ff).float()
        bf = torch.from_numpy(bf).float()
        if masks is not None:
            masks = torch.from_numpy(masks).float()
        return img4d, ms, md, ff, bf, masks

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Resize:
    def __init__(self, p: float, size: tuple[int, int, int]):
        self.p = p
        self.size = size

    def __call__(self, img4d, ms, md, ff, bf, masks):
        if np.random.rand() < self.p:
            NZ, NY, NX, _, _ = ff.shape
            img4d = torch.from_numpy(img4d).float()  # NZ, NY, NX, NT
            img4d = img4d.unsqueeze(0).permute(4, 0, 1, 2, 3)  # NT, CH, NZ, NY, NX
            ms = torch.from_numpy(ms).float().unsqueeze(0).unsqueeze(0)
            md = torch.from_numpy(md).float().unsqueeze(0).unsqueeze(0)
            ff = torch.from_numpy(ff).float().permute(4, 3, 0, 1, 2)  # NT, CH, NZ, NY, NX
            bf = torch.from_numpy(bf).float().permute(4, 3, 0, 1, 2)

            img4d = F.interpolate(img4d, size=self.size, align_corners=True, mode='trilinear').squeeze()
            ms = F.interpolate(ms, size=self.size, align_corners=True, mode='trilinear').squeeze()
            md = F.interpolate(md, size=self.size, align_corners=True, mode='trilinear').squeeze()
            ff = F.interpolate(ff, size=self.size, align_corners=True, mode='trilinear')
            bf = F.interpolate(bf, size=self.size, align_corners=True, mode='trilinear')

            img4d = img4d.permute(1, 2, 3, 0).numpy()
            ms = ms.numpy()
            md = md.numpy()
            ff = ff.permute(2, 3, 4, 1, 0).numpy()
            bf = bf.permute(2, 3, 4, 1, 0).numpy()

            if masks is not None:
                masks = torch.from_numpy(masks).float().unsqueeze(0)  # C, Z, Y, X, T
                masks = masks.permute(4, 0, 1, 2, 3)  # T, C, Z, Y, X
                masks = F.interpolate(masks, size=self.size, align_corners=True, mode='trilinear').squeeze()
                masks = masks.permute(1, 2, 3, 0).numpy()

            # correct scale factor
            NZ2, NY2, NX2, _, _ = ff.shape
            ff[..., 0, :] /= (float(NX) / float(NX2))
            ff[..., 1, :] /= (float(NY) / float(NY2))
            ff[..., 2, :] /= (float(NZ) / float(NZ2))
        return img4d, ms, md, ff, bf, masks

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

    def __call__(self, img4d, ms, md, ff, bf, masks):
        if np.random.rand() < self.p:
            NZ, NY, NX, NT = img4d.shape
            R, offset = self.create_rot_mat(NZ, NY, NX)
            grid_t = self.scale_grid(self.generate_rotation_grid(R, offset, NZ, NY, NX).unsqueeze(0))

            # Rotate masks
            ms_rot, md_rot = self.rotate_masks(ms, md, grid_t)

            # Rotate images 4d
            grid_t = grid_t.repeat(NT, 1, 1, 1, 1)
            img4d_rot = self.rotate_imgs(img4d, grid_t)

            # Rotate optical flow imgs and vectors
            ff_rot = self.rotate_flow_img(ff, grid_t)
            bf_rot = self.rotate_flow_img(bf, grid_t)
            ff_rot = self.rotate_flow_vectors(ff_rot, R)
            bf_rot = self.rotate_flow_vectors(bf_rot, R)

            return img4d_rot.numpy(), ms_rot.numpy(), md_rot.numpy(), ff_rot.numpy(), bf_rot.numpy(), masks
        else:
            return img4d, ms, md, ff, bf, masks

    def scale_grid(self, grid):
        # scale grid to [-1,1]
        _, NZ, NY, NX, _ = grid.shape
        grid[..., 0] = 2.0 * grid[..., 0] / max(NX - 1, 1) - 1.0
        grid[..., 1] = 2.0 * grid[..., 1] / max(NY - 1, 1) - 1.0
        grid[..., 2] = 2.0 * grid[..., 2] / max(NZ - 1, 1) - 1.0
        return grid

    def rotate_flow_img(self, of, grid_t):
        NT = of.shape[-1]
        grid_t = grid_t[:NT]
        of = np.transpose(of, (4, 3, 0, 1, 2))
        of = torch.from_numpy(of).float()
        of_rot = F.grid_sample(of, grid_t, mode='bilinear', padding_mode=self.boundary, align_corners=True)
        of_rot = torch.permute(of_rot, (2, 3, 4, 1, 0))
        return of_rot

    def rotate_imgs(self, img4d, grid_t):
        NT = img4d.shape[-1]
        img4d = np.transpose(img4d, (3, 0, 1, 2))
        img4d = torch.from_numpy(img4d).float().unsqueeze(1)
        img4d_rot = F.grid_sample(img4d, grid_t, mode='bilinear', padding_mode=self.boundary, align_corners=True).squeeze()
        img4d_rot = torch.permute(img4d_rot, (1, 2, 3, 0))
        return img4d_rot

    def rotate_masks(self, ms, md, grid_t):
        ms = torch.from_numpy(ms).float().unsqueeze(0).unsqueeze(0)
        md = torch.from_numpy(md).float().unsqueeze(0).unsqueeze(0)
        ms_rot = F.grid_sample(ms, grid_t, mode='bilinear', padding_mode=self.boundary, align_corners=True).squeeze()
        md_rot = F.grid_sample(md, grid_t, mode='bilinear', padding_mode=self.boundary, align_corners=True).squeeze()
        return ms_rot, md_rot

    def rotate_flow_vectors(self, of, R):
        R = R.T
        x = of[..., 0, :].clone()
        y = of[..., 1, :].clone()
        z = of[..., 2, :].clone()
        of[..., 0, :] = (x * R[0, 0] + y * R[0, 1] + z * R[0, 2])
        of[..., 1, :] = (x * R[1, 0] + y * R[1, 1] + z * R[1, 2])
        of[..., 2, :] = (x * R[2, 0] + y * R[2, 1] + z * R[2, 2])
        return of

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

    def __call__(self, img4d, ms, md, ff, bf, masks):
        if np.random.rand() < self.p:
            sigma = np.random.uniform(self.sigma_range[0], self.sigma_range[1])
            if self.axis_str == 'zyx':
                axis = [(0, 1, 2)] * 5
            else:
                axis = [(1, 2)] * 5
            [img4d_d, ms_d, md_d, ff_d, bf_d] = ed.deform_random_grid([img4d, ms, md, ff, bf], sigma,
                                                                      points=self.points, mode=self.boundary,
                                                                      prefilter=self.prefilter,
                                                                      axis=axis, order=self.order)
            return img4d_d, ms_d, md_d, ff_d, bf_d, masks
        else:
            return img4d, ms, md, ff, bf, masks

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
