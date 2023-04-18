import numpy as np
import torch
import torch.nn.functional as F
import elasticdeform as ed
from utilities.basic_transforms import rotx, roty, rotz


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img4d, ms, md, ff, bf):
        for t in self.transforms:
            img4d, ms, md, ff, bf = t(img4d, ms, md, ff, bf)
        return (img4d, ms, md, ff, bf)

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

    def __call__(self, img4d: np.array, ms: np.array, md: np.array, ff: np.array, bf: np.array):
        idx = np.random.randint(self.n)
        return self.transforms[idx](img4d, ms, md, ff, bf)


class RandomVerticalFlip:
    def __init__(self, p):
        self.p = p

    def __call__(self, img4d: np.array, ms: np.array, md: np.array, ff: np.array, bf: np.array):
        if np.random.rand() < self.p:
            img4d_flip = np.flip(img4d, axis=1).copy()
            ms_flip = np.flip(ms, axis=1).copy()
            md_flip = np.flip(md, axis=1).copy()
            ff_flip = np.flip(ff, axis=1).copy()
            bf_flip = np.flip(bf, axis=1).copy()

            ff_flip[..., 1, :] *= -1
            bf_flip[..., 1, :] *= -1
            return (img4d_flip, ms_flip, md_flip, ff_flip, bf_flip)
        else:
            return (img4d, ms, md, ff, bf)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class RandomHorizontalFlip:
    def __init__(self, p):
        self.p = p

    def __call__(self, img4d: np.array, ms: np.array, md: np.array, ff: np.array, bf: np.array):
        if np.random.rand() < self.p:
            img4d_flip = np.flip(img4d, axis=2).copy()
            ms_flip = np.flip(ms, axis=2).copy()
            md_flip = np.flip(md, axis=2).copy()
            ff_flip = np.flip(ff, axis=2).copy()
            bf_flip = np.flip(bf, axis=2).copy()

            ff_flip[..., 0, :] *= -1
            bf_flip[..., 0, :] *= -1
            return (img4d_flip, ms_flip, md_flip, ff_flip, bf_flip)
        else:
            return (img4d, ms, md, ff, bf)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class RandomDepthFlip:
    def __init__(self, p):
        self.p = p

    def __call__(self, img4d: np.array, ms: np.array, md: np.array, ff: np.array, bf: np.array):
        if np.random.rand() < self.p:
            img4d_flip = np.flip(img4d, axis=0).copy()
            ms_flip = np.flip(ms, axis=0).copy()
            md_flip = np.flip(md, axis=0).copy()
            ff_flip = np.flip(ff, axis=0).copy()
            bf_flip = np.flip(bf, axis=0).copy()

            ff_flip[..., 2, :] *= -1
            bf_flip[..., 2, :] *= -1
            return (img4d_flip, ms_flip, md_flip, ff_flip, bf_flip)
        else:
            return (img4d, ms, md, ff, bf)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class MutiplicativeScaling:
    def __init__(self, p, scale_range: tuple, clip_interval: tuple):
        self.p = p
        self.scale_range = scale_range
        self.clip_interval = clip_interval

    def __call__(self, img4d: np.array, ms: np.array, md: np.array, ff: np.array, bf: np.array):
        if np.random.rand() < self.p:
            sigma = np.random.uniform(self.scale_range[0], self.scale_range[1])
            img4d = np.clip(sigma * img4d, a_min=self.clip_interval[0], a_max=self.clip_interval[1])
        return (img4d, ms, md, ff, bf)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class AdditiveScaling:
    def __init__(self, p, mean, std, clip_interval):
        self.p = p
        self.mean = mean
        self.std = std
        self.clip_interval = clip_interval

    def __call__(self, img4d: np.array, ms: np.array, md: np.array, ff: np.array, bf: np.array):
        if np.random.rand() < self.p:
            sigma = np.random.normal(self.mean, self.std)
            img4d = np.clip(sigma + img4d, a_min=self.clip_interval[0], a_max=self.clip_interval[1])
        return (img4d, ms, md, ff, bf)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class GammaScaling:
    def __init__(self, p, gamma_range: tuple):
        self.p = p
        self.gamma_range = gamma_range

    def __call__(self, img4d: np.array, ms: np.array, md: np.array, ff: np.array, bf: np.array):
        if np.random.rand() < self.p:
            gamma = np.random.uniform(self.gamma_range[0], self.gamma_range[1])
            # print(gamma, np.min(img4d), np.max(img4d), np.isnan(img4d).any())
            img4d = np.float_power(img4d, gamma)
            # img4d = img4d**gamma

        return (img4d, ms, md, ff, bf)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class AdditiveGaussianNoise:
    def __init__(self, p, mu, sigma, clip_interval):
        self.p = p
        self.mu = mu
        self.sigma = sigma
        self.clip_interval = clip_interval

    def __call__(self, img4d: np.array, ms: np.array, md: np.array, ff: np.array, bf: np.array):
        if np.random.rand() < self.p:
            noise = np.random.normal(self.mu, self.sigma, size=img4d.shape)
            img4d = np.clip(noise + img4d, a_min=self.clip_interval[0], a_max=self.clip_interval[1])
        return (img4d, ms, md, ff, bf)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class BinarizeMasks:
    def __init__(self, th):
        self.th = th

    def __call__(self, img4d: np.array, ms: np.array, md: np.array, ff: np.array, bf: np.array):
        ms = np.where(ms > self.th, 1.0, 0.0)
        md = np.where(md > self.th, 1.0, 0.0)
        return (img4d, ms, md, ff, bf)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class ToTensor:
    def __init__(self):
        pass

    def __call__(self, img4d: np.array, ms: np.array, md: np.array, ff: np.array, bf: np.array):
        img4d = torch.from_numpy(img4d).float()
        ms = torch.from_numpy(ms).float()
        md = torch.from_numpy(md).float()
        ff = torch.from_numpy(ff).float()
        bf = torch.from_numpy(bf).float()
        return (img4d, ms, md, ff, bf)

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

    def __call__(self, img4d: np.array, ms: np.array, md: np.array, ff: np.array, bf: np.array):
        if np.random.rand() < self.p:
            NZ, NY, NX, NT = img4d.shape
            R, offset = self.create_rot_mat(NZ, NY, NX)
            grid_t = self.scale_grid(self.generate_rotation_grid(R, offset, NZ, NY, NX).unsqueeze(0))

            # Rotate masks
            ms_rot, md_rot = self.rotate_masks(ms, md, grid_t)

            # Rotate images 4d
            grid_t = grid_t.repeat(NT, 1, 1, 1, 1)
            img4d_rot = self.rotate_imgs(img4d, grid_t)
            img4d_rot = torch.clip(img4d_rot, min=self.clip_interval[0], max=self.clip_interval[1])

            # Rotate optical flow imgs and vectors
            ff_rot = self.rototate_flow_img(ff, grid_t)
            bf_rot = self.rototate_flow_img(bf, grid_t)
            ff_rot = self.rotate_flow_vectors(ff_rot, R)
            bf_rot = self.rotate_flow_vectors(bf_rot, R)

            return (img4d_rot.numpy(), ms_rot.numpy(), md_rot.numpy(), ff_rot.numpy(), bf_rot.numpy())
        else:
            return (img4d, ms, md, ff, bf)

    def scale_grid(self, grid):
        # scale grid to [-1,1]
        _, NZ, NY, NX, _ = grid.shape
        grid[..., 0] = 2.0 * grid[..., 0] / max(NX - 1, 1) - 1.0
        grid[..., 1] = 2.0 * grid[..., 1] / max(NY - 1, 1) - 1.0
        grid[..., 2] = 2.0 * grid[..., 2] / max(NZ - 1, 1) - 1.0
        return grid

    def rototate_flow_img(self, of, grid_t):
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
    def __init__(self, p, sigma_range, points, boundary, prefilter, axis, clip_interval):
        self.p = p
        self.sigma_range = sigma_range
        self.points = points
        self.boundary = boundary
        self.prefilter = prefilter
        self.axis_str = axis
        self.clip_interval = clip_interval

    def __call__(self, img4d: np.array, ms: np.array, md: np.array, ff: np.array, bf: np.array):
        if np.random.rand() < self.p:
            sigma = np.random.uniform(self.sigma_range[0], self.sigma_range[1])
            if self.axis_str == 'zyx':
                axis = [(0, 1, 2)] * 5
            else:
                axis = [(1, 2)] * 5
            [img4d_d, ms_d, md_d, ff_d, bf_d] = ed.deform_random_grid([img4d, ms, md, ff, bf], sigma,
                                                                      points=self.points, mode=self.boundary,
                                                                      prefilter=self.prefilter,
                                                                      axis=axis)
            img4d_d = np.clip(img4d_d, self.clip_interval[0], self.clip_interval[1])
            return (img4d_d, ms_d, md_d, ff_d, bf_d)
        else:
            return (img4d, ms, md, ff, bf)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# class RandomRotate:
#     def __init__(self, p=0.5, range_x: tuple = (0, 0), range_y: tuple = (0, 0), range_z: tuple = (0, 0), total: int = None, boundary='nearest'):
#         self.p = p
#         self.range_x = range_x
#         self.range_y = range_y
#         self.range_z = range_z
#         self.total = total
#         self.boundary = boundary

#         self.i = 1
#         if self.total:
#             step_x = abs(self.range_x[1] - self.range_x[0]) / self.total
#             step_y = abs(self.range_y[1] - self.range_y[0]) / self.total
#             step_z = abs(self.range_z[1] - self.range_z[0]) / self.total
#             self.angles_x = np.linspace(self.range_x[0] + step_x, self.range_x[1] - step_x, self.total)
#             self.angles_y = np.linspace(self.range_y[0] + step_y, self.range_y[1] - step_y, self.total)
#             self.angles_z = np.linspace(self.range_z[0] + step_z, self.range_z[1] - step_z, self.total)
#             np.random.shuffle(self.angles_x)
#             np.random.shuffle(self.angles_y)
#             np.random.shuffle(self.angles_z)
#             print(self.angles_x)
#             print(self.angles_y)
#             print(self.angles_z)
#             self.random_angles = False
#         else:
#             self.angles_x = self.angles_y = self.angles_z = None
#             self.random_angles = True

#     def __call__(self, img4d: np.array, ms: np.array, md: np.array):
#         angx = angy = angz = 0
#         if np.random.rand() < self.p:
#             if self.random_angles:
#                 angx = np.random.uniform(self.range_x[0], self.range_x[1])
#                 angy = np.random.uniform(self.range_y[0], self.range_y[1])
#                 angz = np.random.uniform(self.range_z[0], self.range_z[1])
#             else:
#                 angx = self.angles_x[self.i]
#                 angy = self.angles_y[self.i]
#                 angz = self.angles_z[self.i]
#                 self.i += 1
#                 if self.i >= self.total:
#                     self.i = 0
#                     np.random.shuffle(self.angles_x)
#                     np.random.shuffle(self.angles_y)
#                     np.random.shuffle(self.angles_z)

#             NZ, NY, NX, NT = img4d.shape
#             CZ, CY, CX = NZ // 2, NY // 2, NX // 2

#             # Rotation about the image center
#             Rx = rotx(angx)
#             Ry = roty(angy)
#             Rz = rotz(angz)
#             Rot = Rz @ Ry @ Rx
#             Rot = Rot.T
#             tx = CX - Rot[0, 0] * CX - Rot[0, 1] * CY - Rot[0, 2] * CZ
#             ty = CY - Rot[1, 0] * CX - Rot[1, 1] * CY - Rot[1, 2] * CZ
#             tz = CZ - Rot[2, 0] * CX - Rot[2, 1] * CY - Rot[2, 2] * CZ
#             offset = np.array([tx, ty, tz])

#             img4d_rot = np.zeros(shape=(NX, NY, NZ, NT))
#             for t in range(NT):
#                 img4d_rot[:, :, :, t] = ndimage.affine_transform(np.swapaxes(img4d[:, :, :, t], 0, 2),
#                                                                  matrix=Rot, offset=offset, order=3, mode=self.boundary)
#             ms_rot = ndimage.affine_transform(np.swapaxes(ms, 0, 2), matrix=Rot, offset=offset, order=3, mode=self.boundary)
#             md_rot = ndimage.affine_transform(np.swapaxes(md, 0, 2), matrix=Rot, offset=offset, order=3, mode=self.boundary)
#             return (np.swapaxes(img4d_rot, 0, 2), np.swapaxes(ms_rot, 0, 2), np.swapaxes(md_rot, 0, 2))
#         else:
#             return (img4d, ms, md)

#     def __repr__(self) -> str:
#         return f"{self.__class__.__name__}()"
