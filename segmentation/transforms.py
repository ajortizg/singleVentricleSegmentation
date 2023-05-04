import numpy as np
import torch
import torch.nn.functional as F
import elasticdeform as ed
import os
import os.path as osp
import sys
from abc import ABCMeta, abstractmethod

import monai
import monai.transforms
import monai.data
from scipy import ndimage
from skimage.transform import resize

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
from utilities.transforms.basic_transforms import rotx, roty, rotz, rot_x_rad, rot_y_rad, rot_z_rad

metadata_subfix = '_meta'
flow_subfix = 'flow'

# All transforms assume images and labels with shape (T, C, Z, Y, X)
# and flow with shape (T, 3, Z, Y, X)

# Note: The data is saved as:
# imgs: xyzt
# mask: xyzt
# flow: t3xyz


class BaseTransform(object, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, keys):
        self.keys = keys
        self.cur_key = None

    def apply_transform(self, data):
        for key in self.keys:
            self.cur_key = key
            x = data[key]
            metadata_key = key + metadata_subfix
            metadata = data[metadata_key] if metadata_key in data else None
            x_new = self._transform_impl(x, metadata)
            data[key] = x_new
        return data

    @abstractmethod
    def __call__(self, data):
        pass

    @abstractmethod
    def _transform_impl(self, x, metadata=None):
        pass

    def class_name(self):
        return str(type(self).__name__)

    def items(self):
        return self.__dict__

    def __repr__(self):
        ret_str = str(type(self).__name__) + "( " + ", ".join(
            [key + " = " + repr(val) for key, val in self.__dict__.items()]) + " )"
        return ret_str


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

# -----------------------------------------------------------
#               Conversion transformations
# -----------------------------------------------------------


class DoNothing(BaseTransform):
    def __init__(self, keys=[]):
        super().__init__(keys)

    def __call__(self, data):
        return data

    def _transform_impl(self, x, metadata=None):
        return x


class ToArray(BaseTransform):
    def __init__(self, keys=['image', 'label']):
        super(ToArray, self).__init__(keys)

    def __call__(self, data):
        return super().apply_transform(data)

    def _transform_impl(self, x, metadata=None):
        return x.cpu().detach().numpy()


class ToTensor(BaseTransform):
    def __init__(self, keys=['image', 'label']):
        super(ToTensor, self).__init__(keys)

    def __call__(self, data):
        return super().apply_transform(data)

    def _transform_impl(self, x, metadata=None):
        return torch.from_numpy(x).float()


class AddDimAt(BaseTransform):
    def __init__(self, axis, keys=['image', 'label']):
        super().__init__(keys)
        self.axis = axis

    def __call__(self, data):
        return super().apply_transform(data)

    def _transform_impl(self, x, metadata=None):
        return np.expand_dims(x, axis=self.axis)


class RemoveDimAt(BaseTransform):
    def __init__(self, axis, keys=['image', 'label']):
        super(RemoveDimAt, self).__init__(keys)
        self.axis = axis

    def __call__(self, data):
        return super().apply_transform(data)

    def _transform_impl(self, x, metadata=None):
        return np.squeeze(x, axis=self.axis)


class AddNLeadingDims(BaseTransform):
    """
    Args:
        n: number of leading dimensions to add
    """

    def __init__(self, n, keys=['image', 'label']):
        super().__init__(keys)
        self.n = n

    def __call__(self, data):
        return super().apply_transform(data)

    def _transform_impl(self, x, metadata=None):
        for _ in range(self.n):
            x = np.expand_dims(x, axis=0)
        return x


class RemoveNLeadingDims(BaseTransform):
    """
    Args:
        n: number of leading dimensions to remove
    """

    def __init__(self, n, keys=['image', 'label']):
        super().__init__(keys)
        self.n = n

    def __call__(self, data):
        return super().apply_transform(data)

    def _transform_impl(self, x, metadata=None):
        for _ in range(self.n):
            x = np.squeeze(x, axis=0)
        return x


class OneHotEncoding(BaseTransform):
    def __init__(self, n, keys=['label']):
        """
        Args:
            n: Number of classes
        """
        super().__init__(keys)
        self.tr = monai.transforms.AsDiscrete(to_onehot=n)
        self.n = n

    def __call__(self, data):
        return super().apply_transform(data)

    def _transform_impl(self, x, metadata=None):
        bs, ch, d1, d2, d3 = x.shape
        assert ch == 1, 'Labels must have a channel dimension of len 1'
        x_onehot = np.empty((bs, self.n, d1, d2, d3), dtype=np.float32)
        for b in range(bs):
            x_onehot[b] = self.tr(x[b])
        return x_onehot


class BCXYZ_To_BCZYX(BaseTransform):
    def __init__(self, keys=['image', 'label']):
        super().__init__(keys)

    def __call__(self, data):
        return super().apply_transform(data)

    def _transform_impl(self, x, metadata=None):
        return np.transpose(x, (0, 1, 4, 3, 2))


# -----------------------------------------------------------
#               Intensity transformations
# -----------------------------------------------------------

class MinMaxNormalization(BaseTransform):
    def __init__(self, q1=0, q2=100, keys=['image']):
        super(MinMaxNormalization, self).__init__(keys)
        self.q1 = q1
        self.q2 = q2

    def __call__(self, data):
        return super().apply_transform(data)

    def _transform_impl(self, x, metadata=None):
        per1 = np.percentile(x, self.q1)
        per2 = np.percentile(x, self.q2)
        x = np.clip(x, per1, per2)
        amin = np.amin(x)
        amax = np.amax(x)
        x = (x - amin) / (amax - amin)
        return x


class ZScoreNormalization(BaseTransform):
    def __init__(self, keys=['image']):
        super(ZScoreNormalization, self).__init__(keys)

    def __call__(self, data):
        return super().apply_transform(data)

    def _transform_impl(self, x, metadata=None):
        sigma = np.std(x)
        mu = np.mean(x)
        x = (x - mu) / sigma
        return x


class QuadraticNormalization(BaseTransform):
    def __init__(self, q2=95, use_label=True, keys=['image'], label_key='label'):
        super(QuadraticNormalization, self).__init__(keys)
        self.q2 = q2
        self.label_key = label_key
        self.use_label = use_label

    def __call__(self, data):
        self.label = data[self.label_key].copy()
        bs = self.label.shape[0]
        if bs == 1:
            self.t = None
        elif bs == 2:
            self.t = [data['es'], data['ed']]
        elif bs > 2:
            self.t = [data['es'], data['ed']]
            self.label = self.label[self.t]
        else:
            # this should not happen
            self.t = None
            raise ValueError('There is something strange with bs in QuadraticNormalization')

        return super().apply_transform(data)

    def _transform_impl(self, x, metadata=None):
        per2 = np.percentile(x, self.q2)
        x = np.clip(x, 0, per2)
        if self.t is not None:
            avg = np.mean(x[self.t], where=self.label.astype('bool')) if self.use_label else np.mean(x)
        else:
            avg = np.mean(x, where=self.label.astype('bool')) if self.use_label else np.mean(x)
        # normalization n(I) = a I/sqrt(1+beta I**2)
        norm_a = np.sqrt(per2 * per2 - avg * avg) / (np.sqrt(3) * per2 * avg)
        norm_b = (per2 * per2 - 4. * avg * avg) / (3. * per2 * per2 * avg * avg)
        x_norm = norm_a * x / np.sqrt(1 + norm_b * x**2)
        return x_norm


class MultiplicativeScaling(BaseTransform):
    def __init__(self, p, scale_range, keys=['image']):
        super(MultiplicativeScaling, self).__init__(keys)
        self.p = p
        self.scale_range = scale_range

    def __call__(self, data):
        if np.random.uniform() < self.p:
            data = super().apply_transform(data)
        return data

    def _transform_impl(self, x, metadata=None):
        sigma = np.random.uniform(self.scale_range[0], self.scale_range[1])
        x *= sigma
        return x


class AdditiveScaling(BaseTransform):
    def __init__(self, p, mean, std, keys=['image']):
        super(AdditiveScaling, self).__init__(keys)
        self.p = p
        self.mean = mean
        self.std = std

    def __call__(self, data):
        if np.random.uniform() < self.p:
            data = super().apply_transform(data)
        return data

    def _transform_impl(self, x, metadata=None):
        sigma = np.random.normal(self.mean, self.std)
        x += sigma
        return x


class ContrastAugmentation(BaseTransform):
    def __init__(self, p, contrast_range, preserve_range=True, keys=['image']):
        super(ContrastAugmentation, self).__init__(keys)
        self.p = p
        self.contrast_range = contrast_range
        self.preserve_range = preserve_range

    def __call__(self, data):
        if np.random.uniform() < self.p:
            data = super().apply_transform(data)
        return data

    def _transform_impl(self, x, metadata=None):
        if np.random.random() < 0.5 and self.contrast_range[0] < 1:
            factor = np.random.uniform(self.contrast_range[0], 1)
        else:
            factor = np.random.uniform(max(self.contrast_range[0], 1), self.contrast_range[1])

        mn = x.mean()
        if self.preserve_range:
            minm = x.min()
            maxm = x.max()

        x = (x - mn) * factor + mn

        if self.preserve_range:
            x[x < minm] = minm
            x[x > maxm] = maxm
        return x


class GammaCorrection(BaseTransform):
    def __init__(self, p, gamma_range, invert_image=False, retain_stats=False, keys=['image']):
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
        super(GammaCorrection, self).__init__(keys)
        self.p = p
        self.retain_stats = retain_stats
        self.gamma_range = gamma_range
        self.invert_image = invert_image
        self.epsilon = 1e-7

    def __call__(self, data):
        if np.random.uniform() < self.p:
            data = super().apply_transform(data)
        return data

    def _transform_impl(self, x, metadata=None):
        if self.invert_image:
            x = - x
        if self.retain_stats:
            mn = x.mean()
            sd = x.std()

        if np.random.random() < 0.5 and self.gamma_range[0] < 1:
            gamma = np.random.uniform(self.gamma_range[0], 1)
        else:
            gamma = np.random.uniform(max(self.gamma_range[0], 1), self.gamma_range[1])

        minm = x.min()
        rnge = x.max() - minm
        x = np.power(((x - minm) / float(rnge + self.epsilon)), gamma) * float(rnge + self.epsilon) + minm

        if self.retain_stats:
            x = x - x.mean()
            x = x / (x.std() + 1e-8) * sd
            x = x + mn
        if self.invert_image:
            x = - x
        return x


class GaussialBlur(BaseTransform):
    def __init__(self, p, sigma_range, keys=['image']):
        super(GaussialBlur, self).__init__(keys)
        self.p = p
        self.sigma_range = sigma_range

    def __call__(self, data):
        if np.random.uniform() < self.p:
            data = super().apply_transform(data)
        return data

    def _transform_impl(self, x, metadata=None):
        sigma = np.random.uniform(self.sigma_range[0], self.sigma_range[1])
        # batch_dim, chan_dim = x.shape[:2]
        # for b in range(batch_dim):
        # for c in range(chan_dim):
        # x[b, c] = ndimage.gaussian_filter(x[b, c], sigma, order=0)
        x = ndimage.gaussian_filter(x, sigma, order=0)
        return x


class AdditiveGaussianNoise(BaseTransform):
    def __init__(self, p, sigma_range, mu=0, keys=['image']):
        super(AdditiveGaussianNoise, self).__init__(keys)
        self.p = p
        self.mu = mu
        self.sigma_range = sigma_range

    def __call__(self, data):
        if np.random.uniform() < self.p:
            data = super().apply_transform(data)
        return data

    def _transform_impl(self, x, metadata=None):
        sigma = np.random.uniform(self.sigma_range[0], self.sigma_range[1])
        noise = np.random.normal(self.mu, sigma, size=x.shape)
        x = noise + x
        return x


class Discretize(BaseTransform):
    def __init__(self, th, keys=['label']):
        super(Discretize, self).__init__(keys)
        self.th = th

    def __call__(self, data):
        return super().apply_transform(data)

    def _transform_impl(self, x, metadata=None):
        return np.where(x > self.th, 1.0, 0.0)


class ClipRange(BaseTransform):
    def __init__(self, min, max, keys=['image']):
        super().__init__(keys)
        self.min = min
        self.max = max

    def __call__(self, data):
        return super().apply_transform(data)

    def _transform_impl(self, x, metadata=None):
        return np.clip(x, self.min, self.max)


# -----------------------------------------------------------
#               Geometric transformations
# -----------------------------------------------------------

class RandomFlip(BaseTransform):
    def __init__(self, p, axis, keys=['image', 'label']):
        """
        Args:
            axis: 2, 3, 4 for depth, vertical and horizontal flips
        """
        super(RandomFlip, self).__init__(keys)
        self.p = p
        self.axis = axis

    def __call__(self, data):
        if np.random.uniform() < self.p:
            data = super().apply_transform(data)
        return data

    def _transform_impl(self, x, metadata=None):
        x_flip = np.flip(x, axis=self.axis).copy()

        if flow_subfix in self.cur_key:
            x_flip = self.flow_transform(x_flip)
        return x_flip

    def flow_transform(self, flow):
        '''
        Args:
            flow: Numpy array with shape (T, 3, Z, Y, X)
        '''
        if self.axis == 2:
            flow[:, 2] *= -1
        elif self.axis == 3:
            flow[:, 1] *= -1
        elif self.axis == 4:
            flow[:, 0] *= -1
        else:
            raise ValueError('Axis should be 2, 3 or 4')
        return flow


class ToRAS(BaseTransform):
    def __init__(self, keys=['image', 'label']):
        super(ToRAS, self).__init__(keys)
        self.ortr = monai.transforms.Orientation(axcodes='RAS')

    def __call__(self, data):
        return super().apply_transform(data)

    def _transform_impl(self, x, metadata=None):
        affine = metadata['affine']
        batch_dim = x.shape[0]
        for b in range(batch_dim):
            xc_new = self.ortr(monai.data.MetaTensor(x[b], affine=affine))
            try:
                x[b] = xc_new.numpy()
            except ValueError:
                # TODO: I don't know why this happens with svd :(
                x[b] = xc_new.permute(0, 2, 3, 1).numpy()
                # print('ToRAS ValueError')
        return x


class Resize(BaseTransform):
    def __init__(self, p, new_shape, keys=['image', 'label'], label_key='label'):
        super().__init__(keys)
        self.p = p
        self.new_shape = list(new_shape) if type(new_shape) == tuple else new_shape
        self.label_key = label_key
        assert len(new_shape) == 3, 'size must be 3d'

    def __call__(self, data):
        if np.random.uniform() < self.p:
            data = super().apply_transform(data)
        return data

    def _transform_impl(self, x, metadata=None):
        mode = 'nearest' if self.cur_key == self.label_key else 'trilinear'
        self.orig_shape = x.shape[2:]
        for i in range(3):
            if self.new_shape[i] == -1:
                self.new_shape[i] = self.orig_shape[i]
        x = torch.from_numpy(x).float()
        x = F.interpolate(x, size=self.new_shape, mode=mode)

        if flow_subfix in self.cur_key:
            x = self.flow_transform(x)

        return x.numpy()

    def flow_transform(self, flow):
        '''
        Args:
            flow: Numpy array with shape (T, 3, Z, Y, X)
        '''
        flow[:, 0] /= (float(self.orig_shape[2]) / float(self.new_shape[2]))
        flow[:, 1] /= (float(self.orig_shape[1]) / float(self.new_shape[1]))
        flow[:, 2] /= (float(self.orig_shape[0]) / float(self.new_shape[0]))
        return flow


class CropForeground(BaseTransform):
    def __init__(self, tol: int = 10, keys=['image', 'label'], label_key='label'):
        super(CropForeground, self).__init__(keys)
        self.tol = tol
        self.label_key = label_key

    def __call__(self, data):
        *_, NZ, NY, NX = data[self.label_key].shape
        self.zmin, self.zmax, self.ymin, self.ymax, self.xmin, self.xmax = self.mask_range(data[self.label_key])
        self.zmin = max(0, self.zmin - self.tol)
        self.zmax = min(NZ - 1, self.zmax + self.tol)
        self.ymin = max(0, self.ymin - self.tol)
        self.ymax = min(NY - 1, self.ymax + self.tol)
        self.xmin = max(0, self.xmin - self.tol)
        self.xmax = min(NX - 1, self.xmax + self.tol)
        return super().apply_transform(data)

    def mask_range(self, label):
        *_, z, y, x = np.nonzero(label)
        xmin = np.min(x)
        xmax = np.max(x)
        ymin = np.min(y)
        ymax = np.max(y)
        zmin = np.min(z)
        zmax = np.max(z)
        return zmin, zmax, ymin, ymax, xmin, xmax

    def _transform_impl(self, x, metadata=None):
        return x[..., self.zmin:self.zmax + 1, self.ymin:self.ymax + 1, self.xmin:self.xmax + 1]


def _create_zero_centered_coordinate_mesh(shape):
    tmp = tuple([torch.arange(i) for i in shape])
    zz, yy, xx = torch.meshgrid(*tmp, indexing='ij')
    coords = torch.stack((xx, yy, zz), dim=3).float().unsqueeze(0)

    NZ, NY, NX = shape
    ordered_shape = (NX, NY, NZ)
    for d in range(len(ordered_shape)):
        coords[..., d] -= ((torch.tensor(ordered_shape).float() - 1) / 2.)[d]
    return coords


def _normalize_coords(coords):
    # normalize coords to [-1,1]
    _, NZ, NY, NX, _ = coords.shape
    shape = (NX, NY, NZ)
    for i in range(len(shape)):
        coords[..., i] = 2 * (coords[..., i] / (shape[i] - 1) - 0.5)
    return coords


def _find_center(coords):
    # Find a nice center location
    _, NZ, NY, NX, _ = coords.shape
    shape = (NX, NY, NZ)
    for d in range(3):
        ctr = shape[d] / 2. - 0.5
        coords[..., d] += ctr
    return coords


class RandomRotate(BaseTransform):
    def __init__(self, p, range_x, range_y, range_z, boundary='zeros',
                 keys=['image', 'label'], label_key='label'):
        super(RandomRotate, self).__init__(keys)
        self.p = p
        self.range_x = range_x
        self.range_y = range_y
        self.range_z = range_z
        self.boundary = boundary
        self.label_key = label_key

    def __call__(self, data):
        if np.random.rand() < self.p:
            NZ, NY, NX = data[self.label_key].shape[2:]
            self.R, offset = self.create_rot(NZ, NY, NX)
            self.coords = _normalize_coords(self.rotate_coords(self.R, offset, NZ, NY, NX))
            data = super().apply_transform(data)
        return data

    def _transform_impl(self, x, metadata=None):
        mode = 'nearest' if self.cur_key == self.label_key else 'bilinear'
        x = torch.from_numpy(x).float()
        x_new = F.grid_sample(x, self.coords.repeat(x.shape[0], 1, 1, 1, 1), mode=mode, padding_mode=self.boundary, align_corners=False)

        if flow_subfix in self.cur_key:
            x_new = self.flow_transform(x_new)
        return x_new.numpy()

    def rotate_coords(self, rot, offset, NZ, NY, NX):
        zz, yy, xx = torch.meshgrid(torch.arange(NZ), torch.arange(NY), torch.arange(NX), indexing="ij")
        xx_t = (rot[0, 0] * xx + rot[0, 1] * yy + rot[0, 2] * zz) + offset[0]
        yy_t = (rot[1, 0] * xx + rot[1, 1] * yy + rot[1, 2] * zz) + offset[1]
        zz_t = (rot[2, 0] * xx + rot[2, 1] * yy + rot[2, 2] * zz) + offset[2]
        return torch.stack((xx_t, yy_t, zz_t), dim=3).float().unsqueeze(0)

    def create_rot(self, NZ, NY, NX):
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

    def flow_transform(self, flow):
        R = self.R.T
        x = flow[:, 0].clone()
        y = flow[:, 1].clone()
        z = flow[:, 2].clone()
        flow[:, 0] = (x * R[0, 0] + y * R[0, 1] + z * R[0, 2])
        flow[:, 1] = (x * R[1, 0] + y * R[1, 1] + z * R[1, 2])
        flow[:, 2] = (x * R[2, 0] + y * R[2, 1] + z * R[2, 2])
        return flow


class RandomScale(BaseTransform):
    def __init__(self, p, scale_range, boundary='zeros', keys=['image', 'label'], label_key='label'):
        super().__init__(keys)
        self.p = p
        self.scale_range = scale_range
        self.label_key = label_key
        self.boundary = boundary

    def __call__(self, data):
        if np.random.uniform() < self.p:
            spatial_shape = data[self.label_key].shape[2:]
            coords = self.scale_coords(_create_zero_centered_coordinate_mesh(spatial_shape))
            self.coords = _normalize_coords(_find_center(coords))
            data = super().apply_transform(data)
        return data

    def _transform_impl(self, x, metadata=None):
        mode = 'nearest' if self.cur_key == self.label_key else 'bilinear'
        x = torch.from_numpy(x).float()
        x_new = F.grid_sample(x, self.coords.repeat(x.shape[0], 1, 1, 1, 1), mode=mode, padding_mode=self.boundary, align_corners=False)

        if flow_subfix in self.cur_key:
            x_new = self.flow_transform(x_new)
        return x_new.numpy()

    def scale_coords(self, coords):
        if np.random.random() < 0.5 and self.scale_range[0] < 1:
            self.sc = np.random.uniform(self.scale_range[0], 1)
        else:
            self.sc = np.random.uniform(max(self.scale_range[0], 1), self.scale_range[1])
        coords *= self.sc
        return coords

    def flow_transform(self, flow):
        flow /= self.sc
        return flow


class ElasticDeformation(BaseTransform):
    def __init__(self, p, sigma_range, points, boundary, axis,
                 keys=['image', 'label'], label_key='label'):
        super(ElasticDeformation, self).__init__(keys)
        self.p = p
        self.sigma_range = sigma_range
        self.points = points
        self.boundary = boundary
        self.axis_str = axis
        self.label_key = label_key

    def __call__(self, data):
        if np.random.rand() < self.p:
            self.displacement = self.create_displacement(data)
            data = super().apply_transform(data)
        return data

    def _transform_impl(self, x, metadata=None):
        order = 0 if self.cur_key == self.label_key else 3
        [x_new] = ed.deform_grid([x],
                                 self.displacement,
                                 order=order,
                                 mode=self.boundary,
                                 prefilter=False,
                                 axis=self.axis)
        return x_new

    def create_displacement(self, data):
        Xs = [data[self.label_key]]
        axis = [(2, 3, 4)] if self.axis_str == 'zyx' else [(3, 4)]
        self.axis, deform_shape = self.normalize_axis_list(axis, Xs)

        if not isinstance(self.points, (list, tuple)):
            self.points = [self.points] * len(deform_shape)

        sigma = np.random.uniform(self.sigma_range[0], self.sigma_range[1])
        return np.random.randn(len(deform_shape), *self.points) * sigma

    def normalize_axis_list(self, axis, Xs):
        if axis is None:
            axis = [tuple(range(x.ndim)) for x in Xs]
        elif isinstance(axis, int):
            axis = (axis,)
        if isinstance(axis, tuple):
            axis = [axis] * len(Xs)
        assert len(axis) == len(Xs), 'Number of axis tuples should match number of inputs.'
        input_shapes = []
        for x, ax in zip(Xs, axis):
            assert isinstance(ax, tuple), 'axis should be given as a tuple'
            assert all(isinstance(a, int) for a in ax), 'axis must contain ints'
            assert len(ax) == len(axis[0]), 'All axis tuples should have the same length.'
            assert ax == tuple(set(ax)), 'axis must be sorted and unique'
            assert all(0 <= a < x.ndim for a in ax), 'invalid axis for input'
            input_shapes.append(tuple(x.shape[d] for d in ax))
        assert len(set(input_shapes)) == 1, 'All inputs should have the same shape.'
        deform_shape = input_shapes[0]
        return axis, deform_shape


class SpatialTransform(BaseTransform):
    def __init__(self, p_rot, p_rot_per_axis, angle_x, angle_y, angle_z,    # rot params
                 p_scale, scale,                                            # scale params
                 p_ed=0, alpha=(0, 0), sigma=(0, 0),                        # ed params
                 border_mode='constant',                                    # nearest
                 keys=['image', 'label'], label_key='label'):
        super(SpatialTransform, self).__init__(keys)
        self.label_key = label_key
        # elastic deformation
        self.p_el_per_sample = p_ed
        self.alpha = alpha
        self.sigma = sigma
        # rotation
        self.p_rot_per_sample = p_rot
        self.p_rot_per_axis = p_rot_per_axis  # TODO: experiment with this
        self.angle_x = angle_x
        self.angle_y = angle_y
        self.angle_z = angle_z
        # scaling
        self.p_scale_per_sample = p_scale
        self.scale = scale
        # border mode
        self.border_mode = border_mode

    def __call__(self, data):
        patch_size = data[self.label_key].shape[2:]
        self.coords, self.modified_coords = self.transform_coords(patch_size)

        # Find a nice center location
        if self.modified_coords:
            for d in range(3):
                ctr = data[self.label_key].shape[d + 2] / 2. - 0.5
                self.coords[d] += ctr

        return super().apply_transform(data)

    def _transform_impl(self, x, metadata=None):
        if self.modified_coords:
            x_result = np.zeros(x.shape, dtype=np.float32)
            order = 0 if self.cur_key == self.label_key else 3

            for b in range(x.shape[0]):
                for c in range(x.shape[1]):
                    # x_result[channel_id] = self.interpolate_img(x[channel_id], self.coords, order, self.border_mode, cval=0)
                    x_result[b, c] = ndimage.map_coordinates(x[b, c].astype(float),
                                                             self.coords, order=order, mode=self.border_mode, cval=0).astype(x.dtype)
            return x_result
        else:
            return x

    def transform_coords(self, patch_size):
        coords = self.create_zero_centered_coordinate_mesh(patch_size)
        modified_coords = False

        # Elastic deformation
        if np.random.uniform() < self.p_el_per_sample:
            a = np.random.uniform(self.alpha[0], self.alpha[1])
            s = np.random.uniform(self.sigma[0], self.sigma[1])
            coords = self.elastic_deform_coordinates(coords, a, s)
            modified_coords = True

        # Rotation
        if np.random.uniform() < self.p_rot_per_sample:
            if np.random.uniform() <= self.p_rot_per_axis:
                a_x = np.random.uniform(self.angle_x[0], self.angle_x[1])
            else:
                a_x = 0

            if np.random.uniform() <= self.p_rot_per_axis:
                a_y = np.random.uniform(self.angle_y[0], self.angle_y[1])
            else:
                a_y = 0

            if np.random.uniform() <= self.p_rot_per_axis:
                a_z = np.random.uniform(self.angle_z[0], self.angle_z[1])
            else:
                a_z = 0

            coords = self.rotate_coords_3d(coords, a_x, a_y, a_z)
            modified_coords = True

        # Scaling
        if np.random.uniform() < self.p_scale_per_sample:
            if np.random.random() < 0.5 and self.scale[0] < 1:
                sc = np.random.uniform(self.scale[0], 1)
            else:
                sc = np.random.uniform(max(self.scale[0], 1), self.scale[1])

            coords = self.scale_coords(coords, sc)
            modified_coords = True

        return coords, modified_coords

    def create_zero_centered_coordinate_mesh(self, shape):
        tmp = tuple([np.arange(i) for i in shape])
        coords = np.array(np.meshgrid(*tmp, indexing='ij')).astype(float)
        for d in range(len(shape)):
            coords[d] -= ((np.array(shape).astype(float) - 1) / 2.)[d]
        return coords

    def elastic_deform_coordinates(self, coordinates, alpha, sigma):
        n_dim = len(coordinates)
        offsets = []
        for _ in range(n_dim):
            offsets.append(ndimage.filters.gaussian_filter((np.random.random(coordinates.shape[1:]) * 2 - 1), sigma, mode="constant", cval=0) * alpha)
        offsets = np.array(offsets)
        indices = offsets + coordinates
        return indices

    def rotate_coords_3d(self, coords, angle_x, angle_y, angle_z):
        rot_matrix = np.identity(len(coords))
        rot_matrix = rot_x_rad(angle_x, rot_matrix)
        rot_matrix = rot_y_rad(angle_y, rot_matrix)
        rot_matrix = rot_z_rad(angle_z, rot_matrix)
        coords = np.dot(coords.reshape(len(coords), -1).transpose(), rot_matrix).transpose().reshape(coords.shape)
        return coords

    def scale_coords(self, coords, scale):
        if isinstance(scale, (tuple, list, np.ndarray)):
            assert len(scale) == len(coords)
            for i in range(len(scale)):
                coords[i] *= scale[i]
        else:
            coords *= scale
        return coords


class SimulateLowResolution(BaseTransform):
    def __init__(self, p, zoom_range, keys=['image', 'label'], label_key='label'):
        super().__init__(keys)
        self.p = p
        self.zoom_range = zoom_range
        self.label_key = label_key

    def __call__(self, data):
        if np.random.uniform() < self.p:
            self.zoom = np.random.uniform(self.zoom_range[0], self.zoom_range[1])
            data = super().apply_transform(data)
        return data

    def _transform_impl(self, x, metadata=None):
        order = 0 if self.cur_key == self.label_key else 3
        shp = np.array(x.shape[2:])
        target_shape = np.round(shp * self.zoom).astype(int)

        for b in range(x.shape[0]):
            for c in range(x.shape[1]):
                down = resize(x[b, c].astype(float), target_shape, order=order, mode='edge', anti_aliasing=False)
                x[b, c] = resize(down, shp, order=0, mode='edge', anti_aliasing=False)
        return x


class XYZT_To_TZYX(BaseTransform):
    def __init__(self, keys=['image', 'label']):
        super().__init__(keys)

    def __call__(self, data):
        return super().apply_transform(data)

    def _transform_impl(self, x, metadata=None):
        return np.transpose(x, (3, 2, 1, 0))


class TZYX_To_XYZT(BaseTransform):
    def __init__(self, keys=['image', 'label']):
        super().__init__(keys)

    def __call__(self, data):
        return super().apply_transform(data)

    def _transform_impl(self, x, metadata=None):
        return np.transpose(x, (3, 2, 1, 0))


class Flow_T3XYZ_To_T3ZYX(BaseTransform):
    def __init__(self, keys=['forward_flow', 'backward_flow']):
        super().__init__(keys)

    def __call__(self, data):
        return super().apply_transform(data)

    def _transform_impl(self, x, metadata=None):
        return np.transpose(x, (0, 1, 4, 3, 2))


class FlowChannelToLastDim(BaseTransform):
    """
    Expects optical flow with shape (t,3,z,y,x)
    """

    def __init__(self, keys=['forward_flow', 'backward_flow']):
        super().__init__(keys)

    def __call__(self, data):
        return super().apply_transform(data)

    def _transform_impl(self, x, metadata=None):
        return np.transpose(x, (0, 2, 3, 4, 1))


class Pad(BaseTransform):
    def __init__(self, desired_dims=(40, 1, 80, 80, 80), keys=['image']):
        super().__init__(keys)
        self.desired_dims = np.array(desired_dims)

    def __call__(self, data):
        return super().apply_transform(data)

    def _transform_impl(self, x, metadata=None):
        shape = np.array(x.shape)
        difs = self.desired_dims - shape
        return np.pad(x, ((0, difs[0]), (0, difs[1]), (0, difs[2]), (0, difs[3]), (0, difs[4])))


class ExtremaPoints(BaseTransform):
    """
    This transform will add 4 keys to the data dict correspondig with the initial and
    final masks (mi, mf) and times (ti, tf). This functions expect 5d labels with shape (nt, ch, d1, d2, d3)
    """

    def __init__(self, keys=['label']):
        super().__init__(keys)

    def __call__(self, data):
        label_key = self.keys[0]
        es = data['es']
        ed = data['ed']
        ts = data[label_key].shape[0]
        test = ts > 2

        ti = min(es, ed)
        tf = max(es, ed)
        if not test:
            if ti == es:
                mi, mf = data[label_key][0], data[label_key][1]
            else:
                mi, mf = data[label_key][1], data[label_key][0]
        else:
            if ti == es:
                mi, mf = data[label_key][es], data[label_key][ed]
            else:
                mi, mf = data[label_key][ed], data[label_key][es]
        data['mi'] = np.expand_dims(mi, 0)
        data['mf'] = np.expand_dims(mf, 0)
        data['ti'] = ti
        data['tf'] = tf
        return data

    def _transform_impl(self, x, metadata=None):
        return x

# For 2D data


class RandomRotate2D(BaseTransform):
    def __init__(self, p, angle_range=(-30, 30), keys=['image', 'label'], label_key='label'):
        super().__init__(keys)
        self.p = p
        self.angle_range = angle_range
        self.label_key = label_key

    def __call__(self, data):
        if np.random.rand() < self.p:
            self.angle = np.random.randint(self.angle_range[0], self.angle_range[1])
            data = super().apply_transform(data)
        return data

    def _transform_impl(self, x, metadata=None):
        order = 0 if self.cur_key == self.label_key else 3
        x_rot = ndimage.rotate(x, self.angle, axes=(4, 3), order=order, reshape=False)
        return x_rot


# Functions
import torch
import torch.nn.functional as F


def one_hot(x: torch.Tensor, n: int, argmax=False, dim=1) -> torch.Tensor:
    x = x.argmax(dim=dim) if argmax else x.squeeze(1).long()

    if x.ndim == 3:
        return F.one_hot(x, n).permute(0, 3, 1, 2).float()
    elif x.ndim == 4:
        return F.one_hot(x, n).permute(0, 4, 1, 2, 3).float()
    else:
        raise ValueError('Only 4D and 5D tensors are supported in one_hot')
