import numpy as np
import torch
import torch.nn.functional as F
import elasticdeform as ed
import os
import os.path as osp
import sys
from abc import ABCMeta, abstractmethod

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)

# from segmentation.transforms import (
#     BaseTransform,
#     Compose,
#     MinMaxNormalization,
#     ZScoreNormalization,
#     QuadraticNormalization as _QuadraticNormalization,
#     ToRAS,
#     ToTensor,
#     CropForeground,
#     Resize as _Resize
# )

# All transforms assume images and labels with shape (T, Z, Y, X)
# and flow with shape (T, 3, Z, Y, X)

# Note: The data is aved as:
# imgs: xyzt
# mask: xyzt
# flow: t3xyz


# class XYZT_To_TZYX(BaseTransform):
#     def __init__(self, keys=['image', 'label']):
#         super().__init__(keys)

#     def __call__(self, data):
#         return super().apply_transform(data)

#     def _transform_impl(self, x, metadata=None):
#         return np.transpose(x, (3, 2, 1, 0))


# class Flow_T3XYZ_To_T3ZYX(BaseTransform):
#     def __init__(self, keys=['forward_flow', 'backward_flow']):
#         super().__init__(keys)

#     def __call__(self, data):
#         return super().apply_transform(data)

#     def _transform_impl(self, x, metadata=None):
#         return np.transpose(x, (0, 1, 4, 3, 2))


# class TZYX_To_XYZT(BaseTransform):
#     def __init__(self, keys=['image', 'label']):
#         super().__init__(keys)

#     def __call__(self, data):
#         return super().apply_transform(data)

#     def _transform_impl(self, x, metadata=None):
#         return np.transpose(x, (3, 2, 1, 0))


# class QuadraticNormalization(_QuadraticNormalization):
#     def __init__(self, q2=95, keys=['image'], label_key='label'):
#         super().__init__(q2, True, keys, label_key)

#     def __call__(self, data):
#         self.t = [data['es'], data['ed']]
#         self.label = data[self.label_key].copy()
#         if self.label.shape[0] > 2:  # test labels
#             self.label = self.label[self.t]
#         return super().apply_transform(data)

#     def _transform_impl(self, x, metadata=None):
#         per2 = np.percentile(x, self.q2)
#         x = np.clip(x, 0, per2)
#         avg = np.mean(x[self.t], where=self.label.astype('bool'))
#         # normalization n(I) = a I/sqrt(1+beta I**2)
#         norm_a = np.sqrt(per2 * per2 - avg * avg) / (np.sqrt(3) * per2 * avg)
#         norm_b = (per2 * per2 - 4. * avg * avg) / (3. * per2 * per2 * avg * avg)
#         x_norm = norm_a * x / np.sqrt(1 + norm_b * x**2)
#         return x_norm


# class Resize(_Resize):
#     def __init__(self, p, size, keys=['image', 'label'], label_key='label'):
#         super().__init__(p, size, keys, label_key)

#     def _transform_impl(self, x, metadata=None):
#         mode = 'nearest' if self.cur_key == self.label_key else 'trilinear'
#         x = torch.from_numpy(x).float().unsqueeze(dim=1)    # add channel dim
#         x = F.interpolate(x, size=self.size, mode=mode)
#         return x.squeeze(1).numpy()                         # remove channel dim and convert back to numpy
