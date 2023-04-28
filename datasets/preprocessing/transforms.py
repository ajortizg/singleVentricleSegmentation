import os.path as osp
import sys

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../../'))
sys.path.append(ROOT_DIR)
import segmentation.transforms as T


def _str_to_tuple(x, dtype):
    return tuple(map(dtype, x.strip().replace('(', '').replace(')', '').split(',')))


def get_transforms(cfg):
    to_ras = cfg.getboolean('to_ras')
    crop_fg = cfg.getboolean('crop_fg')
    norm = cfg.getboolean('norm')
    q2 = cfg.getfloat('norm_q2')
    size = _str_to_tuple(cfg['size'], int)
    resize = -1 not in size

    transforms_list = [T.XYZT_To_TZYX(keys=['image', 'label']),
                       T.AddDimAt(axis=1, keys=['image', 'label'])]
    if to_ras:
        transforms_list.append(T.ToRAS(keys=['image', 'label']))
    if crop_fg:
        transforms_list.append(T.CropForeground(tol=10, keys=['image', 'label'], label_key='label'))
    if norm:
        transforms_list.append(T.QuadraticNormalization(q2=q2, keys=['image'], label_key='label'))
    if resize:
        transforms_list.append(T.Resize(p=1.0, new_shape=size, keys=['image', 'label'], label_key='label'))
    transforms_list.append(T.RemoveDimAt(axis=1, keys=['image', 'label']))
    transforms_list.append(T.TZYX_To_XYZT(keys=['image', 'label']))
    return T.Compose(transforms_list)


def get_viz_transforms(cfg):
    size = _str_to_tuple(cfg['size'], int)
    resize = -1 not in size

    transforms_list = [T.XYZT_To_TZYX(keys=['image', 'label']),
                       T.AddDimAt(axis=1, keys=['image', 'label'])]
    if resize:
        transforms_list.append(T.Resize(p=1.0, new_shape=size, keys=['image', 'label'], label_key='label'))
    transforms_list.append(T.RemoveDimAt(axis=1, keys=['image', 'label']))
    transforms_list.append(T.ToTensor())
    return T.Compose(transforms_list)
