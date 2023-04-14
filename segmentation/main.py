from typing import Tuple, Union, List
import os.path as osp
import sys
import configparser

import torch
import nibabel as nib
from nibabel import io_orientation
from tqdm import tqdm
import pandas as pd
from nibabel.processing import conform
import numpy as np

from monai.transforms import AsDiscrete

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
from utils import plots
from segmentation.dataset import SVDataset, SegmentationDataset
import segmentation.transforms as T


if __name__ == '__main__':

    transforms = T.Compose([
        T.AddChannelDim(keys=['image', 'label']),
        T.CXYZ_To_CZYX(keys=['image', 'label']),
        T.ToRAS(keys=['image', 'label']),
        # T.CropForeground(keys=['image', 'label'], label_key='label'),
        # T.Resize(1.0, (96, 96, 96), keys=['image', 'label']),
        # T.ElasticDeformation(1.0, (0.5, 2.0), 8, 'constant', 'yx')
        # T.RandomRotate(1.0, (0, 360), (0, 360), (0, 360), keys=['image', 'label'], label_key='label'),
        T.ToTensor(keys=['image', 'label'])
    ])

    ds = SegmentationDataset('data/nnUNet_raw/Dataset012_SVDraw', 'train', transforms)

    save_dir = plots.createSaveDirectory('results', 'TESTS')

    pbar = tqdm(total=len(ds))
    for i, data in enumerate(ds):
        img = data['image']
        label = data['label']

        plots.save_overlaped_img_mask(img.squeeze(0),
                                      label.squeeze(0),
                                      f'img_{i}.png',
                                      save_dir, th=0.5,
                                      alpha=0.3)
        pbar.update(1)

    # transforms = T.Compose([
    #     T.CropForeground(p=1.0, tol=10),
    #     T.Resize(p=1.0, size=(96, 96, 96)),
    #     # T.ZScoreNormalization(p=1.0),
    #     # T.QuadraticNormalization(p=1.0),
    #     T.MinMaxNormalization(p=1.0),
    #     # T.RandomRotate(p=1.0, range_z=(0, 360), boundary='border'),
    #     # T.RandomVerticalFlip(p=1.0),
    #     # T.RandomHorizontalFlip(p=1.0),
    #     # T.RandomDepthFlip(p=0.5),
    #     T.ElasticDeformation(1.0, (1.4, 1.5), 8, 'nearest', False, 'yx', 1),
    #     T.BinarizeMasks(th=0.5),
    #     T.ToTensor(add_ch_dim=False)
    #     # T.ElasticDeformati),
    #     # T6.GammaScaling(P['gamma_scaling_prob'], P['gamma_scaling_range']),
    #     # T6.MutiplicativeScaling(P['mult_scaling_prob'], P['gamma_scaling_range'], P['clip_interval']),
    #     # T6.AdditiveScaling(P['add_scaling_prob'], P['add_scaling_mean'], P['add_scaling_std'], P['clip_interval']),
    #     # T6.AdditiveGaussianNoise(P['noise_prob'], P['noise_mu'], P['noise_std'], P['clip_interval']),
    # ])

    # ds = SVDSegmentation('data/svd_segmentation', mode='test', transforms=transforms)
    # red = [0, 0, 1]
    # saving_transforms = T.Compose([T.ToArray(), T.Resize(p=1.0, size=(16, 96, 96)), T.ToTensor(add_ch_dim=False)])
    # for data in ds:
    #     data = saving_transforms(data)

    #     NT = data['img'].shape[-1]
    #     patient_dir = plots.createSubDirectory(save_dir, data['patient'])
    #     for t in range(NT):
    #         img = data['img'][..., t]
    #         mask = data['mask'][..., t]

    #         plots.save_img_masks(img, [mask], f'img_{t}.png', patient_dir, 0.5, [0.2], [red])
