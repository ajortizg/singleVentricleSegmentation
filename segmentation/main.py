import os.path as osp
import sys
import configparser
import json

import torch
from tqdm import tqdm
import pandas as pd
from nibabel.processing import conform
import numpy as np


ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
import utilities.file_paths_utils as fpu
# from utilities import stuff
from utilities import plots
from segmentation.dataset import SegmentationDataset
import segmentation.transforms as T


if __name__ == '__main__':
    # stuff.seeding(42)
    # transforms = T.Compose([
    #     T.AddNLeadingDims(n=2, keys=['image', 'label']),
    #     T.BCXYZ_To_BCZYX(keys=['image', 'label']),
    #     T.ToRAS(keys=['image', 'label']),
    #     T.CropForeground(keys=['image', 'label'], label_key='label'),
    #     T.QuadraticNormalization(q2=99, use_label=True, keys=['image'], label_key='label'),
    #     T.Resize(1.0, (96, 96, 96), keys=['image', 'label']),

    #     # T.RandomRotate(0.2, (-30, 30), (-30, 30), (-30, 30), keys=['image', 'label'], label_key='label'),
    #     # T.RandomScale(0.2, (0.7, 1.4), keys=['image', 'label'], label_key='label'),
    #     # T.RandomFlip(0.5, 2, keys=['image', 'label']),
    #     # T.RandomFlip(0.5, 3, keys=['image', 'label']),
    #     # T.RandomFlip(0.5, 4, keys=['image', 'label']),
    #     # T.ElasticDeformation(0.1, (0.5, 2.0), 8, 'constant', 'zyx'),

    #     # T.AdditiveGaussianNoise(0.1, sigma_range=(0.0, 0.1), mu=0.0, keys=['image']),
    #     T.GaussialBlur(1.0, sigma_range=(0.5, 1.), keys=['image']),
    #     # T.MultiplicativeScaling(0.15, (0.75, 1.25), keys=['image']),
    #     # T.ContrastAugmentation(0.15, (0.75, 1.25), keys=['image']),
    #     # T.GammaCorrection(0.1, (0.7, 1.5), True, True, keys=['image']),
    #     # T.GammaCorrection(0.3, (0.7, 1.5), False, True, keys=['image']),

    #     T.Resize(1.0, (16, 96, 96), keys=['image', 'label']),  # For vizualization

    #     # T.OneHotEncoding(n=2, keys=['label']),
    #     T.RemoveNLeadingDims(n=1, keys=['image', 'label']),
    #     T.ToTensor(keys=['image', 'label'])
    # ])

    test_transforms = T.Compose([
        T.XYZT_To_TZYX(keys=['image', 'label']),
        T.AddDimAt(axis=1, keys=['image', 'label']),
        T.ToRAS(keys=['image', 'label']),
        T.CropForeground(keys=['image', 'label'], label_key='label'),
        T.QuadraticNormalization(q2=99, use_label=True, keys=['image'], label_key='label'),
        T.Resize(1.0, (16, 96, 96), keys=['image', 'label']),
        # T.Resize(1.0, (16, 96, 96), keys=['image', 'label']),  # For vizualization
        T.OneHotEncoding(2, keys=['label']),
        # T.RemoveDimAt(axis=1, keys=['image']),
        T.ToTensor(keys=['image', 'label'])
    ])

    ds = SegmentationDataset('data/nnUNet_raw/Dataset012_SVDraw', 'test', test_transforms)

    save_dir = fpu.create_save_dir('results', 'SEG-TESTS')
    # fpu.save_transforms_to_json(transforms, osp.join(save_dir, 'transforms.json'))

    pbar = tqdm(total=len(ds))
    for i, data in enumerate(ds):
        img = data['image']
        label = data['label']

        print(img.shape)
        print(label.shape)
        print(data['es'], data['ed'])
        print(data['patient'])

        # for t in range(img.shape[0]):
        #     plots.save_overlaped_img_mask(img[t],
        #                                   label[t],
        #                                   f'img_{i}_{t}.png',
        #                                   save_dir,
        #                                   alpha=0.3)

        pbar.update(1)
