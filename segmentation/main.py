from typing import Tuple, Union, List
import os.path as osp
import sys
import configparser
import json

import torch
import nibabel as nib
from nibabel import io_orientation
from tqdm import tqdm
import pandas as pd
from nibabel.processing import conform
import numpy as np


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
        T.CropForeground(keys=['image', 'label'], label_key='label'),
        # T.RandomFlip(0.5, 1),
        # T.RandomFlip(0.5, 2),
        # T.RandomFlip(0.5, 3),
        T.Resize(1.0, (96, 96, 96), keys=['image', 'label']),
        # T.CZYX_To_CXYZ(keys=['image', 'label']),
        T.SpatialTransform(p_rot=1.0, p_scale=1.0, p_ed=0.0),
        # T.CXYZ_To_CZYX(keys=['image', 'label']),
        # T.ElasticDeformation(1.0, (0.5, 2.0), 8, 'constant', 'yx'),
        # T.RandomRotate(1.0, (-30, 30), (-30, 30), (-30, 30), keys=['image', 'label'], label_key='label'),
        # T.Resize(1.0, (16, 96, 96)),
        # T.GammaCorrection(1.0, (0.5, 1.7), False, False, keys=['image']),
        # T.GaussialBlur(0.2, sigma_range=(0.5, 1.), keys=['image']),
        # T.MultiplicativeScaling(0.15, (0.75, 1.25), keys=['image']),
        # T.ContrastAugmentation(0.15, (0.75, 1.25), keys=['image']),
        T.Resize(1.0, (16, 96, 96), keys=['image', 'label']),
        T.ToTensor(keys=['image', 'label'])
    ])

    ds = SegmentationDataset('data/nnUNet_raw/Dataset012_SVDraw', 'train', transforms)

    save_dir = plots.createSaveDirectory('results', 'TESTS')
    plots.save_transforms_to_json(transforms, osp.join(save_dir, 'transforms.json'))

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
