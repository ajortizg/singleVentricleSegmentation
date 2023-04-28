import os.path as osp
import sys
import configparser
import json

import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
from monai import transforms

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
import utilities.path_utils as path_utils
# from utilities import stuff
from utilities import plots
from datasets.segmentation_dataset import SegmentationDataset
import segmentation.transforms as T


if __name__ == '__main__':
    # stuff.seeding(42)
    # transforms = T.Compose([
    #     T.AddNLeadingDims(n=2, keys=['image', 'label']),
    #     T.BCXYZ_To_BCZYX(keys=['image', 'label']),
    #     T.CropForeground(keys=['image', 'label'], label_key='label'),
    #     T.QuadraticNormalization(q2=99, use_label=True, keys=['image'], label_key='label'),
    #     T.Resize(1.0, (96, 96, 96), keys=['image', 'label']),

    #     # T.OneHotEncoding(n=2, keys=['label']),
    #     T.RemoveNLeadingDims(n=2, keys=['image', 'label']),
    #     T.ToTensor(keys=['image', 'label'])
    # ])

    ds = SegmentationDataset('data/nnUNet_raw/Dataset013_SVDraw', 'train', None)

    save_dir = path_utils.create_save_dir('results', 'SEG-TESTS')
    # fpu.save_transforms_to_json(transforms, osp.join(save_dir, 'transforms.json'))

    # transforms.Spacing()
    pbar = tqdm(total=len(ds))
    for i, data in enumerate(ds):
        img = data['image']
        label = data['label']

        print(data['image_meta']['pix_dim'])
        print(data['label_meta']['pix_dim'])

        print(img.shape, label.shape)

        # plots.save_overlaped_img_mask(img,
        #                               label,
        #                               f'img_{i}.png',
        #                               save_dir,
        #                               alpha=0.3)

        pbar.update(1)
