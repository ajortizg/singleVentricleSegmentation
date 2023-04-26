import os
import os.path as osp
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../../'))
sys.path.append(ROOT_DIR)
from datasets.segmentation_dataset import SegmentationDataset
import segmentation.transforms as T
import utilities.path_utils as path_utils


def save_overlaped_img_mask(img2d, mask2d, file, alpha):
    plt.figure()
    plt.imshow(img2d, cmap="gray", interpolation='none')
    if mask2d is not None:
        plt.imshow(mask2d, cmap='jet', alpha=alpha, interpolation='none')
    plt.axis('off')
    plt.savefig(file, dpi=100)
    plt.close('all')


transforms = T.Compose([
    T.AddNLeadingDims(n=3),
    T.RandomFlip(0.5, axis=3, keys=['image', 'label']),
    T.RandomFlip(0.5, axis=4, keys=['image', 'label']),
    T.RandomRotate2D(1.0, (0, 360)),
    T.ElasticDeformation(1.0,(0.5,2.0),8,'constant','yx'),
    T.SimulateLowResolution(0.25, (0.5, 1.0)),
    T.GammaCorrection(0.3, (0.7, 1.5), retain_stats=True, invert_image=False),
    T.GammaCorrection(0.1, (0.7, 1.5), retain_stats=True, invert_image=True),
    T.MultiplicativeScaling(0.15, (0.75, 1.25)),
    T.AdditiveGaussianNoise(0.1, (0.0, 0.1)),
    T.ContrastAugmentation(0.15, (0.75, 1.25)),
    T.GaussialBlur(0.2, (0.5, 1.0)),
    T.RemoveNLeadingDims(n=3, keys=['image', 'label'])
])
ds = SegmentationDataset(root_dir='data/Dataset_SVD_crop_2d', mode='test', transforms=transforms)
save_dir = path_utils.create_save_dir('results', 'overlay2d')

pbar = tqdm(total=len(ds))
for i, data in enumerate(ds):
    img = data['image']
    label = data['label']

    save_overlaped_img_mask(img, label, osp.join(save_dir, f'image_{i}.png'), 0.3)
    pbar.update(1)
