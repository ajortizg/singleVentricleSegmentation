import os
import sys
import os.path as osp
import nibabel as nib
from glob import glob
import numpy as np
import torch
from monai.metrics.meandice import compute_dice
from monai.metrics.hausdorff_distance import compute_hausdorff_distance

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
import segmentation.transforms as T

patient = 'Adolescent_53'
ti, tf = 14, 28
pred_labels_dir = f'results/nnunet_test/{patient}'
true_labels_dir = f'results/svd_nnunet_raw_20230422-115342/test_nnunet_labels/{patient}'

pred_files = sorted(glob(osp.join(pred_labels_dir, '*.nii.gz')))[ti:tf + 1]
true_files = sorted(glob(osp.join(true_labels_dir, '*.nii.gz')))[ti:tf + 1]


for predf, truef in zip(pred_files, true_files):
    y_pred = nib.load(predf).get_fdata()
    y_true = nib.load(truef).get_fdata()

    print(compute_dice(torch.from_numpy(y_pred), torch.from_numpy(y_true)))
