import os.path as osp
import sys

import pandas as pd

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
from utils import plots
from preprocessing.transforms import get_transforms

if __name__ == '__main__':
    # Source paths
    root_dir = 'data/singleVentricleData'
    imgs_folder = 'NIFTI_4D_Datasets'
    labels_folder = 'NIFTI_Single_Ventricle_Segmentations'
    excel_filename = 'Segmentation_volumes.xlsx'

    src_imgs_dir = osp.join(root_dir, imgs_folder)
    src_labels_dir = osp.join(root_dir, labels_folder)

    # Split in training and testing
    df = pd.read_excel(osp.join(root_dir, excel_filename))
    df_train = df[(df['Split'] == 'train') | (df['Split'] == 'val')]
    df_train.reset_index(inplace=True, drop=True)
    df_test = df[df['Split'] == 'test']
    df_test.reset_index(inplace=True, drop=True)

    # Output paths
    results_folder = 'results'
    save_dir = plots.createSaveDirectory(results_folder, 'singleVentricleData_prep')
