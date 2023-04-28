'''
Save the SV dataset in nnUNet format
The dasatet must be previously preprocessed with datasets/preprocessing/preprocess_svd.py
'''

import os
import os.path as osp
import sys
import shutil
import json

from tqdm import tqdm
import pandas as pd
import nibabel as nib
import numpy as np


ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../../'))
sys.path.append(ROOT_DIR)
from utilities import path_utils as fpu
from thirdparty.nnUNet.nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
import segmentation.transforms as T
from utilities import stuff


def save_np_to_nifty(x, save_dir, filename, affine, hdr_old, remove_last_zoom):
    hdr = nib.nifti1.Nifti1Header.from_header(hdr_old)
    hdr.set_data_shape(x.shape)
    hdr.set_qform(hdr_old.get_qform())
    hdr.set_sform(hdr_old.get_sform())
    if remove_last_zoom:
        hdr.set_zooms(hdr_old.get_zooms()[:-1])
    else:
        hdr.set_zooms(hdr_old.get_zooms() + (1.0,))
    nii_data = nib.Nifti1Image(x, affine=affine, header=hdr)
    nib.save(nii_data, osp.join(save_dir, filename))


if __name__ == '__main__':
    # Source paths
    root_dir = 'results/preprocessed/singleVentricleData_affineIdentity_hdr'
    imgs_dir = osp.join(root_dir, 'images')
    labels_dir = osp.join(root_dir, 'labels')

    # Split in training and testing
    df = pd.read_excel(osp.join(root_dir, 'info.xlsx'))
    df_train = df[(df['Split'] == 'train')]
    df_train.reset_index(inplace=True, drop=True)
    df_test = df[df['Split'] == 'test']
    df_test.reset_index(inplace=True, drop=True)

    # Output paths
    save_dir = fpu.create_save_dir('results', 'svd_nnunet_raw')
    train_imgs_dir = fpu.create_sub_dir(save_dir, 'imagesTr')
    train_labels_dir = fpu.create_sub_dir(save_dir, 'labelsTr')
    test_imgs_dir = fpu.create_sub_dir(save_dir, 'imagesTs')
    test_labels_dir = fpu.create_sub_dir(save_dir, 'labelsTs')

    # Save training data
    k = 1
    for i in tqdm(range(len(df_train))):
        df_row = df_train.iloc[[i]]
        patient_name = df_row.loc[i, 'Name']
        es = df_row.loc[i, 'Systole']
        ed = df_row.loc[i, 'Diastole']

        times = [ed, es]
        endings = ['_Diastole_Labelmap.nii.gz', '_Systole_Labelmap.nii.gz']
        nib_img = nib.load(osp.join(imgs_dir, patient_name + '.nii.gz'))

        for t, ending in zip(times, endings):
            img = nib_img.get_fdata()[..., t]
            label = nib.load(osp.join(labels_dir, patient_name, patient_name + ending))

            save_np_to_nifty(img, train_imgs_dir, 'SVD_{:03d}_0001.nii.gz'.format(k), nib_img.affine, nib_img.header, remove_last_zoom=True)
            nib.save(label, osp.join(train_labels_dir, 'SVD_{:03d}.nii.gz'.format(k)))
            k += 1

    generate_dataset_json(output_folder=save_dir,
                          channel_names={1: 'T1'},
                          labels={'background': 0, 'SV': 1},
                          num_training_cases=k - 1,
                          file_ending='.nii.gz',
                          dataset_name='SVD',
                          overwrite_image_reader_writer='NibabelIOWithReorient')

    # Save test data
    k = 1
    test_json_dict = {}
    for i in tqdm(range(len(df_test))):
        df_row = df_test.iloc[[i]]
        patient_name = df_row.loc[i, 'Name']
        es = int(df_row.loc[i, 'Systole'])
        ed = int(df_row.loc[i, 'Diastole'])

        nib_img = nib.load(osp.join(imgs_dir, patient_name + '.nii.gz'))
        img_xyzt = nib_img.get_fdata()

        nib_label = nib.load(osp.join(labels_dir, patient_name, patient_name + f'_Labelmap.nii.gz'))
        label_xyzt = nib_label.get_fdata()
        for t in range(img_xyzt.shape[-1]):
            save_np_to_nifty( img_xyzt[..., t], test_imgs_dir, 'SVD_{:03d}_0001.nii.gz'.format(k),nib_img.affine, nib_img.header, remove_last_zoom=True)
            save_np_to_nifty(label_xyzt[..., t], test_labels_dir, 'SVD_{:03d}.nii.gz'.format(k), nib_label.affine, nib_label.header, remove_last_zoom=True)
            test_json_dict['SVD_{:03d}_0001.nii.gz'.format(k)] = {'patient': patient_name, 'es': es, 'ed': ed}
            k += 1

    # shutil.copy(osp.join(root_dir, 'Segmentation_volumes.xlsx'), osp.join(save_dir, 'info.xlsx'))
    stuff.save_json(test_json_dict, osp.join(save_dir, 'test.json'))