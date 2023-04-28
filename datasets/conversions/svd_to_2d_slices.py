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


def preprocess(patient, es, ed, transforms, imgs_dir, labels_dir, is_test):
    # Read images
    nib_img = nib.load(osp.join(imgs_dir, patient + '.nii.gz'))
    img_xyzt = nib_img.get_fdata()

    # Read labels
    if is_test:
        label_xyzt = np.zeros(img_xyzt.shape, dtype=img_xyzt.dtype)
        for t in range(img_xyzt.shape[-1]):
            nib_label = nib.load(osp.join(labels_dir, patient_name, patient_name + f'_{t}_Labelmap.nii'))
            label_xyzt[..., t] = nib_label.get_fdata()
            label_meta = {'affine': nib_label.affine}
    else:
        nib_label_es = nib.load(osp.join(labels_dir, patient, patient + '_Systole_Labelmap.nii'))
        nib_label_ed = nib.load(osp.join(labels_dir, patient, patient + '_Diastole_Labelmap.nii'))
        label_xyzt = np.stack((nib_label_es.get_fdata(), nib_label_ed.get_fdata()), axis=3)
        label_meta = {'affine': nib_label_ed.affine}

    # Save in dictionary for preprocessing
    data = {}
    data['es'] = es
    data['ed'] = ed
    data['image'] = img_xyzt
    data['image_meta'] = {'affine': nib_img.affine}
    data['label'] = label_xyzt
    data['label_meta'] = label_meta

    data = transforms(data)
    return data


if __name__ == '__main__':
    # Source paths
    root_dir = 'data/singleVentricleData'
    imgs_dir = osp.join(root_dir, 'NIFTI_4D_Datasets')
    labels_dir = osp.join(root_dir, 'NIFTI_Single_Ventricle_Segmentations')

    # Split in training and testing
    df = pd.read_excel(osp.join(root_dir, 'Segmentation_volumes.xlsx'))
    df_train = df[(df['Split'] == 'train')]
    df_train.reset_index(inplace=True, drop=True)
    df_test = df[df['Split'] == 'test']
    df_test.reset_index(inplace=True, drop=True)

    # Output paths
    save_dir = fpu.create_save_dir('results', 'svd_2d_prep')
    train_imgs_dir = fpu.create_sub_dir(save_dir, 'imagesTr')
    train_labels_dir = fpu.create_sub_dir(save_dir, 'labelsTr')
    test_imgs_dir = fpu.create_sub_dir(save_dir, 'imagesTs')
    test_labels_dir = fpu.create_sub_dir(save_dir, 'labelsTs')

    transforms = T.Compose([
        T.XYZT_To_TZYX(keys=['image', 'label']),
        T.AddDimAt(axis=1, keys=['image', 'label']),
        T.ToRAS(keys=['image', 'label']),
        T.CropForeground(tol=10, keys=['image', 'label'], label_key='label'),
        T.QuadraticNormalization(95, True, ['image'], 'label'),
        T.Resize(p=1, new_shape=(-1, 96, 96), keys=['image', 'label'], label_key='label'),
        T.RemoveDimAt(axis=1, keys=['image', 'label']),
        T.TZYX_To_XYZT(keys=['image', 'label'])
    ])

    # Save training data
    k = 1
    for idx in tqdm(range(len(df_train))):
        df_row = df_train.iloc[[idx]]
        patient_name = df_row.loc[idx, 'Name']
        es = df_row.loc[idx, 'Systole']
        ed = df_row.loc[idx, 'Diastole']

        data = preprocess(patient_name, es, ed, transforms, imgs_dir, labels_dir, is_test=False)

        for j, heart_cycle in enumerate([es, ed]):
            img3d = data['image'][..., heart_cycle]
            label3d = data['label'][..., j]

            for z in range(img3d.shape[-1]):
                img2d = img3d[..., z]
                label2d = label3d[..., z]
                if np.count_nonzero(label2d):
                    np.save(osp.join(train_imgs_dir, 'SVD_{:04d}_0001.npy'.format(k)), img2d)
                    np.save(osp.join(train_labels_dir, 'SVD_{:04d}.npy'.format(k)), label2d)
                    k += 1

    generate_dataset_json(output_folder=save_dir,
                          channel_names={1: 'T1'},
                          labels={'background': 0, 'SV': 1},
                          num_training_cases=k - 1,
                          file_ending='.npy',
                          dataset_name='SVD',
                          overwrite_image_reader_writer='None')

    # Save test data
    k = 1
    json_dict = {}
    for idx in tqdm(range(len(df_test))):
        df_row = df_test.iloc[[idx]]
        patient_name = df_row.loc[idx, 'Name']
        es = int(df_row.loc[idx, 'Systole'])
        ed = int(df_row.loc[idx, 'Diastole'])

        data = preprocess(patient_name, es, ed, transforms, imgs_dir, labels_dir, is_test=True)

        for t in range(data['image'].shape[-1]):
            img3d = data['image'][..., t]
            label3d = data['label'][..., t]

            for z in range(img3d.shape[-1]):
                img2d = img3d[..., z]
                label2d = label3d[..., z]
                if np.count_nonzero(label2d):
                    np.save(osp.join(test_imgs_dir, 'SVD_{:04d}_0001.npy'.format(k)), img2d)
                    np.save(osp.join(test_labels_dir, 'SVD_{:04d}.npy'.format(k)), label2d)
                    json_dict['SVD_{:04d}_0001.npy'.format(k)] = {'patient': patient_name, 'es': es, 'ed': ed}
                    k += 1

    # shutil.copy(osp.join(root_dir, 'Segmentation_volumes.xlsx'), osp.join(save_dir, 'info.xlsx'))
    fpu.save_json(json_dict, osp.join(save_dir, 'test.json'))
