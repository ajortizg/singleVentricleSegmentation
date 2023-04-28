import os.path as osp
import sys
import shutil
import configparser

import pandas as pd
from tqdm import tqdm
import nibabel as nib
import numpy as np

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../../'))
sys.path.append(ROOT_DIR)
import utilities.path_utils as path_utils
from utilities import stuff
from datasets.preprocessing.transforms import get_transforms
import datasets.preprocessing.viz as viz
import datasets.preprocessing.nib_utils as nib_utils
from thirdparty.nnUNet.nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json


def preprocess_train_split(data, transforms, src_labels_dir, dst_imgs_dir, dst_labels_dir):
    patient_name = data['patient']
    dst_labels_patient_dir = path_utils.create_sub_dir(dst_labels_dir, patient_name)

    nib_label_es = nib.load(osp.join(src_labels_dir, patient_name, patient_name + '_Systole_Labelmap.nii'))
    nib_label_ed = nib.load(osp.join(src_labels_dir, patient_name, patient_name + '_Diastole_Labelmap.nii'))
    label_xyzt = np.stack((nib_label_es.get_fdata(), nib_label_ed.get_fdata()), axis=3)
    data['label'] = label_xyzt
    data['label_meta'] = {'affine': nib_label_ed.affine}

    # Save transformed data
    data = transforms(data)

    np.save(osp.join(dst_imgs_dir, f'{patient_name}.npy'), data['image'])
    np.save(osp.join(dst_labels_patient_dir, f'{patient_name}_Systole_Labelmap.npy'), data['label'][..., 0])
    np.save(osp.join(dst_labels_patient_dir, f'{patient_name}_Diastole_Labelmap.npy'), data['label'][..., 1])

    nib_utils.save_np_to_nifty(data['image'], dst_imgs_dir, f'{patient_name}.nii.gz', data['image_nib'].header)
    nib_utils.save_np_to_nifty(data['label'][..., 0], dst_labels_patient_dir, f'{patient_name}_Systole_Labelmap.nii.gz', nib_label_es.header)
    nib_utils.save_np_to_nifty(data['label'][..., 1], dst_labels_patient_dir, f'{patient_name}_Diastole_Labelmap.nii.gz', nib_label_ed.header)
    return data


def preprocess_test_split(data, transforms, src_labels_dir, dst_imgs_dir, dst_labels_dir):
    patient_name = data['patient']
    img_xyzt = data['image']
    dst_labels_patient_dir = path_utils.create_sub_dir(dst_labels_dir, patient_name)

    label_xyzt = np.empty(shape=img_xyzt.shape, dtype=img_xyzt.dtype)
    for t in range(label_xyzt.shape[3]):
        nib_label = nib.load(osp.join(src_labels_dir, patient_name, f'{patient_name}_{t}_Labelmap.nii'))
        label_xyzt[..., t] = nib_label.get_fdata()
    data['label'] = label_xyzt
    data['label_meta'] = {'affine': nib_label.affine}

    # Save transformed data
    data = transforms(data)

    np.save(osp.join(dst_imgs_dir, f'{patient_name}.npy'), data['image'])
    np.save(osp.join(dst_labels_patient_dir, f'{patient_name}_Labelmap.npy'), data['label'])
    np.save(osp.join(dst_labels_patient_dir, f'{patient_name}_Systole_Labelmap.npy'), data['label'][..., data['es']])
    np.save(osp.join(dst_labels_patient_dir, f'{patient_name}_Diastole_Labelmap.npy'), data['label'][..., data['ed']])

    nib_utils.save_np_to_nifty(data['image'], dst_imgs_dir, f'{patient_name}.nii.gz', data['image_nib'].header)
    # for t in range(data['label'].shape[3]):
    #     nib_utils.save_np_to_nifty(data['label'][..., t], dst_labels_patient_dir,
    #                                f'{patient_name}_{t}_Labelmap.nii.gz', data['label_nib'].affine, data['label_nib'].header)

    nib_utils.save_np_to_nifty(data['label'], dst_labels_patient_dir,
                               f'{patient_name}_Labelmap.nii.gz', nib_label.header, zooms=nib_label.header.get_zooms() + (1.0,))
    nib_utils.save_np_to_nifty(data['label'][..., data['es']], dst_labels_patient_dir, f'{patient_name}_Systole_Labelmap.nii.gz', nib_label.header)
    nib_utils.save_np_to_nifty(data['label'][..., data['ed']], dst_labels_patient_dir, f'{patient_name}_Diastole_Labelmap.nii.gz', nib_label.header)
    return data


if __name__ == '__main__':
    # Read configuration file
    cfg = configparser.ConfigParser()
    cfg.read('parser/preprocessing.ini')
    data_cfg = cfg['DATA']
    transforms_cfg = cfg['TRANSFORMS']
    viz_cfg = cfg['VIZ']

    # Source paths
    root_dir = data_cfg['root_dir']
    src_imgs_dir = osp.join(root_dir, 'NIFTI_4D_Datasets')
    src_labels_dir = osp.join(root_dir, 'NIFTI_Single_Ventricle_Segmentations')
    df = pd.read_excel(osp.join(root_dir, 'Segmentation_volumes.xlsx'))

    # Output paths
    save_dir = path_utils.create_save_dir(data_cfg['output_dir'], f'{osp.basename(root_dir)}')
    stuff.save_config(cfg, save_dir)
    dst_imgs_dir = path_utils.create_sub_dir(save_dir, data_cfg['dst_images_folder'])
    dst_labels_dir = path_utils.create_sub_dir(save_dir, data_cfg['dst_labels_folder'])

    # Preprocessing transforms
    transforms = get_transforms(transforms_cfg)
    stuff.save_transforms_to_json(transforms, osp.join(save_dir, 'transforms.json'))

    pbar = tqdm(total=len(df))
    num_training_casses = 0
    for i in range(len(df)):
        df_row = df.iloc[[i]]
        patient_name = df_row.loc[i, 'Name']
        es = df_row.loc[i, 'Systole']
        ed = df_row.loc[i, 'Diastole']
        split = df_row.loc[i, 'Split']

        pbar.set_postfix_str(f'{split} - {patient_name}')

        # Read image data
        nib_img = nib.load(osp.join(src_imgs_dir, patient_name + '.nii.gz'))
        img_xyzt = nib_img.get_fdata()
        data = {'es': es, 'ed': ed, 'patient': patient_name,
                'image_nib': nib_img, 'image': img_xyzt, 'image_meta': {'affine': nib_img.affine}}

        # Process and save data
        if split == 'train':
            data = preprocess_train_split(data, transforms, src_labels_dir, dst_imgs_dir, dst_labels_dir)
            num_training_casses += 1
        elif split == 'test':
            data = preprocess_test_split(data, transforms, src_labels_dir, dst_imgs_dir, dst_labels_dir)
        else:
            # This should not happen
            raise ValueError(f'{split} is not a valid split')

        # Save images
        if viz_cfg.getboolean('save_png'):
            viz_dir = path_utils.create_sub_dir(osp.join(save_dir, viz_cfg['viz_dir']), patient_name)
            viz.save_png(data, viz_cfg=viz_cfg, save_dir=viz_dir)
            if viz_cfg.getboolean('save_gif'):
                viz.save_gif(viz_dir, viz_cfg.getint('dur_gif'))
        pbar.update(1)

    shutil.copy(osp.join(root_dir, 'Segmentation_volumes.xlsx'), osp.join(save_dir, 'info.xlsx'))
    generate_dataset_json(output_folder=save_dir,
                          channel_names={1: 'T1'},
                          labels={'background': 0, 'SV': 1},
                          num_training_cases=num_training_casses,
                          file_ending='.npy',
                          dataset_name='SVD',
                          overwrite_image_reader_writer='None')
