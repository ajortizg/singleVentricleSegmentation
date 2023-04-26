import os
import os.path as osp
import sys
import shutil
import configparser

from tqdm import tqdm
from natsort import natsorted
import nibabel as nib
import numpy as np

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../../'))
sys.path.append(ROOT_DIR)
from utilities import path_utils as fpu
from thirdparty.nnUNet.nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json


def read_configfile(file_dir):
    with open(file_dir, 'r') as f:
        config_string = '[dummy_section]\n' + f.read()
    config = configparser.ConfigParser()
    config.read_string(config_string)
    return config['dummy_section']


if __name__ == '__main__':
    split_dir = osp.join('data/acdc', 'training')
    patients_list_dir = [osp.join(split_dir, p) for p in natsorted(os.listdir(split_dir))]

    save_dir = fpu.create_save_dir('results', 'acdc_2d_raw')
    train_imgs_dir = fpu.create_sub_dir(save_dir, 'imagesTr')
    train_labels_dir = fpu.create_sub_dir(save_dir, 'labelsTr')

    k = 1
    for patient_dir in tqdm(patients_list_dir):
        info_cfg = read_configfile(osp.join(patient_dir, 'Info.cfg'))
        ed = info_cfg.getint('ED')
        es = info_cfg.getint('ES')
        patient_name = osp.basename(patient_dir)

        times = [ed, es]
        for t in times:
            img_nib = nib.load(osp.join(patient_dir, '{}_frame{:02d}.nii.gz'.format(patient_name, t)))
            img3d = img_nib.get_fdata()

            label_nib = nib.load(osp.join(patient_dir, '{}_frame{:02d}_gt.nii.gz'.format(patient_name, t)))
            label3d = label_nib.get_fdata()

            for z in range(img3d.shape[-1]):
                img2d = img3d[..., z]
                label2d = label3d[..., z]
                np.save(osp.join(train_imgs_dir, 'ACDC_{:04d}_0001.npy'.format(k)), img2d)
                np.save(osp.join(train_labels_dir, 'ACDC_{:04d}.npy'.format(k)), label2d)
                k += 1

    generate_dataset_json(output_folder=save_dir,
                          channel_names={1: 'T1'},
                          labels={'background': 0, 'RV': 1, 'MY': 2, 'LV': 3},
                          num_training_cases=k - 1,
                          file_ending='.npy',
                          dataset_name='ACDC')