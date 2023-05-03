from configparser import ConfigParser
import os.path as osp
import os
import sys
import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../../'))
sys.path.append(ROOT_DIR)
import utilities.path_utils as path_utils
from utilities import stuff
from datasets.preprocessing.transforms import get_transforms
import datasets.preprocessing.viz as viz
from thirdparty.nnUNet.nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json


def read_info(file_dir):
    with open(file_dir, 'r') as f:
        config_string = '[dummy_section]\n' + f.read()
    config = ConfigParser()
    config.read_string(config_string)
    return config['dummy_section']


if __name__ == '__main__':
    cfg = ConfigParser()
    cfg.read('parser/preprocessing.ini')
    data = cfg['DATA']
    vizcfg = cfg['VIZ']

    # Source paths
    root_dir = data['root_dir']

    # Output paths
    save_dir = path_utils.create_save_dir(data['output_dir'], f'{osp.basename(root_dir)}')
    stuff.save_config(cfg, save_dir)
    dst_imgs_dir = path_utils.create_sub_dir(save_dir, data['dst_images_folder'])
    dst_labels_dir = path_utils.create_sub_dir(save_dir, data['dst_labels_folder'])
    metadata = pd.DataFrame(columns=['Name', 'Systole', 'Diastole', 'Group', 'Height', 'NbFrame', 'Weight', 'Split'])

    # Preprocessing transforms
    transforms = get_transforms(cfg['TRANSFORMS'])
    stuff.save_transforms_to_json(transforms, osp.join(save_dir, 'transforms.json'))

    train_patients_dir = [osp.join(osp.join(root_dir, 'training'), p) for p in sorted(os.listdir(osp.join(root_dir, 'training')))]
    num_training_casses = 0
    for patient_dir in tqdm(train_patients_dir):
        info = read_info(osp.join(patient_dir, 'Info.cfg'))
        ed = info.getint('ED')
        es = info.getint('ES')
        patient_name = osp.basename(patient_dir)

        # Read data
        nib_img4d = nib.load(osp.join(patient_dir, f'{patient_name}_4d.nii.gz'))
        nib_label_ed = nib.load(osp.join(patient_dir, '{}_frame{:02d}_gt.nii.gz'.format(patient_name, ed)))
        nib_label_es = nib.load(osp.join(patient_dir, '{}_frame{:02d}_gt.nii.gz'.format(patient_name, es)))

        # Preprocess data
        data = {
            'es': es,
            'ed': ed,
            'patient': patient_name,
            'image': nib_img4d.get_fdata(),
            'image_meta': {'affine': nib_img4d.affine},
            'label': np.stack((nib_label_es.get_fdata(), nib_label_ed.get_fdata()), axis=3),
            'label_meta': {'affine': nib_label_es.affine}
        }
        data = transforms(data)

        # Save new data
        dst_labels_patient_dir = path_utils.create_sub_dir(dst_labels_dir, patient_name)
        np.save(osp.join(dst_imgs_dir, f'{patient_name}.npy'), data['image'])
        np.save(osp.join(dst_labels_patient_dir, f'{patient_name}_Systole_Labelmap.npy'), data['label'][..., 0])
        np.save(osp.join(dst_labels_patient_dir, f'{patient_name}_Diastole_Labelmap.npy'), data['label'][..., 1])
        metadata.loc[len(metadata)] = [patient_name, es, ed, info['Group'], info.getfloat(
            'Height'), info.getint('NbFrame'), info.getfloat('Weight'), 'train']
        num_training_casses += 1

        # Save images
        if vizcfg.getboolean('save_png'):
            viz_dir = path_utils.create_sub_dir(osp.join(save_dir, vizcfg['viz_dir']), patient_name)
            viz.save_png(data, viz_cfg=vizcfg, save_dir=viz_dir)
            if vizcfg.getboolean('save_gif'):
                viz.save_gif(viz_dir, vizcfg.getint('dur_gif'))

    metadata.to_excel(osp.join(save_dir, 'info.xlsx'))
    generate_dataset_json(output_folder=save_dir,
                          channel_names={1: 'T1'},
                          labels={'Background': 0, 'RV': 1, 'MY': 2, 'LV': 3},
                          num_training_cases=num_training_casses,
                          file_ending='.npy',
                          dataset_name='ACDC',
                          overwrite_image_reader_writer='None')
