import sys
from tqdm import tqdm
import os
import pandas as pd
import glob
import configparser
import shutil
import os.path as osp
import shutil

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../../../'))
sys.path.append(ROOT_DIR)
from utilities import path_utils, stuff
from deprecated.dataset.singleVentricleDataset import SingleVentricleDataset


def copy(patient_name, src_img4d_dir, dst_img4d_dir, dst_segmentations_dir, dset):
    shutil.copy(src_img4d_dir, dst_img4d_dir)
    src_segmentation_files = glob.glob(osp.sep.join([dset.segmentations_path, patient_name, '*.nii.gz']))
    dst_segmentations_dir = path_utils.create_sub_dir(dst_segmentations_dir, patient_name)
    [shutil.copy(f, dst_segmentations_dir) for f in src_segmentation_files]
    return dset.get_row(patient_name)


if __name__ == "__main__":
    # load config parser
    config = configparser.ConfigParser()
    config.read('parser/deprecated/configPreprocessing.ini')

    # create save directory
    saveDir = path_utils.create_save_dir(config.get('DATA', 'OUTPUT_PATH'), "preprocessing_split")
    logger = stuff.create_logger(saveDir)
    logger.info('Preprocessing data: split train, validation and test dataset')
    logger.info(f'Save directory: {saveDir}')

    stuff.save_config(config, saveDir, 'config.ini')

    dset = SingleVentricleDataset(config)

    # load specific patients for validation and testing
    val_patients = set(config.get('SPLIT', 'validation_patients').replace('{', '').replace('}', '').replace('\n', '').split(','))
    test_patients = set(config.get('SPLIT', 'test_patients').replace('{', '').replace('}', '').replace('\n', '').split(','))

    # Paths for training dataset
    save_dir_train = path_utils.create_sub_dir(saveDir, 'train')
    img4d_dir_train = path_utils.create_sub_dir(save_dir_train, dset.volumes_subdir_path)
    segmentations_dir_train = path_utils.create_sub_dir(save_dir_train, dset.segmentations_subdir_path)

    # Paths for validation dataset
    save_dir_val = path_utils.create_sub_dir(saveDir, 'val')
    img4d_dir_val = path_utils.create_sub_dir(save_dir_val, dset.volumes_subdir_path)
    segmentations_dir_val = path_utils.create_sub_dir(save_dir_val, dset.segmentations_subdir_path)

    # Paths for test dataset
    save_dir_test = path_utils.create_sub_dir(saveDir, 'test')
    img4d_dir_test = path_utils.create_sub_dir(save_dir_test, dset.volumes_subdir_path)
    segmentations_dir_test = path_utils.create_sub_dir(save_dir_test, dset.segmentations_subdir_path)

    imgs4d_dir_full = glob.glob(osp.join(dset.volumes_path, '*.nii.gz'))
    logger.info(f'Found {len(imgs4d_dir_full)} .nii.gz files')
    pbar = tqdm(total=len(dset))
    df_val = pd.DataFrame()
    df_train = pd.DataFrame()
    df_test = pd.DataFrame()

    for src_img4d_dir in imgs4d_dir_full:
        patient_name = src_img4d_dir.split('/')[-1].split('.')[0]

        if patient_name in val_patients:
            row = copy(patient_name, src_img4d_dir, img4d_dir_val, segmentations_dir_val, dset)
            df_val = pd.concat([df_val, row])

        elif patient_name in test_patients:
            row = copy(patient_name, src_img4d_dir, img4d_dir_test, segmentations_dir_test, dset)
            df_test = pd.concat([df_test, row])

        else:
            row = copy(patient_name, src_img4d_dir, img4d_dir_train, segmentations_dir_train, dset)
            df_train = pd.concat([df_train, row])

        pbar.update(1)
    pbar.close()

    output_df_file_train = osp.sep.join([save_dir_train, dset.segmentations_filename])
    df_train.to_excel(output_df_file_train, index=False)
    logger.info(f'Saved: {output_df_file_train}')

    output_df_file_val = osp.sep.join([save_dir_val, dset.segmentations_filename])
    df_val.to_excel(output_df_file_val, index=False)
    logger.info(f'Saved: {output_df_file_val}')

    output_df_file_test = osp.sep.join([save_dir_test, dset.segmentations_filename])
    df_test.to_excel(output_df_file_test, index=False)
    logger.info(f'Saved: {output_df_file_test}')
