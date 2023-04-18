import configparser
import os.path as osp
import sys
from tqdm import tqdm
import pandas as pd
import torch
import nibabel as nib
import numpy as np

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../..'))
sys.path.append(ROOT_DIR)
from utilities import plots
import utilities.quaternary_transforms as T
from dataset.singleVentricleDataset import SingleVentricleDataset, SingleVentriclePatient


def save_np_to_nifty(file: np.array, saveDir: str, fileName: str, hdr_old):
    # header
    hdr = nib.nifti1.Nifti1Header()
    hdr.set_data_shape(file.shape)
    hdr.set_qform(hdr_old.get_qform())
    hdr.set_sform(hdr_old.get_sform())
    hdr.set_zooms(hdr_old.get_zooms())
    # img
    ni_img = nib.Nifti1Image(file, affine=None, header=hdr)
    # save
    outputFile = osp.sep.join([saveDir, fileName])
    nib.save(ni_img, outputFile)


def save_data(patient: SingleVentriclePatient, saveDir4D: str, saveDirSegmentations: str):
    saveDirPatient = plots.createSubDirectory(saveDirSegmentations, patient.name)
    save_np_to_nifty(patient.nii_data_xyzt, saveDir4D, patient.name + ".nii.gz", patient.nii_header_xyzt)
    save_np_to_nifty(patient.nii_mask_diastole_xyz, saveDirPatient, patient.name + "_Diastole_Labelmap.nii", patient.hdr_mask_diastole)
    save_np_to_nifty(patient.nii_mask_systole_xyz, saveDirPatient, patient.name + "_Systole_Labelmap.nii", patient.hdr_mask_systole)

    # row = {'Name': patient.name, 'Systole': patient.tSystole, 'Diastole': patient.tDiastole}
    # df_patient = pd.DataFrame(row, index=[0])
    # return df_patient
    return patient.df_row


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('parser/configPreprocessing.ini')

    N = config.getint('DATA_AUGMENTATION', 'CREATE_NEW')

    # Flip
    flip_prob_z = config.getfloat('DATA_AUGMENTATION', 'FLIP_Z_PROB')
    flip_prob_y = config.getfloat('DATA_AUGMENTATION', 'FLIP_Y_PROB')
    flip_prob_x = config.getfloat('DATA_AUGMENTATION', 'FLIP_X_PROB')

    # Rotation
    prob_rot = config.getfloat('DATA_AUGMENTATION', 'ROT_PROB')
    rot_range_z = tuple(map(float, config.get('DATA_AUGMENTATION', 'ROT_Z_RANGE').split(',')))
    rot_range_y = tuple(map(float, config.get('DATA_AUGMENTATION', 'ROT_Y_RANGE').split(',')))
    rot_range_x = tuple(map(float, config.get('DATA_AUGMENTATION', 'ROT_X_RANGE').split(',')))
    rot_bdryMode = config.get('DATA_AUGMENTATION', 'ROT_DEFORM_BDRYMODE')

    # Elastic deformation
    ed_prob = config.getfloat('DATA_AUGMENTATION', 'ELASTIC_DEFORM_PROB')
    ed_grid = config.getint('DATA_AUGMENTATION', 'ELASTIC_DEFORM_GRID')
    ed_sigma_min = config.getfloat('DATA_AUGMENTATION', 'ELASTIC_DEFORM_SIGMA_MIN')
    ed_sigma_max = config.getfloat('DATA_AUGMENTATION', 'ELASTIC_DEFORM_SIGMA_MAX')
    ed_bdryMode = config.get('DATA_AUGMENTATION', 'ELASTIC_DEFORM_BDRYMODE')
    ed_usePrefilter = config.getboolean('DATA_AUGMENTATION', 'ELASTIC_DEFORM_USE_PREFILTER')

    transf_tern = T.ComposeTernary([
        T.RandomRotate(p=prob_rot, range_z=rot_range_z, range_y=rot_range_y, range_x=rot_range_x, total=None, boundary=rot_bdryMode),
        T.ElasticDeformation(p=ed_prob, sigma_range=(ed_sigma_min, ed_sigma_max), points=ed_grid, boundary=ed_bdryMode, prefilter=ed_usePrefilter)
    ])

    train_ds = SingleVentricleDataset(config, mode='train')
    val_ds = SingleVentricleDataset(config, mode='val')

    saveDir = plots.createSaveDirectory(config.get('DATA', 'OUTPUT_PATH'), 'preprocessing_da')
    logger = plots.create_logger(saveDir)

    logger.info("===========================================================")
    logger.info("Data augmentation")
    logger.info("===========================================================")
    logger.info(f'\t* Create: {N} new patients')
    logger.info(f'\t* Flip probs: {flip_prob_x}, {flip_prob_y}, {flip_prob_z}')
    logger.info(f'\t* Rot prob: {prob_rot}, with ranges: {rot_range_x}, {rot_range_y}, {rot_range_z}')
    logger.info(
        f'\t* Elastic def prob: {ed_prob}, grid: {ed_grid}, sigma: {ed_sigma_min,ed_sigma_max}, boundary: {ed_bdryMode}, prefilter: {ed_usePrefilter}')
    logger.info("===========================================================")
    logger.info('\nsave directory: ' + saveDir)

    # train paths
    saveDir_train = plots.createSubDirectory(saveDir, 'train')
    saveDir4D_train = plots.createSubDirectory(saveDir_train, train_ds.volumes_subdir_path)
    saveDirSegmentations_train = plots.createSubDirectory(saveDir_train, train_ds.segmentations_subdir_path)

    # validation paths
    saveDir_val = plots.createSubDirectory(saveDir, 'val')
    saveDir4D_val = plots.createSubDirectory(saveDir_val, val_ds.volumes_subdir_path)
    saveDirSegmentations_val = plots.createSubDirectory(saveDir_val, val_ds.segmentations_subdir_path)

    # save config file to save directory
    conifg_output = osp.sep.join([saveDir, "config.ini"])
    with open(conifg_output, 'w') as configfile:
        config.write(configfile)

    df_train = pd.DataFrame()
    df_val = pd.DataFrame()
    pbar = tqdm(total=len(train_ds) + (len(val_ds)))
    logger.info(f'Found {len(train_ds)} training patientes')
    logger.info(f'Found {len(val_ds)} validation patients')

    # Generate augmented training dataset
    for index in range(0, len(train_ds)):
        patient = train_ds[index]
        pbar.set_postfix_str(f'P: {patient.name}')
        logger.info(f'[Train]: {index} -> {patient.name}')

        # Save original data
        df = save_data(patient, saveDir4D_train, saveDirSegmentations_train)
        df_train = pd.concat([df_train, df])

        # Generate new data
        for i in range(N):
            img4d_t, ms_t, md_t = transf_tern(patient.nii_data_zyxt,
                                              patient.nii_mask_systole,
                                              patient.nii_mask_diastole)

            newPatient = SingleVentriclePatient()
            newPatient.name = patient.name + f'_A_{i}'
            newPatient.tSystole = patient.tSystole
            newPatient.tDiastole = patient.tDiastole
            newPatient.df_row = patient.df_row
            newPatient.df_row.loc[index, 'Name'] = newPatient.name

            # save new images with header
            newPatient.nii_data_xyzt = np.swapaxes(img4d_t, 0, 2)
            newPatient.nii_header_xyzt = patient.nii_header_xyzt

            # save new masks with header
            newPatient.nii_mask_systole_xyz = np.swapaxes(ms_t, 0, 2)
            newPatient.hdr_mask_systole = patient.hdr_mask_systole
            newPatient.nii_mask_diastole_xyz = np.swapaxes(md_t, 0, 2)
            newPatient.hdr_mask_diastole = patient.hdr_mask_diastole

            df = save_data(newPatient, saveDir4D_train, saveDirSegmentations_train)
            df_train = pd.concat([df_train, df])
        pbar.update(1)

    # Save validation data
    for index in range(0, len(val_ds)):
        patient = val_ds[index]
        pbar.set_postfix_str(f'P: {patient.name}')
        logger.info(f'[Val]: {index} -> {patient.name}')

        df = save_data(patient, saveDir4D_val, saveDirSegmentations_val)
        df_val = pd.concat([df_val, df])
        pbar.update(1)

    df_train.to_excel(osp.join(saveDir_train, train_ds.segmentations_filename), index=False)
    df_val.to_excel(osp.join(saveDir_val, val_ds.segmentations_filename), index=False)
