
import sys
import numpy as np
import nibabel as nib
from tqdm import tqdm
import os
import pandas as pd
import configparser
import time
import os.path as osp

from intensity_normalization.typing import Modality, TissueType
from intensity_normalization.normalize.fcm import FCMNormalize


ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../../'))
sys.path.append(ROOT_DIR)
from utils import plots
import utils.transforms as T
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

    row = {'Name': patient.name, 'Systole': patient.tSystole, 'Diastole': patient.tDiastole}
    df_patient = pd.DataFrame(row, index=[0])
    return df_patient


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('parser/configPreprocessing.ini')

    # old: normalize to [0,1]
    #transf_tern = T.ComposeUnary([T.Normalize()])

    train_ds = SingleVentricleDataset(config, mode='train')
    val_ds = SingleVentricleDataset(config, mode='val')

    saveDir = plots.createSaveDirectory(config.get('DATA', 'OUTPUT_PATH'), 'preprocessing_normalization')
    logger = plots.create_logger(saveDir)

    logger.info("===========================================================")
    logger.info("Data normalization")
    logger.info("===========================================================")
    logger.info('save directory: ' + saveDir)

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

    df_train = pd.DataFrame(columns=['Name', 'Systole', 'Diastole'])
    df_val = pd.DataFrame(columns=['Name', 'Systole', 'Diastole'])
    pbar = tqdm(total=len(train_ds) + (len(val_ds)))
    logger.info(f'Found {len(train_ds)} training patientes')
    logger.info(f'Found {len(val_ds)} validation patients')
    
    # Generate augmented training dataset
    for index in range(0, len(train_ds)):
        patient = train_ds[index]
        pbar.set_postfix_str(f'P: {patient.name}')
        logger.info(f'[Train]: {index} -> {patient.name}')

        # Generate new data
        
        #old: normalize to [0,1]
        #vol_t = transf_tern(patient.nii_data_zyxt)
        #new: use better normalization
        vol_t = fcm_norm(patient.nii_data_zyxt, patient.nii_mask_diastole)

        newPatient = SingleVentriclePatient()
        newPatient.name = patient.name
        newPatient.tSystole = patient.tSystole
        newPatient.tDiastole = patient.tDiastole

        # save new images with header
        newPatient.nii_data_xyzt = np.swapaxes(vol_t, 0, 2)
        newPatient.nii_header_xyzt = patient.nii_header_xyzt

        # save new masks with header
        newPatient.nii_mask_systole_xyz = np.swapaxes(patient.nii_mask_systole, 0, 2)
        newPatient.hdr_mask_systole = patient.hdr_mask_systole
        newPatient.nii_mask_diastole_xyz = np.swapaxes(patient.nii_mask_diastole, 0, 2)
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