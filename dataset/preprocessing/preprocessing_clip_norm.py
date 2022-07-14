
import sys
import numpy as np
import nibabel as nib
from tqdm import tqdm
import os
import pandas as pd
import configparser
import time
import os.path as osp

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


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('parser/configPreprocessing.ini')

    img4d_transf = T.ComposeUnary([T.PercentileClip(1, 99), T.Normalize()])
    mask_transf = T.ComposeUnary([T.Round(0.5)])
    dset = SingleVentricleDataset(config, mode='full')

    saveDir = plots.createSaveDirectory(config.get('DATA', 'OUTPUT_PATH'), 'preprocessing_normClip')
    logger = plots.create_logger(saveDir)

    logger.info("===========================================================")
    logger.info("Clip normalization")
    logger.info("===========================================================")
    logger.info('save directory: ' + saveDir)

    # saving paths
    saveDir4D = plots.createSubDirectory(saveDir, dset.volumes_subdir_path)
    saveDirSegmentations = plots.createSubDirectory(saveDir, dset.segmentations_subdir_path)

    # save config file to save directory
    conifg_output = osp.sep.join([saveDir, "config.ini"])
    with open(conifg_output, 'w') as configfile:
        config.write(configfile)

    pbar = tqdm(total=len(dset))
    logger.info(f'Found {len(dset)} patientes')

    for index in range(0, len(dset)):
        patient = dset[index]
        pbar.set_postfix_str(f'{patient.name}')
        logger.info(f'{patient.name}')

        # Normalize data
        img4d_xyzt = img4d_transf(patient.nii_data_xyzt)

        newPatient = SingleVentriclePatient()
        newPatient.name = patient.name
        newPatient.tSystole = patient.tSystole
        newPatient.tDiastole = patient.tDiastole

        # save new images with header
        newPatient.nii_data_xyzt = img4d_xyzt
        newPatient.nii_header_xyzt = patient.nii_header_xyzt

        # save new masks with header
        newPatient.nii_mask_systole_xyz = mask_transf(patient.nii_mask_systole_xyz)
        newPatient.hdr_mask_systole = patient.hdr_mask_systole
        newPatient.nii_mask_diastole_xyz = mask_transf(patient.nii_mask_diastole_xyz)
        newPatient.hdr_mask_diastole = patient.hdr_mask_diastole

        save_data(newPatient, saveDir4D, saveDirSegmentations)
        pbar.update(1)

    output_df = dset.df.copy()
    output_df_file = os.path.sep.join([saveDir, dset.segmentations_filename])
    output_df.to_excel(output_df_file, index=False)
