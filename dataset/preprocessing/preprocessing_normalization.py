
import sys
import numpy as np
import nibabel as nib
from tqdm import tqdm
import os
import pandas as pd
import configparser
import time
import os.path as osp
import math

# from intensity_normalization.typing import Modality, TissueType
# from intensity_normalization.normalize.fcm import FCMNormalize


ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../../'))
sys.path.append(ROOT_DIR)
from utils import plots
# import utils.transforms as T
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

    # old: normalize to [0,1]
    #transf_tern = T.ComposeUnary([T.Normalize()])

    ds = SingleVentricleDataset(config, mode='full')

    saveDir = plots.createSaveDirectory(config.get('DATA', 'OUTPUT_PATH'), 'preprocessing_normalization')
    logger = plots.create_logger(saveDir)

    logger.info("===========================================================")
    logger.info("Data normalization")
    logger.info("===========================================================")
    logger.info('save directory: ' + saveDir)

    # paths
    saveDir4D = plots.createSubDirectory(saveDir, ds.volumes_subdir_path)
    saveDirSegmentations = plots.createSubDirectory(saveDir, ds.segmentations_subdir_path)

    # save config file to save directory
    conifg_output = osp.sep.join([saveDir, "config.ini"])
    with open(conifg_output, 'w') as configfile:
        config.write(configfile)

    pbar = tqdm(total=len(ds))
    logger.info(f'Found {len(ds)} patientes')

    for index in range(0, len(ds)):
        patient = ds[index]
        pbar.set_postfix_str(f'{patient.name}')
        logger.info(f'[Patient]: {index} -> {patient.name}')

        # normalize data for newPatient
        per95 = np.percentile(patient.nii_data_zyxt, 95)
        new_nii_data_zyxt = np.clip(patient.nii_data_zyxt, 0, per95)

        # print("old min, max :", np.min(patient.nii_data_zyxt), np.max(patient.nii_data_zyxt) )
        # print("clip min, max :", np.min(new_nii_data_zyxt), np.max(new_nii_data_zyxt) )

        avg_diastole = np.mean(new_nii_data_zyxt[:, :, :, patient.tDiastole], where=patient.nii_mask_diastole.astype('bool'))
        avg_systole = np.mean(new_nii_data_zyxt[:, :, :, patient.tSystole], where=patient.nii_mask_systole.astype('bool'))
        avg = 0.5 * (avg_diastole + avg_systole)

        # print("avg = ", avg, "per95 = ", per95)
        # print("denom= ", (per95*avg*(per95-avg)))

        # old: quadratic normalization n(I) = a I**2 + b I
        # norm_a = (per95 - 2.*avg)/(2.*per95*avg*(avg - per95))
        # norm_b = (2*avg*avg - per95*per95)/(2.*per95*avg*(avg - per95))
        # new_nii_data_zyxt = norm_a * (new_nii_data_zyxt**2) + norm_b * new_nii_data_zyxt
        # print("norm(per95) = ", norm_a*per95*per95+norm_b*per95)
        # print("norm(avg) = ", norm_a*avg*avg+norm_b*avg)
        # print("norm(0) = ", norm_a*0.+norm_b*0.)

        # new: normalization n(I) = a I/sqrt(1+beta I**2)
        norm_a = math.sqrt(per95 * per95 - avg * avg) / (math.sqrt(3) * per95 * avg)
        norm_b = (per95 * per95 - 4. * avg * avg) / (3. * per95 * per95 * avg * avg)
        new_nii_data_zyxt = norm_a * new_nii_data_zyxt / np.sqrt(1 + norm_b * new_nii_data_zyxt**2)
        logger.info("norm(per95) = ", norm_a * per95 / math.sqrt(1 + norm_b * per95 * per95))
        logger.info("norm(avg) = ", norm_a * avg / math.sqrt(1 + norm_b * avg * avg))

        logger.info(np.min(new_nii_data_zyxt), np.max(new_nii_data_zyxt))

        newPatient = SingleVentriclePatient()
        newPatient.name = patient.name
        newPatient.tSystole = patient.tSystole
        newPatient.tDiastole = patient.tDiastole

        # save new images with header
        newPatient.nii_data_xyzt = np.swapaxes(new_nii_data_zyxt, 0, 2)
        newPatient.nii_header_xyzt = patient.nii_header_xyzt

        # save new masks with header
        newPatient.nii_mask_systole_xyz = patient.nii_mask_systole_xyz
        newPatient.hdr_mask_systole = patient.hdr_mask_systole
        newPatient.nii_mask_diastole_xyz = patient.nii_mask_diastole_xyz
        newPatient.hdr_mask_diastole = patient.hdr_mask_diastole

        save_data(newPatient, saveDir4D, saveDirSegmentations)
        pbar.update(1)

    logger.info("\n")
    logger.info("==================================")
    logger.info("save database to excel file")
    output_df = ds.df.copy()
    output_df_file = os.path.sep.join([saveDir, ds.segmentations_filename])
    output_df.to_excel(output_df_file, index=False)
