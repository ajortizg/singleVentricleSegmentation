
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


ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../../../'))
sys.path.append(ROOT_DIR)
from utilities import path_utils
from deprecated.dataset.singleVentricleDataset import SingleVentricleDataset, SingleVentriclePatient
import utilities.transforms.unary_transforms as T1

__all__ = ['normalize_patient']


def save_np_to_nifty(file: np.array, saveDir: str, fileName: str, hdr_old, affine):
    # header
    # hdr = nib.nifti1.Nifti1Header()
    # hdr.set_data_shape(file.shape)
    # hdr.set_qform(hdr_old.get_qform())
    # hdr.set_sform(hdr_old.get_sform())
    # hdr.set_zooms(hdr_old.get_zooms())
    # img
    ni_img = nib.Nifti1Image(file, affine=affine, header=hdr_old)
    # save
    outputFile = osp.sep.join([saveDir, fileName])
    nib.save(ni_img, outputFile)


def save_data(patient: SingleVentriclePatient, saveDir4D: str, saveDirSegmentations: str):
    saveDirPatient = path_utils.create_sub_dir(saveDirSegmentations, patient.name)
    save_np_to_nifty(patient.nii_data_xyzt, saveDir4D, patient.name + ".nii.gz", patient.nii_header_xyzt, patient.nii_xyzt.affine)
    save_np_to_nifty(patient.nii_mask_diastole_xyz, saveDirPatient,
                     patient.name + "_Diastole_Labelmap.nii.gz", patient.hdr_mask_diastole, patient.nii_mask_diastole_load.affine)
    save_np_to_nifty(patient.nii_mask_systole_xyz, saveDirPatient,
                     patient.name + "_Systole_Labelmap.nii.gz", patient.hdr_mask_systole, patient.nii_mask_systole_load.affine)

    if patient.full_cycle:
        for t in range(patient.NT):
            save_np_to_nifty(
                patient.nii_masks_xyz[t],
                saveDirPatient, patient.masks_dirs[t].split('/')[-1],
                patient.nii_masks_load[t].header, patient.nii_masks_load[t].affine)


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('parser/deprecated/configPreprocessing.ini')
    minmax_norm = config.getboolean('NORMALIZATION', 'MIN_MAX_NORM')

    ds = SingleVentricleDataset(config, mode='full')
    saveDir = path_utils.create_save_dir(config.get('DATA', 'OUTPUT_PATH'), 'preprocessing_normalization')

    print("===========================================================")
    print("Data normalization")
    print("===========================================================")
    print('save directory: ' + saveDir)

    # paths
    saveDir4D = path_utils.create_sub_dir(saveDir, ds.volumes_subdir_path)
    saveDirSegmentations = path_utils.create_sub_dir(saveDir, ds.segmentations_subdir_path)

    # save config file to save directory
    conifg_output = osp.sep.join([saveDir, "config.ini"])
    with open(conifg_output, 'w') as configfile:
        config.write(configfile)

    pbar = tqdm(total=len(ds))
    print(f'Found {len(ds)} patientes')

    norm_fn = T1.Normalize()

    for index in range(0, len(ds)):
        patient = ds[index]
        pbar.set_postfix_str(f'{patient.name}')
        print(f'[Patient]: {index} -> {patient.name}')

        # normalize data for newPatient
        if minmax_norm:
            new_nii_data_zyxt = norm_fn(patient.nii_data_zyxt)
            print(np.min(new_nii_data_zyxt), np.max(new_nii_data_zyxt))
        else:
            per95 = np.percentile(patient.nii_data_zyxt, 95)
            new_nii_data_zyxt = np.clip(patient.nii_data_zyxt, 0, per95)

            # print("old min, max :", np.min(patient.nii_data_zyxt), np.max(patient.nii_data_zyxt) )
            # print("clip min, max :", np.min(new_nii_data_zyxt), np.max(new_nii_data_zyxt) )

            avg_diastole = np.mean(new_nii_data_zyxt[:, :, :, patient.tDiastole],
                                   where=patient.nii_mask_diastole.astype('bool'))
            avg_systole = np.mean(new_nii_data_zyxt[:, :, :, patient.tSystole],
                                  where=patient.nii_mask_systole.astype('bool'))
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
            print("norm(per95) = ", norm_a * per95 / math.sqrt(1 + norm_b * per95 * per95))
            print("norm(avg) = ", norm_a * avg / math.sqrt(1 + norm_b * avg * avg))

            print(np.min(new_nii_data_zyxt), np.max(new_nii_data_zyxt))

        newPatient = SingleVentriclePatient()
        newPatient.name = patient.name
        newPatient.tSystole = patient.tSystole
        newPatient.tDiastole = patient.tDiastole

        # save new images with header
        newPatient.nii_xyzt = patient.nii_xyzt
        newPatient.nii_data_xyzt = np.swapaxes(new_nii_data_zyxt, 0, 2)
        newPatient.nii_header_xyzt = patient.nii_header_xyzt

        # save new masks with header
        newPatient.nii_mask_systole_xyz = patient.nii_mask_systole_xyz
        newPatient.hdr_mask_systole = patient.hdr_mask_systole
        newPatient.nii_mask_systole_load = patient.nii_mask_systole_load
        newPatient.nii_mask_diastole_xyz = patient.nii_mask_diastole_xyz
        newPatient.hdr_mask_diastole = patient.hdr_mask_diastole
        newPatient.nii_mask_diastole_load = patient.nii_mask_diastole_load

        # Save full cycle masks
        newPatient.full_cycle = patient.full_cycle
        newPatient.NT = patient.NT
        newPatient.nii_masks_load = patient.nii_masks_load
        newPatient.masks_dirs = patient.masks_dirs
        newPatient.nii_masks_xyz = patient.nii_masks_xyz

        save_data(newPatient, saveDir4D, saveDirSegmentations)
        pbar.update(1)

    print("\n")
    print("==================================")
    print("save database to excel file")
    output_df = ds.df.copy()
    output_df_file = os.path.sep.join([saveDir, ds.segmentations_filename])
    output_df.to_excel(output_df_file, index=False)
