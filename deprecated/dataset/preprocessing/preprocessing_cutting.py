
import sys
import numpy as np
import nibabel as nib
from tqdm import tqdm
import os
import configparser
import os.path as osp

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../../../'))
sys.path.append(ROOT_DIR)
from utilities import path_utils
from deprecated.dataset import singleVentricleDataset

__all__ = ['cut_patient']


def getRangeOfMask_xyz(mask, printRange=False, name=""):
    x, y, z = np.nonzero(mask)
    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(y)
    ymax = np.max(y)
    zmin = np.min(z)
    zmax = np.max(z)
    if printRange:
        print("\nrange of mask", name, ":")
        print("(xmin, xmax) = ", xmin, ",", xmax)
        print("(ymin, ymax) = ", ymin, ",", ymax)
        print("(zmin, zmax) = ", zmin, ",", zmax)
    return zmin, zmax, ymin, ymax, xmin, xmax


def save_np_to_nifty(file, saveDir, fileName, hdr_old):
    # header
    hdr = nib.nifti1.Nifti1Header()
    hdr.set_data_shape(file.shape)
    hdr.set_qform(hdr_old.get_qform())
    hdr.set_sform(hdr_old.get_sform())
    hdr.set_zooms(hdr_old.get_zooms())
    # img
    ni_img = nib.Nifti1Image(file, affine=None, header=hdr)
    # save
    outputFile = os.path.sep.join([saveDir, fileName])
    nib.save(ni_img, outputFile)
    # print("old header:")
    # print(hdr_old)
    # print("new header:")
    # print(hdr)


if __name__ == "__main__":

    # plots.printConsoleOutput_Header("preprocessing data: cutting out heart region")

    # load config parser
    config = configparser.ConfigParser()
    config.read('parser/deprecated/configPreprocessing.ini')

    # create save directory
    saveDir = path_utils.create_save_dir(config.get('DATA', 'OUTPUT_PATH'), "preprocessing_cut")

    # save config file to save directory
    conifgOutput = os.path.sep.join([saveDir, "config.ini"])
    with open(conifgOutput, 'w') as configfile:
        config.write(configfile)

    dataSet = singleVentricleDataset.SingleVentricleDataset(config)

    # load specific patients
    xTol = config.getint('CUTTING', 'xTol')
    yTol = config.getint('CUTTING', 'yTol')
    zTol = config.getint('CUTTING', 'zTol')

    # generate columns for (x,y,z)-shifts
    xshifts = np.zeros(len(dataSet))
    yshifts = np.zeros(len(dataSet))
    zshifts = np.zeros(len(dataSet))
    original_NX = np.zeros(len(dataSet))
    original_NY = np.zeros(len(dataSet))
    original_NZ = np.zeros(len(dataSet))
    original_NT = np.zeros(len(dataSet))

    #
    saveDir4D = path_utils.create_sub_dir(saveDir, dataSet.volumes_subdir_path)
    saveDirSegmentations = path_utils.create_sub_dir(saveDir, dataSet.segmentations_subdir_path)

    # iterate over all patients
    pbar = tqdm(total=len(dataSet))
    for index in range(0, len(dataSet)):
        patient = dataSet[index]

        pbar.set_postfix_str(f'P: {patient.name}')

        # get range of diastole and systole
        zmin_dia, zmax_dia, ymin_dia, ymax_dia, xmin_dia, xmax_dia = getRangeOfMask_xyz(
            patient.nii_mask_diastole_xyz, printRange=False, name="diastole")
        zmin_sys, zmax_sys, ymin_sys, ymax_sys, xmin_sys, xmax_sys = getRangeOfMask_xyz(
            patient.nii_mask_systole_xyz, printRange=False, name="systole")

        # Original shape
        NX, NY, NZ, NT = patient.nii_data_xyzt.shape

        # extend range by tolerance
        xmin_total = max(0, min(xmin_dia, xmin_sys) - xTol)
        xmax_total = min(patient.NX - 1, max(xmax_dia, xmax_sys) + xTol)
        ymin_total = max(0, min(ymin_dia, ymin_sys) - yTol)
        ymax_total = min(patient.NY - 1, max(ymax_dia, ymax_sys) + yTol)
        zmin_total = max(0, min(zmin_dia, zmin_sys) - zTol)
        zmax_total = min(patient.NZ - 1, max(zmax_dia, zmax_sys) + zTol)
        # print("\ntotal range:")
        # print("(xmin, xmax) = ", xmin_total, ",", xmax_total)
        # print("(ymin, ymax) = ", ymin_total, ",", ymax_total)
        # print("(zmin, zmax) = ", zmin_total, ",", zmax_total)

        xshifts[index] = xmin_total
        yshifts[index] = ymin_total
        zshifts[index] = zmin_total
        original_NX[index] = NX
        original_NY[index] = NY
        original_NZ[index] = NZ
        original_NT[index] = NT

        NX_cut = xmax_total - xmin_total + 1
        NY_cut = ymax_total - ymin_total + 1
        NZ_cut = zmax_total - zmin_total + 1

        cutting_4d = patient.nii_data_xyzt[xmin_total:xmax_total + 1, ymin_total:ymax_total + 1, zmin_total:zmax_total + 1, :]
        cutting_diastole = patient.nii_mask_diastole_xyz[xmin_total:xmax_total + 1, ymin_total:ymax_total + 1, zmin_total:zmax_total + 1]
        cutting_systole = patient.nii_mask_systole_xyz[xmin_total:xmax_total + 1, ymin_total:ymax_total + 1, zmin_total:zmax_total + 1]

        # save to nifty
        saveDirPatient = path_utils.create_sub_dir(saveDirSegmentations, patient.name)
        save_np_to_nifty(cutting_4d, saveDir4D, patient.name + ".nii.gz", patient.nii_header_xyzt)
        save_np_to_nifty(cutting_diastole, saveDirPatient, patient.name + "_Diastole_Labelmap.nii.gz", patient.hdr_mask_diastole)
        save_np_to_nifty(cutting_systole, saveDirPatient, patient.name + "_Systole_Labelmap.nii.gz", patient.hdr_mask_systole)

        if patient.full_cycle:
            for t in range(patient.NT):
                mask_cutted = patient.nii_masks_xyz[t][xmin_total:xmax_total + 1, ymin_total:ymax_total + 1, zmin_total:zmax_total + 1]
                mask_filename = patient.masks_dirs[t].split('/')[-1]
                save_np_to_nifty(mask_cutted, saveDirPatient, mask_filename, patient.nii_masks_load[t].header)

        pbar.update(1)

    # save data base with shifts
    print("\n")
    print("==================================")
    print("save database to excel file")
    output_df = dataSet.df.copy()
    output_df['xshift'] = xshifts
    output_df['yshift'] = yshifts
    output_df['zshift'] = zshifts
    output_df['original_NX'] = original_NX
    output_df['original_NY'] = original_NY
    output_df['original_NZ'] = original_NZ
    output_df['original_NT'] = original_NT
    output_df_file = os.path.sep.join([saveDir, dataSet.segmentations_filename])
    output_df.to_excel(output_df_file, index=False)
    