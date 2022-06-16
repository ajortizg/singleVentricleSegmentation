
import sys
import numpy as np
import nibabel as nib
import os
import pandas
import configparser
import time
import os.path as osp

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
from utils import plots

def flip_mask(patient_name, mask_xyz,flip_all):
    mask_zyx = np.swapaxes(mask_xyz, 0, 2)
    mask_flipped = None
    if patient_name in flip_all:
        mask_flipped = np.flip(mask_zyx, (0, 1, 2)).copy()
    else:
        mask_flipped = np.flip(mask_zyx, 1).copy()
    return np.swapaxes(mask_flipped, 0, 2)  # return xyz mask


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

    print("\n\n")
    print("==================================================")
    print("==================================================")
    print("  preprocessing data: flipping nifty files        ")
    print("==================================================")
    print("==================================================")
    print("\n\n")

    # load config parser
    config = configparser.ConfigParser()
    config.read('parser/configPreprocessing.ini')

    # create save directory
    saveDir = plots.createSaveDirectory(config.get('DATA', 'OUTPUT_PATH'), "preprocessing_flip")

    # save config file to save directory
    conifgOutput = os.path.sep.join([saveDir, "config.ini"])
    with open(conifgOutput, 'w') as configfile:
        config.write(configfile)

    # load data base
    BASE_PATH_3D = config.get('DATA', 'BASE_PATH_3D')
    VOLUMES_SUBDIR_PATH = config.get('DATA', 'VOLUMES_SUBDIR_PATH')
    VOLUMES_PATH = os.path.sep.join([BASE_PATH_3D, VOLUMES_SUBDIR_PATH])
    SEGMENTATIONS_FILE_NAME = config.get('DATA', 'SEGMENTATIONS_FILE_NAME')
    SEGMENTATIONS_FILE = os.path.sep.join([BASE_PATH_3D, SEGMENTATIONS_FILE_NAME])
    df = pandas.read_excel(SEGMENTATIONS_FILE)
    numDataFiles = df.shape[0]
    print("number of data files = ", numDataFiles)
    SEGMENTATIONS_SUBDIR_PATH = config.get('DATA', 'SEGMENTATIONS_SUBDIR_PATH')
    SEGMENTATIONS_PATH = os.path.sep.join([BASE_PATH_3D, SEGMENTATIONS_SUBDIR_PATH])

    #load specific patients
    flip_all = config.get('FLIPPING', 'flip_all')

    #generate columns for flipping axis
    xflip = np.zeros(numDataFiles)
    yflip = np.zeros(numDataFiles)
    zflip = np.zeros(numDataFiles)

    #
    saveDir4D = plots.createSubDirectory(saveDir, VOLUMES_SUBDIR_PATH)
    saveDirSegmentations = plots.createSubDirectory(saveDir, SEGMENTATIONS_SUBDIR_PATH)

    for index, row in df.iterrows():

        # Load 4D nifty [x,y,z,t]
        print("=======================================")
        PATIENT_NAME = row['Name']
        print("load data for patient: ", PATIENT_NAME)
        vol = nib.load(os.path.sep.join([VOLUMES_PATH, PATIENT_NAME + ".nii.gz"]))
        vol_hdr = vol.header
        #vol_affine = vol.affine
        nii_data_xyzt = vol.get_fdata()
        NX = nii_data_xyzt.shape[0]
        NY = nii_data_xyzt.shape[1]
        NZ = nii_data_xyzt.shape[2]
        NT = nii_data_xyzt.shape[3]
        print("   * (NX,NY,NZ,NT) = ", NX, NY, NZ, NT)
        #
        saveDirPatient = plots.createSubDirectory(saveDirSegmentations, PATIENT_NAME)
        # read time steps for diastole and systole
        tDiastole = row["Diastole"]
        tSystole = row["Systole"]
        print("   * systole at time:  ", tSystole)
        print("   * diastole at time: ", tDiastole)
        print("=======================================")

        if PATIENT_NAME in flip_all:
           xflip[index] = 1
           yflip[index] = 1
           zflip[index] = 1
        else:
           yflip[index] = 1

        # get input masks for diastole
        nii_mask_diastole_load = nib.load(os.path.sep.join([SEGMENTATIONS_PATH, PATIENT_NAME, PATIENT_NAME + "_Diastole_Labelmap.nii"]))
        hdr_mask_diastole = nii_mask_diastole_load.header
        #affine_mask_diastole = nii_mask_diastole_load.affine
        nii_mask_diastole_xyz = nii_mask_diastole_load.get_fdata()
        nii_mask_diastole_xyz_flip = flip_mask(PATIENT_NAME, nii_mask_diastole_xyz,flip_all)

        # get input masks for systole
        nii_mask_systole_load = nib.load(os.path.sep.join([SEGMENTATIONS_PATH, PATIENT_NAME, PATIENT_NAME + "_Systole_Labelmap.nii"]))
        hdr_mask_systole = nii_mask_systole_load.header
        #affine_mask_systole = nii_mask_systole_load.affine
        nii_mask_systole_xyz = nii_mask_systole_load.get_fdata()
        nii_mask_systole_xyz_flip = flip_mask(PATIENT_NAME, nii_mask_systole_xyz,flip_all)

        # save to nifty
        save_np_to_nifty(nii_data_xyzt, saveDir4D, PATIENT_NAME + ".nii.gz", vol_hdr)
        save_np_to_nifty(nii_mask_diastole_xyz_flip, saveDirPatient, PATIENT_NAME + "_Diastole_Labelmap.nii", hdr_mask_diastole)
        save_np_to_nifty(nii_mask_systole_xyz_flip, saveDirPatient, PATIENT_NAME + "_Systole_Labelmap.nii", hdr_mask_systole)

    # save data base with shifts
    df['xflip'] = xflip
    df['yflip'] = yflip
    df['zflip'] = zflip
    output_df = os.path.sep.join([saveDir, SEGMENTATIONS_FILE_NAME])
    df.to_excel(output_df)
