
import sys
import numpy as np
import nibabel as nib
from tqdm import tqdm
import os
import pandas
import configparser
import time
import os.path as osp

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../../'))
sys.path.append(ROOT_DIR)
from utils import plots
from dataset import singleVentricleDataset

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

    plots.printConsoleOutput_Header("preprocessing data: flipping nifty files")

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
    dataSet = singleVentricleDataset.SingleVentricleDataset(config)

    #load specific patients
    flip_all = config.get('FLIPPING', 'flip_all')

    #generate columns for flipping axis
    xflip = np.zeros(len(dataSet))
    yflip = np.zeros(len(dataSet))
    zflip = np.zeros(len(dataSet))

    #
    saveDir4D = plots.createSubDirectory(saveDir, dataSet.volumes_subdir_path)
    saveDirSegmentations = plots.createSubDirectory(saveDir, dataSet.segmentations_subdir_path)

    #iterate over all patients
    pbar = tqdm(total=len(dataSet))
    for index in range(0,len(dataSet)):

        patient = dataSet[index]

        pbar.set_postfix_str(f'P: {patient.name}')

        #flip masks for diastole and systole
        nii_mask_diastole_xyz_flip = flip_mask(patient.name, patient.nii_mask_diastole_xyz,flip_all)
        nii_mask_systole_xyz_flip = flip_mask(patient.name, patient.nii_mask_systole_xyz,flip_all)

        # save to nifty
        saveDirPatient = plots.createSubDirectory(saveDirSegmentations, patient.name)
        save_np_to_nifty(patient.nii_data_xyzt, saveDir4D, patient.name + ".nii.gz", patient.nii_header_xyzt)
        save_np_to_nifty(nii_mask_diastole_xyz_flip, saveDirPatient, patient.name + "_Diastole_Labelmap.nii", patient.hdr_mask_diastole)
        save_np_to_nifty(nii_mask_systole_xyz_flip, saveDirPatient, patient.name + "_Systole_Labelmap.nii", patient.hdr_mask_systole)

        if patient.name in flip_all:
           xflip[index] = 1
           yflip[index] = 1
           zflip[index] = 1
        else:
           yflip[index] = 1

        pbar.update(1)


    # save data base with shifts
    print("\n")
    print("==================================")
    print("save database to excel file")
    output_df = dataSet.df.copy()
    output_df['xflip'] = xflip
    output_df['yflip'] = yflip
    output_df['zflip'] = zflip
    output_df_file = os.path.sep.join([saveDir, dataSet.segmentations_filename])
    output_df.to_excel(output_df_file,index=False)