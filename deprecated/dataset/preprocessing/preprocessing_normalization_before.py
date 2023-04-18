
import sys
import numpy as np
import nibabel as nib
from tqdm import tqdm
import os
import configparser
import os.path as osp

from intensity_normalization.typing import Modality, TissueType
from intensity_normalization.normalize.fcm import FCMNormalize

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../../'))
sys.path.append(ROOT_DIR)
from utilities import plots
from dataset import singleVentricleDataset

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

    plots.printConsoleOutput_Header("preprocessing data: cutting out heart region")

    # load config parser
    config = configparser.ConfigParser()
    config.read('parser/configPreprocessing.ini')

    # create save directory
    saveDir = plots.createSaveDirectory(config.get('DATA', 'OUTPUT_PATH'), "preprocessing_normalize")

    # save config file to save directory
    conifgOutput = os.path.sep.join([saveDir, "config.ini"])
    with open(conifgOutput, 'w') as configfile:
        config.write(configfile)

    dataSet = singleVentricleDataset.SingleVentricleDataset(config)

    #
    saveDir4D = plots.createSubDirectory(saveDir, dataSet.volumes_subdir_path)
    saveDirSegmentations = plots.createSubDirectory(saveDir, dataSet.segmentations_subdir_path)

    # iterate over all patients
    pbar = tqdm(total=len(dataSet))
    for index in range(0, len(dataSet)):
        patient = dataSet[index]

        pbar.set_postfix_str(f'P: {patient.name}')

        fcm_norm = FCMNormalize(tissue_type=TissueType.WM)
        normalized_xyzt = fcm_norm(patient.nii_data_xyzt)

        # save to nifty
        saveDirPatient = plots.createSubDirectory(saveDirSegmentations, patient.name)
        save_np_to_nifty(normalized_xyzt, saveDir4D, patient.name + ".nii.gz", patient.nii_header_xyzt)
        save_np_to_nifty(patient.nii_mask_diastole_xyz, saveDirPatient, patient.name + "_Diastole_Labelmap.nii", patient.hdr_mask_diastole)
        save_np_to_nifty(patient.nii_mask_systole_xyz, saveDirPatient, patient.name + "_Systole_Labelmap.nii", patient.hdr_mask_systole)

        pbar.update(1)

    # save data base with shifts
    print("\n")
    print("==================================")
    print("save database to excel file")
    output_df = dataSet.df.copy()
    output_df_file = os.path.sep.join([saveDir, dataSet.segmentations_filename])
    output_df.to_excel(output_df_file, index=False)
