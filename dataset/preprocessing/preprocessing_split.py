import sys
import numpy as np
import nibabel as nib
from tqdm import tqdm
import os
import pandas as pd
import configparser
import os.path as osp

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../../'))
sys.path.append(ROOT_DIR)
from utils import plots
from dataset.singleVentricleDataset import SingleVentricleDataset


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


def save_data(patient, saveDir4D, saveDirSegmentations):
    saveDirPatient = plots.createSubDirectory(saveDirSegmentations, patient.name)
    save_np_to_nifty(patient.nii_data_xyzt, saveDir4D, patient.name + ".nii.gz", patient.nii_header_xyzt)
    save_np_to_nifty(patient.nii_mask_diastole_xyz, saveDirPatient, patient.name + "_Diastole_Labelmap.nii", patient.hdr_mask_diastole)
    save_np_to_nifty(patient.nii_mask_systole_xyz, saveDirPatient, patient.name + "_Systole_Labelmap.nii", patient.hdr_mask_systole)

    row = {'Name': patient.name, 'Systole': patient.tSystole, 'Diastole': patient.tDiastole}
    df_patient = pd.DataFrame(row, index=[0])
    return df_patient


if __name__ == "__main__":
    plots.printConsoleOutput_Header("preprocessing data: split train and validation dataset")

    # load config parser
    config = configparser.ConfigParser()
    config.read('parser/configPreprocessing.ini')

    # create save directory
    saveDir = plots.createSaveDirectory(config.get('DATA', 'OUTPUT_PATH'), "preprocessing_split")

    # save config file to save directory
    conifgOutput = os.path.sep.join([saveDir, "config.ini"])
    with open(conifgOutput, 'w') as configfile:
        config.write(configfile)

    dataSet = SingleVentricleDataset(config)

    # load specific patients for validation
    val_patients = set(config.get('SPLIT', 'validation_patients').replace('{', '').replace('}', '').replace('\n', '').split(','))

    # Paths for training dataset
    saveDir_train = plots.createSubDirectory(saveDir, 'train')
    saveDir4D_train = plots.createSubDirectory(saveDir_train, dataSet.volumes_subdir_path)
    saveDirSegmentations_train = plots.createSubDirectory(saveDir_train, dataSet.segmentations_subdir_path)

    # Paths for validation dataset
    saveDir_val = plots.createSubDirectory(saveDir, 'val')
    saveDir4D_val = plots.createSubDirectory(saveDir_val, dataSet.volumes_subdir_path)
    saveDirSegmentations_val = plots.createSubDirectory(saveDir_val, dataSet.segmentations_subdir_path)

    # iterate over all patients
    pbar = tqdm(total=len(dataSet))
    df_val = pd.DataFrame(columns=['Name', 'Systole', 'Diastole'])
    df_train = pd.DataFrame(columns=['Name', 'Systole', 'Diastole'])
    for index in range(0, len(dataSet)):
        patient = dataSet[index]
        pbar.set_postfix_str(f'P: {patient.name}')

        if patient.name in val_patients:
            df = save_data(patient, saveDir4D_val, saveDirSegmentations_val)
            df_val = pd.concat([df_val, df])
        else:
            df = save_data(patient, saveDir4D_train, saveDirSegmentations_train)
            df_train = pd.concat([df_train, df])

        pbar.update(1)

    pbar.close()

    # save data base
    print("\n")
    print("==================================")
    print("save database to excel file")
    output_df_file_val = os.path.sep.join([saveDir_val, dataSet.segmentations_filename])
    df_val.to_excel(output_df_file_val, index=False)

    output_df_file_train = os.path.sep.join([saveDir_train, dataSet.segmentations_filename])
    df_train.to_excel(output_df_file_train, index=False)
