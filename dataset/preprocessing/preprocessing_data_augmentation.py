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
from utils import plots
import utils.transforms as T
from dataset.singleVentricleDataset import SingleVentricleDataset, SingleVentriclePatient


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
    outputFile = osp.sep.join([saveDir, fileName])
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
    config = configparser.ConfigParser()
    config.read('parser/configPreprocessing.ini')

    flip_prob_z = config.getfloat('DATA_AUGMENTATION', 'FLIP_Z_PROB')
    flip_prob_y = config.getfloat('DATA_AUGMENTATION', 'FLIP_Y_PROB')
    flip_prob_x = config.getfloat('DATA_AUGMENTATION', 'FLIP_X_PROB')

    prob_rot = config.getfloat('DATA_AUGMENTATION', 'ROT_PROB')
    rot_range_z = tuple(map(float, config.get('DATA_AUGMENTATION', 'ROT_Z_RANGE').split(',')))
    rot_range_y = tuple(map(float, config.get('DATA_AUGMENTATION', 'ROT_Y_RANGE').split(',')))
    rot_range_x = tuple(map(float, config.get('DATA_AUGMENTATION', 'ROT_X_RANGE').split(',')))

    ed_prob = config.getfloat('DATA_AUGMENTATION', 'ELASTIC_DEFORM_PROB')
    ed_grid = config.getint('DATA_AUGMENTATION', 'ELASTIC_DEFORM_GRID')
    ed_sigma = config.getfloat('DATA_AUGMENTATION', 'ELASTIC_DEFORM_SIGMA')

    save_original_data = config.getboolean('DATA_AUGMENTATION', 'SAVE_ORIGINAL_DATA')
    save_val_data = config.getboolean('DATA_AUGMENTATION', 'SAVE_VAL_DATA')
    N = config.getint('DATA_AUGMENTATION', 'CREATE_NEW')

    transf = T.ComposeTernary([
        T.RandomFlipZ(p=flip_prob_z),
        T.RandomFlipY(p=flip_prob_y),
        T.RandomFlipX(p=flip_prob_x),
        T.RandomRotate(p=prob_rot, range_z=rot_range_z, range_y=rot_range_y, range_x=rot_range_x),
        T.ElasticDeformation(p=ed_prob, sigma_range=(ed_sigma, ed_sigma), points_range=(ed_grid, ed_grid))])

    print("===========================================================")
    print("Data augmentation")
    print("===========================================================")
    print(f'\t* Create: {N} new patientes')
    print(f'\t* Flip probs: {flip_prob_x}, {flip_prob_y}, {flip_prob_z}')
    print(f'\t* Rot prob: {prob_rot}, with ranges: {rot_range_x}, {rot_range_y}, {rot_range_z}')
    print(f'\t* Elastic def prob: {ed_prob}, grid: {ed_grid}, sigma: {ed_sigma}')
    print()

    train_ds = SingleVentricleDataset(config, mode='train')
    val_ds = SingleVentricleDataset(config, mode='val')

    saveDir = plots.createSaveDirectory(config.get('DATA', 'OUTPUT_PATH'), 'preprocessing_da')
    saveDir_train = plots.createSubDirectory(saveDir, 'train')
    saveDir_val = plots.createSubDirectory(saveDir, 'val')

    # train paths
    saveDir4D_train = plots.createSubDirectory(saveDir_train, train_ds.volumes_subdir_path)
    saveDirSegmentations_train = plots.createSubDirectory(saveDir_train, train_ds.segmentations_subdir_path)

    # validation paths
    saveDir4D_val = plots.createSubDirectory(saveDir_val, val_ds.volumes_subdir_path)
    saveDirSegmentations_val = plots.createSubDirectory(saveDir_val, val_ds.segmentations_subdir_path)

    # save config file to save directory
    conifg_output = osp.sep.join([saveDir, "config.ini"])
    with open(conifg_output, 'w') as configfile:
        config.write(configfile)

    df_train = pd.DataFrame(columns=['Name', 'Systole', 'Diastole'])
    df_val = pd.DataFrame(columns=['Name', 'Systole', 'Diastole'])
    pbar = tqdm(total=len(train_ds) + (len(val_ds) if save_val_data else 0))

    # Generate augmented training dataset
    for index in range(0, len(train_ds)):
        patient = train_ds[index]

        pbar.set_postfix_str(f'P: {patient.name}')

        # Save original data
        if save_original_data:
            df = save_data(patient, saveDir4D_train, saveDirSegmentations_train)
            df_train = pd.concat([df_train, df])

        # Generate new data
        for i in range(N):
            vol_t, ms_t, md_t = transf(torch.from_numpy(patient.nii_data_zyxt),
                                       torch.from_numpy(patient.nii_mask_systole),
                                       torch.from_numpy(patient.nii_mask_diastole))

            newPatient = SingleVentriclePatient()
            newPatient.name = patient.name + f'_A_{i}'
            newPatient.tSystole = patient.tSystole
            newPatient.tDiastole = patient.tDiastole

            # save new images with header
            newPatient.nii_data_xyzt = np.swapaxes(vol_t.numpy(), 0, 2)
            newPatient.nii_header_xyzt = patient.nii_header_xyzt

            # save new masks with header
            newPatient.nii_mask_systole_xyz = np.swapaxes(ms_t.numpy(), 0, 2)
            newPatient.hdr_mask_systole = patient.hdr_mask_systole
            newPatient.nii_mask_diastole_xyz = np.swapaxes(md_t.numpy(), 0, 2)
            newPatient.hdr_mask_diastole = patient.hdr_mask_diastole

            df = save_data(newPatient, saveDir4D_train, saveDirSegmentations_train)
            df_train = pd.concat([df_train, df])
        pbar.update(1)

    # Save validation data
    if save_val_data:
        for index in range(0, len(val_ds)):
            patient = val_ds[index]
            pbar.set_postfix_str(f'P: {patient.name}')

            df = save_data(patient, saveDir4D_val, saveDirSegmentations_val)
            df_val = pd.concat([df_val, df])
            pbar.update(1)

    df_train.to_excel(osp.join(saveDir_train, train_ds.segmentations_filename), index=False)
    df_val.to_excel(osp.join(saveDir_val, val_ds.segmentations_filename), index=False)
