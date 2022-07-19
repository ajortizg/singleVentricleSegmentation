import numpy as np
from glob import glob
import os
import nibabel as nib
import os.path as osp
import sys
from tqdm import tqdm
import pandas as pd

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
from utils import plots


class ACDCPatient:
    def __init__(self, name, img4d_nii, mask_systole_nii, mask_diastole_nii, tsystole, tdiastole, group, height, nbframe, weight):
        self.name = name
        self.img4d_nii = img4d_nii
        self.mask_systole_nii = mask_systole_nii
        self.mask_diastole_nii = mask_diastole_nii
        self.tsystole = tsystole
        self.tdiastole = tdiastole
        self.group = group
        self.height = height
        self.nbframe = nbframe
        self.weight = weight


class ACDCDataset:
    def __init__(self, base_path, mode):
        self.base_path = base_path
        self.mode = mode
        self.patient_dirs = sorted(os.listdir(base_path))

    def __getitem__(self, idx):
        name = self.patient_dirs[idx]
        patient_dir = osp.join(self.base_path, name)
        img4d_nii = nib.load(osp.join(patient_dir, name + '_4d.nii.gz'))

        with open(osp.join(patient_dir, 'Info.cfg'), 'r') as f:
            tdiastole = int(f.readline().replace(' ', '').replace('\n', '').split(':')[1])
            tsystole = int(f.readline().replace(' ', '').replace('\n', '').split(':')[1])
            if self.mode == 'train':
                group = f.readline().replace(' ', '').replace('\n', '').split(':')[1]
            else:
                group = ''
            height = f.readline().replace(' ', '').replace('\n', '').split(':')[1]
            nbframe = f.readline().replace(' ', '').replace('\n', '').split(':')[1]
            weight = f.readline().replace(' ', '').replace('\n', '').split(':')[1]
        if self.mode == 'train':
            mask_diastole_nii = nib.load(osp.sep.join([self.base_path, name, name + '_frame%02d_gt.nii.gz' % (tdiastole)]))
            mask_systole_nii = nib.load(osp.sep.join([self.base_path, name, name + '_frame%02d_gt.nii.gz' % (tsystole)]))
        else:
            mask_diastole_nii = nib.load(osp.sep.join([self.base_path, name, name + '_frame%02d.nii.gz' % (tdiastole)]))
            mask_systole_nii = nib.load(osp.sep.join([self.base_path, name, name + '_frame%02d.nii.gz' % (tsystole)]))

        patient = ACDCPatient(name, img4d_nii, mask_systole_nii, mask_diastole_nii, tsystole, tdiastole, group, height, nbframe, weight)
        return patient

    def __len__(self):
        return len(self.patient_dirs)


def save_data(patient, saveDir4D, saveDirSegmentations):
    # Save 4d data
    img4d_nii = nib.Nifti1Image(patient.img4d_nii.get_fdata(), affine=patient.img4d_nii.affine, header=patient.img4d_nii.header)
    outputFile = osp.sep.join([saveDir4D, patient.name + ".nii.gz"])
    nib.save(img4d_nii, outputFile)

    save_patient_seg_dir = plots.createSubDirectory(saveDirSegmentations, patient.name)

    # Save systole mask
    mask_systole_nii = nib.Nifti1Image(patient.mask_systole_nii.get_fdata(),
                                       affine=patient.mask_systole_nii.affine, header=patient.mask_systole_nii.header)
    outputFile = osp.sep.join([save_patient_seg_dir, patient.name + "_Systole_Labelmap.nii"])
    nib.save(mask_systole_nii, outputFile)

    # Save diastole mask
    mask_diastole_nii = nib.Nifti1Image(patient.mask_diastole_nii.get_fdata(),
                                        affine=patient.mask_diastole_nii.affine, header=patient.mask_diastole_nii.header)
    outputFile = osp.sep.join([save_patient_seg_dir, patient.name + "_Diastole_Labelmap.nii"])
    nib.save(mask_diastole_nii, outputFile)

    row = {'Name': patient.name, 'Systole': patient.tsystole, 'Diastole': patient.tdiastole}
    df_patient = pd.DataFrame(row, index=[0])
    return df_patient


if __name__ == "__main__":
    save_dir = 'results/ACDCData'
    save4d_dir = plots.createSubDirectory(save_dir, 'NIFTI_4D_Datasets')
    save_segmentations_dir = plots.createSubDirectory(save_dir, 'NIFTI_Single_Ventricle_Segmentations')
    df_dset = pd.DataFrame(columns=['Name', 'Systole', 'Diastole'])

    train_ds = ACDCDataset('data/acdc/training/training', mode='train')
    # test_ds = ACDCDataset('data/ACDC/testing/testing', mode='test')
    dsets = [train_ds]
    pbar = tqdm(total=len(train_ds))
    for ds in dsets:
        for i in range(len(ds)):
            patient = ds[i]
            df = save_data(patient, save4d_dir, save_segmentations_dir)
            df_dset = pd.concat([df_dset, df])
            pbar.update(1)

    df_dset.to_excel(osp.join(save_dir, 'Segmentation_volumes.xlsx'), index=False)
