import numpy as np
from glob import glob
import os
import nibabel as nib
import os.path as osp
import sys
from tqdm import tqdm
import pandas as pd

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../../'))
sys.path.append(ROOT_DIR)
# from utilities import plots
from utilities import path_utils


class ACDCPatient:
    def __init__(self, name, img4d_nii, mask_systole_nii, mask_diastole_nii, tsystole, tdiastole):
        self.name = name
        self.img4d_nii = img4d_nii
        self.mask_systole_nii = mask_systole_nii
        self.mask_diastole_nii = mask_diastole_nii
        self.tsystole = tsystole
        self.tdiastole = tdiastole

    def save(self, save4d_dir, save_segmentations_dir):
        # Save 4d data
        outputFile = osp.sep.join([save4d_dir, self.name + ".nii.gz"])
        nib.save(self.img4d_nii, outputFile)

        # Save masks
        if self.mask_systole_nii is not None and self.mask_diastole_nii is not None:
            save_patient_seg_dir = path_utils.create_sub_dir(save_segmentations_dir, self.name)
            outputFile = osp.sep.join([save_patient_seg_dir, self.name + "_Systole_Labelmap.nii.gz"])
            nib.save(self.mask_systole_nii, outputFile)
            outputFile = osp.sep.join([save_patient_seg_dir, self.name + "_Diastole_Labelmap.nii.gz"])
            nib.save(self.mask_diastole_nii, outputFile)

        row = {'Name': self.name, 'Systole': self.tsystole, 'Diastole': self.tdiastole}
        df_patient = pd.DataFrame(row, index=[0])
        return df_patient


class ACDCDataset:
    def __init__(self, base_path):
        self.base_path = base_path
        self.patient_dirs = sorted(os.listdir(base_path))

    def __getitem__(self, idx):
        name = self.patient_dirs[idx]
        patient_dir = osp.join(self.base_path, name)
        img4d_nii = nib.load(osp.join(patient_dir, name + '_4d.nii.gz'))

        with open(osp.join(patient_dir, 'Info.cfg'), 'r') as f:
            tdiastole = int(f.readline().replace(' ', '').replace('\n', '').split(':')[1])
            tsystole = int(f.readline().replace(' ', '').replace('\n', '').split(':')[1])

        mode = self.base_path.split(osp.sep)[-1]
        if mode == 'training':
            mask_diastole_nii = nib.load(osp.sep.join([self.base_path, name, name + '_frame%02d_gt.nii.gz' % (tdiastole)]))
            mask_systole_nii = nib.load(osp.sep.join([self.base_path, name, name + '_frame%02d_gt.nii.gz' % (tsystole)]))
            # mask_diastole_nii = self.extract_label(mask_diastole_nii)
            # mask_systole_nii = self.extract_label(mask_systole_nii)
        else:
            mask_diastole_nii = None
            mask_systole_nii = None

        patient = ACDCPatient(name, img4d_nii, mask_systole_nii, mask_diastole_nii, tsystole, tdiastole)
        return patient

    # def extract_label(self, mask_nii):
    #     mask_data = mask_nii.get_fdata()
    #     # mask_data_label = np.where(np.logical_or(mask_data == self.label1, mask_data == self.label2), 1.0, 0.0)
    #     mask_data_label = np.where(mask_data == self.label, 1.0, 0.0)
    #     mask_nii = nib.Nifti1Image(mask_data_label, affine=mask_nii.affine, header=mask_nii.header)
    #     return mask_nii

    def __len__(self):
        return len(self.patient_dirs)


if __name__ == "__main__":
    # labels = {'background': 0,
    #           'right_ventricle': 1,
    #           'myocardium': 2,
    #           'left_ventricle': 3}

    save_dir = path_utils.create_save_dir('results', 'ACDCData')
    save4d_dir = path_utils.create_sub_dir(save_dir, 'NIFTI_4D_Datasets')
    save_segmentations_dir = path_utils.create_sub_dir(save_dir, 'NIFTI_Single_Ventricle_Segmentations')
    df_dset = pd.DataFrame(columns=['Name', 'Systole', 'Diastole'])

    train_ds = ACDCDataset('data/acdc/training')
    # test_ds = ACDCDataset('data/ACDC/testing/testing', mode='test')
    dsets = [train_ds]
    pbar = tqdm(total=len(train_ds))
    for ds in dsets:
        for i in range(len(ds)):
            patient = ds[i]
            df = patient.save(save4d_dir, save_segmentations_dir)
            df_dset = pd.concat([df_dset, df])
            pbar.update(1)

    df_dset.to_excel(osp.join(save_dir, 'Segmentation_volumes.xlsx'), index=False)
