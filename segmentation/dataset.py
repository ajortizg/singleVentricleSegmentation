import os.path as osp
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import nibabel as nib
from torch.utils.data.dataloader import default_collate
from collections import UserDict


class SVDSegmentation(Dataset):
    def __init__(self, root_dir, mode='train', transforms=None):
        self.transforms = transforms
        self.is_test = mode == 'test'
        self.imgs_dir = osp.join(root_dir, 'NIFTI_4D_Datasets')
        self.masks_dir = osp.join(root_dir, 'NIFTI_Single_Ventricle_Segmentations')
        df = pd.read_excel(osp.join(root_dir, 'Segmentation_volumes.xlsx'))
        self.df_split = df.loc[df['Split'] == mode]
        self.df_split.reset_index(inplace=True)

    def __len__(self):
        return len(self.df_split)

    def __getitem__(self, idx):
        df_row = self.df_split.iloc[[idx]]
        patient_name = df_row.loc[idx, 'Name']
        es = df_row.loc[idx, 'Systole']
        ed = df_row.loc[idx, 'Diastole']

        # Load 4D nifty
        img_zyxt = np.swapaxes(nib.load(osp.join(self.imgs_dir, patient_name + '.nii.gz')).get_fdata(), 0, 2)
        
        if self.is_test:
            # Load data for the full cardiac cycle
            imgs = img_zyxt
            masks = np.empty(shape=img_zyxt.shape, dtype=img_zyxt.dtype)
            for t in range(masks.shape[3]):
                masks[..., t] = np.swapaxes(nib.load(osp.join(self.masks_dir, patient_name, f'{patient_name}_{t}_Labelmap.nii')).get_fdata(), 0, 2)
        else:
            # Load data only for ed and es time points
            imgs = np.stack((img_zyxt[..., es], img_zyxt[..., ed]), axis=3)
            mask_zyx_es = np.swapaxes(nib.load(osp.join(self.masks_dir, patient_name, patient_name + '_Systole_Labelmap.nii')).get_fdata(), 0, 2)
            mask_zyx_ed = np.swapaxes(nib.load(osp.join(self.masks_dir, patient_name, patient_name + '_Diastole_Labelmap.nii')).get_fdata(), 0, 2)
            masks = np.stack((mask_zyx_es, mask_zyx_ed), axis=3)

        # Group data in a dictionary
        data = {'img': imgs, 'mask': masks, 'patient': patient_name}
        if self.transforms is not None:
            data = self.transforms(data)
        return data

    @staticmethod
    def collate_fn(batch):
        ret = UserDict(**default_collate(batch))
        return ret
