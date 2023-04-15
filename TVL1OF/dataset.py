import os.path as osp
import configparser
import torch
import os
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import nibabel as nib
from torch.utils.data.dataloader import default_collate
from collections import UserDict
from sklearn.model_selection import KFold
from glob import glob


def get_bounds(es, ed, masks, fwd, test=False):
    mi, mf = None, None

    if es < ed:
        ti, tf = es, ed
        if masks is not None:
            if not test:
                mi, mf = masks[0, ...], masks[1, ...]
            else:
                mi, mf = masks[ti, ...], masks[tf, ...]
    else:
        ti, tf = ed, es
        if masks is not None:
            if not test:
                mi, mf = masks[1, ...], masks[0, ...]
            else:
                mi, mf = masks[ti, ...], masks[tf, ...]
    indices = torch.arange(ti, tf + 1, 1)

    if fwd:
        return ti, tf, mi, mf, indices
    else:
        return tf, ti, mf, mi, torch.flip(indices, (0,))


class SVDataset(Dataset):
    def __init__(self, root_dir, mode='train', transforms=None, load_flow=False, fold_indices=None):
        """
        Args:
            mode: train, test, full
            vxm: Set to True when computing optical flow
            load_flow: Set to True to load optical flows
            fold_indices: Indices for cross valiation
        """
        self.root_dir = root_dir
        self.transforms = transforms
        self.is_test = mode == 'test'
        self.load_flow = load_flow
        self.class_names = ['background', 'sv']
        self.num_classes = len(self.class_names)

        self.imgs_dir = osp.join(root_dir, 'NIFTI_4D_Datasets')
        self.masks_dir = osp.join(root_dir, 'NIFTI_Single_Ventricle_Segmentations')
        df = pd.read_excel(osp.join(root_dir, 'Segmentation_volumes.xlsx'))

        if mode == 'full':
            self.df_split = df
        else:
            self.df_split = df[df['Split'] == mode]
            self.df_split.reset_index(inplace=True, drop=True)

            if fold_indices is not None and mode != 'test':
                self.df_split = self.df_split.iloc[fold_indices]
                self.df_split.reset_index(inplace=True, drop=True)

    def __len__(self):
        return len(self.df_split)

    def __getitem__(self, idx):
        df_row = self.df_split.iloc[[idx]]
        patient_name = df_row.loc[idx, 'Name']
        es = df_row.loc[idx, 'Systole']
        ed = df_row.loc[idx, 'Diastole']

        # Load nifty images and masks
        nib_image = nib.load(osp.join(self.imgs_dir, patient_name + '.nii.gz'))
        image_xyzt = nib_image.get_fdata()

        if self.is_test:
            label_xyzt = np.empty(shape=image_xyzt.shape, dtype=image_xyzt.dtype)
            for t in range(label_xyzt.shape[3]):
                nib_label = nib.load(osp.join(self.masks_dir, patient_name, f'{patient_name}_{t}_Labelmap.nii'))
                label_xyzt[..., t] = nib_label.get_fdata()
        else:
            nib_label = nib.load(osp.join(self.masks_dir, patient_name, patient_name + '_Systole_Labelmap.nii'))
            label_xyz_es = nib_label.get_fdata()
            nib_label = nib.load(osp.join(self.masks_dir, patient_name, patient_name + '_Diastole_Labelmap.nii'))
            label_xyz_ed = nib_label.get_fdata()
            label_xyzt = np.stack((label_xyz_es, label_xyz_ed), axis=3)

        # Load metadata
        # img_meta = dict(nii_img.header)
        image_meta = {'affine': nib_image.affine}

        # mask_meta = dict(nii_mask.header)
        label_meta = {'affine': nib_label.affine}

        # Group data in a dictionary
        data = {'image': image_xyzt,
                'label': label_xyzt,
                'image_meta': image_meta,
                'label_meta': label_meta,
                'patient': patient_name,
                'es': es, 'ed': ed}

        # Load optical flow
        if self.load_flow:
            directions = ['forward', 'backward']
            for d in directions:
                patient_dir = osp.join(self.root_dir, 'optical_flow', d, patient_name)
                *_, indices = get_bounds(es, ed, None, fwd=d == 'forward')
                indices = indices[:-1]
                flows_list = [np.load(osp.join(patient_dir, f'time{t}', 'it0', 'flow_m_it0.npy')) for t in indices]
                flows_np = np.stack([flow for flow in flows_list], axis=4)
                data[d + '_flow'] = flows_np

        # Apply transformations to data
        if self.transforms is not None:
            data = self.transforms(data)

        return data

    @staticmethod
    def collate_fn(batch):
        ret = UserDict(**default_collate(batch))
        return ret
