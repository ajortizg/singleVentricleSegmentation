import os.path as osp
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import nibabel as nib
import json


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


class FlowUNetDataset(Dataset):
    def __init__(self, root_dir, mode='train', transforms=None, load_flow=False, fold_indices=None):
        """
        Args:
            mode: train, test, full
            load_flow: Set to True to load optical flows
            fold_indices: Indices for cross valiation
        """
        self.root_dir = root_dir
        self.transforms = transforms
        self.is_test = mode == 'test'
        self.load_flow = load_flow

        self.imgs_dir = osp.join(root_dir, 'images')
        self.masks_dir = osp.join(root_dir, 'labels')
        df = pd.read_excel(osp.join(root_dir, 'info.xlsx'))

        # Read json
        self.json_ds = self.read_json(osp.join(root_dir, 'dataset.json'))

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
                nib_label = nib.load(osp.join(self.masks_dir, patient_name, f'{patient_name}_{t}_Labelmap.nii.gz'))
                label_xyzt[..., t] = nib_label.get_fdata()
        else:
            nib_label = nib.load(osp.join(self.masks_dir, patient_name, patient_name + '_Systole_Labelmap.nii.gz'))
            label_xyz_es = nib_label.get_fdata()
            nib_label = nib.load(osp.join(self.masks_dir, patient_name, patient_name + '_Diastole_Labelmap.nii.gz'))
            label_xyz_ed = nib_label.get_fdata()
            label_xyzt = np.stack((label_xyz_es, label_xyz_ed), axis=3)

        # Group data in a dictionary
        data = {'image': image_xyzt,
                'label': label_xyzt,
                'image_meta': {'affine': nib_image.affine},
                'label_meta': {'affine': nib_label.affine},
                'patient': patient_name,
                'es': es, 'ed': ed}

        # Load optical flow
        if self.load_flow:
            directions = ['forward', 'backward']
            for d in directions:
                key = d + '_flow'
                data[key] = np.load(osp.join(self.root_dir, 'optical_flow', d, f'{patient_name}_{d}_flow.npy'))
                data[key + '_meta'] = 0

        # Apply transformations to data
        if self.transforms is not None:
            data = self.transforms(data)
        return data

    def class_names(self):
        self.class_names = self.json_ds['labels']

    def num_classes(self):
        return len(self.class_names())

    def dataset_name(self):
        return self.json_ds['name']

    def file_ending(self):
        return self.json_ds['file_ending']

    def num_training_casses(self):
        return self.json_ds['numTraining']

    def read_json(self, filepath):
        f = open(filepath, mode='r')
        data = json.load(f)
        f.close()
        return data
