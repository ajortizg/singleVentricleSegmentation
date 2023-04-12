import os.path as osp
import configparser
import torch
import os
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import random
import nibabel as nib
from torch.utils.data.dataloader import default_collate
from collections import UserDict
from natsort import natsorted


def get_bounds(es, ed, masks, fwd, test=False):
    mi, mf = None, None

    if es < ed:
        ti, tf = es, ed
        if masks is not None:
            if not test:
                mi, mf = masks[..., 0], masks[..., 1]
            else:
                mi, mf = masks[..., ti], masks[..., tf]
    else:
        ti, tf = ed, es
        if masks is not None:
            if not test:
                mi, mf = masks[..., 1], masks[..., 0]
            else:
                mi, mf = masks[..., ti], masks[..., tf]
    indices = torch.arange(ti, tf + 1, 1)

    if fwd:
        return ti, tf, mi, mf, indices
    else:
        return tf, ti, mf, mi, torch.flip(indices, (0,))


class SVDataset(Dataset):
    def __init__(self, root_dir, mode='train', transforms=None, vxm=False, load_flow=False):
        """
        Args:
            mode: train, val, test, full
            vxm: Set to True when computing optical flow
            load_flow: Set to True to load optical flows
        """
        self.root_dir = root_dir
        self.transforms = transforms
        self.is_test = mode == 'test'
        self.vxm = vxm
        self.load_flow = load_flow
        self.class_names = ['background', 'sv']
        self.num_classes = len(self.class_names)

        self.imgs_dir = osp.join(root_dir, 'NIFTI_4D_Datasets')
        self.masks_dir = osp.join(root_dir, 'NIFTI_Single_Ventricle_Segmentations')
        df = pd.read_excel(osp.join(root_dir, 'Segmentation_volumes.xlsx'))

        if mode != 'full':
            self.df_split = df.loc[df['Split'] == mode]
            self.df_split.reset_index(inplace=True)
        else:
            self.df_split = df

    def __len__(self):
        return len(self.df_split)

    def __getitem__(self, idx):
        df_row = self.df_split.iloc[[idx]]
        patient_name = df_row.loc[idx, 'Name']
        es = df_row.loc[idx, 'Systole']
        ed = df_row.loc[idx, 'Diastole']

        # Load nifty images and masks
        nii_img = nib.load(osp.join(self.imgs_dir, patient_name + '.nii.gz'))
        img_zyxt = np.swapaxes(nii_img.get_fdata(), 0, 2)

        if self.is_test:
            imgs = img_zyxt
            masks = np.empty(shape=img_zyxt.shape, dtype=img_zyxt.dtype)
            for t in range(masks.shape[3]):
                nii_mask = nib.load(osp.join(self.masks_dir, patient_name, f'{patient_name}_{t}_Labelmap.nii'))
                masks[..., t] = np.swapaxes(nii_mask.get_fdata(), 0, 2)
        else:
            imgs = img_zyxt if self.vxm else np.stack((img_zyxt[..., es], img_zyxt[..., ed]), axis=3)
            nii_mask = nib.load(osp.join(self.masks_dir, patient_name, patient_name + '_Systole_Labelmap.nii'))
            mask_zyx_es = np.swapaxes(nii_mask.get_fdata(), 0, 2)
            nii_mask = nib.load(osp.join(self.masks_dir, patient_name, patient_name + '_Diastole_Labelmap.nii'))
            mask_zyx_ed = np.swapaxes(nii_mask.get_fdata(), 0, 2)
            masks = np.stack((mask_zyx_es, mask_zyx_ed), axis=3)

        # Load metadata
        # img_meta = dict(nii_img.header)
        img_meta = {}
        img_meta['affine'] = nii_img.affine
        # mask_meta = dict(nii_mask.header)
        mask_meta = {}
        mask_meta['affine'] = nii_mask.affine

        # Group data in a dictionary
        data = {'img': imgs,
                'mask': masks,
                'img_meta': img_meta,
                'mask_meta': mask_meta,
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


class ACDCDataset(Dataset):
    def __init__(self, root_dir, mode='train', transforms=None, vxm=False):
        self.transforms = transforms
        self.is_test = mode == 'test'
        self.vxm = vxm
        self.class_names = ['background', 'rv', 'my', 'lv']
        self.num_classes = len(self.class_names)

        split_dir = osp.join(root_dir, 'testing' if self.is_test else 'training')
        self.patients_list_dir = [osp.join(split_dir, p) for p in natsorted(os.listdir(split_dir))]

    def __len__(self):
        return len(self.patients_list_dir)

    def __getitem__(self, idx):
        patient_dir = self.patients_list_dir[idx]
        info_cfg = self.read_configfile(osp.join(patient_dir, 'Info.cfg'))
        ed = info_cfg.getint('ED')
        es = info_cfg.getint('ES')
        patient_name = osp.basename(patient_dir)

        # End diastolic data
        nii_img_ed = nib.load(osp.join(patient_dir, '{}_frame{:02d}.nii.gz'.format(patient_name, ed)))
        nii_mask_ed = nib.load(osp.join(patient_dir, '{}_frame{:02d}_gt.nii.gz'.format(patient_name, ed)))
        img_ed = np.swapaxes(nii_img_ed.get_fdata(), 0, 2)
        mask_ed = np.swapaxes(nii_mask_ed.get_fdata(), 0, 2)

        # End systolic data
        nii_img_es = nib.load(osp.join(patient_dir, '{}_frame{:02d}.nii.gz'.format(patient_name, es)))
        nii_mask_es = nib.load(osp.join(patient_dir, '{}_frame{:02d}_gt.nii.gz'.format(patient_name, es)))
        img_es = np.swapaxes(nii_img_es.get_fdata(), 0, 2)
        mask_es = np.swapaxes(nii_mask_es.get_fdata(), 0, 2)

        imgs = np.stack((img_es, img_ed), axis=3)
        masks = np.stack((mask_es, mask_ed), axis=3)

        img_meta = dict(nii_img_ed.header)
        img_meta['affine'] = nii_img_ed.affine
        mask_meta = dict(nii_mask_ed.header)
        mask_meta['affine'] = nii_mask_ed.affine

        data = {'img': imgs,
                'mask': masks,
                'img_meta': img_meta,
                'mask_meta': mask_meta,
                'patient': patient_name, 'ed': ed, 'es': es,
                'raw_imgs': [nii_img_es, nii_img_ed],
                'raw_labels': [nii_mask_es, nii_mask_ed]
                }
        if self.transforms is not None:
            data = self.transforms(data)
        return data

    def read_configfile(self, file_dir):
        with open(file_dir, 'r') as f:
            config_string = '[dummy_section]\n' + f.read()
        config = configparser.ConfigParser()
        config.read_string(config_string)
        return config['dummy_section']

    @staticmethod
    def collate_fn(batch):
        ret = UserDict(**default_collate(batch))
        return ret
