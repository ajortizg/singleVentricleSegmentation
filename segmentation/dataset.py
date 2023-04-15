import os.path as osp
import configparser
import torch
import os
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import random
import nibabel as nib
import nibabel.processing as nip
from torch.utils.data.dataloader import default_collate
from collections import UserDict
from natsort import natsorted
from sklearn.model_selection import KFold
import json
from glob import glob
import re


def create_5fold(n, seed=12345):
    splits = []
    indices = np.arange(n)
    kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
    for i, (train_idx, test_idx) in enumerate(kfold.split(indices)):
        train_keys = np.array(indices)[train_idx]
        test_keys = np.array(indices)[test_idx]
        splits.append({})
        splits[-1]['train'] = list(train_keys)
        splits[-1]['val'] = list(test_keys)
    return splits


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
                'patient': patient_name, 'ed': ed, 'es': es}
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


class SegmentationDataset(Dataset):
    def __init__(self, root_dir, mode='train', transforms=None, fold_indices=None):
        """
        Args:
            mode: train, test
            fold_indices: Indices for cross valiation
        """
        self.transforms = transforms
        self.json_ds = self.read_json(osp.join(root_dir, 'dataset.json'))

        if mode == 'train':
            self.img_paths = sorted(glob(osp.join(root_dir, 'imagesTr', f'*{self.file_ending()}')))
            self.label_paths = sorted(glob(osp.join(root_dir, 'labelsTr', f'*{self.file_ending()}')))
            assert len(self.img_paths) == self.num_training() and len(self.label_paths) == self.num_training()
        elif mode == 'test':
            raise NotImplementedError(self.__class__.__name__ + ' test no implemented yet')
        else:
            raise ValueError('{} is not a valid mode. Use train or test'.format(mode))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Read image
        nib_image = nib.load(self.img_paths[idx])
        image_xyz = nib_image.get_fdata()

        # Read mask
        nib_label = nib.load(self.label_paths[idx])
        label_xyz = nib_label.get_fdata()

        # Read metadata
        image_meta = {'affine': nib_image.affine}
        label_meta = {'affine': nib_label.affine}

        data = {'image': image_xyz,
                'label': label_xyz,
                'image_meta': image_meta,
                'label_meta': label_meta}

        if self.transforms is not None:
            data = self.transforms(data)
        return data

    def read_json(self, filepath):
        f = open(filepath, mode='r')
        data = json.load(f)
        f.close()
        return data

    def classes(self):
        return self.json_ds['labels']

    def num_training(self):
        return self.json_ds['numTraining']

    def file_ending(self):
        return self.json_ds['file_ending']
