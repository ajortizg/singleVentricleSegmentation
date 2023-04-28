import os.path as osp
import json
from glob import glob

from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
import pandas as pd

from .utils.image_reader import ImageReader


class SegmentationDataset(Dataset):
    def __init__(self, root_dir, mode='train', transforms=None, fold_idxs=None):
        """
        Args:
            mode: train, test
            fold_indices: Indices for cross valiation
        """
        self.transforms = transforms
        self.json_ds = self.read_json(osp.join(root_dir, 'dataset.json'))
        self.test_json = None
        self.reader = ImageReader(self.file_ending())

        if mode == 'train':
            self.img_paths = sorted(glob(osp.join(root_dir, 'imagesTr', f'*{self.file_ending()}')))
            self.label_paths = sorted(glob(osp.join(root_dir, 'labelsTr', f'*{self.file_ending()}')))
            assert len(self.img_paths) == self.num_training() and len(self.label_paths) == self.num_training()

            if fold_idxs is not None:
                self.img_paths = np.array(self.img_paths)[fold_idxs].tolist()
                self.label_paths = np.array(self.label_paths)[fold_idxs].tolist()
        elif mode == 'test':
            self.img_paths = sorted(glob(osp.join(root_dir, 'imagesTs', f'*{self.file_ending()}')))
            self.label_paths = sorted(glob(osp.join(root_dir, 'labelsTs', f'*{self.file_ending()}')))
            self.test_json = self.read_json(osp.join(root_dir, 'test.json'))
        else:
            raise ValueError('{} is not a valid mode. Use train or test'.format(mode))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Read image in xyz or xyzt order for train and test respectively
        # nib_image = nib.load(self.img_paths[idx])
        # image = nib_image.get_fdata()
        data = self.reader(self.img_paths[idx])
        img = data['data']
        img_meta = data['meta']

        # Read mask in xyz or xyzt order for train and test respectively
        # nib_label = nib.load(self.label_paths[idx])
        # label = nib_label.get_fdata()
        data = self.reader(self.label_paths[idx])
        label = data['data']
        label_meta = data['meta']

        # # Read metadata
        # image_meta = {'affine': nib_image.affine}
        # label_meta = {'affine': nib_label.affine}

        data = {'image': img,
                'label': label}

        if img_meta is not None:
            data['image_meta'] = img_meta
        if label_meta is not None:
            data['label_meta'] = label_meta

        # Read some metada for the test split
        if self.test_json is not None:
            img_file = osp.basename(self.img_paths[idx])
            data['es'] = self.test_json[img_file]['es']
            data['ed'] = self.test_json[img_file]['ed']
            data['patient'] = self.test_json[img_file]['patient']

        if self.transforms is not None:
            data = self.transforms(data)
        return data

    def filter_patient(self, patient_to_keep):
        found = False
        for i in range(len(self.img_paths)):
            img_file = osp.basename(self.img_paths[i])
            patient = self.test_json[img_file]['patient']
            if patient == patient_to_keep:
                found = True
                idx = i
                break
        if not found:
            raise ValueError(f'Patient: {patient_to_keep} not found in test split')
        else:
            self.img_paths = [self.img_paths[idx]]
            self.label_paths = [self.label_paths[idx]]

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
