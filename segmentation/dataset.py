import os.path as osp
import json
from glob import glob

from torch.utils.data import Dataset
import nibabel as nib
import numpy as np


class SegmentationDataset(Dataset):
    def __init__(self, root_dir, mode='train', transforms=None, fold_idxs=None):
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

            if fold_idxs is not None:
                self.img_paths = np.array(self.img_paths)[fold_idxs].tolist()
                self.label_paths = np.array(self.img_paths)[fold_idxs].tolist()
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
