import os.path as osp
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import json
from collections import defaultdict
import torch.nn.functional as F

from .utils.image_reader import ImageReader


def get_bounds(es, ed, masks, fwd, test=False):
    mi, mf = None, None

    if es < ed:
        ti, tf = es, ed
        if masks is not None:
            if not test:
                mi, mf = masks[0], masks[1]
            else:
                mi, mf = masks[ti], masks[tf]
    else:
        ti, tf = ed, es
        if masks is not None:
            if not test:
                mi, mf = masks[1], masks[0]
            else:
                mi, mf = masks[ti], masks[tf]
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
        self.reader = ImageReader(self.file_ending(), read_meta=False)

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
        ext = self.file_ending()

        # Load images and masks
        # image_xyzt = np.load(osp.join(self.imgs_dir, patient_name + ext))
        image_xyzt = self.reader(osp.join(self.imgs_dir, patient_name + ext))['data']

        if self.is_test:
            # label_xyzt = np.load(osp.join(self.masks_dir, patient_name, f'{patient_name}_Labelmap{ext}'))
            label_xyzt = self.reader(osp.join(self.masks_dir, patient_name, f'{patient_name}_Labelmap{ext}'))['data']
        else:
            # label_xyz_es = np.load(osp.join(self.masks_dir, patient_name, patient_name + f'_Systole_Labelmap{ext}'))
            # label_xyz_ed = np.load(osp.join(self.masks_dir, patient_name, patient_name + f'_Diastole_Labelmap{ext}'))
            label_xyz_es = self.reader(osp.join(self.masks_dir, patient_name, patient_name + f'_Systole_Labelmap{ext}'))['data']
            label_xyz_ed = self.reader(osp.join(self.masks_dir, patient_name, patient_name + f'_Diastole_Labelmap{ext}'))['data']
            label_xyzt = np.stack((label_xyz_es, label_xyz_ed), axis=3)

        # Group data in a dictionary
        data = {'image': image_xyzt,
                'label': label_xyzt,
                'patient': patient_name,
                'es': es, 'ed': ed}

        # Load optical flow
        if self.load_flow:
            directions = ['forward', 'backward']
            for d in directions:
                key = d + '_flow'
                data[key] = np.load(osp.join(self.root_dir, 'optical_flow', d, f'{patient_name}_{d}_flow.npy'))

        # Apply transformations to data
        if self.transforms is not None:
            data = self.transforms(data)
        return data

    def class_names(self):
        return self.json_ds['labels']

    def num_classes(self):
        # remove background class
        return len(self.class_names()) - 1

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

    def filter_patient(self, patient_to_keep):
        row_patient = self.df_split[self.df_split['Name'] == patient_to_keep]
        # index_patient = row_patient.index[0]
        self.df_split = row_patient
        self.df_split.reset_index(inplace=True, drop=True)

    @staticmethod
    def collate(data_list):
        """
        Returns tensors with shape (nt, bs, ch, d1, d2, d3), except for optical flow
        where ch dim is the last one
        """
        keys_to_collate = ['image', 'label', 'mi', 'mf', 'forward_flow', 'backward_flow']

        tss_per_key = defaultdict(list)
        collated_dict = defaultdict(list)
        offsets_per_key = defaultdict(list)

        # Get the maximum time in the batch
        for data in data_list:
            for k, v in data.items():
                if k in keys_to_collate:
                    tss_per_key[k].append(v.shape[0])
                collated_dict[k].append(v)

        max_ts_per_key = {}
        for k, v in tss_per_key.items():
            max_ts_per_key[k] = max(v)

        keys_to_collate = max_ts_per_key.keys()

        # Pad to fit the maximum time
        for k, v in collated_dict.items():
            if k in keys_to_collate:
                list_of_padded_tensors = []
                for tensor in v:
                    dif = max_ts_per_key[k] - tensor.shape[0]
                    offsets_per_key[k].append(dif)
                    padded_tensor = F.pad(tensor, (0, 0,
                                                   0, 0,
                                                   0, 0,
                                                   0, 0,
                                                   0, dif))
                    list_of_padded_tensors.append(padded_tensor)
                # collated_dict[k] = torch.permute(torch.stack(list_of_padded_tensors), (0, 2, 3, 4, 5, 1))
                collated_dict[k] = torch.stack(list_of_padded_tensors, dim=1)
        collated_dict['offsets'] = offsets_per_key

        # Create time sequence forward and backward
        ti = torch.tensor(collated_dict['ti'])
        tf = torch.tensor(collated_dict['tf'])
        max_time_gap = (tf - ti).max().item()
        times_fwd = [ti]
        times_bwd = [tf]
        for _ in range(max_time_gap):
            next_time = times_fwd[-1] + 1
            times_fwd.append(torch.where(next_time > tf, tf, next_time))
            prev_time = times_bwd[-1] - 1
            times_bwd.append(torch.where(prev_time < ti, ti, prev_time))
        times_fwd = torch.stack(times_fwd, dim=0)
        times_bwd = torch.stack(times_bwd, dim=0)
        collated_dict['times_fwd'] = times_fwd
        collated_dict['times_bwd'] = times_bwd

        # Convert numeric lists to tensors
        for k, v in collated_dict.items():
            if type(v) == list:
                if type(v[0]) is not str:
                    collated_dict[k] = torch.tensor(v)

        return collated_dict
