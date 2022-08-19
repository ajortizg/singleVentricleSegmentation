from torch.utils.data import Dataset
import os.path as osp
import nibabel as nib
import numpy as np
import pandas
from enum import Enum


class DatasetMode(Enum):
    TRAIN = 1
    VAL = 2
    TEST = 3


class Timepoint(Enum):
    ED = 1
    ES = 2


class ACDCDataset(Dataset):
    def __init__(self, config, mode, timepoint, transforms=None):
        self.config = config
        self.mode = mode
        self.timepoint = timepoint
        self.transforms = transforms

        self.base_path = config.get('DATA', 'BASE_PATH_3D')
        if mode == DatasetMode.TRAIN:
            self.base_path = osp.join(self.base_path, 'train')
        elif mode == DatasetMode.VAL:
            self.base_path = osp.join(self.base_path, 'val')
        elif mode == DatasetMode.TEST:
            self.base_path = osp.join(self.base_path, 'test')

        self.segmentations_subdir_path = config.get('DATA', 'SEGMENTATIONS_SUBDIR_PATH')
        self.segmentations_path = osp.join(self.base_path, self.segmentations_subdir_path)

        self.volumes_subdir_path = config.get('DATA', 'VOLUMES_SUBDIR_PATH')
        self.volumes_path = osp.join(self.base_path, self.volumes_subdir_path)

        self.segmetations_filename = config.get('DATA', 'SEGMENTATIONS_FILE_NAME')
        self.segmetations_file = osp.join(self.base_path, self.segmetations_filename)
        self.df = pandas.read_excel(self.segmetations_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        df_row = self.df.iloc[[idx]]
        patient_name = df_row.loc[idx, 'Name']
        es = df_row.loc[idx, 'Systole']
        ed = df_row.loc[idx, 'Diastole']

        # Load 4D nifty
        img4d = nib.load(osp.join(self.volumes_path, patient_name + '.nii.gz'))
        img4d_zyxt = np.swapaxes(img4d.get_fdata(), 0, 2)

        # Load segmentations masks
        if self.timepoint == Timepoint.ES:
            mask_zyx = self.load_mask(patient_name, '_Systole_Labelmap.nii')
            img_zyx = img4d_zyxt[..., es]
        elif self.timepoint == Timepoint.ED:
            mask_zyx = self.load_mask(patient_name, '_Diastole_Labelmap.nii')
            img_zyx = img4d_zyxt[..., ed]

        # Add channel dimension
        img_zyx = np.expand_dims(img_zyx, 0)
        mask_zyx = np.expand_dims(mask_zyx, 0)

        # Image transformations
        if self.transforms is not None:
            img_zyx, mask_zyx = self.transforms(img_zyx, mask_zyx)

        return (img_zyx, mask_zyx)

    def systole_diastole_time(self, idx):
        ts = self.df.iloc[idx]['Systole']
        td = self.df.iloc[idx]['Diastole']
        return (ts, td)

    def systole_diastole_mask(self, patient_name):
        mask_syst_zyx = self.load_mask(patient_name, '_Systole_Labelmap.nii')
        mask_diast_zyx = self.load_mask(patient_name, '_Diastole_Labelmap.nii')
        return (mask_syst_zyx, mask_diast_zyx)

    def get_patient_name(self, idx):
        return self.df.iloc[idx]['Name']

    def get_systole_time(self, idx):
        return self.df.iloc[idx]['Systole']

    def full_cycle(self, idx):
        return self.df.iloc[idx]['Full']

    def get_diastole_time(self, idx):
        return self.df.iloc[idx]['Diastole']

    def load_mask(self, patient_name, ending):
        mask = nib.load(osp.sep.join([self.segmentations_path, patient_name, patient_name + ending]))
        mask = np.swapaxes(mask.get_fdata(), 0, 2)
        return mask

    def index_for_patient(self, patient_name):
        row_patient = self.df[self.df['Name'] == patient_name]
        index_patient = row_patient.index[0]
        return index_patient

    def save_patients(self, save_dir: str, filename: str):
        output_df_file = osp.join(save_dir, filename)
        self.df.to_excel(output_df_file, index=False)
