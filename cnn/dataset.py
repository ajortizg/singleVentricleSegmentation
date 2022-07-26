from torch.utils.data import Dataset
import os.path as osp
import nibabel as nib
import numpy as np
import os
import sys
import pandas
import torch
from enum import Enum
import yaml
import re


class DatasetMode(Enum):
    TRAIN = 1
    VAL = 2
    FULL = 3


class LoadFlowMode(Enum):
    NO_LOAD_OF = 1
    TRAIN_OF = 2
    PREDICT_OF = 3


class SingleVentricleDataset(Dataset):
    def __init__(self, config, mode, flow_mode, img4d_transforms=None, mask_transforms=None):
        self.config = config
        self.mode = mode
        self.flow_mode = flow_mode
        self.img4d_transforms = img4d_transforms                # transformations applied only on data
        self.mask_transforms = mask_transforms                  # transformations applied only on masks

        self.base_path = config.get('DATA', 'BASE_PATH_3D')
        if mode == DatasetMode.TRAIN:
            self.base_path = osp.join(self.base_path, 'train')
        elif mode == DatasetMode.VAL:
            self.base_path = osp.join(self.base_path, 'val')

        self.segmentations_subdir_path = config.get('DATA', 'SEGMENTATIONS_SUBDIR_PATH')
        self.segmentations_path = osp.join(self.base_path, self.segmentations_subdir_path)

        self.volumes_subdir_path = config.get('DATA', 'VOLUMES_SUBDIR_PATH')
        self.volumes_path = osp.join(self.base_path, self.volumes_subdir_path)

        self.segmetations_filename = config.get('DATA', 'SEGMENTATIONS_FILE_NAME')
        self.segmetations_file = osp.join(self.base_path, self.segmetations_filename)
        self.df = pandas.read_excel(self.segmetations_file)

        # Optical flow parameters
        self.fwdof_dir = None
        self.bwdof_dir = None
        self.flow_name = None
        self.flow_level = None
        self.load_flow = False
        if self.flow_mode != LoadFlowMode.NO_LOAD_OF:
            self.load_flow = True
            use_filtered_flow = config.getboolean('PARAMETERS', 'USE_MEDIAN_FILTERED_FLOW')
            self.flow_name = 'flow_m_it0.pt' if use_filtered_flow else 'flow_it0.pt'
            self.flow_level = 'it0'
            self.fwdof_dir = osp.sep.join([config.get('DATA', 'BASE_PATH_3D'), 'optical_flow', 'forward'])
            self.bwdof_dir = osp.sep.join([config.get('DATA', 'BASE_PATH_3D'), 'optical_flow', 'backward'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        df_row = self.df.iloc[[idx]]
        patient_name = df_row.loc[idx, 'Name']
        tsyst = df_row.loc[idx, 'Systole']
        tdias = df_row.loc[idx, 'Diastole']

        # Load 4D nifty [x,y,z,t]
        img4d = nib.load(osp.join(self.volumes_path, patient_name + '.nii.gz'))
        img4d_zyxt = np.swapaxes(img4d.get_fdata(), 0, 2)

        # Load segmentations masks
        mask_syst_zyx, mask_diast_zyx = self.systole_diastole_mask(patient_name)
        if self.mask_transforms is not None:
            mask_syst_zyx = self.mask_transforms(mask_syst_zyx)
            mask_diast_zyx = self.mask_transforms(mask_diast_zyx)

        if self.img4d_transforms is not None:
            img4d_zyxt = self.img4d_transforms(img4d_zyxt)

        m0, mk, init_ts, final_ts = self.prepare_masks(tsyst, tdias, mask_syst_zyx, mask_diast_zyx)

        # Load optical flow if needed
        ff, bf = None, None
        if self.load_flow:
            ff, bf = self.optflow_for_patient(patient_name, init_ts, final_ts, idx)

        return (patient_name, img4d_zyxt, m0, mk, init_ts, final_ts, ff, bf)

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

    def get_diastole_time(self, idx):
        return self.df.iloc[idx]['Diastole']

    def get_original_NT(self, idx):
        return self.df.iloc[idx]['original_NT']

    def load_mask(self, patient_name, ending):
        mask = nib.load(osp.sep.join([self.segmentations_path, patient_name, patient_name + ending]))
        mask = np.swapaxes(mask.get_fdata(), 0, 2)
        return mask

    def index_for_patient(self, patient_name):
        row_patient = self.df[self.df['Name'] == patient_name]
        index_patient = row_patient.index[0]
        return index_patient

    def optflow_for_patient(self, patient, init_ts, final_ts, idx):
        fwd_flows = []
        bwd_flows = []
        fwd_patient_dir = osp.join(self.fwdof_dir, patient)
        bwd_patient_dir = osp.join(self.bwdof_dir, patient)

        times_fwd, times_bwd = self.create_timeline(init_ts, final_ts, self.get_original_NT(idx))
        assert len(times_fwd) == len(times_bwd)

        for i in range(len(times_fwd)):
            fwd_file = osp.sep.join([fwd_patient_dir, f'time{times_fwd[i]}', self.flow_level, self.flow_name])
            bwd_file = osp.sep.join([bwd_patient_dir, f'time{times_bwd[i]}', self.flow_level, self.flow_name])
            fwd_flows.append(torch.load(fwd_file, map_location='cpu'))
            bwd_flows.append(torch.load(bwd_file, map_location='cpu'))

        fwd_t = torch.stack([x.float() for x in fwd_flows], dim=0)
        bwd_t = torch.stack([x.float() for x in bwd_flows], dim=0)
        return (fwd_t, bwd_t)

    def create_timeline(self, init_ts, final_ts, original_NT):
        if self.flow_mode == LoadFlowMode.TRAIN_OF:
            times_fwd = np.arange(init_ts, final_ts, 1)
            times_bwd = np.arange(final_ts, init_ts, -1)
        elif self.flow_mode == LoadFlowMode.PREDICT_OF:
            times_fwd = np.concatenate((np.arange(init_ts, original_NT, 1), np.arange(0, init_ts)))
            times_bwd = np.concatenate((np.arange(final_ts, -1, -1), np.arange(original_NT - 1, final_ts, -1)))
        return (times_fwd, times_bwd)

    def prepare_masks(self, tsyst, tdias, msyst, mdias):
        init_ts = min(tdias, tsyst)
        final_ts = max(tdias, tsyst)

        m0 = mk = None
        if init_ts == tsyst:
            m0 = msyst
            mk = mdias
        else:
            m0 = mdias
            mk = msyst
        return (m0, mk, init_ts, final_ts)

    def save_patients(self, save_dir: str, filename: str):
        output_df_file = osp.join(save_dir, filename)
        self.df.to_excel(output_df_file, index=False)

    def optflow(self, idx):
        patient_name = self.get_patient_name(idx)
        return self.optflow_for_patient(patient_name)
