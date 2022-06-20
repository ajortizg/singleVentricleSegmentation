from torch.utils.data import Dataset
import os.path as osp
from glob import glob
import nibabel as nib
import numpy as np
import os
import pandas
import torch
from enum import Enum


class DatasetMode(Enum):
    TRAIN = 1
    VAL = 2
    FULL = 3


class SingleVentricleDataset(Dataset):
    def __init__(self, config, mode, load_flow,
                 data_transforms=None,
                 mask_transforms=None,
                 data_mask_transforms=None,
                 flow_transforms=None):
        self.config = config
        self.load_flow = load_flow
        self.mode = mode
        self.data_transforms = data_transforms              # transformations applied only on data
        self.mask_tranforms = mask_transforms               # transformations applied only on masks
        self.data_mask_transforms = data_mask_transforms    # transformations applied on data and masks
        self.flow_transforms = flow_transforms              # transformations applied on optical flow

        self.base_path = config.get('DATA', 'BASE_PATH_3D')
        if mode == DatasetMode.TRAIN:
            self.base_path = osp.join(self.base_path, 'train')
            self.mode_str = 'train'
        elif mode == DatasetMode.VAL:
            self.base_path = osp.join(self.base_path, 'val')
            self.mode_str = 'val'
        else:
            self.mode_str = 'full'

        self.segmentations_subdir_path = config.get('DATA', 'SEGMENTATIONS_SUBDIR_PATH')
        self.masks_root = osp.join(self.base_path, self.segmentations_subdir_path)

        self.volumes_subdir_path = config.get('DATA', 'VOLUMES_SUBDIR_PATH')
        self.volumes_path = osp.join(self.base_path, self.volumes_subdir_path)
        self.volume_files = glob(osp.join(self.volumes_path, '*.nii.gz'))

        self.segmetations_filename = config.get('DATA', 'SEGMENTATIONS_FILE_NAME')
        self.segmetations_path = osp.join(self.base_path, self.segmetations_filename)
        self.df = pandas.read_excel(self.segmetations_path)

        # Optical flow parameters
        self.fwdof_dir = None
        self.bwdof_dir = None
        self.flow_name = None
        self.flow_level = None

        if self.load_flow:
            use_filtered_flow = config.getboolean('PARAMETERS', 'USE_MEDIAN_FILTERED_FLOW')
            self.flow_name = 'flow_m_it0.pt' if use_filtered_flow else 'flow_it0.pt'
            self.flow_level = 'it0'
            self.fwdof_dir = self.config.get('DATA', 'FWD_OPTFLOW_DIR')
            self.bwdof_dir = self.config.get('DATA', 'BWD_OPTFLOW_DIR')

    def __len__(self):
        return len(self.volume_files)

    def __getitem__(self, idx):
        # Load 4D nifty [x,y,z,t]
        vol = nib.load(self.volume_files[idx])
        vol_zyxt = torch.from_numpy(np.swapaxes(vol.get_fdata(), 0, 2)).float()

        # Read time steps for systole and diastole
        patient_name = self.get_patient_name(idx)
        tsyst, tdias = self.systole_diastole_time(patient_name)

        # Load segmentations masks and convert them to torch tensors
        mask_syst_zyx, mask_diast_zyx = self.systole_diastole_mask(patient_name)

        if self.data_mask_transforms is not None:
            vol_zyxt, mask_syst_zyx, mask_diast_zyx = self.data_mask_transforms(vol_zyxt, mask_syst_zyx, mask_diast_zyx)

        if self.mask_tranforms is not None:
            mask_syst_zyx = self.mask_tranforms(mask_syst_zyx)
            mask_diast_zyx = self.mask_tranforms(mask_diast_zyx)

        if self.data_transforms is not None:
            vol_zyxt = self.data_transforms(vol_zyxt)

        m0, mk, init_ts, final_ts = self.prepare_masks(tsyst, tdias, mask_syst_zyx, mask_diast_zyx)

        # Load optical flow if needed
        ff, bf = None, None
        if self.load_flow:
            ff, bf = self.optflow_for_patient(patient_name)

        return (patient_name, vol_zyxt, m0, mk, init_ts, final_ts, ff, bf)

    def systole_diastole_time(self, patient_name):
        row_patient = self.df[self.df['Name'] == patient_name]
        index_patient = row_patient.index[0]
        tsyst = int(row_patient.loc[index_patient, 'Systole'])
        tdias = int(row_patient.loc[index_patient, 'Diastole'])
        return (tsyst, tdias)

    def systole_diastole_mask(self, patient_name):
        mask_syst_zyx = self.load_mask(patient_name, '_Systole_Labelmap.nii')
        mask_syst_zyx = torch.from_numpy(mask_syst_zyx).float()
        mask_diast_zyx = self.load_mask(patient_name, '_Diastole_Labelmap.nii')
        mask_diast_zyx = torch.from_numpy(mask_diast_zyx).float()
        return (mask_syst_zyx, mask_diast_zyx)

    def get_patient_name(self, idx):
        return (self.volume_files[idx].split(os.sep)[-1]).split(".")[0]

    def load_mask(self, patient_name, ending):
        mask = nib.load(osp.sep.join([self.masks_root, patient_name, patient_name + ending]))
        mask = np.swapaxes(mask.get_fdata(), 0, 2)
        return mask

    def index_for_patient(self, patient_name):
        found = False
        for idx in range(len(self.volume_files)):
            query = self.get_patient_name(idx)
            if patient_name == query:
                found = True
                return (idx, found)
        return (-1, found)

    def optflow(self, idx):
        patient_name = self.get_patient_name(idx)
        return self.optflow_for_patient(patient_name)

    def optflow_for_patient(self, patient):
        fwd_flows = []
        bwd_flows = []
        fwd_patient_dir = osp.join(self.fwdof_dir, patient)
        fwd_time_dirs = sorted(os.listdir(fwd_patient_dir))
        bwd_patient_dir = osp.join(self.bwdof_dir, patient)
        # bwd_time_dirs = sorted(os.listdir(bwd_patient_dir))
        bwd_time_dirs = sorted(os.listdir(bwd_patient_dir), reverse=True)
        assert len(fwd_time_dirs) == len(bwd_time_dirs)

        for fwd_dir, bwd_dir in zip(fwd_time_dirs, bwd_time_dirs):
            # Read forward optical flow
            fwd_path = osp.join(fwd_patient_dir, fwd_dir)
            if osp.isdir(fwd_path):
                flow_path = osp.sep.join([fwd_path, self.flow_level, self.flow_name])
                u = torch.load(flow_path, map_location='cpu')
                if self.flow_transforms is not None:
                    u = self.flow_transforms(u)
                fwd_flows.append(u)

            # Read backward optical flow
            bwd_path = osp.join(bwd_patient_dir, bwd_dir)
            if osp.isdir(bwd_path):
                flow_path = osp.sep.join([bwd_path, self.flow_level, self.flow_name])
                u = torch.load(flow_path, map_location='cpu')
                if self.flow_transforms is not None:
                    u = self.flow_transforms(u)
                bwd_flows.append(u)

        fwd_t = torch.stack([x.float() for x in fwd_flows], dim=0)
        bwd_t = torch.stack([x.float() for x in bwd_flows], dim=0)

        # return (fwd_flows, bwd_flows)
        return (fwd_t, bwd_t)

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
        filepath = osp.join(save_dir, filename)
        with open(filepath, 'w') as pfile:
            for i in range(len(self.volume_files)):
                pfile.write(self.get_patient_name(i) + '\n')
        pfile.close()
