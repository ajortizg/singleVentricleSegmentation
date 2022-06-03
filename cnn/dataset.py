from torch.utils.data import Dataset
import os.path as osp
from glob import glob
import nibabel as nib
import numpy as np
import os
import pandas
import torch
from enum import Enum


def split_train_val_dataset(total_patients: int, val_percent: float = 0.2):
    val_patients = int(total_patients * val_percent)
    train_patients = total_patients - val_patients
    assert((val_patients + train_patients) == total_patients)

    total_idxs = np.arange(total_patients).astype(int)
    val_idxs = total_idxs[:val_patients]  # First N patients for evaluation
    train_idxs = total_idxs[val_patients:]  # The rest for training
    return (train_idxs, val_idxs)


class DatasetMode(Enum):
    TRAIN = 1
    VAL = 2
    FULL = 3


class SingleVentricleDataset(Dataset):
    def __init__(self, config, mode, transforms, load_flow):
        self.config = config
        self.load_flow = load_flow
        self.mode = mode
        self.transforms = transforms
        self.base_path = config.get('DATA', 'BASE_PATH_3D')

        self.segmentations_subdir_path = config.get('DATA', 'SEGMENTATIONS_SUBDIR_PATH')
        self.masks_root = osp.join(self.base_path, self.segmentations_subdir_path)

        self.volumes_subdir_path = config.get('DATA', 'VOLUMES_SUBDIR_PATH')
        self.volumes_path = osp.join(self.base_path, self.volumes_subdir_path)
        self.volume_files = glob(osp.join(self.volumes_path, '*.nii.gz'))

        # Split for Train and Test
        train_idxs, val_idxs = split_train_val_dataset(len(self.volume_files), 0.2)
        if mode == DatasetMode.TRAIN:
            self.volume_files = np.asarray(self.volume_files)[train_idxs]
        elif mode == DatasetMode.VAL:
            self.volume_files = np.asarray(self.volume_files)[val_idxs]

        self.segmetations_filename = config.get('DATA', 'SEGMENTATIONS_FILE_NAME')
        self.segmetations_path = osp.join(self.base_path, self.segmetations_filename)
        self.df = pandas.read_excel(self.segmetations_path)

        # These patients need to be flipped along the three axis
        self.flip_all = {'Adolescent_7',
                         'Adolescent_26',
                         'Adolescent_54',
                         'Adolescent_75',
                         'Adult_6',
                         'Adult_11',
                         'Adult_16',
                         'Adult_17',
                         'Child_27'}
        self.flip_masks = config.getboolean('PARAMETERS', 'FLIP_MASKS')

        # Optical flow parameters
        self.fwdof_dir = None
        self.bwdof_dir = None
        self.flow_name = None
        self.flow_level = None

        if self.load_flow:
            use_filtered_flow = config.getboolean('PARAMETERS', 'USE_MEDIAN_FILTERED_FLOW')
            self.flow_name = 'flow_m_it0.pt' if use_filtered_flow else 'flow_it0.pt'
            self.flow_level = 'it0'
            self.fwdof_dir = self.config.get('DATA', 'FWD_OPTFLOW_RESULTS_DIR')
            self.bwdof_dir = self.config.get('DATA', 'BWD_OPTFLOW_RESULTS_DIR')

    def __len__(self):
        return len(self.volume_files)

    def __getitem__(self, idx):
        # Load 4D nifty [x,y,z,t]
        vol = nib.load(self.volume_files[idx])
        vol_zyxt = torch.from_numpy(np.swapaxes(vol.get_fdata(), 0, 2)).float()
        if self.transforms is not None:
            for t in self.transforms:
                vol_zyxt = t(vol_zyxt)
        # vol_zyxt = torch_utils.normalize(vol_zyxt)
        # vol_zyxt = (vol_zyxt - MEAN) / STD

        # Read time steps for systole and diastole
        patient_name = self.get_patient_name(idx)
        tsyst, tdias = self.systole_diastole_time(patient_name)

        # Load segmentations masks and convert them to torch tensors
        mask_syst_zyx, mask_diast_zyx = self.systole_diastole_mask(patient_name)

        ff, bf = None, None
        if self.load_flow:
            ff, bf = self.optflow_for_patient(patient_name)

        m0, mk, init_ts, final_ts = self.prepare_masks(tsyst, tdias, mask_syst_zyx, mask_diast_zyx)
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
        if self.flip_masks:
            mask_flipped = np.flip(mask, (0, 1, 2)).copy() if patient_name in self.flip_all else np.flip(mask, 1).copy()
            return mask_flipped
        else:
            return mask

    def index_for_patient(self, patient_name):
        found = False
        for idx in range(self.__len__()):
            query = self.get_patient_name(idx)
            if patient_name == query:
                found = True
                return (idx, found)
        return (-1, found)

    def data_for_patient(self, patient_name):
        idx, found = self.index_for_patient(patient_name)
        if found:
            return self.__getitem__(idx)
        else:
            return (None, None, None, None, None, None, None, None)

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
                fwd_flows.append(u)

            # Read backward optical flow
            bwd_path = osp.join(bwd_patient_dir, bwd_dir)
            if osp.isdir(bwd_path):
                flow_path = osp.sep.join([bwd_path, self.flow_level, self.flow_name])
                u = torch.load(flow_path, map_location='cpu')
                bwd_flows.append(u)

        return (fwd_flows, bwd_flows)

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


# class SingleVentricleDatasetBatch(Dataset):
#     def __init__(self, config, mode, load_flow):
#         self.config = config
#         self.load_flow = load_flow
#         self.mode = mode
#         self.base_path = config.get('DATA', 'BASE_PATH_3D')

#         self.segmentations_subdir_path = config.get('DATA', 'SEGMENTATIONS_SUBDIR_PATH')
#         self.masks_root = osp.join(self.base_path, self.segmentations_subdir_path)

#         self.volumes_subdir_path = config.get('DATA', 'VOLUMES_SUBDIR_PATH')
#         self.volumes_path = osp.join(self.base_path, self.volumes_subdir_path)
#         self.volume_files = glob(osp.join(self.volumes_path, '*.nii.gz'))

#         # Split for Train and Test
#         train_idxs, val_idxs = split_train_val_dataset(len(self.volume_files), 0.2)
#         if mode == DatasetMode.TRAIN:
#             self.volume_files = np.asarray(self.volume_files)[train_idxs]
#         elif mode == DatasetMode.VAL:
#             self.volume_files = np.asarray(self.volume_files)[val_idxs]

#         masks_file = osp.join(self.base_path, config.get('DATA', 'SEGMENTATIONS_FILE_NAME'))
#         self.df = pandas.read_excel(masks_file)

#         # These patients need to be flipped along the three axis
#         self.flip_all = {'Adolescent_7',
#                          'Adolescent_26',
#                          'Adolescent_54',
#                          'Adolescent_75',
#                          'Adult_6',
#                          'Adult_11',
#                          'Adult_16',
#                          'Adult_17',
#                          'Child_27'}
#         self.flip_masks = config.getboolean('PARAMETERS', 'FLIP_MASKS')

#         # Optical flow parameters
#         self.fwdof_dir = None
#         self.bwdof_dir = None
#         self.flow_name = None
#         self.flow_level = None

#         if self.load_flow:
#             use_filtered_flow = config.getboolean('PARAMETERS', 'USE_MEDIAN_FILTERED_FLOW')
#             self.flow_name = 'flow_m_it0.pt' if use_filtered_flow else 'flow_it0.pt'
#             self.flow_level = 'it0'
#             self.fwdof_dir = self.config.get('DATA', 'FWD_OPTFLOW_RESULTS_DIR')
#             self.bwdof_dir = self.config.get('DATA', 'BWD_OPTFLOW_RESULTS_DIR')

#     def __len__(self):
#         return len(self.volume_files)

#     def __getitem__(self, idx):
#         # Load 4D nifty [x,y,z,t]
#         vol = nib.load(self.volume_files[idx])
#         vol_zyxt = torch.from_numpy(np.swapaxes(vol.get_fdata(), 0, 2)).float()
#         vol_zyxt = torch_utils.normalize(vol_zyxt)

#         # Read time steps for systole and diastole
#         patient_name = self.get_patient_name(idx)
#         row_patient = self.df[self.df['Name'] == patient_name]
#         index_patient = row_patient.index[0]
#         tsyst = int(row_patient.loc[index_patient, 'Systole'])
#         tdias = int(row_patient.loc[index_patient, 'Diastole'])

#         # Load segmentations masks and convert them to torch tensors
#         mask_syst_zyx = self.load_mask(patient_name, '_Systole_Labelmap.nii')
#         mask_syst_zyx = torch.from_numpy(mask_syst_zyx).float()
#         mask_diast_zyx = self.load_mask(patient_name, '_Diastole_Labelmap.nii')
#         mask_diast_zyx = torch.from_numpy(mask_diast_zyx).float()

#         # Pad dataset
#         if self.mode != DatasetMode.FULL:
#             vol_zyxt = self.pad(vol_zyxt)
#             mask_syst_zyx = self.pad(mask_syst_zyx)
#             mask_diast_zyx = self.pad(mask_diast_zyx)

#         ff, bf = None, None
#         if self.load_flow:
#             ff, bf = self.optflow_for_patient(patient_name)

#         m0, mk, init_ts, final_ts = self.prepare_masks(tsyst, tdias, mask_syst_zyx, mask_diast_zyx)
#         return (patient_name, vol_zyxt, m0, mk, init_ts, final_ts, ff, bf)

#     def pad(self, x):
#         nz, ny, nx, nt = 30, 124, 113, 40
#         if len(x.shape) == 5:
#             NZ, NY, NX, _, NT = x.shape
#             diff_z = nz - NZ
#             diff_y = ny - NY
#             diff_x = nx - NX
#             diff_t = nt - NT
#             return F.pad(x,
#                          [0, diff_t,
#                           0, 0,
#                           diff_x // 2, diff_x - diff_x // 2,
#                           diff_y // 2, diff_y - diff_y // 2,
#                           diff_z // 2, diff_z - diff_z // 2])
#         if len(x.shape) == 4:
#             NZ, NY, NX, NT = x.shape
#             diff_z = nz - NZ
#             diff_y = ny - NY
#             diff_x = nx - NX
#             diff_t = nt - NT
#             return F.pad(x,
#                          [0, diff_t,
#                           diff_x // 2, diff_x - diff_x // 2,
#                           diff_y // 2, diff_y - diff_y // 2,
#                           diff_z // 2, diff_z - diff_z // 2])
#         elif len(x.shape) == 3:
#             NZ, NY, NX = x.shape
#             diff_z = nz - NZ
#             diff_y = ny - NY
#             diff_x = nx - NX
#             return F.pad(x,
#                          [diff_x // 2, diff_x - diff_x // 2,
#                           diff_y // 2, diff_y - diff_y // 2,
#                           diff_z // 2, diff_z - diff_z // 2])
#         else:
#             return x

#     def get_patient_name(self, idx):
#         return (self.volume_files[idx].split(os.sep)[-1]).split(".")[0]

#     def load_mask(self, patient_name, ending):
#         mask = nib.load(osp.sep.join([self.masks_root, patient_name, patient_name + ending]))
#         mask = np.swapaxes(mask.get_fdata(), 0, 2)
#         if self.flip_masks:
#             mask_flipped = np.flip(mask, (0, 1, 2)).copy() if patient_name in self.flip_all else np.flip(mask, 1).copy()
#             return mask_flipped
#         else:
#             return mask

#     def index_for_patient(self, patient_name):
#         found = False
#         for idx in range(self.__len__()):
#             query = self.get_patient_name(idx)
#             if patient_name == query:
#                 found = True
#                 return (idx, found)
#         return (-1, found)

#     def data_for_patient(self, patient_name):
#         idx, found = self.index_for_patient(patient_name)
#         if found:
#             return self.__getitem__(idx)
#         else:
#             return (None, None, None, None, None, None, None, None)

#     def optflow(self, idx):
#         patient_name = self.get_patient_name(idx)
#         return self.optflow_for_patient(patient_name)

#     def optflow_for_patient(self, patient):
#         fwd_flows = []
#         bwd_flows = []
#         fwd_patient_dir = osp.join(self.fwdof_dir, patient)
#         fwd_time_dirs = sorted(os.listdir(fwd_patient_dir))
#         bwd_patient_dir = osp.join(self.bwdof_dir, patient)
#         bwd_time_dirs = sorted(os.listdir(bwd_patient_dir), reverse=True)
#         assert len(fwd_time_dirs) == len(bwd_time_dirs)

#         for fwd_dir, bwd_dir in zip(fwd_time_dirs, bwd_time_dirs):
#             # Read forward optical flow
#             fwd_path = osp.join(fwd_patient_dir, fwd_dir)
#             if osp.isdir(fwd_path):
#                 flow_path = osp.sep.join([fwd_path, self.flow_level, self.flow_name])
#                 u = torch.load(flow_path, map_location='cpu')
#                 fwd_flows.append(u)

#             # Read backward optical flow
#             bwd_path = osp.join(bwd_patient_dir, bwd_dir)
#             if osp.isdir(bwd_path):
#                 flow_path = osp.sep.join([bwd_path, self.flow_level, self.flow_name])
#                 u = torch.load(flow_path, map_location='cpu')
#                 bwd_flows.append(u)

#         fwd_t = torch.stack([x.float() for x in fwd_flows], dim=4)
#         bwd_t = torch.stack([x.float() for x in bwd_flows], dim=4)

#         if self.mode != DatasetMode.FULL:
#             fwd_t = self.pad(fwd_t)
#             bwd_t = self.pad(bwd_t)

#         return (fwd_t, bwd_t)

#     def prepare_masks(self, tsyst, tdias, msyst, mdias):
#         init_ts = min(tdias, tsyst)
#         final_ts = max(tdias, tsyst)

#         m0 = mk = None
#         if init_ts == tsyst:
#             m0 = msyst
#             mk = mdias
#         else:
#             m0 = mdias
#             mk = msyst
#         return (m0, mk, init_ts, final_ts)
