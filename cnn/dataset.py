from torch.utils.data import Dataset
import os.path as osp
import nibabel as nib
import numpy as np
import pandas
from enum import Enum
import json

__all__ = ['LoadFlowMode', 'DatasetMode', 'SingleVentricleDataset']


class DatasetMode(Enum):
    TRAIN = 1
    VAL = 2
    FULL = 3
    TEST = 4


class LoadFlowMode(Enum):
    NO_LOAD = 1
    ED_ES = 2           # Load optical flow from ED to ES and viceversa
    WHOLE_CYCLE = 3     # Load optical flow for the complete carciac cycle


class SingleVentricleDataset(Dataset):
    def __init__(self, config, mode, flow_mode,
                 img4d_transforms=None,
                 mask_transforms=None,
                 flow_transforms=None,
                 full_transforms=None,
                 test_masks_transforms=None):
        self.config = config
        self.mode = mode
        self.flow_mode = flow_mode
        self.img4d_transforms = img4d_transforms    # transformations applied only on data
        self.mask_transforms = mask_transforms      # transformations applied only on masks
        self.flow_transforms = flow_transforms      # transformations applied only on optical flow
        self.full_transforms = full_transforms      # transformations applied on img, mask and optical flow
        self.test_masks_transforms = test_masks_transforms

        self.base_path = config.get('DATA', 'BASE_PATH_3D')
        self.json_ds = self.read_json(osp.join(self.base_path, 'dataset.json'))
        self.img_ext = self.img_file_ending()
        self.label_ext = self.label_file_ending()

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

        # if mode == 'full':
        #     self.df = df
        # else:
        #     self.df = df[df['Split'] == mode]
        #     self.df.reset_index(inplace=True, drop=True)

        #     if fold_indices is not None and mode != 'test':
        #         self.df = self.df.iloc[fold_indices]
        #         self.df.reset_index(inplace=True, drop=True)

        # Optical flow parameters
        self.fwdof_dir = None
        self.bwdof_dir = None
        self.flow_name = None
        self.flow_level = None
        self.load_flow = False
        if self.flow_mode != LoadFlowMode.NO_LOAD:
            self.load_flow = True
            # use_filtered_flow = config.getboolean('PARAMETERS', 'USE_MEDIAN_FILTERED_FLOW')
            use_filtered_flow = True
            self.flow_name = 'flow_m_it0.npy' if use_filtered_flow else 'flow_it0.npy'
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
        try:
            full_cycle = df_row.loc[idx, 'Full']
        except:
            full_cycle = False

        # Load 4D nifty
        img4d = nib.load(osp.join(self.volumes_path, patient_name + self.img_ext))
        img4d_zyxt = np.swapaxes(img4d.get_fdata(), 0, 2)

        # Load segmentations masks
        mask_syst_zyx = self.load_mask(patient_name, f'_Systole_Labelmap{self.label_ext}')
        mask_diast_zyx = self.load_mask(patient_name, f'_Diastole_Labelmap{self.label_ext}')

        # Load whole cycle masks
        if full_cycle:
            orig_NT = df_row.loc[idx, 'orig_NT']
            masks = np.empty(shape=(*mask_syst_zyx.shape, orig_NT), dtype=mask_syst_zyx.dtype)
            for t in range(orig_NT):
                masks[..., t] = self.load_mask(patient_name, f'_{t}_Labelmap{self.label_ext}')
        else:
            masks = None

        m0, mk, init_ts, final_ts = self.prepare_masks(tsyst, tdias, mask_syst_zyx, mask_diast_zyx)

        # Load optical flow
        ff, bf = None, None
        if self.load_flow:
            ff, bf = self.optflow_for_patient(patient_name, init_ts, final_ts, idx)

        # Full transforms
        if self.full_transforms is not None:
            img4d_zyxt, m0, mk, ff, bf, masks = self.full_transforms(img4d_zyxt, m0, mk, ff, bf, masks)

        # Mask transformations
        if self.mask_transforms is not None:
            m0 = self.mask_transforms(m0)
            mk = self.mask_transforms(mk)

        # Complete cardiac cycle masks transforms
        if self.test_masks_transforms is not None:
            masks = self.test_masks_transforms(masks)

        # Image transformations
        if self.img4d_transforms is not None:
            img4d_zyxt = self.img4d_transforms(img4d_zyxt)

        # Optical flow transformation
        if self.flow_transforms is not None:
            ff = self.flow_transforms(ff)
            bf = self.flow_transforms(bf)

        return (patient_name, img4d_zyxt, m0, mk, masks, init_ts, final_ts, ff, bf)

    def num_classes(self):
        return self.json_ds['n_classes']

    def name(self):
        return self.json_ds['name']

    def img_file_ending(self):
        return self.json_ds['img_file_ending']

    def label_file_ending(self):
        return self.json_ds['label_file_ending']

    def read_json(self, filepath):
        f = open(filepath, mode='r')
        data = json.load(f)
        f.close()
        return data

    def header(self, idx):
        df_row = self.df.iloc[[idx]]
        patient_name = df_row.loc[idx, 'Name']
        mask_nii = nib.load(osp.sep.join([self.segmentations_path, patient_name,
                            patient_name + '_Systole_Labelmap.nii']))
        return mask_nii.header

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
        try:
            full_cycle = self.df.iloc[idx]['Full']
        except:
            full_cycle = False
        return full_cycle

    def cutted_shape(self, idx):
        NX = self.df.iloc[idx]['cut_NX']
        NY = self.df.iloc[idx]['cut_NY']
        NZ = self.df.iloc[idx]['cut_NZ']
        return NZ, NY, NX

    def orig_shape(self, idx):
        NX = self.df.iloc[idx]['original_NX']
        NY = self.df.iloc[idx]['original_NY']
        NZ = self.df.iloc[idx]['original_NZ']
        return NZ, NY, NX

    def cut_bounds(self, idx):
        x_min = self.df.iloc[idx]['x_min']
        x_max = self.df.iloc[idx]['x_max']
        y_min = self.df.iloc[idx]['y_min']
        y_max = self.df.iloc[idx]['y_max']
        z_min = self.df.iloc[idx]['z_min']
        z_max = self.df.iloc[idx]['z_max']
        return z_min, z_max, y_min, y_max, x_min, x_max

    def get_diastole_time(self, idx):
        return self.df.iloc[idx]['Diastole']

    def get_original_NT(self, idx):
        return self.df.iloc[idx]['orig_NT']

    def load_mask(self, patient_name, ending):
        mask = nib.load(osp.sep.join([self.segmentations_path, patient_name, patient_name + ending]))
        mask = np.swapaxes(mask.get_fdata(), 0, 2)
        return mask

    def index_for_patient(self, patient_name):
        try:
            row_patient = self.df[self.df['Name'] == patient_name]
            index_patient = row_patient.index[0]
        except:
            return -1, False
        return index_patient, True

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
            fwd_flows.append(np.load(fwd_file))
            bwd_flows.append(np.load(bwd_file))

        fwd_t = np.stack([x for x in fwd_flows], axis=4)
        bwd_t = np.stack([x for x in bwd_flows], axis=4)
        return (fwd_t, bwd_t)

    def create_timeline(self, init_ts, final_ts, original_NT):
        if self.flow_mode == LoadFlowMode.ED_ES:
            times_fwd = np.arange(init_ts, final_ts, 1)
            times_bwd = np.arange(final_ts, init_ts, -1)
        elif self.flow_mode == LoadFlowMode.WHOLE_CYCLE:
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
