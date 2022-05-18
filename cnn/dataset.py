from torch.utils.data import Dataset
import os.path as osp
from glob import glob
import nibabel as nib
import numpy as np
import os
import cv2
import pandas
import torch


class SingleVentricleDataset(Dataset):
    def __init__(self, config, load_flow):
        self.config = config
        self.load_flow = load_flow
        base = config.get('DATA', 'BASE_PATH_3D')

        self.masks_root = osp.join(base, config.get('DATA', 'SEGMENTATIONS_SUBDIR_PATH'))
        self.volume_files = glob(osp.join(osp.join(base, config.get('DATA', 'VOLUMES_SUBDIR_PATH')), '*.nii.gz'))

        masks_file = osp.join(base, config.get('DATA', 'SEGMENTATIONS_FILE_NAME'))
        self.df = pandas.read_excel(masks_file)

        # Optical flow parameters
        self.fwdof_dir = None
        self.bwdof_dir = None
        self.flow_name = 'flow_it0.pt'
        self.flow_level = 'it0'

        if self.load_flow:
            self.fwdof_dir = self.config.get('DATA', 'FWD_OPTFLOW_RESULTS_DIR')
            self.bwdof_dir = self.config.get('DATA', 'BWD_OPTFLOW_RESULTS_DIR')

    def __len__(self):
        return len(self.volume_files)

    def __getitem__(self, idx):
        # Load 4D nifty [x,y,z,t]
        vol = nib.load(self.volume_files[idx])
        vol_zyxt = torch.from_numpy(np.swapaxes(vol.get_fdata(), 0, 2)).float()

        # Read time steps for systole and diastole
        patient_name = self.get_patient_name(idx)
        row_patient = self.df[self.df['Name'] == patient_name]
        index_patient = row_patient.index[0]
        tsyst = int(row_patient.loc[index_patient, 'Systole'])
        tdias = int(row_patient.loc[index_patient, 'Diastole'])

        # Load segmentations masks and convert them to torch tensors
        mask_syst_zyx = self.load_mask(patient_name, '_Systole_Labelmap.nii')
        mask_syst_zyx = torch.from_numpy(mask_syst_zyx).float()
        mask_diast_zyx = self.load_mask(patient_name, '_Diastole_Labelmap.nii')
        mask_diast_zyx = torch.from_numpy(mask_diast_zyx).float()

        ff, bf = None, None
        if self.load_flow:
            ff, bf = self.optflow_results_for_patient(patient_name)
        return (patient_name, vol_zyxt, mask_syst_zyx, mask_diast_zyx, tsyst, tdias, ff, bf)

    def get_patient_name(self, idx):
        return (self.volume_files[idx].split(os.sep)[-1]).split(".")[0]

    def load_mask(self, patient_name, ending):
        mask = nib.load(osp.sep.join([self.masks_root, patient_name, patient_name + ending]))
        mask = np.swapaxes(mask.get_fdata(), 0, 2)
        # TODO! Improve this to avoid flip by slices
        NZ, _, _ = mask.shape
        for z in range(NZ):
            mask[z, :, :] = np.flip(mask[z, :, :], 0)
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

    def optflow_results(self, idx):
        patient_name = self.get_patient_name(idx)
        return self.optflow_results_for_patient(patient_name)

    def optflow_results_for_patient(self, patient):
        fwd_flows = []
        bwd_flows = []
        fwd_patient_dir = osp.join(self.fwdof_dir, patient)
        fwd_time_dirs = sorted(os.listdir(fwd_patient_dir))
        bwd_patient_dir = osp.join(self.bwdof_dir, patient)
        bwd_time_dirs = sorted(os.listdir(bwd_patient_dir))
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
