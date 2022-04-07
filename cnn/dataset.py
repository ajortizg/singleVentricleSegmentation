from torch.utils.data import Dataset
import os.path as osp
from glob import glob
import nibabel as nib
import numpy as np
import os
import pandas


class SingleVentricleDataset(Dataset):
    def __init__(self, config, transforms=None):
        self.transforms = transforms
        base = config.get("DATA", "BASE_PATH_3D")

        self.masks_root = osp.join(base, config.get(
            "DATA", "SEGMENTATIONS_SUBDIR_PATH"))
        self.volume_files = glob(osp.join(
            osp.join(base, config.get("DATA", "VOLUMES_SUBDIR_PATH")), "*.nii.gz"))

        segmentations_file = osp.join(
            base, config.get("DATA", "SEGMENTATIONS_FILE_NAME"))
        self.df = pandas.read_excel(segmentations_file)

    def __len__(self):
        return len(self.volume_files)

    def __getitem__(self, idx):
        # Load 4D nifty [x,y,z,t]
        vol = nib.load(self.volume_files[idx])
        vol_zyxt = np.swapaxes(vol.get_fdata(), 0, 2)

        # Read time steps for systole and diastole
        patient_name = self.get_patient_name(idx)
        row_patient = self.df[self.df["Name"] == patient_name]
        index_patient = row_patient.index[0]
        tsyst = row_patient.loc[index_patient, "Systole"]
        tdias = row_patient.loc[index_patient, "Diastole"]

        # Load segmentations masks
        mask_syst_zyx = self.load_mask(patient_name, "_Systole_Labelmap.nii")
        mask_diast_zyx = self.load_mask(patient_name, "_Diastole_Labelmap.nii")

        if self.transforms is not None:
            vol_zyxt = self.transforms(vol_zyxt)
            mask_syst_zyx = self.transforms(mask_syst_zyx)
            mask_diast_zyx = self.transforms(mask_diast_zyx)

        return (vol_zyxt, mask_syst_zyx, mask_diast_zyx, tsyst, tdias)

    def get_patient_name(self, idx):
        return (self.volume_files[idx].split(os.sep)[-1]).split(".")[0]

    def load_mask(self, patient_name, ending):
        mask = nib.load(osp.sep.join(
            [self.masks_root, patient_name, patient_name + ending]))
        return np.swapaxes(mask.get_fdata(), 0, 2)

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
            return (None, None, None, None, None)

        
                