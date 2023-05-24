from torch.utils.data import Dataset
import os.path as osp
import nibabel as nib
import numpy as np
import os
import pandas


class SingleVentriclePatient:
    def __init__(self, idx=None, df_row=None, volumes_path=None, segmentations_path=None):
        # load from database
        # indexPatient = row.index[0]
        # if indexPatient == idx:
        #     print("index correct")
        # else:
        #     print("index wrong")
        if idx is not None and df_row is not None and volumes_path is not None and segmentations_path is not None:
            self.df_row = df_row
            self.name = df_row.loc[idx, "Name"]
            self.tDiastole = df_row.loc[idx, "Diastole"]
            self.tSystole = df_row.loc[idx, "Systole"]
            self.init_ts = min(self.tDiastole, self.tSystole)
            self.final_ts = max(self.tDiastole, self.tSystole)

            # Load 4D nifty [x,y,z,t]
            self.nii_xyzt = nib.load(osp.join(volumes_path, f'{self.name}.nii.gz'))
        
            self.nii_data_xyzt = self.nii_xyzt.get_fdata()
            self.nii_header_xyzt = self.nii_xyzt.header
            self.nii_data_zyxt = np.swapaxes(self.nii_data_xyzt, 0, 2)

            self.NX = self.nii_data_xyzt.shape[0]
            self.NY = self.nii_data_xyzt.shape[1]
            self.NZ = self.nii_data_xyzt.shape[2]
            self.NT = self.nii_data_xyzt.shape[3]

            if 'orig_NT' in df_row.columns:
               self.NT = df_row.loc[idx, "orig_NT"]

            # get input masks for diastole
            self.nii_mask_diastole_load = nib.load(os.path.sep.join([segmentations_path, self.name, self.name + "_Diastole_Labelmap.nii.gz"]))
            self.hdr_mask_diastole = self.nii_mask_diastole_load.header
            #affine_mask_diastole = nii_mask_diastole_load.affine
            self.nii_mask_diastole_xyz = self.nii_mask_diastole_load.get_fdata()
            self.nii_mask_diastole = np.swapaxes(self.nii_mask_diastole_xyz, 0, 2)

            # get input masks for systole
            self.nii_mask_systole_load = nib.load(os.path.sep.join([segmentations_path, self.name, self.name + "_Systole_Labelmap.nii.gz"]))
            self.hdr_mask_systole = self.nii_mask_systole_load.header
            #affine_mask_systole = nii_mask_systole_load.affine
            self.nii_mask_systole_xyz = self.nii_mask_systole_load.get_fdata()
            self.nii_mask_systole = np.swapaxes(self.nii_mask_systole_xyz, 0, 2)

            try:
                self.full_cycle = df_row.loc[idx, "Full"]
            except KeyError:
                self.full_cycle = False
                
            # Load whole cycle segmentations 
            if self.full_cycle:
                self.nii_masks_load = []
                self.masks_dirs = []
                self.nii_masks_xyz = []
                self.nii_masks_zyx = []
                for t in range(0, self.NT):
                    mask_filename = osp.sep.join([segmentations_path, self.name, self.name + f'_{t}_Labelmap.nii.gz'])
                    self.masks_dirs.append(mask_filename)
                    nii_mask = nib.load(mask_filename)
                    self.nii_masks_load.append(nii_mask)
                    mask_xyz = nii_mask.get_fdata()
                    self.nii_masks_xyz.append(mask_xyz)
                    self.nii_masks_zyx.append(np.swapaxes(mask_xyz, 0, 2))   
            else:
                self.nii_masks_load = self.masks_dirs = self.nii_masks_xyz = self.nii_masks_zyx = None        
        else:
            self.df_row = None
            self.name = self.tDiastole = self.tSystole = None
            self.nii_xyzt = self.nii_data_xyzt = self.nii_header_xyzt = self.nii_data_zyxt = None
            self.NX = self.NY = self.NZ = self.NT = None
            # get input masks for diastole
            self.nii_mask_diastole_load = self.hdr_mask_diastole = self.nii_mask_diastole_xyz = None
            self.nii_mask_diastole = None  # zyx

            # get input masks for systole
            self.nii_mask_systole_load = self.hdr_mask_systole = self.nii_mask_systole_xyz = None
            self.nii_mask_systole = None  # zyx

            # Full cycle masks
            self.full_cycle = self.nii_masks_load = self.masks_dirs = self.nii_masks_xyz = self.nii_masks_zyx = None


class SingleVentricleDataset(Dataset):

    def __init__(self, config, mode='full'):
        # mode = full, train, val, test
        self.config = config
        self.mode = mode

        self.base_path = config.get('DATA', 'BASE_PATH_3D')

        # load data base
        if mode == 'train' or mode == 'val' or mode == 'test':
            self.base_path = osp.join(self.base_path, mode)

        self.segmentations_subdir_path = config.get('DATA', 'SEGMENTATIONS_SUBDIR_PATH')
        self.segmentations_path = osp.join(self.base_path, self.segmentations_subdir_path)

        self.volumes_subdir_path = config.get('DATA', 'VOLUMES_SUBDIR_PATH')
        self.volumes_path = osp.join(self.base_path, self.volumes_subdir_path)

        self.segmentations_filename = config.get('DATA', 'SEGMENTATIONS_FILE_NAME')
        self.segmentations_file = osp.join(self.base_path, self.segmentations_filename)

        self.df = pandas.read_excel(self.segmentations_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        patient = SingleVentriclePatient(idx, self.df.iloc[[idx]], self.volumes_path, self.segmentations_path)
        return patient

    def index_for_patient(self, patient_name):
        row_patient = self.df[self.df['Name'] == patient_name]
        index_patient = row_patient.index[0]
        return index_patient
    
    def get_row(self, patient_name):
        row_patient = self.df[self.df['Name'] == patient_name]
        return row_patient
    
    def get_patient_name(self, idx):
        return self.df.iloc[idx]['Name']
    
    def get_systole_time(self, idx):
        return self.df.iloc[idx]['Systole']
    
    def get_diastole_time(self, idx):
        return self.df.iloc[idx]['Diastole']
    
    