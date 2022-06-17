from torch.utils.data import Dataset
import os.path as osp
from glob import glob
import nibabel as nib
import numpy as np
import os
import pandas
import torch
from enum import Enum


# def split_train_val_dataset(total_patients: int, val_percent: float = 0.2):
#     val_patients = int(total_patients * val_percent)
#     train_patients = total_patients - val_patients
#     assert((val_patients + train_patients) == total_patients)

#     total_idxs = np.arange(total_patients).astype(int)
#     val_idxs = total_idxs[:val_patients]  # First N patients for evaluation
#     train_idxs = total_idxs[val_patients:]  # The rest for training
#     return (train_idxs, val_idxs)


# class DatasetMode(Enum):
#     TRAIN = 1
#     VAL = 2
#     FULL = 3


# val_patients = {'Adolescent_1',
#                 'Adolescent_20',
#                 'Adolescent_53',
#                 'Adolescent_62',
#                 'Adult_3',
#                 'Adult_11',
#                 'Adult_24',
#                 'Adult_36',
#                 'Child_2',
#                 'Child_14',
#                 'Child_24',
#                 'Child_36'}


class SingleVentriclePatient:
    def __init__(self, idx, df_row, volume_file, segmentations_path):
        
        # Load 4D nifty [x,y,z,t]
        self.nii_xyzt = nib.load(volume_file)
        self.nii_data_xyzt = self.nii_xyzt.get_fdata()
        self.nii_header_xyzt = self.nii_xyzt.header
        self.nii_data_zyxt = np.swapaxes(self.nii_data_xyzt, 0, 2)

        self.NX = self.nii_data_xyzt.shape[0]
        self.NY = self.nii_data_xyzt.shape[1]
        self.NZ = self.nii_data_xyzt.shape[2]
        self.NT = self.nii_data_xyzt.shape[3]

        # load from database
        # indexPatient = row.index[0]
        # if indexPatient == idx:
        #     print("index correct")
        # else:
        #     print("index wrong")
        self.name = df_row.loc[idx, "Name"]
        self.tDiastole = df_row.loc[idx, "Diastole"]
        self.tSystole = df_row.loc[idx, "Systole"]

        # get input masks for diastole
        self.nii_mask_diastole_load = nib.load(os.path.sep.join([segmentations_path, self.name, self.name + "_Diastole_Labelmap.nii"]))
        self.hdr_mask_diastole = self.nii_mask_diastole_load.header
        #affine_mask_diastole = nii_mask_diastole_load.affine
        self.nii_mask_diastole_xyz = self.nii_mask_diastole_load.get_fdata()
        self.nii_mask_diastole = np.swapaxes(self.nii_mask_diastole_xyz, 0, 2)

        # get input masks for systole
        self.nii_mask_systole_load = nib.load(os.path.sep.join([segmentations_path, self.name, self.name + "_Systole_Labelmap.nii"]))
        self.hdr_mask_systole = self.nii_mask_systole_load.header
        #affine_mask_systole = nii_mask_systole_load.affine
        self.nii_mask_systole_xyz = self.nii_mask_systole_load.get_fdata()
        self.nii_mask_systole = np.swapaxes(self.nii_mask_systole_xyz, 0, 2)


        # # Load segmentations masks and convert them to torch tensors
        # mask_syst_zyx, mask_diast_zyx = self.systole_diastole_mask(patient_name)

        # if self.data_mask_transforms is not None:
        #     vol_zyxt, mask_syst_zyx, mask_diast_zyx = self.data_mask_transforms(vol_zyxt, mask_syst_zyx, mask_diast_zyx)

        # if self.mask_tranforms is not None:
        #     mask_syst_zyx = self.mask_tranforms(mask_syst_zyx)
        #     mask_diast_zyx = self.mask_tranforms(mask_diast_zyx)

        # if self.data_transforms is not None:
        #     vol_zyxt = self.data_transforms(vol_zyxt)

        # m0, mk, init_ts, final_ts = self.prepare_masks(tSystole, tDiastole, mask_syst_zyx, mask_diast_zyx)

        # # Load optical flow if needed
        # ff, bf = None, None
        # if self.load_flow:
        #     ff, bf = self.optflow_for_patient(patient_name)

        # return (patient_name, nii_data_xyzt, nii_header_xyzt, vol_zyxt)

        # return (patient_name, vol_zyxt, m0, mk, init_ts, final_ts, ff, bf)


class SingleVentricleDataset(Dataset):

    def __init__(self, config):
                #   , mode, load_flow,
                #  data_transforms=None,
                #  mask_transforms=None,
                #  data_mask_transforms=None,
                #  flow_transforms=None):


        self.config = config
        # self.load_flow = load_flow
        # self.mode = mode
        # self.data_transforms = data_transforms              # transformations applied only on data
        # self.mask_tranforms = mask_transforms               # transformations applied only on masks
        # self.data_mask_transforms = data_mask_transforms    # transformations applied on data and masks
        # self.flow_transforms = flow_transforms              # transformations applied on optical flow
        
        # load data base
        self.base_path = config.get('DATA', 'BASE_PATH_3D') #old: BASE_PATH_3D = config.get('DATA', 'BASE_PATH_3D')

        self.segmentations_subdir_path = config.get('DATA', 'SEGMENTATIONS_SUBDIR_PATH')  #old: SEGMENTATIONS_SUBDIR_PATH = config.get('DATA', 'SEGMENTATIONS_SUBDIR_PATH')
        self.segmentations_path = osp.join(self.base_path, self.segmentations_subdir_path) #old: SEGMENTATIONS_PATH = os.path.sep.join([BASE_PATH_3D, SEGMENTATIONS_SUBDIR_PATH])

        self.volumes_subdir_path = config.get('DATA', 'VOLUMES_SUBDIR_PATH') #old: VOLUMES_SUBDIR_PATH = config.get('DATA', 'VOLUMES_SUBDIR_PATH')
        self.volumes_path = osp.join(self.base_path, self.volumes_subdir_path) #old: VOLUMES_PATH = os.path.sep.join([BASE_PATH_3D, VOLUMES_SUBDIR_PATH])
        self.volume_files = glob(osp.join(self.volumes_path, '*.nii.gz'))
 
        self.segmentations_filename = config.get('DATA', 'SEGMENTATIONS_FILE_NAME') #old: SEGMENTATIONS_FILE_NAME = config.get('DATA', 'SEGMENTATIONS_FILE_NAME')
        self.segmentations_file = osp.join(self.base_path, self.segmentations_filename) #old: SEGMENTATIONS_FILE = os.path.sep.join([BASE_PATH_3D, SEGMENTATIONS_FILE_NAME])

        self.df = pandas.read_excel(self.segmentations_file) #old: df = pandas.read_excel(SEGMENTATIONS_FILE)
        # self.numDataFiles = self.df.shape[0]
        # print("number of data files = ", self.numDataFiles)
        # print("compare with length = ", self.__len__())

    def __len__(self):
        return len(self.volume_files)

    def __getitem__(self, idx):
        patient = SingleVentriclePatient(idx, self.df.iloc[[idx]], self.volume_files[idx], self.segmentations_path)
        return patient


        # Split for Train and Test
        # train_idxs, val_idxs = split_train_val_dataset(len(self.volume_files), 0.2)
        # if mode == DatasetMode.TRAIN:
        #     train_idxs = self.get_train_idxs()
        #     self.volume_files = np.asarray(self.volume_files)[train_idxs]
        # elif mode == DatasetMode.VAL:
        #     val_idxs = self.get_val_idxs()
        #     self.volume_files = np.asarray(self.volume_files)[val_idxs]



        # These patients need to be flipped along the three axis
        # self.flip_all = {'Adolescent_7',
        #                  'Adolescent_26',
        #                  'Adolescent_54',
        #                  'Adolescent_75',
        #                  'Adult_6',
        #                  'Adult_11',
        #                  'Adult_16',
        #                  'Adult_17',
        #                  'Child_27'}
        # self.flip_masks = config.getboolean('PARAMETERS', 'FLIP_MASKS')

        # Optical flow parameters
        # self.fwdof_dir = None
        # self.bwdof_dir = None
        # self.flow_name = None
        # self.flow_level = None

        # if self.load_flow:
        #     use_filtered_flow = config.getboolean('PARAMETERS', 'USE_MEDIAN_FILTERED_FLOW')
        #     self.flow_name = 'flow_m_it0.pt' if use_filtered_flow else 'flow_it0.pt'
        #     self.flow_level = 'it0'
        #     self.fwdof_dir = self.config.get('DATA', 'FWD_OPTFLOW_RESULTS_DIR')
        #     self.bwdof_dir = self.config.get('DATA', 'BWD_OPTFLOW_RESULTS_DIR')

    # def get_train_idxs(self):
    #     train_idxs = []
    #     for i in range(len(self.volume_files)):
    #         pname = self.get_patient_name(i)
    #         if pname in val_patients:
    #             continue
    #         else:
    #             train_idxs.append(i)
    #     return np.asarray(train_idxs)

    # def get_val_idxs(self):
    #     val_idxs = []
    #     for i in range(len(self.volume_files)):
    #         pname = self.get_patient_name(i)
    #         if pname in val_patients:
    #             val_idxs.append(i)
    #     return np.asarray(val_idxs)

    # def systole_diastole_mask(self, patient_name):
    #     mask_syst_zyx = self.load_mask(patient_name, '_Systole_Labelmap.nii')
    #     mask_syst_zyx = torch.from_numpy(mask_syst_zyx).float()
    #     mask_diast_zyx = self.load_mask(patient_name, '_Diastole_Labelmap.nii')
    #     mask_diast_zyx = torch.from_numpy(mask_diast_zyx).float()
    #     return (mask_syst_zyx, mask_diast_zyx)

    # def get_patient_name(self, idx):
    #     return (self.volume_files[idx].split(os.sep)[-1]).split(".")[0]

    # def load_mask(self, patient_name, ending):
    #     mask = nib.load(osp.sep.join([self.segmentations_path, patient_name, patient_name + ending]))
    #     mask = np.swapaxes(mask.get_fdata(), 0, 2)
    #     if self.flip_masks:
    #         mask_flipped = np.flip(mask, (0, 1, 2)).copy() if patient_name in self.flip_all else np.flip(mask, 1).copy()
    #         return mask_flipped
    #     else:
    #         return mask

    # def index_for_patient(self, patient_name):
    #     found = False
    #     for idx in range(self.__len__()):
    #         query = self.get_patient_name(idx)
    #         if patient_name == query:
    #             found = True
    #             return (idx, found)
    #     return (-1, found)

    # def data_for_patient(self, patient_name):
    #     idx, found = self.index_for_patient(patient_name)
    #     if found:
    #         return self.__getitem__(idx)
    #     else:
    #         return (None, None, None, None, None, None, None, None)

    # def optflow(self, idx):
    #     patient_name = self.get_patient_name(idx)
    #     return self.optflow_for_patient(patient_name)

    # def optflow_for_patient(self, patient):
    #     fwd_flows = []
    #     bwd_flows = []
    #     fwd_patient_dir = osp.join(self.fwdof_dir, patient)
    #     fwd_time_dirs = sorted(os.listdir(fwd_patient_dir))
    #     bwd_patient_dir = osp.join(self.bwdof_dir, patient)
    #     # bwd_time_dirs = sorted(os.listdir(bwd_patient_dir))
    #     bwd_time_dirs = sorted(os.listdir(bwd_patient_dir), reverse=True)
    #     assert len(fwd_time_dirs) == len(bwd_time_dirs)

    #     for fwd_dir, bwd_dir in zip(fwd_time_dirs, bwd_time_dirs):
    #         # Read forward optical flow
    #         fwd_path = osp.join(fwd_patient_dir, fwd_dir)
    #         if osp.isdir(fwd_path):
    #             flow_path = osp.sep.join([fwd_path, self.flow_level, self.flow_name])
    #             u = torch.load(flow_path, map_location='cpu')
    #             if self.flow_transforms is not None:
    #                 u = self.flow_transforms(u)
    #             fwd_flows.append(u)

    #         # Read backward optical flow
    #         bwd_path = osp.join(bwd_patient_dir, bwd_dir)
    #         if osp.isdir(bwd_path):
    #             flow_path = osp.sep.join([bwd_path, self.flow_level, self.flow_name])
    #             u = torch.load(flow_path, map_location='cpu')
    #             if self.flow_transforms is not None:
    #                 u = self.flow_transforms(u)
    #             bwd_flows.append(u)

    #     fwd_t = torch.stack([x.float() for x in fwd_flows], dim=0)
    #     bwd_t = torch.stack([x.float() for x in bwd_flows], dim=0)

    #     # return (fwd_flows, bwd_flows)
    #     return (fwd_t, bwd_t)

    # def prepare_masks(self, tsyst, tdias, msyst, mdias):
    #     init_ts = min(tdias, tsyst)
    #     final_ts = max(tdias, tsyst)

    #     m0 = mk = None
    #     if init_ts == tsyst:
    #         m0 = msyst
    #         mk = mdias
    #     else:
    #         m0 = mdias
    #         mk = msyst
    #     return (m0, mk, init_ts, final_ts)

    # def save_patients(self, save_dir: str, filename: str):
    #     filepath = osp.join(save_dir, filename)
    #     with open(filepath, 'w') as pfile:
    #         for i in range(len(self.volume_files)):
    #             pfile.write(self.get_patient_name(i) + '\n')
    #     pfile.close()