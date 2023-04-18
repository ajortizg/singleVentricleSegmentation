import numpy as np
import utilities
import os.path as osp
import nibabel as nib
import configparser
import pandas


# load config parser
config = configparser.ConfigParser()
config.read('parser/configSynthetic3D.ini')
cuda_availabe = config.get('DEVICE', 'cuda_availabe')
DEVICE = "cuda" if cuda_availabe else "cpu"


# Load 4D nifty [x,y,z,t]
print("=======================================")
BASE_PATH_3D = config.get('DATA', 'BASE_PATH_3D')
PATIENT_NAME = config.get('DATA', 'PATIENT_NAME')
VOLUMES_SUBDIR_PATH = config.get('DATA', 'VOLUMES_SUBDIR_PATH')
VOLUMES_PATH = osp.sep.join([BASE_PATH_3D, VOLUMES_SUBDIR_PATH])
print("load data for patient: ", PATIENT_NAME)
vol = nib.load(osp.sep.join([VOLUMES_PATH, PATIENT_NAME + ".nii.gz"]))
nii_data_xyzt = vol.get_fdata()
NX = nii_data_xyzt.shape[0]
NY = nii_data_xyzt.shape[1]
NZ = nii_data_xyzt.shape[2]
NT = nii_data_xyzt.shape[3]

# ==================================
# scaling of data
totalMinValue = np.amin(nii_data_xyzt)
totalMaxValue = np.amax(nii_data_xyzt)
scaleMinValue = 0.
scaleMaxValue = 1.
# scaleMaxValue = 255.
print(
    f"\t* scaling of data in range {totalMinValue,totalMaxValue} to {scaleMinValue,scaleMaxValue}")
nii_data_xyzt *= scaleMaxValue / totalMaxValue

# swap from nibabel (X,Y,Z) to cuda-compatible (Z,Y,X):
# print("swap axes (X,Y,Z,T) to (Z,Y,X,T)")
nii_data = np.swapaxes(nii_data_xyzt, 0, 2)
print(f"\t* dimensions: (Z,Y,X,T) = {nii_data.shape}")

# read time steps for diastole and systole
SEGMENTATIONS_FILE_NAME = config.get('DATA', 'SEGMENTATIONS_FILE_NAME')
SEGMENTATIONS_FILE = osp.sep.join(
    [BASE_PATH_3D, SEGMENTATIONS_FILE_NAME])
df = pandas.read_excel(SEGMENTATIONS_FILE)
rowPatient = df[df['Name'] == PATIENT_NAME]
indexPatient = rowPatient.index[0]
tDiastole = rowPatient.loc[indexPatient, "Diastole"]
tSystole = rowPatient.loc[indexPatient, "Systole"]
print("\t* systole at time:  ", tSystole)
print("\t* diastole at time: ", tDiastole)
numTimeSteps = abs(tDiastole - tSystole)
initTimeStep = min(tDiastole, tSystole)
finalTimeStep = max(tDiastole, tSystole)
print("=======================================")
print("\n")

# load input masks
SEGMENTATIONS_SUBDIR_PATH = config.get('DATA', 'SEGMENTATIONS_SUBDIR_PATH')
SEGMENTATIONS_PATH = osp.sep.join([BASE_PATH_3D, SEGMENTATIONS_SUBDIR_PATH])

nii_mask_load_systole = nib.load(osp.sep.join(
    [SEGMENTATIONS_PATH, PATIENT_NAME, PATIENT_NAME + "_Systole_Labelmap.nii"]))
nii_mask_xyz_systole = nii_mask_load_systole.get_fdata()
nii_mask_systole = np.swapaxes(nii_mask_xyz_systole, 0, 2)
# mask_systole = torch.from_numpy(nii_mask_systole).float().to(DEVICE)

nii_mask_load_diastole = nib.load(osp.sep.join(
    [SEGMENTATIONS_PATH, PATIENT_NAME, PATIENT_NAME + "_Diastole_Labelmap.nii"]))
nii_mask_xyz_diastole = nii_mask_load_diastole.get_fdata()
nii_mask_diastole = np.swapaxes(nii_mask_xyz_diastole, 0, 2)
# mask_diastole = torch.from_numpy(nii_mask_diastole).float().to(DEVICE)


for t in range(initTimeStep, finalTimeStep+1):
    data = nii_data[:, :, :, t]
    utilities.plot_slices(data, str=f"data_t{t}", block=True)

utilities.plot_slices(nii_mask_systole, str=f"mask_systole", block=False)
utilities.plot_slices(nii_mask_diastole, str=f"mask_diastole", block=True)
