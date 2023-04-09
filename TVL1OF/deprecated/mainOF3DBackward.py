import sys
import nibabel as nib
import numpy as np
import os
import pandas

from TVL1OF.TVL1OF3D import *

utils_lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'utils'))
sys.path.append(utils_lib_path)
import plots

from opticalFlow_cuda_ext import opticalFlow


if __name__ == "__main__":

    print("\n\n")
    print("==================================================")
    print("==================================================")
    print("        compute TV-L1 optical flow:")
    print("==================================================")
    print("==================================================")
    print("\n\n")

    # load config parser
    config = configparser.ConfigParser()
    config.read('parser/configTVL1OF3D.ini')
    cuda_availabe = config.get('DEVICE', 'cuda_availabe')
    DEVICE = "cuda" if cuda_availabe and torch.cuda.is_available() else "cpu"

    # create save directory
    saveDir = plots.createSaveDirectory(config.get('DATA', 'OUTPUT_PATH'), "TVL1OF3DBackward")

    # save config file to save directory
    conifgOutput = os.path.sep.join([saveDir, "config.ini"])
    with open(conifgOutput, 'w') as configfile:
        config.write(configfile)

    # Load 4D nifty [x,y,z,t]
    print("=======================================")
    BASE_PATH_3D = config.get('DATA', 'BASE_PATH_3D')
    PATIENT_NAME = config.get('DATA', 'PATIENT_NAME')
    VOLUMES_SUBDIR_PATH = config.get('DATA', 'VOLUMES_SUBDIR_PATH')
    VOLUMES_PATH = os.path.sep.join([BASE_PATH_3D, VOLUMES_SUBDIR_PATH])
    print("load data for patient: ", PATIENT_NAME)
    vol = nib.load(os.path.sep.join([VOLUMES_PATH, PATIENT_NAME + ".nii.gz"]))
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
    print(f"   * scaling of data in range {totalMinValue,totalMaxValue} to {scaleMinValue,scaleMaxValue}")
    nii_data_xyzt *= scaleMaxValue / totalMaxValue

    # swap from nibabel (X,Y,Z) to cuda-compatible (Z,Y,X):
    #print("swap axes (X,Y,Z,T) to (Z,Y,X,T)")
    nii_data = np.swapaxes(nii_data_xyzt, 0, 2)
    print(f"   * dimensions: (Z,Y,X,T) = {nii_data.shape}")

    # read time steps for diastole and systole
    SEGMENTATIONS_FILE_NAME = config.get('DATA', 'SEGMENTATIONS_FILE_NAME')
    SEGMENTATIONS_FILE = os.path.sep.join([BASE_PATH_3D, SEGMENTATIONS_FILE_NAME])
    df = pandas.read_excel(SEGMENTATIONS_FILE)
    rowPatient = df[df['Name'] == PATIENT_NAME]
    indexPatient = rowPatient.index[0]
    tDiastole = rowPatient.loc[indexPatient, "Diastole"]
    tSystole = rowPatient.loc[indexPatient, "Systole"]
    print("   * systole at time:  ", tSystole)
    print("   * diastole at time: ", tDiastole)
    numTimeSteps = abs(tDiastole - tSystole)
    initTimeStep = min(tDiastole, tSystole)
    finalTimeStep = max(tDiastole, tSystole)
    print("=======================================")
    print("\n")

    # load input masks
    SEGMENTATIONS_SUBDIR_PATH = config.get('DATA', 'SEGMENTATIONS_SUBDIR_PATH')
    SEGMENTATIONS_PATH = os.path.sep.join([BASE_PATH_3D, SEGMENTATIONS_SUBDIR_PATH])

    nii_mask_load_systole = nib.load(os.path.sep.join([SEGMENTATIONS_PATH, PATIENT_NAME, PATIENT_NAME + "_Systole_Labelmap.nii"]))
    nii_mask_xyz_systole = nii_mask_load_systole.get_fdata()
    nii_mask_systole = np.swapaxes(nii_mask_xyz_systole, 0, 2)
    mask_systole = torch.from_numpy(nii_mask_systole).float().to(DEVICE)

    nii_mask_load_diastole = nib.load(os.path.sep.join([SEGMENTATIONS_PATH, PATIENT_NAME, PATIENT_NAME + "_Diastole_Labelmap.nii"]))
    nii_mask_xyz_diastole = nii_mask_load_diastole.get_fdata()
    nii_mask_diastole = np.swapaxes(nii_mask_xyz_diastole, 0, 2)
    mask_diastole = torch.from_numpy(nii_mask_diastole).float().to(DEVICE)

    saveDirInitTime = plots.createSubDirectory(saveDir, f"time{initTimeStep}")
    # saveDirInitTime = os.path.sep.join([saveDir, f"time{initTimeStep}"])
    # if not os.path.exists(saveDirInitTime):
    #     os.makedirs(saveDirInitTime)

    # initialization of optical flow and mask
    u = torch.zeros([NZ, NY, NX, 3]).float().to(DEVICE)
    p = torch.zeros([NZ, NY, NX, 3, 3]).float().to(DEVICE)
    mask = None
    if initTimeStep == tSystole:
        mask = mask_systole.clone().detach()
    else:
        mask = mask_diastole.clone().detach()

    for t in range(initTimeStep, finalTimeStep):
        saveDirTimeStep = plots.createSubDirectory(saveDir, f"time{t}")
        # saveDirTimeStep = os.path.sep.join([saveDir, f"time{t}"])
        # if not os.path.exists(saveDirTimeStep):
        #   os.makedirs(saveDirTimeStep)
        # convert to torch for given time steps
        t0, t1 = t + 1, t
        I0 = torch.from_numpy(nii_data[:, :, :, t0]).float().to(DEVICE)
        I1 = torch.from_numpy(nii_data[:, :, :, t1]).float().to(DEVICE)

        # Compute the optical flow
        alg = TVL1OpticalFlow3D(saveDirTimeStep, config)
        u, p = alg.computeOnPyramid(I0, I1, u, p)
        # flowName = "flow_it0.pt"
        # fileNameFlow = os.path.join(saveDirStep, flowName)
        # u = torch.load(fileNameFlow, map_location=torch.device(DEVICE))

        # save the old mask
        save3D_torch_to_nifty(mask, saveDirTimeStep, f"mask_time{t1}.nii")
        save_slices(mask, f"mask.png", saveDirTimeStep)
        save_single_zslices(mask, saveDirTimeStep, "mask_slices", 1., 2)

        # warp mask with the computed optical flow
        mask = alg.warpMask(mask, u, t0, saveDirTimeStep)
