'''
Forward mask propagation, from m0 to mk
'''
import sys
import nibabel as nib
import numpy as np
import os.path as osp
import os
from TVL1OF.TVL1OF3D import *
from cnn.dataset import SingleVentricleDataset

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
    DEVICE = 'cuda' if cuda_availabe and torch.cuda.is_available() else 'cpu'
    # PATIENT_NAME = config.get('DATA', 'PATIENT_NAME')

    # create save directory
    saveDir = plots.createSaveDirectory(config.get('DATA', 'OUTPUT_PATH'), 'TVL1OF3DForward')

    # save config file to save directory
    conifgOutput = os.path.sep.join([saveDir, "config.ini"])
    with open(conifgOutput, 'w') as configfile:
        config.write(configfile)

    ds = SingleVentricleDataset(config, load_flow=False)

    # idx, found = ds.index_for_patient(PATIENT_NAME)
    # if not found:
    #     print(PATIENT_NAME + " not found!")
    #     sys.exit()

    for idx in range(len(ds)):
        (pname, data, mask_systole, mask_diastole, systole_time, diastole_time, _, _) = ds[idx]
        data = data.squeeze().to(DEVICE)
        mask_systole = mask_systole.squeeze().to(DEVICE)
        mask_diastole = mask_diastole.squeeze().to(DEVICE)
        NZ, NY, NX, NT = data.shape

        print("=======================================")
        print("load data for patient: ", pname)
        print(f"   * dimensions: (Z,Y,X,T) = {data.shape}")
        print("   * systole at time:  ", systole_time)
        print("   * diastole at time: ", diastole_time)
        numTimeSteps = abs(diastole_time - systole_time)
        initTimeStep = min(diastole_time, systole_time)
        finalTimeStep = max(diastole_time, systole_time)
        print("=======================================")
        print("\n")

        patientDir = osp.join(saveDir, pname)

        # initialization of optical flow and mask
        u = torch.zeros([NZ, NY, NX, 3]).float().to(DEVICE)
        p = torch.zeros([NZ, NY, NX, 3, 3]).float().to(DEVICE)
        mask = None
        if initTimeStep == systole_time:
            mask = mask_systole.clone().detach()
            print('mask = mask_systole')
        else:
            mask = mask_diastole.clone().detach()
            print('mask = mask_diastole')

        for t in range(initTimeStep, finalTimeStep):
            saveDirTimeStep = plots.createSubDirectory(patientDir, f'time{t}')
            t0, t1 = t + 1, t
            I0 = data[:, :, :, t0]
            I1 = data[:, :, :, t1]
            print(f'{t1}->{t0}')

            # Compute the optical flow
            alg = TVL1OpticalFlow3D(saveDirTimeStep, config)
            u, p = alg.computeOnPyramid(I0, I1, u, p)

            # save the old mask
            save3D_torch_to_nifty(mask, saveDirTimeStep, f'mask_time{t1}.nii')
            save_slices(mask, f'mask.png', saveDirTimeStep)
            save_single_zslices(mask, saveDirTimeStep, 'mask_slices', 1., 2)

            # warp mask with the computed optical flow
            mask = alg.warpMask(mask, u, t0, saveDirTimeStep)
