import sys
import nibabel as nib
import numpy as np
import os.path as osp
import os
from enum import Enum
from TVL1OF.TVL1OF3D import *
from cnn.dataset import SingleVentricleDataset

utils_lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'utils'))
sys.path.append(utils_lib_path)
import plots

from opticalFlow_cuda_ext import opticalFlow


class OpticalFlowMode(Enum):
    FORWARD = 1
    BACKWARD = 2
    UNKNOWN = 0


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

    mode_str = config.get('PARAMETERS', 'mode')
    mode = OpticalFlowMode.FORWARD if mode_str == 'FORWARD' else OpticalFlowMode.BACKWARD
    print("=======================================")
    print('Mode: ' + mode_str + ' Optical Flow')
    print("=======================================")
    print("\n")

    # create save directory
    saveDir = plots.createSaveDirectory(config.get('DATA', 'OUTPUT_PATH'), f'TVL1OF3D{mode_str}')

    # save config file to save directory
    conifgOutput = os.path.sep.join([saveDir, "config.ini"])
    with open(conifgOutput, 'w') as configfile:
        config.write(configfile)

    ds = SingleVentricleDataset(config, load_flow=False)

    PATIENT_NAME = config.get('DATA', 'PATIENT_NAME')
    idx, found = ds.index_for_patient(PATIENT_NAME)
    if not found:
        print(PATIENT_NAME + " not found!")
        sys.exit()

    for _ in range(1):
        (pname, data, mask_systole, mask_diastole, systole_time, diastole_time, _, _) = ds[idx]
        data = data.to(DEVICE)
        mask_systole = mask_systole.to(DEVICE)
        mask_diastole = mask_diastole.to(DEVICE)
        NZ, NY, NX, NT = data.shape

        print("=======================================")
        print("load data for patient: ", pname)
        print(f"   * dimensions: (Z,Y,X,T) = {data.shape}")
        print("   * systole at time:  ", systole_time)
        print("   * diastole at time: ", diastole_time)
        num_ts = abs(diastole_time - systole_time)
        init_ts = min(diastole_time, systole_time)
        final_ts = max(diastole_time, systole_time)
        print("=======================================")
        print("\n")

        patientDir = osp.join(saveDir, pname)

        # initialization of optical flow and mask
        u = torch.zeros([NZ, NY, NX, 3]).float().to(DEVICE)
        p = torch.zeros([NZ, NY, NX, 3, 3]).float().to(DEVICE)

        mask = None
        m0 = mk = None
        from_t, to_t, inc_t = 0, 0, 0

        if init_ts == systole_time:
            m0 = mask_systole.clone().detach()
            mk = mask_diastole.clone().detach()
            print('m0 = mask_systole')
            print('mk = mask_diastole')
        else:
            m0 = mask_diastole.clone().detach()
            mk = mask_systole.clone().detach()
            print('m0 = mask_diastole')
            print('mk = mask_systole')

        if mode == OpticalFlowMode.FORWARD:
            mask = m0.clone().detach()
            from_t, to_t, inc_t = init_ts, final_ts, 1
        elif mode == OpticalFlowMode.BACKWARD:
            mask = mk.clone().detach()
            from_t, to_t, inc_t = final_ts, init_ts, -1

        for t in range(from_t, to_t, inc_t):
            saveDirTimeStep = plots.createSubDirectory(patientDir, f'time{t}')
            t0, t1 = t + inc_t, t
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
