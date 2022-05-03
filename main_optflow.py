import sys
import nibabel as nib
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
import os
from enum import Enum
from TVL1OF.TVL1OF3D import *
from cnn.dataset import SingleVentricleDataset
import csv
utils_lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'utils'))
sys.path.append(utils_lib_path)
import plots
import torch_utils


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
    use_cuda = config.get('DEVICE', 'cuda_availabe')
    if use_cuda and torch.cuda.is_available():
        DEVICE = 'cuda'
        CUDA_DEVICE = config.getint('DEVICE', 'cuda_device')
        torch.cuda.set_device(CUDA_DEVICE)
    else:
        DEVICE = 'cpu'

    mode_str = config.get('PARAMETERS', 'mode')
    mode = OpticalFlowMode.FORWARD if mode_str == 'Forward' else OpticalFlowMode.BACKWARD
    print("=======================================")
    print('Mode: ' + mode_str + ' Optical Flow')
    print("=======================================")
    print("\n")

    # create save directory
    save_dir = plots.createSaveDirectory(config.get('DATA', 'OUTPUT_PATH'), f'TVL1OF3D{mode_str}')

    # save config file to save directory
    conifg_output = os.path.sep.join([save_dir, "config.ini"])
    with open(conifg_output, 'w') as configfile:
        config.write(configfile)

    ds = SingleVentricleDataset(config, load_flow=False)

    PATIENT_NAME = config.get('DATA', 'PATIENT_NAME')
    idx, found = ds.index_for_patient(PATIENT_NAME)
    if not found:
        print(PATIENT_NAME + " not found!")
        sys.exit()
    STEP = config.getint('PARAMETERS', 'step')

    # for idx in range(len(ds)):
    for _ in range(1):
        (pname, data, mask_systole, mask_diastole, systole_time, diastole_time, _, _) = ds[idx]
        data = torch_utils.normalize(data.to(DEVICE))
        NZ, NY, NX, NT = data.shape
        # grid = torch_utils.create_grid(NZ, NY, NX).numpy()

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

        patient_dir = osp.join(save_dir, pname)
        if not os.path.exists(patient_dir):
            os.makedirs(patient_dir)

        # initialization of optical flow and mask
        u = torch.zeros([NZ, NY, NX, 3]).float().to(DEVICE)
        p = torch.zeros([NZ, NY, NX, 3, 3]).float().to(DEVICE)

        mask = None
        m0 = mk = None
        from_t, to_t, inc_t = 0, 0, 0
        if init_ts == systole_time:
            m0 = mask_systole.to(DEVICE)
            mk = mask_diastole.to(DEVICE)
            print('m0 = mask_systole', '\tmk = mask_diastole')
        else:
            m0 = mask_diastole.to(DEVICE)
            mk = mask_systole.to(DEVICE)
            print('m0 = mask_diastole', '\tmk = mask_systole')
        save_slices(m0, 'm0.png', patient_dir)
        save_slices(mk, 'mk.png', patient_dir)

        # idxs = torch.linspace(init_ts, final_ts, STEP).int()
        idxs = None
        if mode == OpticalFlowMode.FORWARD:
            mask = m0.clone()
            idxs = torch.arange(init_ts, final_ts + 1, 1) if STEP == -1 else torch.linspace(init_ts, final_ts, STEP).int()
            # from_t, to_t, inc_t = init_ts, final_ts, 1
        elif mode == OpticalFlowMode.BACKWARD:
            mask = mk.clone()
            idxs = torch.arange(final_ts, init_ts - 1, -1) if STEP == -1 else torch.flip(torch.linspace(init_ts, final_ts, STEP).int(), dims=(0,))
            # from_t, to_t, inc_t = final_ts, init_ts, -1

        # for t in range(from_t, to_t, inc_t):
        print(idxs)
        for i in range(len(idxs) - 1):
            # saveDirTimeStep = plots.createSubDirectory(patient_dir, f'time{t}')
            # t0, t1 = t + inc_t, t
            t0 = idxs[i + 1].item()
            t1 = idxs[i].item()
            I0 = data[:, :, :, t0]
            I1 = data[:, :, :, t1]
            print(f'{t1}->{t0}')
            saveDirTimeStep = plots.createSubDirectory(patient_dir, f'time{t1}')

            # Compute the optical flow
            alg = TVL1OpticalFlow3D(saveDirTimeStep, config)
            u, p = alg.computeOnPyramid(I0, I1, u, p)

            np.savetxt(osp.join(patient_dir, f'u_{i}.txt'), u.cpu().detach().numpy().reshape((-1, 3)))

            # save the old mask
            save3D_torch_to_nifty(mask, saveDirTimeStep, f'mask_time{t1}.nii')
            save_slices(mask, f'mask.png', saveDirTimeStep)
            save_single_zslices(mask, saveDirTimeStep, 'mask_slices', 1., 2)

            # warp mask with the computed optical flow
            mask = alg.warpMask(mask, u, I0, t0, saveDirTimeStep)
