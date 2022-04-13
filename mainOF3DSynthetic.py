import numpy as np
import os.path as osp
import torch
import configparser
import sys
from cnn.dataset import SingleVentricleDataset
from TVL1OF.TVL1OF3D import *
from torchvision.transforms import Compose
import cnn.custom_transforms as ct


utils_lib_path = osp.abspath(osp.join(osp.dirname(__file__), 'utils'))
sys.path.append(utils_lib_path)
import plots


if __name__ == "__main__":
    print("\n\n")
    print("==================================================")
    print("==================================================")
    print("        compute TV-L1 optical flow:")
    print("==================================================")
    print("==================================================")
    print("\n\n")

    config = configparser.ConfigParser()
    config.read('parser/configTVL1OF3D.ini')
    cuda_availabe = config.get('DEVICE', 'cuda_availabe')
    DEVICE = 'cuda' if cuda_availabe and torch.cuda.is_available() else 'cpu'
    # PATIENT_NAME = config.get('DATA', 'PATIENT_NAME')

    # create save directory
    save_dir = plots.createSaveDirectory(config.get('DATA', 'OUTPUT_PATH'), 'TVL1OF3DForward')

    # save config file to save directory
    conifg_output = os.path.sep.join([save_dir, 'config.ini'])
    with open(conifg_output, 'w') as config_file:
        config.write(config_file)

    transforms = Compose([ct.ToTensor()])
    ds = SingleVentricleDataset(config, transforms)

    # idx, found = ds.index_for_patient(PATIENT_NAME)
    # if not found:
    #     print(PATIENT_NAME + " not found!")
    #     sys.exit()

    for idx in range(len(ds)):
        (data, mask_systole, mask_diastole, systole_time, diastole_time, pname) = ds[idx]
        NZ, NY, NX, NT = data.shape
        print('\n====================================')
        print('Load data for patient: ' + pname)
        print(f'\t* (NZ, NY, NX, NT) = ({NZ}, {NY}, {NX}, {NT})')
        print(f'\t* Systole at time: {systole_time}')
        print(f'\t* Diastole at time: {diastole_time}')
        print('====================================\n')

        # num_timesteps = abs(diastole_time - systole_time)
        init_timestep = min(diastole_time, systole_time)
        final_timestep = max(diastole_time, systole_time)

        patient_dir = osp.join(save_dir, pname)

        saveDirInitTime = plots.createSubDirectory(patient_dir, f"time{init_timestep}")

        # initialization of optical flow and mask
        u = torch.zeros([NZ, NY, NX, 3]).float().to(DEVICE)
        p = torch.zeros([NZ, NY, NX, 3, 3]).float().to(DEVICE)

        mask = None
        if init_timestep == systole_time:
            mask = mask_systole.clone().detach().to(DEVICE)
        else:
            mask = mask_diastole.clone().detach().to(DEVICE)

        for t in range(init_timestep, final_timestep):
            save_timestep_dir = plots.createSubDirectory(patient_dir, f"time{t}")

            # Compute the optical flow for given time steps
            # t0, t1 = t + 1, t  # bwd
            t0, t1 = t, t + 1 # fwd
            print(f"{t1} -> {t0}")
            I0 = data[:, :, :, t0].to(DEVICE)
            I1 = data[:, :, :, t1].to(DEVICE)
            alg = TVL1OpticalFlow3D(save_timestep_dir, config)
            u, p = alg.computeOnPyramid(I0, I1, u, p)

            # save the old mask
            save3D_torch_to_nifty(mask, save_timestep_dir, f"mask_time{t1}.nii")
            save_slices(mask, f"mask.png", save_timestep_dir)
            save_single_zslices(mask, save_timestep_dir, "mask_slices", 1., 2)

            # warp mask with the computed optical flow
            mask = alg.warpMask(mask, u, t0, save_timestep_dir)
