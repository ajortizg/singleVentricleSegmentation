import os.path as osp
import os
from enum import Enum
from TVL1OF.TVL1OF3D import *
from cnn.dataset import SingleVentricleDataset, DatasetMode
from utils import plots
import cnn.transforms as T


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
    if cuda_availabe and torch.cuda.is_available():
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

    ds = SingleVentricleDataset(config, DatasetMode.FULL, load_flow=False, data_transforms=[T.Normalize()])

    # PATIENT_NAME = config.get('DATA', 'PATIENT_NAME')
    # idx, found = ds.index_for_patient(PATIENT_NAME)
    # if not found:
    #     print(PATIENT_NAME + " not found!")
    #     sys.exit()
    STEP = config.getint('PARAMETERS', 'step')
    N = len(ds)
    pbar = tqdm(total=N)
    for idx in range(N):
        # for _ in range(1):
        (pname, data, _, _, init_ts, final_ts, _, _) = ds[idx]
        data = T.Normalize()(data.to(DEVICE))
        NZ, NY, NX, NT = data.shape

        # print("=======================================")
        # print("load data for patient: ", pname)
        # print(f"   * dimensions: (Z,Y,X,T) = {data.shape}")
        # print("=======================================")
        # print("\n")

        patient_dir = osp.join(save_dir, pname)
        if not os.path.exists(patient_dir):
            os.makedirs(patient_dir)

        # initialization of optical flow and mask
        u = torch.zeros([NZ, NY, NX, 3]).float().to(DEVICE)
        p = torch.zeros([NZ, NY, NX, 3, 3]).float().to(DEVICE)

        idxs = None
        if mode == OpticalFlowMode.FORWARD:
            # print("do forward compuation of optical flow")
            # mask = m0.to(DEVICE)
            idxs = torch.arange(init_ts, final_ts + 1, 1) if STEP == -1 else torch.linspace(init_ts, final_ts, STEP).int()
        elif mode == OpticalFlowMode.BACKWARD:
            # print("do backward compuation of optical flow")
            # mask = mk.to(DEVICE)
            idxs = torch.arange(final_ts, init_ts - 1, -1) if STEP == -1 else torch.flip(torch.linspace(init_ts, final_ts, STEP).int(), dims=(0,))

        for i in range(len(idxs) - 1):
            # t0, t1 = t + inc_t, t
            t0 = idxs[i + 1].item()
            t1 = idxs[i].item()
            I0 = data[:, :, :, t0]
            I1 = data[:, :, :, t1]
            # print(f'{t1}->{t0}')
            pbar.set_postfix_str(f'P: {pname}, M: {mode_str}, ({t1}->{t0})')
            saveDirTimeStep = plots.createSubDirectory(patient_dir, f'time{t1}')

            # Compute the optical flow
            alg = TVL1OpticalFlow3D(saveDirTimeStep, config)
            u, p = alg.computeOnPyramid(I0, I1, u, p)

            # save the old mask
            # save3D_torch_to_nifty(mask, saveDirTimeStep, f'mask_time{t1}.nii')
            # save_slices(mask, f'mask.png', saveDirTimeStep)
            # save_single_zslices(mask, saveDirTimeStep, 'mask_slices', 1., 2)

            # warp mask with the computed optical flow
            # mask = alg.warpMask(mask, u, I1, t0, saveDirTimeStep)

        pbar.update(1)
