import os.path as osp
import os
from enum import Enum
from TVL1OF.deprecated.TVL1SymOF3D import *
from cnn.dataset import SingleVentricleDataset
from utilities import plots
from utilities.deprecated import torch_utils


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

        # test
        device_id = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(range(device_id))
        print("CUDA_DEVICE = ", CUDA_DEVICE)
        print("device_id = ", device_id)
        print("device_name = ", device_name)
    else:
        DEVICE = 'cpu'

    mode_str = config.get('PARAMETERS', 'mode')
    mode = OpticalFlowMode.FORWARD if mode_str == 'Forward' else OpticalFlowMode.BACKWARD
    print("=======================================")
    print('Mode: ' + mode_str + ' Optical Flow')
    print("=======================================")
    print("\n")

    # create save directory
    save_dir = plots.createSaveDirectory(config.get('DATA', 'OUTPUT_PATH'), f'TVL1SymOF3D{mode_str}')

    # save config file to save directory
    conifg_output = os.path.sep.join([save_dir, "config.ini"])
    with open(conifg_output, 'w') as configfile:
        config.write(configfile)

    ds = SingleVentricleDataset(config, load_flow=False)

    computeForAllPatients = config.getboolean('DATA', 'COMPUTE_ALL_PATIENTS')
   
    if computeForAllPatients:

        for idx in range(len(ds)):
            (pname, data, mask_systole, mask_diastole, systole_time, diastole_time, _, _) = ds[idx]
            data = torch_utils.normalize(data.to(DEVICE))
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
                print("do forward compuation of optical flow")
                mask = m0.clone()
                #idxs = torch.arange(init_ts, final_ts + 1, 1) if STEP == -1 else torch.linspace(init_ts, final_ts, STEP).int()
                idxs = torch.arange(init_ts - 1, final_ts + 1, 1)
            elif mode == OpticalFlowMode.BACKWARD:
                print("do backward compuation of optical flow")
                mask = mk.clone()
                #idxs = torch.arange(final_ts, init_ts - 1, -1) if STEP == -1 else torch.flip(torch.linspace(init_ts, final_ts, STEP).int(), dims=(0,))
                idxs = torch.arange(final_ts + 1, init_ts - 1, -1)

            print("time steps = ", idxs)
            for i in range(1, len(idxs) - 1):
                tl = idxs[i - 1].item()
                tc = idxs[i].item()
                tr = idxs[i + 1].item()
                Il = Ic = Ir = None
                try:
                    Il = data[:, :, :, tl]
                    Ic = data[:, :, :, tc]
                    Ir = data[:, :, :, tr]
                except IndexError as e:
                    print(e)
                    continue

                print("iteration = ", i)
                print(f'{tl}<-{tc}->{tr}')
                saveDirTimeStep = plots.createSubDirectory(patient_dir, f'time{tc}')

                # Compute the optical flow
                alg = TVL1SymOpticalFlow3D(saveDirTimeStep, config)
                u, p = alg.computeOnPyramid(Il, Ic, Ir, u, p)

                # np.savetxt(osp.join(patient_dir, f'u_{i}.txt'), u.cpu().detach().numpy().reshape((-1, 3)))

                # save the old mask
                save3D_torch_to_nifty(mask, saveDirTimeStep, f'mask_time{tc}.nii')
                save_slices(mask, f'mask.png', saveDirTimeStep)
                save_single_zslices(mask, saveDirTimeStep, 'mask_slices', 1., 2)

                # warp mask with the computed optical flow
                mask = alg.warpMask(mask, u, Ic, tc, saveDirTimeStep)

    else :

        PATIENT_NAME = config.get('DATA', 'PATIENT_NAME')
        idx, found = ds.index_for_patient(PATIENT_NAME)
        if not found:
            print(PATIENT_NAME + " not found!")
            sys.exit()

        (pname, data, mask_systole, mask_diastole, systole_time, diastole_time, _, _) = ds[idx]
        data = torch_utils.normalize(data.to(DEVICE))
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

        idxs = None
        if mode == OpticalFlowMode.FORWARD:
            print("do forward compuation of optical flow")
            mask = m0.clone()
            #idxs = torch.arange(init_ts, final_ts + 1, 1) if STEP == -1 else torch.linspace(init_ts, final_ts, STEP).int()
            idxs = torch.arange(init_ts - 1, final_ts + 1, 1)
        elif mode == OpticalFlowMode.BACKWARD:
            print("do backward compuation of optical flow")
            mask = mk.clone()
            idxs = torch.arange(final_ts + 1, init_ts - 1, -1)
            if final_ts == NT - 1:
                idxs[0] = final_ts - 1

        print("time steps = ", idxs)
        for i in range(1, len(idxs) - 1):
            tl = idxs[i - 1].item()
            tc = idxs[i].item()
            tr = idxs[i + 1].item()
            Il = Ic = Ir = None
            try:
                Il = data[:, :, :, tl]
                Ic = data[:, :, :, tc]
                Ir = data[:, :, :, tr]
            except IndexError as e:
                print(e)
                continue

            print("iteration = ", i)
            print(f'{tl}<-{tc}->{tr}')
            saveDirTimeStep = plots.createSubDirectory(patient_dir, f'time{tc}')

            # Compute the optical flow
            alg = TVL1SymOpticalFlow3D(saveDirTimeStep, config)
            u, p = alg.computeOnPyramid(Il, Ic, Ir, u, p)

            # np.savetxt(osp.join(patient_dir, f'u_{i}.txt'), u.cpu().detach().numpy().reshape((-1, 3)))

            # save the old mask
            save3D_torch_to_nifty(mask, saveDirTimeStep, f'mask_time{tc}.nii')
            save_slices(mask, f'mask.png', saveDirTimeStep)
            save_single_zslices(mask, saveDirTimeStep, 'mask_slices', 1., 2)

            # warp mask with the computed optical flow
            mask = alg.warpMask(mask, u, Ic, tc, saveDirTimeStep)
