import os
import argparse
from enum import Enum

from TVL1OF.deprecated.TVL1OF3Dnew import *

# cnn_lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../cnn'))
# sys.path.append(cnn_lib_path)
# from dataset import SingleVentricleDataset, DatasetMode

dataset_lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../dataset'))
sys.path.append(dataset_lib_path)
from singleVentricleDataset import SingleVentricleDataset

import plots 
# import transforms as T


class OpticalFlowMode(Enum):
    FORWARD = 1
    BACKWARD = 2
    UNKNOWN = 0


class TVL1OpticalFlow3D_Dataset:
    def __init__(self, saveDir, config):
        self.saveDir = saveDir
        self.config = config


    def compute_optical_flow(self, ds: SingleVentricleDataset, idx: int, mode: OpticalFlowMode, save_dir: str, device: str, step: int, logger):
        #(pname, data, _, _, init_ts, final_ts, _, _) = ds[idx]
        patient = ds[idx]
        print("device in compute_optical_flow =", device)
        #data = data.to(device)
        NZ, NY, NX, NT = patient.nii_data_zyxt.shape

        patient_dir = plots.createSubDirectory(save_dir, patient.name)
        logger.info(f'{idx} - {patient.name}')

        # initialization of optical flow and mask
        u = torch.zeros([NZ, NY, NX, 3]).float().to(device)
        p = torch.zeros([NZ, NY, NX, 3, 3]).float().to(device)

        idxs = None
        if mode == OpticalFlowMode.FORWARD:
            # mask = m0.to(DEVICE)
            idxs = torch.arange(patient.init_ts, patient.final_ts + 1, 1) if step == -1 else torch.linspace(patient.init_ts, patient.final_ts, step).int()
        elif mode == OpticalFlowMode.BACKWARD:
            # mask = mk.to(DEVICE)
            idxs = torch.arange(patient.final_ts, patient.init_ts - 1, -1) if step == -1 else torch.flip(torch.linspace(patient.init_ts, patient.final_ts, step).int(), dims=(0,))

        for i in range(len(idxs) - 1):
            # t0, t1 = t + inc_t, t
            t0 = idxs[i + 1].item()
            t1 = idxs[i].item()

            I0 = torch.from_numpy(patient.nii_data_zyxt[:,:,:,t0]).float().to(device)
            I1 = torch.from_numpy(patient.nii_data_zyxt[:,:,:,t1]).float().to(device)

            print(f'{t1}->{t0}')
            #pbar.set_postfix_str(f'P: {pname}, ({t1}->{t0})')
            save_dir_timestep = plots.createSubDirectory(patient_dir, f'time{t1}')

            # Compute the optical flow
            algPatient = TVL1OpticalFlow3DNew(save_dir_timestep, self.config, device)
            u, p = algPatient.computeOnPyramid(I0, I1, u, p)

            # save the old mask
            # save3D_torch_to_nifty(mask, saveDirTimeStep, f'mask_time{t1}.nii')
            # save_slices(mask, f'mask.png', saveDirTimeStep)
            # save_single_zslices(mask, saveDirTimeStep, 'mask_slices', 1., 2)

            # warp mask with the computed optical flow
            # mask = alg.warpMask(mask, u, I1, t0, saveDirTimeStep)


    def compute(self, mode_str: str,  device: str):
        mode = OpticalFlowMode.FORWARD if mode_str == 'Forward' else OpticalFlowMode.BACKWARD

        # create save directory
        saveDirMode = plots.createSubDirectory(self.saveDir, f'TVL1OF3D{mode_str}')

        logger = plots.create_logger(saveDirMode)
        logger.info(f'Compute TV-L1 optical flow ({mode_str})')

        # save config file to save directory
        conifg_output = os.path.sep.join([saveDirMode, "config.ini"])
        with open(conifg_output, 'w') as configfile:
            config.write(configfile)

        #old: Dataset from cnn
        # data_transf = T.ComposeUnary([T.ToTensor()])
        # train_ds = SingleVentricleDataset(config, DatasetMode.TRAIN, load_flow=False, img4d_transforms=data_transf)
        # val_ds = SingleVentricleDataset(config, DatasetMode.VAL, load_flow=False, img4d_transforms=data_transf)
        #new: Dataset from dataset folder
        train_ds = SingleVentricleDataset(config, mode='train')
        val_ds = SingleVentricleDataset(config, mode='val')

        compute_all_patients = config.get('DATA', 'COMPUTE_ALL_PATIENTS')
        step = config.getint('PARAMETERS', 'step')

        ds_list = []
        use_indices = config.getboolean('PARAMETERS', 'use_indices')
        if use_indices:
            dataset = config.get('PARAMETERS', 'dataset')
            if dataset == 'train':
                ds_list.append(train_ds)
            elif dataset == 'val':
                ds_list.append(val_ds)
            from_idx = config.getint('PARAMETERS', 'from_idx')
            to_idx = config.getint('PARAMETERS', 'to_idx')
        else:
            ds_list = [train_ds, val_ds]
            from_idx = 0

        if compute_all_patients:
            pbar = tqdm(total=to_idx - from_idx if use_indices else len(train_ds) + len(val_ds))
            for ds in ds_list:
                to_idx = to_idx if use_indices else len(ds)

                for idx in range(from_idx, to_idx):
                    self.compute_optical_flow(ds, idx, mode, saveDirMode, device, step, logger)
                    pbar.update(1)
        else:
            pbar = tqdm(total=1)
            patient_name = config.get('DATA', 'PATIENT_NAME')
            idx, found = train_ds.index_for_patient(patient_name)
            if not found:
                logger.info(patient_name + " not found!")
                sys.exit()
            else:
                self.compute_optical_flow(train_ds, idx, mode, saveDirMode, device, step, logger)
                pbar.update(1)


if __name__ == "__main__":

    #load arguments 
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default=0)
    parser.add_argument('--cpu', type=int, default=8)
    args = parser.parse_args()

    # set gpu and cpu
    DEVICE_OF = torch_utils.getTorchDevice(args.gpu)
    torch.set_num_threads(args.cpu)

    # load config parser
    config = configparser.ConfigParser()
    config.read('parser/configTVL1OF3D.ini')
    saveDir = plots.createSaveDirectory(config.get('DATA', 'OUTPUT_PATH'), 'TVL1OF3DNew')

    #compute
    alg = TVL1OpticalFlow3D_Dataset(saveDir,config)

    modeStr = config.get('PARAMETERS', 'mode')
    print("mode =", modeStr)
    alg.compute(mode_str=modeStr,device=DEVICE_OF)
    




