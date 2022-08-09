import os.path as osp
import os
from enum import Enum

from TVL1OF3D import *

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
from cnn.dataset import SingleVentricleDataset, DatasetMode, LoadFlowMode
from utils import plots
import utils.transforms as T


class OpticalFlowMode(Enum):
    FORWARD = 1
    BACKWARD = 2
    UNKNOWN = 0


def compute_optical_flow(ds: SingleVentricleDataset, idx: int, mode: OpticalFlowMode, save_dir: str, device: str, config, logger):
    (pname, data, _, _, _, init_ts, final_ts, _, _) = ds[idx]
    data = data.to(device)
    NZ, NY, NX, NT = data.shape

    patient_dir = plots.createSubDirectory(save_dir, pname)
    logger.info(f'{idx} - {pname} ({data.shape}), ({init_ts}-{final_ts})')

    # initialization of optical flow and mask
    u = torch.zeros([NZ, NY, NX, 3]).float().to(device)
    p = torch.zeros([NZ, NY, NX, 3, 3]).float().to(device)

    original_NT = ds.get_original_NT(idx)
    full_cycle = ds.full_cycle(idx)
    if full_cycle:
        logger.info(f'{pname}, full_cycle, {original_NT}')

    indices = None
    if mode == OpticalFlowMode.FORWARD:
        if full_cycle:
            indices = torch.arange(0, original_NT + 1, 1)
            indices[-1] = 0
        else:
            indices = torch.arange(init_ts, final_ts + 1, 1)
    elif mode == OpticalFlowMode.BACKWARD:
        if full_cycle:
            indices = torch.arange(original_NT - 1, -2, -1)
            indices[-1] = original_NT - 1
        else:
            indices = torch.arange(final_ts, init_ts - 1, -1)

    for i in range(len(indices) - 1):
        # t0, t1 = t + inc_t, t
        t0 = indices[i + 1].item()
        t1 = indices[i].item()
        I0 = data[:, :, :, t0]
        I1 = data[:, :, :, t1]
        # print(f'{t1}->{t0}')
        pbar.set_postfix_str(f'P: {pname}, ({t1}->{t0})')
        save_dir_timestep = plots.createSubDirectory(patient_dir, f'time{t1}')

        # Compute the optical flow
        alg = TVL1OpticalFlow3D(save_dir_timestep, config)
        u, p = alg.computeOnPyramid(I0, I1, u, p)

        # save the old mask
        # save3D_torch_to_nifty(mask, saveDirTimeStep, f'mask_time{t1}.nii')
        # save_slices(mask, f'mask.png', saveDirTimeStep)
        # save_single_zslices(mask, saveDirTimeStep, 'mask_slices', 1., 2)

        # warp mask with the computed optical flow
        # mask = alg.warpMask(mask, u, I1, t0, saveDirTimeStep)


if __name__ == "__main__":
    # load config parser
    config = configparser.ConfigParser()
    config.read('parser/configTVL1OF3D.ini')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    mode_str = config.get('PARAMETERS', 'mode')
    mode = OpticalFlowMode.FORWARD if mode_str == 'Forward' else OpticalFlowMode.BACKWARD

    # create save directory
    save_dir = plots.createSaveDirectory(config.get('DATA', 'OUTPUT_PATH'), f'TVL1OF3D{mode_str}')

    logger = plots.create_logger(save_dir)
    logger.info(f'Compute TV-L1 optical flow ({mode_str})')

    # save config file to save directory
    conifg_output = os.path.sep.join([save_dir, "config.ini"])
    with open(conifg_output, 'w') as configfile:
        config.write(configfile)

    img4d_transforms = T.ComposeUnary([T.ToTensor()])
    # mask_transf = T.ComposeUnary([T.ToTensor()])

    train_ds = SingleVentricleDataset(config, DatasetMode.TRAIN, LoadFlowMode.NO_LOAD_OF, img4d_transforms)
    val_ds = SingleVentricleDataset(config, DatasetMode.VAL, LoadFlowMode.NO_LOAD_OF, img4d_transforms)
    test_ds = SingleVentricleDataset(config, DatasetMode.TEST, LoadFlowMode.NO_LOAD_OF, img4d_transforms)

    compute_all_patients = config.get('DATA', 'COMPUTE_ALL_PATIENTS')

    ds_list = []
    use_indices = config.getboolean('PARAMETERS', 'use_indices')
    if use_indices:
        dataset = config.get('PARAMETERS', 'dataset')
        if dataset == 'train':
            ds_list.append(train_ds)
        elif dataset == 'val':
            ds_list.append(val_ds)
        elif dataset == 'test':
            ds_list.append(test_ds)

        from_idx = config.getint('PARAMETERS', 'from_idx')
        to_idx = config.getint('PARAMETERS', 'to_idx')
    else:
        ds_list = [train_ds, val_ds, test_ds]
        from_idx = 0

    if compute_all_patients:
        pbar = tqdm(total=to_idx - from_idx if use_indices else len(train_ds) + len(val_ds) + len(test_ds))
        for ds in ds_list:
            to_idx = to_idx if use_indices else len(ds)

            for idx in range(from_idx, to_idx):
                compute_optical_flow(ds, idx, mode, save_dir, device, config, logger)
                pbar.update(1)
    else:
        pbar = tqdm(total=1)
        patient_name = config.get('DATA', 'PATIENT_NAME')
        idx, found = train_ds.index_for_patient(patient_name)
        if not found:
            logger.info(patient_name + " not found!")
            sys.exit()
        else:
            compute_optical_flow(train_ds, idx, mode, save_dir, device, config, logger)
            pbar.update(1)
