import os
import os.path as osp
import sys
import configparser

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from tabulate import tabulate

from TVL1OF3D import *

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
from utils import plots
from TVL1OF.dataset import SVDataset, get_bounds
import TVL1OF.transforms as T

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# found = False


def optical_flow(data, mode, cfg, save_dir, pbar):
    # Report for time statistics
    report = pd.DataFrame(columns=['Patient', 'Time', 'NT'])

    img = data['image'].squeeze(0).to(device)
    patient = data['patient'][0]
    # global found
    # if patient == 'Adult_81':
    #     found = True

    # if not found:
    #     return report

    *_, indices = get_bounds(data['es'].item(), data['ed'].item(), None, fwd=mode == 'forward')

    # Create saving directory for current patient
    # patient_dir = plots.createSubDirectory(save_dir, patient)

    # Initialization of optical flow and mask
    NT, NZ, NY, NX = img.shape
    u = torch.zeros((NZ, NY, NX, 3), dtype=torch.float32, device=device)
    us = torch.zeros((len(indices) - 1, 3, NZ, NY, NX), dtype=torch.float32, device=device)
    p = torch.zeros((NZ, NY, NX, 3, 3), dtype=torch.float32, device=device)
    alg = TVL1OpticalFlow3D(cfg)

    tic = time.time()
    for i in range(len(indices) - 1):
        t0 = indices[i + 1].item()
        t1 = indices[i].item()
        I0 = img[t0]
        I1 = img[t1]

        pbar.set_postfix_str(f'P: {patient}, ({t1}->{t0}/{indices[-1]})')
        # save_dir_timestep = plots.createSubDirectory(patient_dir, f'time{t1}')
        # alg.set_save_dir(save_dir_timestep)

        u, p = alg.computeOnPyramid(I0, I1, u, p)
        u = alg.apply_median_filter(u)
        
        us[i] = u.permute(3, 0, 1, 2)

    np.save(osp.join(save_dir, f'{patient}_{mode}_flow.npy'), us.cpu().detach().numpy())
    report.loc[len(report)] = [patient, (time.time() - tic) / 60.0, len(indices)]
    # torch.cuda.empty_cache()
    return report


if __name__ == '__main__':
    # Read configuration parameters
    cfg = configparser.ConfigParser()
    cfg.read('parser/flow_tvl1_3d.ini')
    data_cfg = cfg['DATA']
    param_cfg = cfg['PARAMETERS']

    # Create save directory and console logger
    mode = param_cfg.get('mode').lower()
    save_dir = plots.createSaveDirectory(data_cfg.get('output_path'), f'TVL1OF3D{mode}')
    logger = plots.create_logger(save_dir)
    logger.info(f'Compute TV-L1 optical flow ({mode})')
    plots.save_config(cfg, save_dir)

    # Create dataset
    img_sz = param_cfg.getint('img_sz')
    transforms = T.Compose([
        T.XYZT_To_TZYX(keys=['image', 'label']),
        T.ToRAS(keys=['image', 'label']),
        T.CropForeground(tol=10, keys=['image', 'label'], label_key='label'),
        T.QuadraticNormalization(q2=95, keys=['image'], label_key='label'),
        T.Resize(p=1.0, size=(img_sz, img_sz, img_sz), keys=['image', 'label'], label_key='label'),
        T.ToTensor()
    ])
    plots.save_transforms_to_json(transforms, osp.join(save_dir, 'transforms.json'))

    dataset = SVDataset(data_cfg['base_path_3d'], 'full', transforms)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    pbar = tqdm(total=len(loader))
    tic = time.time()
    for data in loader:
        report = optical_flow(data, mode, cfg, save_dir, pbar)
        logger.info(tabulate(report, headers='keys', tablefmt='psql'))
        pbar.update(1)

    logger.info('Total time {:.3f} hrs.'.format((time.time() - tic) / 3600.0))

    # for data in dset:
    #     img = data['image']
    #     label = data['label']
    #     print(img.shape, label.shape)

    #     times = [data['es'], data['ed']]
    #     index = [0, 1]
    #     patient = data['patient']

    #     for t, i in zip(times, index):
    #         plots.save_overlaped_img_mask(img[t],
    #                                       label[i],
    #                                       f'{patient}_{i}.png',
    #                                       save_dir, th=0.5,
    #                                       alpha=0.3)
