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
from utilities import path_utils as fpu
from utilities import stuff
from datasets.flowunet_dataset import FlowUNetDataset, get_bounds
import segmentation.transforms as T

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def optical_flow(data, mode, cfg, save_dir, pbar):
    # Report for time statistics
    report = pd.DataFrame(columns=['Patient', 'Time', 'NT'])

    img = data['image'].squeeze(0).to(device)
    patient = data['patient'][0]
    *_, indices = get_bounds(data['es'].item(), data['ed'].item(), None, fwd=mode == 'forward')

    # Initialization of optical flow and mask
    NT, NZ, NY, NX = img.shape
    u = torch.zeros((NZ, NY, NX, 3), dtype=torch.float32, device=device)
    us = torch.zeros((len(indices) - 1, 3, NX, NY, NZ), dtype=torch.float32, device=device)
    p = torch.zeros((NZ, NY, NX, 3, 3), dtype=torch.float32, device=device)
    alg = TVL1OpticalFlow3D(cfg)

    tic = time.time()
    for i in range(len(indices) - 1):
        t0 = indices[i + 1].item()
        t1 = indices[i].item()
        I0 = img[t0]
        I1 = img[t1]
        
        pbar.set_postfix_str(f'P: {patient}, ({t1}->{t0}/{indices[-1]})')

        u, p = alg.computeOnPyramid(I0, I1, u, p)
        u = alg.apply_median_filter(u)
        us[i] = u.permute(3, 2, 1, 0)  # change to (3,x,y,z)

    # Save flow with shape (t,3,x,y,z)
    np.save(osp.join(save_dir, f'{patient}_{mode}_flow.npy'), us.cpu().detach().numpy())
    report.loc[len(report)] = [patient, (time.time() - tic) / 60.0, len(indices)]
    # torch.cuda.empty_cache()
    return report


if __name__ == '__main__':
    # Read configuration parameters
    cfg = configparser.ConfigParser()
    cfg.read('parser/flow_compute.ini')
    data_cfg = cfg['DATA']
    param_cfg = cfg['PARAMETERS']

    # Create save directory and console logger
    mode = param_cfg.get('mode').lower()
    save_dir = fpu.create_save_dir(data_cfg.get('output_path'), f'TVL1OF3D{mode}')
    logger = stuff.create_logger(save_dir)
    logger.info(f'Compute TV-L1 optical flow ({mode})')
    stuff.save_config(cfg, save_dir)

    # Create dataset
    img_sz = param_cfg.getint('img_sz')
    transforms = T.Compose([
        T.XYZT_To_TZYX(keys=['image', 'label']),
        T.ToTensor()
    ])
    stuff.save_transforms_to_json(transforms, osp.join(save_dir, 'transforms.json'))

    dataset = FlowUNetDataset(data_cfg['root_dir'], 'full', transforms)
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
