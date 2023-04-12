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
from segmentation.dataset import SVDataset, get_bounds
import segmentation.transforms as T

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# found = False

def optical_flow(data, mode, cfg, save_dir, pbar):
    # Report for time statistics
    report = pd.DataFrame(columns=['Patient', 'Time', 'NT'])

    img = data['img'].squeeze(0).to(device)
    patient = data['patient'][0]
    # # global found
    # if patient != 'Adult_11':
    #     return report
    #     # found = True
    
    # # if not found:
    # #     return report

    *_, indices = get_bounds(data['es'].item(), data['ed'].item(), None, fwd=mode == 'forward')

    # Create saving directory for current patient
    patient_dir = plots.createSubDirectory(save_dir, patient)

    # Initialization of optical flow and mask
    NZ, NY, NX, NT = img.shape
    u = torch.zeros([NZ, NY, NX, 3], dtype=torch.float32, device=device)
    p = torch.zeros([NZ, NY, NX, 3, 3], dtype=torch.float32, device=device)
    alg = TVL1OpticalFlow3D(cfg)

    tic = time.time()
    for i in range(len(indices) - 1):
        t0 = indices[i + 1].item()
        t1 = indices[i].item()
        I0 = img[..., t0]
        I1 = img[..., t1]

        pbar.set_postfix_str(f'P: {patient}, ({t1}->{t0}/{indices[-1]})')
        save_dir_timestep = plots.createSubDirectory(patient_dir, f'time{t1}')

        alg.set_save_dir(save_dir_timestep)
        u, p = alg.computeOnPyramid(I0, I1, u, p)
    report.loc[len(report)] = [patient, (time.time() - tic) / 60.0, len(indices)]
    torch.cuda.empty_cache()
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
        T.ToRAS(),
        # T.CropForeground(p=1.0, tol=10),
        # T.Resize(p=1.0, size=(img_sz, img_sz, img_sz)),
        # T.BinarizeMasks(th=0.5),
        T.MinMaxNormalization(q1=5, q2=95),
        T.ToTensor()
    ])
    dset = SVDataset(data_cfg['base_path_3d'], 'full', transforms, vxm=True)
    loader = DataLoader(dset, batch_size=1, shuffle=False, num_workers=4)

    pbar = tqdm(total=len(loader))
    tic = time.time()
    for data in loader:
        report = optical_flow(data, mode, cfg, save_dir, pbar)
        logger.info(tabulate(report, headers='keys', tablefmt='psql'))
        pbar.update(1)

    logger.info('Total time {:.3f} hrs.'.format((time.time() - tic) / 3600.0))
