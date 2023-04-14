import configparser
import os
import os.path as osp
import sys
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from monai.metrics.meandice import compute_dice
from monai.metrics.hausdorff_distance import compute_hausdorff_distance
from tabulate import tabulate

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
from utils import plots
from segmentation.dataset import SVDataset, get_bounds
import segmentation.transforms as T
from cnn.warp import WarpCNN

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    # Read configuration parameters
    cfg = configparser.ConfigParser()
    cfg.read('parser/flow_warping.ini')
    data_cfg = cfg['DATA']
    param_cfg = cfg['PARAMETERS']

    # Create dataset
    img_sz = param_cfg.getint('img_sz')
    transforms = T.Compose([
        T.ToRAS(),
        T.CropForeground(p=1.0, tol=10),
        T.Resize(p=1.0, size=(img_sz, img_sz, img_sz)),
        T.Discretize(th=0.5),
        T.MinMaxNormalization(q1=5, q2=95),
        T.AddChannelDim(),
        T.ToTensor()
    ])
    dset = SVDataset(data_cfg['base_path_3d'], 'full', transforms, vxm=True, load_flow=True)
    loader = DataLoader(dset, batch_size=1, shuffle=False, num_workers=8)

    # Saving directory for debug outputs
    # save_dir = plots.createSaveDirectory(data_cfg.get('output_path'), 'Warp')
    # plots.save_config(cfg, save_dir)
    # logger = plots.create_logger(save_dir)

    report = pd.DataFrame(columns=['Patient', 'Dice_Fwd', 'Dice_Bwd', 'HD_Fwd', 'HD_Bwd'])

    for data in tqdm(loader):
        patient = data['patient'][0]
        es = data['es'].item()
        ed = data['ed'].item()
        imgs = data['img'].to(device)
        masks = data['mask'].to(device)
        ff = data['forward_flow'].to(device)
        bf = data['backward_flow'].to(device)

        BS, CH, NZ, NY, NX, NT = imgs.shape
        warp = WarpCNN(cfg, NZ, NY, NX)
        *_, mi, mf, indices = get_bounds(es, ed, masks, fwd=True)

        out = {'fwd': [mi], 'bwd': [mf]}
        for i in range(ff.shape[-1]):
            out['fwd'].append(warp(out['fwd'][-1], ff[..., i]))
            out['bwd'].append(warp(out['bwd'][-1], bf[..., i]))
        assert(len(out['fwd']) == len(out['bwd']))

        out['bwd'].reverse()
        mi_bwd = torch.where(out['bwd'][0] > 0.5, 1.0, 0.0)
        mf_fwd = torch.where(out['fwd'][-1] > 0.5, 1.0, 0.0)

        report.loc[len(report)] = [
            patient,
            compute_dice(mf_fwd, mf).mean().item(),
            compute_dice(mi_bwd, mi).mean().item(),
            compute_hausdorff_distance(mf_fwd, mf).mean().item(),
            compute_hausdorff_distance(mi_bwd, mi).mean().item()
        ]

    report.loc[len(report)] = [
        'Mean',
        report['Dice_Fwd'].mean(),
        report['Dice_Bwd'].mean(),
        report['HD_Fwd'].mean(),
        report['HD_Bwd'].mean()
    ]

    print(tabulate(report, headers='keys', tablefmt='psql'))
