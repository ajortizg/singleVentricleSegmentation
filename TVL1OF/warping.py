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
from utilities import path_utils as fpu
from TVL1OF.dataset import FlowUNetDataset, get_bounds
import segmentation.transforms as T
from cnn.warp import WarpCNN

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def propagate(warp, es, ed, labels, flow, fwd, is_test):
    *_, mi, mf, indices = get_bounds(es, ed, labels, fwd=fwd, test=is_test)
    mi = mi.unsqueeze(0)
    mf = mf.unsqueeze(0)

    propagated_mask = mi.clone()
    if is_test:
        est_masks = []

    for i in range(len(indices) - 1):
        propagated_mask = warp(propagated_mask, flow[i].unsqueeze(0))
        if is_test:
            est_masks.append(propagated_mask)

    if is_test:
        y_pred = torch.cat(est_masks).round()
        y_true = labels[indices[1:]]
    else:
        y_pred = propagated_mask.round()
        y_true = mf

    dice = compute_dice(y_pred, y_true, include_background=False).mean().item()
    hd = compute_hausdorff_distance(y_pred, y_true, include_background=False).mean().item()
    return dice, hd


if __name__ == '__main__':
    # Read configuration parameters
    cfg = configparser.ConfigParser()
    cfg.read('parser/flow_warping.ini')
    data_cfg = cfg['DATA']
    param_cfg = cfg['PARAMETERS']

    # Create dataset
    transforms = T.Compose([
        T.XYZT_To_TZYX(keys=['image', 'label']),
        T.AddDimAt(axis=1, keys=['image', 'label']),
        T.Flow_T3XYZ_To_T3ZYX(keys=['forward_flow', 'backward_flow']),
        T.FlowChannelToLastDim(keys=['forward_flow', 'backward_flow']),
        T.OneHotEncoding(n=2, keys=['label']),
        T.ToTensor(keys=['image', 'label', 'forward_flow', 'backward_flow'])
    ])

    full_ds = FlowUNetDataset(data_cfg['root_dir'], 'full', transforms, load_flow=True)
    full_loader = DataLoader(full_ds, batch_size=8, shuffle=False, num_workers=1, collate_fn=FlowUNetDataset.collate)

    for data in full_loader:
        print('-'*5)

    ###############################################################################################
    # full_ds = FlowUNetDataset(data_cfg['root_dir'], 'full', transforms, load_flow=True)
    # test_ds = FlowUNetDataset(data_cfg['root_dir'], 'test', transforms, load_flow=True)
    # full_loader = DataLoader(full_ds, batch_size=1, shuffle=False, num_workers=8)
    # test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=3)

    # # Saving directory for debug outputs
    # # save_dir = plots.createSaveDirectory(data_cfg.get('output_path'), 'Warp')
    # # plots.save_config(cfg, save_dir)
    # # logger = plots.create_logger(save_dir)

    # loaders = [full_loader, test_loader]
    # test_flags = [False, True]

    # for loader, is_test in zip(loaders, test_flags):
    #     report = pd.DataFrame(columns=['Patient', 'Dice_Fwd', 'Dice_Bwd', 'HD_Fwd', 'HD_Bwd'])

    #     for data in tqdm(loader):
    #         labels = data['label'].to(device).squeeze(0)
    #         ff = data['forward_flow'].to(device).squeeze(0)
    #         bf = data['backward_flow'].to(device).squeeze(0)
    #         nz, ny, nx = labels.shape[2:]
    #         warp = WarpCNN(cfg, nz, ny, nx)

    #         dice_fwd, hd_fwd = propagate(warp, data['es'].item(), data['ed'].item(), labels, ff, fwd=True, is_test=is_test)
    #         dice_bwd, hd_bwd = propagate(warp, data['es'].item(), data['ed'].item(), labels, bf, fwd=False, is_test=is_test)
    #         report.loc[len(report)] = [data['patient'][0], dice_fwd, dice_bwd, hd_fwd, hd_bwd]

    #     report.loc[len(report)] = ['Mean', report['Dice_Fwd'].mean(), report['Dice_Bwd'].mean(), report['HD_Fwd'].mean(), report['HD_Bwd'].mean()]
    #     print(tabulate(report, headers='keys', tablefmt='psql'))
