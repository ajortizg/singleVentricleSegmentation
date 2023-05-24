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
from datasets.flowunet_dataset import FlowUNetDataset, get_bounds
import segmentation.transforms as T
from cnn.warp import WarpCNN

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def propagate(warp, mi, mf, times, flow, labels, is_test):
    propagated_mask = mi.clone()
    if is_test:
        est_masks = []

    for i in range(len(times) - 1):
        propagated_mask = warp(propagated_mask, flow[i].unsqueeze(0))
        if is_test:
            est_masks.append(propagated_mask)

    if is_test:
        y_pred = T.one_hot(torch.cat(est_masks), mi.shape[1], argmax=True)
        y_true = labels[times[1:]]
    else:
        y_pred = T.one_hot(propagated_mask, mi.shape[1], argmax=True)
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
        # T.RandomFlip(p=1.0, axis=2, keys=['image', 'label', 'forward_flow', 'backward_flow']),
        # T.RandomFlip(p=1.0, axis=3, keys=['image', 'label', 'forward_flow', 'backward_flow']),
        # T.RandomFlip(p=1.0, axis=4, keys=['image', 'label', 'forward_flow', 'backward_flow']),
        # # T.Resize(1.0, (128, 64, 108), keys=['image', 'label', 'forward_flow', 'backward_flow'], label_key='label'),
        # T.RandomRotate(1.0, (0, 360), (0, 360), (0, 360), 'zeros', keys=['image', 'label', 'forward_flow', 'backward_flow'], label_key='label'),
        # T.RandomScale(1.0, (0.5, 1.5), 'zeros', keys=['image', 'label', 'forward_flow', 'backward_flow'], label_key='label'),
        # T.ElasticDeformation(1.0, (0.5, 1.5), 8, 'constant', 'yx', keys=['image', 'label', 'forward_flow', 'backward_flow'], label_key='label'),
        # T.SimulateLowResolution(1.0, (0.5, 1.0), keys=['image', 'label', 'forward_flow', 'backward_flow'], label_key='label'),
        T.FlowChannelToLastDim(keys=['forward_flow', 'backward_flow']),
        T.ExtremaPoints(['label']),
        T.ToTensor(keys=['image', 'label', 'forward_flow', 'backward_flow', 'mi', 'mf'])
    ])

    full_ds = FlowUNetDataset(data_cfg['base_path_3d'], 'full', transforms, load_flow=True)
    test_ds = FlowUNetDataset(data_cfg['base_path_3d'], 'test', transforms, load_flow=True)
    full_loader = DataLoader(full_ds, batch_size=1, shuffle=False, num_workers=8, collate_fn=FlowUNetDataset.collate)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=3, collate_fn=FlowUNetDataset.collate)
    n_classes = full_ds.num_classes() + 1 # Add background class

    # Saving directory for debug outputs
    # save_dir = plots.createSaveDirectory(data_cfg.get('output_path'), 'Warp')
    # plots.save_config(cfg, save_dir)
    # logger = plots.create_logger(save_dir)

    loaders = [full_loader, test_loader]
    test_flags = [False, True]

    for loader, is_test in zip(loaders, test_flags):
        report = pd.DataFrame(columns=['Patient', 'Dice_Fwd', 'Dice_Bwd', 'HD_Fwd', 'HD_Bwd'])

        for data in tqdm(loader):
            labels = T.one_hot(data['label'].squeeze(1).to(device), n_classes)
            mi = T.one_hot(data['mi'].squeeze(0).to(device), n_classes)
            mf = T.one_hot(data['mf'].squeeze(0).to(device), n_classes)
            ff = data['forward_flow'].squeeze(1).to(device)
            bf = data['backward_flow'].squeeze(1).to(device)
            fwdt = data['times_fwd'].squeeze(1).to(device)
            bwdt = data['times_bwd'].squeeze(1).to(device)
            nz, ny, nx = labels.shape[2:]
            warp = WarpCNN(cfg, nz, ny, nx)

            dice_fwd, hd_fwd = propagate(warp, mi, mf, fwdt, ff, labels, is_test)
            dice_bwd, hd_bwd = propagate(warp, mf, mi, bwdt, bf, labels, is_test)
            report.loc[len(report)] = [data['patient'][0], dice_fwd, dice_bwd, hd_fwd, hd_bwd]

        report.loc[len(report)] = ['Mean', report['Dice_Fwd'].mean(), report['Dice_Bwd'].mean(), report['HD_Fwd'].mean(), report['HD_Bwd'].mean()]
        print(tabulate(report, headers='keys', tablefmt='psql'))
