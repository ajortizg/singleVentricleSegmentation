import os
import os.path as osp
import sys
import configparser
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from monai.metrics.meandice import compute_dice
from monai.metrics.hausdorff_distance import compute_hausdorff_distance
from tabulate import tabulate
from tqdm import tqdm
import pandas as pd

# import voxelmorph with pytorch backend
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph.voxelmorph as vxm   # nopep8

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
import utilities.path_utils as path_utils
from TVL1OF.dataset import FlowUNetDataset


@torch.no_grad()
def validation(loader, model, device, verbose, fwd):
    model.eval()
    report = pd.DataFrame(columns=['Patient', 'Dice', 'HD', 'Time'])

    for data in tqdm(loader):
        img = data['img'].permute(4, 0, 1, 2, 3).to(device)  # NT, CH, NZ, NY, NX
        patient = data['patient'][0]
        indices, mi, mf = bounds(data, device, False, fwd)

        tic = time.time()
        propagated_mask = mi.clone()
        for i in range(len(indices) - 1):
            tm = indices[i].item()
            tf = indices[i + 1].item()
            moving = img[tm, ...].unsqueeze(0)
            fixed = img[tf, ...].unsqueeze(0)

            # Compute flow between two consecutive frames
            _, flow = model(moving, fixed, registration=True)

            # Propagate masks
            propagated_mask = model.transformer(propagated_mask, flow)
            # propagated_mask = torch.where(propagated_mask > 0.5, 1.0, 0.0)

        toc = time.time()
        propagated_mask = torch.where(propagated_mask > 0.5, 1.0, 0.0)
        dice = compute_dice(propagated_mask, mf).item()
        hd = compute_hausdorff_distance(propagated_mask, mf).item()
        report.loc[len(report)] = [patient, dice, hd, toc - tic]

    if verbose:
        print(tabulate(report, headers='keys', tablefmt='psql'))

    return report


@torch.no_grad()
def test(loader, model, device, verbose, fwd):
    model.eval()
    report = pd.DataFrame(columns=['Patient', 'Dice', 'HD', 'Time'])

    for data in tqdm(loader):
        img = data['img'].permute(4, 0, 1, 2, 3).to(device)  # NT, CH, NZ, NY, NX
        patient = data['patient'][0]
        indices, mi, _ = bounds(data, device, True, fwd)

        est_masks = []
        propagated_mask = mi.clone()
        tic = time.time()
        for i in range(len(indices) - 1):
            tm = indices[i].item()
            tf = indices[i + 1].item()
            moving = img[tm, ...].unsqueeze(0)
            fixed = img[tf, ...].unsqueeze(0)

            # Compute flow between two consecutive frames
            _, flow = model(moving, fixed, registration=True)

            propagated_mask = model.transformer(propagated_mask, flow)
            # propagated_mask = torch.where(propagated_mask > 0.5, 1.0, 0.0)
            est_masks.append(propagated_mask)

        toc = time.time()
        true_masks = data['mask'].permute(4, 0, 1, 2, 3)[indices[1:], ...].to(device)

        est_masks = torch.cat(est_masks)
        est_masks = torch.where(est_masks > 0.5, 1.0, 0.0)
        dice = compute_dice(est_masks, true_masks).mean().item()
        hd = compute_hausdorff_distance(est_masks, true_masks).mean().item()
        report.loc[len(report)] = [patient, dice, hd, toc - tic]

    if verbose:
        print(tabulate(report, headers='keys', tablefmt='psql'))

    return report


if __name__ == '__main__':
   # Read configuration options
    config = configparser.ConfigParser()
    config.read('parser/vxm_register.ini')
    root_dir = config.get('DATA', 'ROOT_DIR')
    output_dir = config.get('DATA', 'OUTPUT_DIR')
    model_weights = config.get('DATA', 'MODEL')
    img_sz = config.getint('PARAMETERS', 'IMG_SIZE')
    verbose = config.getboolean('DEBUG', 'VERBOSE')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: ', device)
    print('Weights: ', model_weights)

    # load and set up model
    model = vxm.networks.VxmDense.load(model_weights, device)
    model.to(device)

    transforms = T.Compose([T.CropForeground(p=1.0, tol=10),
                            T.Resize(p=1.0, size=(img_sz, img_sz, img_sz)),
                            T.MinMaxNormalization(p=1.0),
                            T.Discretize(th=0.5),
                            T.ToTensor(add_ch_dim=False)])

    val_ds = FlowUNetDataset(root_dir, 'val', transforms)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)

    test_ds = FlowUNetDataset(root_dir, 'test', transforms, vxm=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=1)

    # save_dir = plots.createSaveDirectory(output_dir, 'REG')

    # for mode in ['fwd', 'bwd']:
    #     report = validation(val_loader, model, device, verbose=verbose, fwd=mode == 'fwd')
    #     print('Mode: {}, mean dice: {:.3f}, mean hd: {:.3f}, mean time: {:.3f}'.format(mode, report['Dice'].mean(), report['HD'].mean(), report['Time'].mean()))

    for mode in ['fwd', 'bwd']:
        report = test(test_loader, model, device, verbose=verbose, fwd=mode == 'fwd')
        print('Mode: {}, mean dice: {:.3f}, mean hd: {:.3f}, mean time: {:.3f}'.format(
            mode, report['Dice'].mean(), report['HD'].mean(), report['Time'].mean()))

    # for t in range(img.shape[0] - 1):
    #     moving = img[t, ...].unsqueeze(0)
    #     fixed = img[t + 1, ...].unsqueeze(0)
    #     mask_moving = mask[t, ...].unsqueeze(0)
    #     moved, warp = model(moving, fixed, registration=True)

    #     mask_moved = model.transformer(mask_moving, warp)

    #     moved = F.interpolate(moved, size=(16, 96, 96), align_corners=True, mode='trilinear').squeeze()
    #     mask_moved = F.interpolate(mask_moved, size=(16, 96, 96), align_corners=True, mode='trilinear').squeeze()
    #     mask_moved = torch.where(mask_moved > 0.5, 1.0, 0.0)

    #     plots.save_slices(moved, f'{patient}_img_t{t}.png', save_dir)
    #     plots.save_slices(mask_moved, f'{patient}_mask_t{t}.png', save_dir)
