import os
import os.path as osp
import sys
import argparse
import configparser
import time

# third party
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from monai.metrics.meandice import compute_dice
from terminaltables import AsciiTable
from tqdm import tqdm

# import voxelmorph with pytorch backend
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph.voxelmorph as vxm   # nopep8

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
from utils import plots
from segmentation.dataset import SVDataset
import segmentation.transforms as T


def bounds(data, device, fwd):
    mask = data['mask'].permute(4, 0, 1, 2, 3).to(device)
    es = data['es'].item()
    ed = data['ed'].item()

    if fwd:
        if es < ed:
            ti, tf = es, ed
            mi, mf = mask[0, ...], mask[1, ...]
        else:
            ti, tf = ed, es
            mi, mf = mask[1, ...], mask[0, ...]
        indices = torch.arange(ti, tf + 1, 1)
    else:
        if ed > es:
            ti, tf = ed, es
            mi, mf = mask[1, ...], mask[0, ...]
        else:
            ti, tf = es, ed
            mi, mf = mask[0, ...], mask[1, ...]
        indices = torch.arange(ti, tf - 1, -1)

    return indices, mi.unsqueeze(0), mf.unsqueeze(0)


@torch.no_grad()
def validation(loader, model, device, verbose, fwd):
    model.eval()
    dices = []

    for data in tqdm(loader):
        img = data['img'].permute(4, 0, 1, 2, 3).to(device)  # NT, CH, NZ, NY, NX
        patient = data['patient'][0]
        indices, mi, mf = bounds(data, device, fwd)

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
        dices.append(dice)

        if verbose:
            print(AsciiTable([
                ['Patient', 'Mode', 'Accuracy', 'Time'],
                [patient, fwd, '{:.3f}'.format(dice), '{:.3f}'.format(toc - tic)]
            ]).table)

    return np.array(dices).mean()


if __name__ == '__main__':
   # Read configuration options
    config = configparser.ConfigParser()
    config.read('parser/vxm_register.ini')
    root_dir = config.get('DATA', 'ROOT_DIR')
    output_dir = config.get('DATA', 'OUTPUT_DIR')
    model_weights = config.get('DATA', 'MODEL')
    warp = config.getboolean('PARAMETERS', 'WARP')
    img_sz = config.getint('PARAMETERS', 'IMG_SIZE')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: ', device)
    print('Weights: ', model_weights)

    # load and set up model
    model = vxm.networks.VxmDense.load(model_weights, device)
    model.to(device)

    transforms = T.Compose([T.CropForeground(p=1.0, tol=10),
                            T.Resize(p=1.0, size=(img_sz, img_sz, img_sz)),
                            T.MinMaxNormalization(p=1.0),
                            T.BinarizeMasks(th=0.5),
                            T.ToTensor(add_ch_dim=False)])

    val_ds = SVDataset(root_dir, 'val', transforms, vxm=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4, collate_fn=SVDataset.collate_fn)

    save_dir = plots.createSaveDirectory(output_dir, 'REG')

    for mode in ['fwd', 'bwd']:
        tic = time.time()
        acc = validation(val_loader, model, device, verbose=False, fwd=mode == 'fwd')
        print(AsciiTable([
            ['Split', 'Mode', 'Accuracy', 'Time (s)'],
            ['val', mode, '{:.3f}'.format(acc), '{:.3f}'.format((time.time() - tic) / len(val_loader))]
        ]).table)

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
