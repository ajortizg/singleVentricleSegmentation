import os
import os.path as osp
import sys
import argparse
import configparser

# third party
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

# import voxelmorph with pytorch backend
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph.voxelmorph as vxm   # nopep8

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
from utils import plots
from segmentation.dataset import SVDataset
import segmentation.transforms as T

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
    model.eval()

    transforms = T.Compose([T.CropForeground(p=1.0, tol=10),
                            T.Resize(p=1.0, size=(img_sz, img_sz, img_sz)),
                            # T.QuadraticNormalization(p=1.0, mean_inside_mask=False),
                            T.MinMaxNormalization(p=1.0),
                            T.BinarizeMasks(th=0.5),
                            T.ToTensor(add_ch_dim=False)])

    train_ds = SVDataset(root_dir, 'test', transforms, vxm=True)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False, num_workers=1, collate_fn=SVDataset.collate_fn)

    save_dir = plots.createSaveDirectory(output_dir, 'REG')

    for data in train_loader:
        img = data['img'].permute(4, 0, 1, 2, 3).to(device)  # NT, CH, NZ, NY, NX
        mask = data['mask'].permute(4, 0, 1, 2, 3).to(device)
        patient = data['patient'][0]

        for t in range(img.shape[0] - 1):
            moving = img[t, ...].unsqueeze(0)
            fixed = img[t + 1, ...].unsqueeze(0)
            mask_moving = mask[t, ...].unsqueeze(0)
            moved, warp = model(moving, fixed, registration=True)

            mask_moved = model.transformer(mask_moving, warp)

            moved = F.interpolate(moved, size=(16, 96, 96), align_corners=True, mode='trilinear').squeeze()
            mask_moved = F.interpolate(mask_moved, size=(16, 96, 96), align_corners=True, mode='trilinear').squeeze()
            mask_moved = torch.where(mask_moved > 0.5, 1.0, 0.0)
            
            plots.save_slices(moved, f'{patient}_img_t{t}.png', save_dir)
            plots.save_slices(mask_moved, f'{patient}_mask_t{t}.png', save_dir)
