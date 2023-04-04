import os
import os.path as osp
import sys
import configparser
import random
import argparse
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch import nn

# import voxelmorph with pytorch backend
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph.voxelmorph as vxm  # nopep8

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
from utils import plots
from segmentation.dataset import SVDataset
import segmentation.transforms as T


def train(train_loader, model, optimizer, losses, weights):
    model.train()
    epoch_total_loss = []

    for data in train_loader:
        img = data['img'].permute(4, 0, 1, 2, 3).to(device)  # NT, CH, NZ, NY, NX
        # img = data['img'].permute(4, 0, 3, 2, 1).to(device)  # NT, CH, NX, NY, NZ

        for t in range(img.shape[0] - 1):
            moving = img[t, ...].unsqueeze(0)
            fixed = img[t + 1, ...].unsqueeze(0)
            y_pred = model(moving, fixed)

            # calculate total loss
            y_true = [fixed, None]
            loss = 0
            for n, loss_function in enumerate(losses):
                curr_loss = loss_function(y_true[n], y_pred[n]) * weights[n]
                loss += curr_loss

            epoch_total_loss.append(loss.item())

            # backpropagate and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return np.array(epoch_total_loss).mean()


if __name__ == '__main__':
    # Seeding for reproducible results
    plots.seeding(42)

    # Read configuration options
    config = configparser.ConfigParser()
    config.read('parser/vxm_train.ini')
    root_dir = config.get('DATA', 'ROOT_DIR')
    output_dir = config.get('DATA', 'OUTPUT_DIR')
    img_sz = config.getint('PARAMETERS', 'IMG_SIZE')
    workers = config.getint('PARAMETERS', 'WORKERS')
    epochs = config.getint('PARAMETERS', 'NUM_EPOCHS')
    batch_size = config.getint('PARAMETERS', 'BATCH_SIZE')
    enc_nf = list(map(int, config.get('PARAMETERS', 'ENC_FEAT').split(',')))
    dec_nf = list(map(int, config.get('PARAMETERS', 'DEC_FEAT').split(',')))
    int_steps = config.getint('PARAMETERS', 'INT_STEPS')
    int_downsize = config.getint('PARAMETERS', 'INT_DOWNSIZE')
    bidir = config.getboolean('PARAMETERS', 'BIDIR')
    img_loss = config.get('PARAMETERS', 'IMG_LOSS')
    lambda_param = config.getfloat('PARAMETERS', 'LAMBDA')
    lr = config.getfloat('PARAMETERS', 'LR')
    cudnn_nondet = config.getboolean('PARAMETERS', 'CUDNN_NONDET')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: ', device)
    # enabling cudnn determinism appears to speed up training by a lot
    torch.backends.cudnn.deterministic = not cudnn_nondet
    print('cudnn.deterministic: ', (not cudnn_nondet))

    save_dir = plots.createSaveDirectory(output_dir, 'VXM')
    plots.save_config(config, save_dir)
    writer = SummaryWriter(log_dir=save_dir)

    transforms = T.Compose([T.CropForeground(p=1.0, tol=10),
                            T.Resize(p=1.0, size=(img_sz, img_sz, img_sz)),
                            # T.RandomRotate(p=1.0, range_z=(0, 360), boundary='border'),
                            T.RandomVerticalFlip(p=0.5),
                            T.RandomHorizontalFlip(p=0.5),
                            T.RandomDepthFlip(p=0.5),
                            # T.ElasticDeformation(p=0.1, sigma_range=(0.5, 2.0), points=8, boundary='nearest', prefilter=False, axis='yx', order=1),
                            # T.GammaScaling(p=0.3, gamma_range=(0.8, 1.2)),
                            T.MinMaxNormalization(p=1.0),
                            # T.ZScoreNormalization(p=1.0),
                            # T.QuadraticNormalization(p=0.1, mean_inside_mask=False),
                            T.BinarizeMasks(th=0.5),
                            T.ToTensor(add_ch_dim=False)])

    train_ds = SVDataset(root_dir, 'train', transforms, vxm=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, collate_fn=SVDataset.collate_fn)

    model = vxm.networks.VxmDense(inshape=(img_sz, img_sz, img_sz),
                                  nb_unet_features=[enc_nf, dec_nf],
                                  bidir=bidir,
                                  int_steps=int_steps,
                                  int_downsize=int_downsize)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if img_loss == 'ncc':
        image_loss_func = vxm.losses.NCC().loss
    elif img_loss == 'mse':
        image_loss_func = vxm.losses.MSE().loss
        # image_loss_func = nn.MSELoss(reduction='sum')
    else:
        raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % img_loss)

    losses = [image_loss_func, vxm.losses.Grad('l2', loss_mult=2).loss]
    weights = [1.0, lambda_param]

    for e in tqdm(range(1, epochs + 1)):
        train_loss = train(train_loader, model, optimizer, losses, weights)

        # Save model every 20 epochs
        if e % 20 == 0:
            model.save(os.path.join(save_dir, 'model.pth'))

        writer.add_scalar('loss', train_loss.item(), e)
        print('Epoch: {}, Loss: {:.6f}'.format(e, train_loss))

    model.save(os.path.join(save_dir, 'model.pth'))
