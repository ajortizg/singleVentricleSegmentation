from dataset import SVDSegmentation
import transforms as T
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import torch
import os.path as osp
import time
import sys
from monai.networks.nets import UNETR
from monai.losses.dice import DiceLoss, DiceCELoss
import matplotlib.pyplot as plt

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
from utils import plots
from cnn.models import model_factory

if __name__ == '__main__':
    num_epochs = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # cudnn.benchmark = True
    # cudnn.deterministic = True

    train_transforms = T.Compose([
        T.CropForeground(p=1.0, tol=10),
        T.Resize(p=1.0, size=(96, 96, 96)),
        T.MinMaxNormalization(p=1.0),
        T.RandomRotate(p=1.0, range_z=(0, 360), boundary='border'),
        T.RandomVerticalFlip(p=0.5),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomDepthFlip(p=0.5),
        T.BinarizeMasks(th=0.5),
        T.ToTensor(add_ch_dim=True)
        # T6.ElasticDeformation(P['ed_prob'], P['ed_sigma_range'], P['ed_grid'], P['ed_boundary'], P['ed_prefilter'], P['ed_axis'], P['ed_order'], P['clip_interval']),
        # T6.GammaScaling(P['gamma_scaling_prob'], P['gamma_scaling_range']),
        # T6.MutiplicativeScaling(P['mult_scaling_prob'], P['gamma_scaling_range'], P['clip_interval']),
        # T6.AdditiveScaling(P['add_scaling_prob'], P['add_scaling_mean'], P['add_scaling_std'], P['clip_interval']),
        # T6.AdditiveGaussianNoise(P['noise_prob'], P['noise_mu'], P['noise_std'], P['clip_interval']),
    ])

    train_ds = SVDSegmentation('data/svd_segmentation', mode='train', transforms=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=8, collate_fn=SVDSegmentation.collate_fn)

    save_dir = plots.createSaveDirectory('results_2023', 'SEG')
    logger = plots.create_logger(save_dir)
    logger.info('Save dir: {}'.format(save_dir))
    logger.info('Device: {}'.format(device))

    net = UNETR(in_channels=1,
                out_channels=1,
                img_size=(96, 96, 96),
                feature_size=16,
                hidden_size=768,
                mlp_dim=3072,
                num_heads=12,
                spatial_dims=3).to(device)

    # loss_fn = DiceCELoss(lambda_ce=0.0, lambda_dice=1.0, sigmoid=True)
    loss_fn = DiceLoss(sigmoid=True)
    optimizer = optim.Adam(net.parameters(), lr=5e-4)

    train_losses = []
    tic = time.time()
    for e in range(1, num_epochs + 1):
        epoch_loss = 0

        for i, data in enumerate(train_loader):
            img = data['img'].to(device)
            mask = data['mask'].to(device)

            for k in range(2):
                output = net(img[..., k])
                loss = loss_fn(output, mask[..., k])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

        train_losses.append(epoch_loss / (len(train_loader) * 2))
        logger.info('Epoch: {}, loss: {:.3f}'.format(e, train_losses[-1]))

    logger.info('\nTotal time taken to train the model: {:.3f} H'.format((time.time() - tic) / 3600.0))

    plt.plot(train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(osp.join(save_dir, 'loss.png'))
