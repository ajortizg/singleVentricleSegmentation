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
from monai.metrics.meandice import compute_dice
import matplotlib.pyplot as plt
from terminaltables import AsciiTable
from torch.utils.tensorboard import SummaryWriter


ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
from utils import plots
from cnn.models import model_factory


def create_dataloaders():
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

    val_transforms = T.Compose([
        T.CropForeground(p=1.0, tol=10),
        T.Resize(p=1.0, size=(96, 96, 96)),
        T.MinMaxNormalization(p=1.0),
        T.ToTensor(add_ch_dim=True)
    ])

    train_ds = SVDSegmentation('data/svd_segmentation', mode='train', transforms=train_transforms)
    val_ds = SVDSegmentation('data/svd_segmentation', mode='val', transforms=val_transforms)
    test_ds = SVDSegmentation('data/svd_segmentation', mode='test', transforms=val_transforms)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=8, collate_fn=SVDSegmentation.collate_fn)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=4, collate_fn=SVDSegmentation.collate_fn)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=1, collate_fn=SVDSegmentation.collate_fn)
    return train_loader, val_loader, test_loader


def train(net, loss_fn, opt, loader, device):
    net.train()
    accum_loss = 0
    accum_dice = 0

    for data in loader:
        img = data['img'].to(device)
        mask = data['mask'].to(device)

        for k in range(2):
            output = net(img[..., k])
            output = F.sigmoid(output)
            loss = loss_fn(output, mask[..., k])
            opt.zero_grad()
            loss.backward()
            opt.step()
            accum_loss += loss.item()

            with torch.no_grad():
                output = torch.where(output > 0.5, 1.0, 0.0)
                dice = compute_dice(output, mask[..., k]).mean()
                accum_dice += dice.item()

    mean_loss = accum_loss / (len(loader) * 2)
    mean_dice = accum_dice / (len(loader) * 2)
    return mean_loss, mean_dice


@torch.no_grad()
def validate(net, loss_fn, loader, device):
    net.eval()
    accum_loss = 0
    accum_dice = 0

    for data in loader:
        img = data['img'].to(device)
        mask = data['mask'].to(device)

        for k in range(2):
            output = net(img[..., k])
            output = F.sigmoid(output)
            loss = loss_fn(output, mask[..., k])
            accum_loss += loss.item()

            output = torch.where(output > 0.5, 1.0, 0.0)
            dice = compute_dice(output, mask[..., k]).mean()
            accum_dice += dice.item()

    mean_loss = accum_loss / (len(loader) * 2)
    mean_dice = accum_dice / (len(loader) * 2)
    return mean_loss, mean_dice


@torch.no_grad()
def test(net, loader, device):
    net.eval()
    accum_dice = 0

    for data in loader:
        img = data['img'].to(device).squeeze(0)
        mask = data['mask'].to(device).squeeze(0)
        img = torch.permute(img, (4, 0, 1, 2, 3))
        mask = torch.permute(mask, (4, 0, 1, 2, 3))

        output = net(img)
        output = F.sigmoid(output)
        output = torch.where(output > 0.5, 1.0, 0.0)
        dice = compute_dice(output, mask).mean()
        accum_dice += dice.item()

    mean_dice = accum_dice / len(loader)
    return mean_dice


if __name__ == '__main__':
    num_epochs = 500
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader, test_loader = create_dataloaders()

    save_dir = plots.createSaveDirectory('results', 'SEG')
    writer = SummaryWriter(log_dir=save_dir,)
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
    loss_fn = DiceLoss(sigmoid=False)
    optimizer = optim.Adam(net.parameters(), lr=5e-4)

    train_losses = []
    train_dices = []
    val_losses = []
    val_dices = []
    test_dices = []

    tic = time.time()
    for e in tqdm(range(1, num_epochs + 1)):
        loss, dice = train(net, loss_fn, optimizer, train_loader, device)
        train_losses.append(loss)
        train_dices.append(dice)

        loss, dice = validate(net, loss_fn, val_loader, device)
        val_losses.append(loss)
        val_dices.append(dice)

        dice = test(net, test_loader, device)
        test_dices.append(dice)

        logger.info(AsciiTable([
            ['Split', 'Loss', 'Dice'],
            ['Train', '{:.3f}'.format(train_losses[-1]), '{:.3f}'.format(train_dices[-1])],
            ['Val', '{:.3f}'.format(val_losses[-1]), '{:.3f}'.format(val_dices[-1])],
            ['Test', 'NA', '{:.3f}'.format(test_dices[-1])]
        ]).table)

        writer.add_scalars('loss', {'train': train_losses[-1], 'val': val_losses[-1]}, e)
        writer.add_scalars('dice', {'train': train_dices[-1], 'val': val_dices[-1], 'test': test_dices[-1]}, e)

    logger.info('\nTotal time taken to train the model: {:.3f} hrs.'.format((time.time() - tic) / 3600.0))

    # Plot loss history
    plt.figure()
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='lower left')
    plt.savefig(osp.join(save_dir, 'loss.png'))

    # Plot dice history
    plt.figure()
    plt.plot(train_dices, label='train')
    plt.plot(val_dices, label='val')
    plt.plot(test_dices, label='test')
    plt.xlabel('Epoch')
    plt.ylabel('Dice')
    plt.legend(loc='lower right')
    plt.savefig(osp.join(save_dir, 'dice.png'))

    writer.close()
