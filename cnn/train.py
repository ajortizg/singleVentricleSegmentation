import torch
import configparser
import sys
from torch.optim import Adam
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from unet_3d import UNet3D, BasicUnet3d
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np
import os.path as osp
import cnn_utils as utils
import os
from torchsummary import summary

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
from utils import plots
import utils.transforms as T
import cnn.dataset as ds
from cnn.trainer import Trainer


if __name__ == "__main__":
    utils.seeding(42)

    config = configparser.ConfigParser()
    config.read('parser/configCNNTrain.ini')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    P = utils.read_train_params(config)

    # Create train and validation datasets
    img4d_transforms = T.ComposeUnary([T.ToTensor()])
    mask_transforms = T.ComposeUnary([T.Round(th=0.5), T.ToTensor()])
    train_ds = ds.SingleVentricleDataset(config, ds.DatasetMode.TRAIN, ds.LoadFlowMode.TRAIN_OF, img4d_transforms, mask_transforms)
    val_ds = ds.SingleVentricleDataset(config, ds.DatasetMode.VAL, ds.LoadFlowMode.TRAIN_OF, img4d_transforms, mask_transforms)

    # Create data loaders
    train_loader = DataLoader(train_ds, batch_size=P['batch_size'], shuffle=True, num_workers=P['workers'], collate_fn=utils.collate_fn)
    val_loader = DataLoader(val_ds, batch_size=P['batch_size'], shuffle=False, num_workers=P['workers'], collate_fn=utils.collate_fn)

    save_dir = plots.createSaveDirectory(config.get('DATA', 'OUTPUT_PATH'), 'CNN')
    logger = plots.create_logger(save_dir)
    writer = SummaryWriter(log_dir=save_dir)

    # Create model
    net = utils.create_net(config, logger).to(device)
    summary(net, input_size=(2, 80, 80, 80), batch_size=P['batch_size'])

    net = torch.nn.DataParallel(net, device_ids=np.arange(P['gpus']).tolist())
    opt = Adam(net.parameters(), lr=P['lr'], weight_decay=P['weight_decay'], betas=(P['beta1'], P['beta2']))
    schedule_lr = StepLR(opt, step_size=P['step_size'], gamma=P['gamma'])

    utils.save_config(config, save_dir, 'config.ini')
    utils.save_model(net, save_dir, 'net.txt')
    train_ds.save_patients(save_dir, 'train.xlsx')
    val_ds.save_patients(save_dir, 'val.xlsx')

    # Steps per epoch for training and evaluation set
    train_steps = len(train_ds) / P['batch_size']
    val_steps = len(val_ds) / P['batch_size']

    # History training info
    H = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    logger.info('Save directory: ' + save_dir)
    logger.info('Trainig CNN')

    pbar = tqdm(total=P['epochs'])
    trainer = Trainer(net, pbar, config, device)
    best_val_loss = 1e10
    best_train_loss = 1e10
    tic = time.time()

    for e in range(P['epochs']):
        train_res = trainer.train_epoch(train_loader, opt)
        val_res = trainer.val_epoch(val_loader)

        train_avg = tuple(x / train_steps for x in train_res)
        val_avg = tuple(x / val_steps for x in val_res)

        utils.log(logger, writer, e, train_avg, val_avg, net, opt, schedule_lr, best_train_loss, best_val_loss, save_dir)
        H = utils.update_train_history(H, train_avg, val_avg)

        schedule_lr.step()
        pbar.update(1)

    toc = time.time()
    logger.info('\nTotal time taken to train the model: {:.3f}s'.format(toc - tic))

    plots.save_loss(H, save_dir)
    plots.save_acc(H, save_dir)
    utils.checkpoint(e, net, opt, train_avg[0], train_avg[4], save_dir, 'weights.pth')
    # torch.save(net, osp.join(save_dir, 'model.pth'))

    pbar.close()
    writer.close()
