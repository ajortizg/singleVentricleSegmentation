import torch
import configparser
import sys
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
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
    # utils.seeding(42)

    config = configparser.ConfigParser()
    config.read('parser/configCNNTrain.ini')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    P = utils.read_train_params(config)

    # Create train and validation datasets
    train_transforms = T.ComposeFull([
        T.RandomRotate(P['rot_prob'], P['rot_range_x'], P['rot_range_y'], P['rot_range_z'], boundary=P['rot_boundary'], clip_interval=P['clip_interval']),
        T.ElasticDeformation(P['ed_prob'], P['ed_sigma_range'], P['ed_grid'], P['ed_boundary'], P['ed_prefilter'], P['ed_axis'], P['clip_interval']),
        T.RandomVerticalFlip(P['vflip_prob']),
        T.RandomHorizontalFlip(P['hflip_prob']),
        T.RandomDepthFlip(P['dflip_prob']),
        T.GammaScaling(P['gamma_scaling_prob'], P['gamma_scaling_range']),
        T.MutiplicativeScaling(P['mult_scaling_prob'], P['gamma_scaling_range'], P['clip_interval']),
        T.AdditiveScaling(P['add_scaling_prob'], P['add_scaling_mean'], P['add_scaling_std'], P['clip_interval']),
        T.AdditiveGaussianNoise(P['noise_prob'], P['noise_mu'], P['noise_std'], P['clip_interval']),
        T.BinarizeMasks(th=0.5),
        T.ToTensorFull()
    ])
    val_transforms = T.ComposeFull([T.BinarizeMasks(th=0.5),
                                    T.ToTensorFull()])

    train_ds = ds.SingleVentricleDataset(config, ds.DatasetMode.TRAIN, ds.LoadFlowMode.TRAIN_VAL_OF,
                                         full_transforms=train_transforms)
    val_ds = ds.SingleVentricleDataset(config, ds.DatasetMode.VAL, ds.LoadFlowMode.TRAIN_VAL_OF,
                                       full_transforms=val_transforms)

    # Create data loaders
    train_loader = DataLoader(train_ds, batch_size=P['batch_size'], shuffle=True, num_workers=P['workers'], collate_fn=utils.collate_fn)
    val_loader = DataLoader(val_ds, batch_size=P['batch_size'], shuffle=False, num_workers=P['workers'], collate_fn=utils.collate_fn)

    save_dir = plots.createSaveDirectory(config.get('DATA', 'OUTPUT_PATH'), 'CNN')
    logger = plots.create_logger(save_dir)
    writer = SummaryWriter(log_dir=save_dir)
    logger.info(f'Using device {device}')

    # Create model
    net = utils.create_net(config, logger).to(device)
    summary(net, input_size=(2, 80, 80, 80), batch_size=P['batch_size'])
    net = torch.nn.DataParallel(net, device_ids=np.arange(P['gpus']).tolist())
    if P['pretrained']:
        checkpoint_file = P['checkpoint_file']
        checkpoint = torch.load(checkpoint_file)
        net.load_state_dict(checkpoint['model_state_dict'], strict=True)
        logger.info(f'Use pretrained weights: {checkpoint_file}')

    opt = optim.Adam(net.parameters(), lr=P['lr'], weight_decay=P['weight_decay'], betas=(P['beta1'], P['beta2']))
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=P['step_size'], gamma=P['gamma'])

    utils.save_config(config, save_dir, 'config.ini')
    utils.save_model(net, save_dir, 'net.txt')
    train_ds.save_patients(save_dir, 'train.xlsx')
    val_ds.save_patients(save_dir, 'val.xlsx')

    # Steps per epoch for training and evaluation set
    train_steps = len(train_loader)
    val_steps = len(val_loader)

    # History training info
    H = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    logger.info('Save directory: ' + save_dir)
    logger.info('Trainig CNN')

    pbar = tqdm(total=P['epochs'])
    trainer = Trainer(net, pbar, config, device, writer)
    best_val_loss = 1e10
    best_train_loss = 1e10
    tic = time.time()

    for e in range(P['epochs']):
        train_res = trainer.train_epoch(train_loader, opt)
        val_res = trainer.val_epoch(val_loader)

        train_avg = tuple(x / train_steps for x in train_res)
        val_avg = tuple(x / val_steps for x in val_res)

        best_train_loss, best_val_loss = utils.log(logger, writer, e, train_avg, val_avg, net, opt,
                                                   scheduler, best_train_loss, best_val_loss, save_dir)
        H = utils.update_train_history(H, train_avg, val_avg)

        scheduler.step()
        pbar.update(1)

    toc = time.time()
    logger.info('\nTotal time taken to train the model: {:.3f}s'.format(toc - tic))

    plots.save_loss(H, save_dir)
    plots.save_acc(H, save_dir)
    utils.checkpoint(e, net, opt, train_avg, val_avg, save_dir, 'checkpoint.pth')
    # torch.save(net, osp.join(save_dir, 'model.pth'))

    pbar.close()
    writer.close()
