import os.path as osp
import os
import sys
from configparser import ConfigParser

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
from utilities import stuff
from utilities import path_utils
from utilities.parser_conversions import str_to_tuple
import segmentation.transforms as T
from datasets.flow_unet_dataset import FlowUNetDataset
from ofunet import utils
from ofunet.models.factory import Factory
from ofunet.trainer import Trainer

if __name__ == '__main__':
    stuff.seeding(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Read configuration
    config = ConfigParser()
    config.read('parser/ofunet_train.ini')
    data = config['DATA']
    params = config['PARAMETERS']

    # Fetch data
    DA = config['DATA_AUGMENTATION']
    img_key, label_key, ff_key, bf_key = 'image', 'label', 'forward_flow', 'backward_flow'
    keys = [img_key, label_key, ff_key, bf_key]

    train_transforms = T.Compose([
        T.XYZT_To_TZYX([img_key, label_key]),
        T.AddDimAt(1, [img_key, label_key]),
        T.Flow_T3XYZ_To_T3ZYX([ff_key, bf_key]),
        T.RandomFlip(DA.getfloat('depth_flip_prob'), 2, keys),
        T.RandomFlip(DA.getfloat('vertical_flip_prob'), 3, keys),
        T.RandomFlip(DA.getfloat('horizontal_flip_prob'), 4, keys),
        T.RandomRotate(DA.getfloat('rot_prob'), str_to_tuple(DA['rot_x_range'], float), str_to_tuple(DA['rot_y_range'], float),
                       str_to_tuple(DA['rot_z_range'], float), DA['rot_boundary'], keys, label_key),
        T.RandomScale(DA.getfloat('scaling_prob'), str_to_tuple(DA['scaling_range'], float), DA['scaling_boundary'], keys, label_key),
        T.ElasticDeformation(DA.getfloat('ed_prob'), str_to_tuple(DA['ed_sigma_range'], float), DA.getint('ed_grid'), DA['ed_boundary'],
                             DA['ed_axis'], keys, label_key),
        T.SimulateLowResolution(DA.getfloat('lowres_prob'), str_to_tuple(DA['lowres_zoom_range'], float), keys, label_key),
        T.GammaCorrection(DA.getfloat('gamma_scaling_prob'), str_to_tuple(DA['gamma_scaling_range'], float), invert_image=False, retain_stats=DA.getboolean('gamma_retain_stats'), keys=[img_key]),
        T.GammaCorrection(DA.getfloat('gamma_scaling_prob') / 2.0, str_to_tuple(DA['gamma_scaling_range'], float), invert_image=True,
                          retain_stats=DA.getboolean('gamma_retain_stats'), keys=[img_key]),
        T.MultiplicativeScaling(DA.getfloat('mult_scaling_prob'), str_to_tuple(DA['mult_scaling_range']), [img_key]),
        T.ContrastAugmentation(DA.getfloat('contrast_prob'), str_to_tuple(DA['contrast_range']), DA.getboolean('contrast_preserve_range'), [img_key]),
        T.AdditiveGaussianNoise(DA.getfloat('noise_prob'), str_to_tuple(DA['noise_std_range'], float), DA.getfloat('noise_mu'), [img_key]),
        T.GaussialBlur(DA.getfloat('blur_prob'), str_to_tuple(DA['blur_sigma_range'], float), [img_key]),
        T.FlowChannelToLastDim([ff_key, bf_key]),
        T.OneHotEncoding(params.getint('num_classes'), [label_key]),
        T.ToTensor(keys)
    ])
    val_transforms = T.Compose([
        T.XYZT_To_TZYX([img_key, label_key]),
        T.AddDimAt(1, [img_key, label_key]),
        T.Flow_T3XYZ_To_T3ZYX([ff_key, bf_key]),
        T.FlowChannelToLastDim([ff_key, bf_key]),
        T.OneHotEncoding(params.getint('num_classes'), [label_key]),
        T.ToTensor(keys)
    ])

    fold, n = utils.get_fold(data)
    train_ds = FlowUNetDataset(data['root_dir'], 'train', train_transforms, True, fold['train'])
    val_ds = FlowUNetDataset(data['root_dir'], 'train', val_transforms, True, fold['val'])
    test_ds = FlowUNetDataset(data['root_dir'], 'test', val_transforms, True)
    train_loader = DataLoader(train_ds, batch_size=params.getint('batch_size'), shuffle=True, num_workers=params.getint('num_workers'))
    val_loader = DataLoader(val_ds, batch_size=params.getint('batch_size'), shuffle=False, num_workers=params.getint('num_workers'))
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)

    # Debug dirs
    save_dir = path_utils.create_save_dir(data['output_dir'], f'fold_{n}')
    stuff.save_config(config, save_dir)
    writer = SummaryWriter(log_dir=save_dir)
    logger = stuff.create_logger(save_dir)
    stuff.save_transforms_to_json(train_loader.dataset.transforms, osp.join(save_dir, 'train_transforms.json'))
    stuff.save_json([fold], osp.join(save_dir, 'fold.json'), default=int)
    logger.info('Save dir: {}'.format(save_dir))
    logger.info('Device: {}'.format(device))
    logger.info('Fold: {}'.format(n))

    # Create model
    net = Factory.create(config).to(device)
    gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    net = torch.nn.DataParallel(net, device_ids=np.arange(gpus).tolist())
    stuff.save_model(net, save_dir, 'net.txt')

    # Optimization
    loss_fn = nn.MSELoss(reduction=params['loss_reduction'])
    optimizer = optim.Adam(net.parameters(), lr=params.getfloat('lr'), weight_decay=params.getfloat('weight_decay'))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=params.getint('step_size'), gamma=params.getfloat('gamma'))

    # Training loop
    trainer = Trainer(config, net, loss_fn, optimizer, device, logger)
    trainer.training_loop(params.getint('num_epochs'),
                          params.getint('patience'),
                          scheduler,
                          train_loader,
                          val_loader,
                          test_loader,
                          writer,
                          save_dir)

    trainer.plot_loss_history(osp.join(save_dir, 'loss.png'))
    trainer.plot_accuracy_history(osp.join(save_dir, 'dice.png'))
    writer.close()
