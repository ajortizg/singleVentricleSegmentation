import os.path as osp
import configparser
import sys
import os

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from monai.losses import DiceCELoss
import numpy as np


ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../../'))
sys.path.append(ROOT_DIR)
import utilities.path_utils as path_utils
from utilities import stuff
from datasets.segmentation_dataset import SegmentationDataset
import segmentation.utils as utils
import segmentation.transforms as T
from segmentation.tridimensional.models.factory import Factory
from utilities.parser_conversions import str_to_tuple
from segmentation.tridimensional.trainer_3d import Trainer3d


if __name__ == '__main__':
    stuff.seeding(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Read configuration
    config = configparser.ConfigParser()
    config.read('parser/seg3d_train.ini')
    data = config['DATA']
    params = config['PARAMETERS']

    # Fetch data
    DA = config['DATA_AUGMENTATION']
    img_sz = params.getint('img_size')
    norm = utils.norm_fn(params['norm'])
    img_key, label_key = 'image', 'label'
    keys = [img_key, label_key]

    train_transforms = T.Compose([
        T.AddNLeadingDims(n=2, keys=keys),
        T.BCXYZ_To_BCZYX(keys=keys),
        T.ToRAS(keys=keys),
        T.CropForeground(keys=keys, label_key=label_key),
        norm,
        T.Resize(1.0, (img_sz, img_sz, img_sz), keys=keys, label_key=label_key),
        T.RandomRotate(DA.getfloat('rot_prob'), str_to_tuple(DA['rot_x_range'], float), str_to_tuple(DA['rot_y_range'], float), str_to_tuple(DA['rot_z_range'], float),
                       DA['rot_boundary'], keys=keys, label_key=label_key),
        T.RandomScale(DA.getfloat('scaling_prob'), str_to_tuple(DA['scaling_range'], float),
                      DA['scaling_boundary'], keys=keys, label_key=label_key),
        T.RandomFlip(DA.getfloat('depth_flip_prob'), 2, keys=keys),
        T.RandomFlip(DA.getfloat('vertical_flip_prob'), 3, keys=keys),
        T.RandomFlip(DA.getfloat('horizontal_flip_prob'), 4, keys=keys),
        T.ElasticDeformation(DA.getfloat('ed_prob'), str_to_tuple(DA['ed_sigma_range'], float), DA.getint('ed_grid'), DA['ed_boundary'], DA['ed_axis'], keys=keys, label_key=label_key),
        T.SimulateLowResolution(DA.getfloat('lowres_prob'), str_to_tuple(DA['lowres_zoom_range'], float), keys=keys, label_key=label_key),
        T.AdditiveGaussianNoise(DA.getfloat('noise_prob'), str_to_tuple(DA['noise_std_range'], float), DA.getfloat('noise_mu'), keys=[img_key]),
        T.GaussialBlur(DA.getfloat('blur_prob'), str_to_tuple(DA['blur_sigma_range'], float), keys=[img_key]),
        T.MultiplicativeScaling(DA.getfloat('mult_scaling_prob'), str_to_tuple(DA['mult_scaling_range']), keys=[img_key]),
        T.ContrastAugmentation(DA.getfloat('contrast_prob'), str_to_tuple(DA['contrast_range']), DA.getboolean('contrast_preserve_range'), keys=[img_key]),
        T.GammaCorrection(DA.getfloat('gamma_scaling_prob'), str_to_tuple(DA['gamma_scaling_range'], float), invert_image=False, retain_stats=DA.getboolean('gamma_retain_stats'), keys=[img_key]),
        T.GammaCorrection(DA.getfloat('gamma_scaling_prob') / 2.0, str_to_tuple(DA['gamma_scaling_range'], float), invert_image=True,
                          retain_stats=DA.getboolean('gamma_retain_stats'), keys=[img_key]),
        T.RemoveNLeadingDims(n=1, keys=keys),
        T.ToTensor(keys=keys)
    ])
    val_transforms = T.Compose([
        T.AddNLeadingDims(n=2, keys=keys),
        T.BCXYZ_To_BCZYX(keys=keys),
        T.ToRAS(keys=keys),
        T.CropForeground(keys=keys, label_key=label_key),
        norm,
        T.Resize(1.0, (img_sz, img_sz, img_sz), keys=keys),
        T.RemoveNLeadingDims(n=1, keys=keys),
        T.ToTensor(keys=keys)
    ])
    test_transforms = T.Compose([
        T.XYZT_To_TZYX(keys=keys),
        T.AddDimAt(axis=1, keys=keys),
        T.ToRAS(keys=keys),
        T.CropForeground(keys=keys, label_key=label_key),
        norm,
        T.Resize(1.0, (img_sz, img_sz, img_sz), keys=keys),
        T.ToTensor(keys=keys)
    ])

    fold, n = utils.get_fold(data)
    train_ds = SegmentationDataset(data['root_dir'], 'train', train_transforms, fold['train'])
    val_ds = SegmentationDataset(data['root_dir'], 'train', val_transforms, fold['val'])
    test_ds = SegmentationDataset(data['root_dir'], 'test', test_transforms)
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
    loss_fn = loss_fn = DiceCELoss(softmax=True, to_onehot_y=True, lambda_dice=1.0, lambda_ce=1.0)
    optimizer = optim.Adam(net.parameters(), lr=params.getfloat('lr'), weight_decay=params.getfloat('weight_decay'))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=params.getint('step_size'), gamma=params.getfloat('gamma'))

    # Training loop
    trainer = Trainer3d(params.getint('num_classes'), net, loss_fn, optimizer, device, logger)
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