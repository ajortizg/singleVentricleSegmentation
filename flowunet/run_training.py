import sys
import os.path as osp
import os
from configparser import ConfigParser

import torch
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.tensorboard import SummaryWriter

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
from utilities import path_utils
from utilities import stuff
from utilities.parser_conversions import str_to_tuple
from datasets.flowunet_dataset import FlowUNetDataset
import segmentation.transforms as T
from flowunet.trainer import FlowUNetTrainer
from flowunet.models.factory import Factory
from flowunet import utils
from flowunet.loss import CustomLoss


if __name__ == '__main__':
    stuff.seeding(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = ConfigParser()
    config.read('parser/flowunet_train.ini')
    data = config['DATA']
    params = config['PARAMETERS']
    aug = config['DATA_AUGMENTATION']

    # Create datasets
    fold, n, n_classes = utils.get_fold(data)
    img_key, label_key, ff_key, bf_key = 'image', 'label', 'forward_flow', 'backward_flow'
    keys = [img_key, label_key, ff_key, bf_key]

    train_transforms = T.Compose([
        T.XYZT_To_TZYX([img_key, label_key]),
        T.AddDimAt(1, [img_key, label_key]),
        T.Flow_T3XYZ_To_T3ZYX([ff_key, bf_key]),
        T.RandomFlip(aug.getfloat('depth_flip_prob'), 2, keys),
        T.RandomFlip(aug.getfloat('vertical_flip_prob'), 3, keys),
        T.RandomFlip(aug.getfloat('horizontal_flip_prob'), 4, keys),
        T.RandomRotate(aug.getfloat('rot_prob'), str_to_tuple(aug['rot_x_range'], float), str_to_tuple(aug['rot_y_range'], float),
                       str_to_tuple(aug['rot_z_range'], float), aug['rot_boundary'], keys, label_key),
        T.RandomScale(aug.getfloat('scaling_prob'), str_to_tuple(aug['scaling_range'], float), aug['scaling_boundary'], keys, label_key),
        T.SimulateLowResolution(aug.getfloat('lowres_prob'), str_to_tuple(aug['lowres_zoom_range'], float), keys, label_key),
        T.ElasticDeformation(aug.getfloat('ed_prob'), str_to_tuple(aug['ed_sigma_range'], float), aug.getint('ed_grid'), aug['ed_boundary'],
                             aug['ed_axis'], keys, label_key),
        T.GammaCorrection(aug.getfloat('gamma_scaling_prob'), str_to_tuple(aug['gamma_scaling_range'], float), invert_image=False,
                          retain_stats=aug.getboolean('gamma_retain_stats'), keys=[img_key]),
        T.MultiplicativeScaling(aug.getfloat('mult_scaling_prob'), str_to_tuple(aug['mult_scaling_range'], float), [img_key]),
        T.ContrastAugmentation(aug.getfloat('contrast_prob'), str_to_tuple(aug['contrast_range'], float),
                               aug.getboolean('contrast_preserve_range'), [img_key]),
        T.GaussialBlur(aug.getfloat('blur_prob'), str_to_tuple(aug['blur_sigma_range'], float), [img_key]),
        T.AdditiveGaussianNoise(aug.getfloat('noise_prob'), str_to_tuple(aug['noise_std_range'], float), aug.getfloat('noise_mu'), [img_key]),
        T.FlowChannelToLastDim(keys=[ff_key, bf_key]),
        T.OneHotEncoding(n_classes + 1, [label_key]) if n_classes > 1 else T.DoNothing(),
        T.ExtremaPoints([label_key]),
        T.ToTensor(keys + ['mi', 'mf'])
    ])

    val_test_transforms = T.Compose([
        T.XYZT_To_TZYX([img_key, label_key]),
        T.AddDimAt(1, [img_key, label_key]),
        T.Flow_T3XYZ_To_T3ZYX([ff_key, bf_key]),
        T.FlowChannelToLastDim(keys=[ff_key, bf_key]),
        T.OneHotEncoding(n_classes + 1, [label_key]) if n_classes > 1 else T.DoNothing(),
        T.ExtremaPoints([label_key]),
        T.ToTensor(keys + ['mi', 'mf'])
    ])

    train_ds = FlowUNetDataset(data['root_dir'], 'train', train_transforms, load_flow=True, fold_indices=fold['train'])
    val_ds = FlowUNetDataset(data['root_dir'], 'train', val_test_transforms, load_flow=True, fold_indices=fold['val'])
    test_ds = FlowUNetDataset(data['root_dir'], 'test', val_test_transforms, load_flow=True)
    train_loader = DataLoader(train_ds, batch_size=params.getint('batch_size'), shuffle=True, num_workers=params.getint('num_workers'),
                              collate_fn=FlowUNetDataset.collate)
    val_loader = DataLoader(val_ds, batch_size=params.getint('batch_size'), shuffle=False, num_workers=params.getint('num_workers'),
                            collate_fn=FlowUNetDataset.collate)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=3, collate_fn=FlowUNetDataset.collate)

    # Debug dirs
    save_dir = path_utils.create_save_dir(data['output_dir'], f'fold_{n}')
    stuff.save_config(config, save_dir)
    stuff.save_transforms_to_json(train_transforms, osp.join(save_dir, 'train_transforms.json'))
    writer = SummaryWriter(log_dir=save_dir)
    logger = stuff.create_logger(save_dir)
    stuff.save_transforms_to_json(train_loader.dataset.transforms, osp.join(save_dir, 'train_transforms.json'))
    stuff.save_json([fold], osp.join(save_dir, 'fold.json'), default=int)
    logger.info('Save dir: {}'.format(save_dir))
    logger.info('Device: {}'.format(device))
    logger.info('Fold: {}'.format(n))

    # Create model
    net = Factory.create(n_classes, config).to(device)
    gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    assert gpus == params.getint('batch_size'), 'Number of gpus and batch size must be equal!'
    net = torch.nn.DataParallel(net, device_ids=np.arange(gpus).tolist())
    stuff.save_model(net,save_dir, 'net.txt')
    
    # Optimization
    loss_fn = CustomLoss(params.getfloat('loss_lambda'), params.getfloat('loss_penalization_gamma'))
    optimizer = optim.Adam(net.parameters(), lr=params.getfloat('lr'), weight_decay=params.getfloat('weight_decay'))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=params.getint('step_size'), gamma=params.getfloat('gamma'))
    trainer = FlowUNetTrainer(config, n_classes, net, loss_fn, optimizer, device, logger)

    trainer.training_loop(params.getint('num_epochs'),
                          scheduler,
                          train_loader,
                          val_loader,
                          test_loader,
                          writer,
                          save_dir,
                          params.getint('patience'),
                          metric_early_stopping='val_dice',
                          initial_best_metric_value=0,
                          comparisson_fn=lambda metric, best_metric: metric > best_metric)

    trainer.plot_loss_history(osp.join(save_dir, 'loss.png'))
    trainer.plot_accuracy_history(osp.join(save_dir, 'dice.png'))
    writer.close()
