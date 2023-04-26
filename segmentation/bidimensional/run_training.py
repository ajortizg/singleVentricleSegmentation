import os.path as osp
import sys
import os
from configparser import ConfigParser

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from monai.losses.dice import DiceCELoss
from torch.utils.tensorboard import SummaryWriter
import numpy as np

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../../'))
sys.path.append(ROOT_DIR)
from datasets.segmentation_dataset import SegmentationDataset
import segmentation.transforms as T
from utilities import stuff
import utilities.path_utils as path_utils
from utilities.parser_conversions import str_to_tuple
import segmentation.utils as utils
from segmentation.bidimensional.models.factory import Factory
from segmentation.bidimensional.fct_trainer import FctTrainer
from segmentation.bidimensional.transunet_trainer import TransUNetTrainer


if __name__ == '__main__':
    stuff.seeding(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Read configuration
    config = ConfigParser()
    config.read('parser/seg2d_train.ini')
    data = config['DATA']
    params = config['PARAMETERS']

    # Fetch data
    DA = config['DATA_AUGMENTATION']
    img_key, label_key = 'image', 'label'
    keys = [img_key, label_key]

    train_transforms = T.Compose([
        T.AddNLeadingDims(n=3, keys=keys),
        T.RandomFlip(DA.getfloat('vertical_flip_prob'), axis=3, keys=keys),
        T.RandomFlip(DA.getfloat('horizontal_flip_prob'), axis=4, keys=keys),
        T.RandomRotate2D(DA.getfloat('rot_prob'), str_to_tuple(DA['rot_range'], float), keys=keys, label_key=label_key),
        T.ElasticDeformation(DA.getfloat('ed_prob'), str_to_tuple(DA['ed_sigma_range'], float), DA.getint('ed_grid'), DA['ed_boundary'], DA['ed_axis'], keys=keys, label_key=label_key),
        T.SimulateLowResolution(DA.getfloat('lowres_prob'), str_to_tuple(DA['lowres_zoom_range'], float), keys=keys, label_key=label_key),
        T.GammaCorrection(DA.getfloat('gamma_scaling_prob'), str_to_tuple(DA['gamma_scaling_range'], float), retain_stats=DA.getboolean('gamma_retain_stats'), invert_image=False, keys=[img_key]),
        T.GammaCorrection(DA.getfloat('gamma_scaling_prob') / 2, str_to_tuple(DA['gamma_scaling_range'], float), retain_stats=DA.getboolean('gamma_retain_stats'), invert_image=True, keys=[img_key]),
        T.MultiplicativeScaling(DA.getfloat('mult_scaling_prob'), str_to_tuple(DA['mult_scaling_range'], float), keys=[img_key]),
        T.AdditiveGaussianNoise(DA.getfloat('noise_prob'), str_to_tuple(DA['noise_std_range'], float), DA.getfloat('noise_mu'), keys=[img_key]),
        T.ContrastAugmentation(DA.getfloat('contrast_prob'), str_to_tuple(DA['contrast_range'], float), DA.getboolean('contrast_preserve_range'), keys=[img_key]),
        T.GaussialBlur(DA.getfloat('blur_prob'), str_to_tuple(DA['blur_sigma_range'], float), keys=[img_key]),
        T.RemoveNLeadingDims(n=2, keys=keys),
        T.ToTensor(keys=keys)
    ])
    val_transforms = T.Compose([
        T.AddNLeadingDims(n=1, keys=keys),
        T.ToTensor(keys=keys)
    ])

    fold, n = utils.get_fold(data)
    train_ds = SegmentationDataset(data['root_dir'], mode='train', transforms=train_transforms, fold_idxs=fold['train'])
    val_ds = SegmentationDataset(data['root_dir'], mode='train', transforms=val_transforms, fold_idxs=fold['val'])
    test_ds = SegmentationDataset(data['root_dir'], mode='test', transforms=val_transforms)
    train_loader = DataLoader(train_ds, batch_size=params.getint('batch_size'), shuffle=True, num_workers=params.getint('num_workers'))
    val_loader = DataLoader(val_ds, batch_size=params.getint('batch_size'), shuffle=False, num_workers=params.getint('num_workers'))
    test_loader = DataLoader(test_ds, batch_size=params.getint('batch_size'), shuffle=False, num_workers=params.getint('num_workers'))

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
    loss_fn = DiceCELoss(softmax=True, to_onehot_y=True, lambda_ce=1.0, lambda_dice=1.0)
    optimizer = optim.Adam(net.parameters(), lr=params.getfloat('lr'), weight_decay=params.getfloat('weight_decay'))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=params.getint('step_size'), gamma=params.getfloat('gamma'))

    # Create trainer
    if params['net'] == 'FCT':
        trainer = FctTrainer(params.getint('img_size'),
                             params.getint('num_classes'),
                             net,
                             loss_fn,
                             optimizer,
                             device,
                             logger)
    elif params['net'] == 'TransUNet':
        trainer = TransUNetTrainer(params.getint('num_classes'),
                                   net,
                                   loss_fn,
                                   optimizer,
                                   device,
                                   logger)
    else:
        raise ValueError('{} not supported!'.format(params['net']))

    trainer.training_loop(
        params.getint('num_epochs'),
        params.getint('patience'),
        scheduler,
        train_loader,
        val_loader,
        test_loader,
        writer,
        save_dir
    )
    trainer.plot_loss_history(osp.join(save_dir, 'loss.png'))
    trainer.plot_accuracy_history(osp.join(save_dir, 'dice.png'))
    writer.close()

    # H = {'train_loss': [], 'train_dice': [], 'val_loss': [], 'val_dice': [], 'test_dice': []}
    # epochs_since_last_improvement = 0
    # best_dice = 0.0

    # # trainer = FCTTrainer(img_size, num_classes, model, loss_fn, optimizer, device)
    # trainer = TransUNetTrainer(num_classes, model, loss_fn, optimizer, device)

    # tic = time.time()
    # for e in tqdm(range(1, epochs + 1)):
    #     epoch_tic = time.time()
    #     report = trainer.train(train_loader)
    #     H['train_loss'].append(report['Loss'].mean())
    #     H['train_dice'].append(report['Dice'].mean())

    #     report = trainer.validate(val_loader)
    #     H['val_loss'].append(report['Loss'].mean())
    #     H['val_dice'].append(report['Dice'].mean())

    #     if do_test:
    #         report = trainer.test(test_loader)
    #         H['test_dice'].append(report['Dice'].mean())
    #     else:
    #         H['test_dice'].append(0)

    #     writer.add_scalar('lr', optimizer.param_groups[0]['lr'], e)
    #     scheduler.step(H['val_loss'][-1])

    #     logger.info(AsciiTable([
    #         ['Split', 'Loss', 'Dice'],
    #         ['Train', '{:.3f}'.format(H['train_loss'][-1]), '{:.3f}'.format(H['train_dice'][-1])],
    #         ['Val', '{:.3f}'.format(H['val_loss'][-1]), '{:.3f}'.format(H['val_dice'][-1])],
    #         ['Test', '-', '{:.3f}'.format(H['test_dice'][-1])],
    #         ['Epoch', e, epochs_since_last_improvement]
    #     ]).table)

    #     if H['val_dice'][-1] > best_dice:
    #         best_dice = H['val_dice'][-1]
    #         epochs_since_last_improvement = 0
    #         utils.create_checkpoint(model, e, optimizer, H, osp.join(save_dir, 'checkpoint_best.pth'))
    #         logger.info(f'Checkpoint updated with dice: {best_dice:,.3f}')
    #     else:
    #         epochs_since_last_improvement += 1

    #     writer.add_scalars('loss', {'train': H['train_loss'][-1], 'val': H['val_loss'][-1]}, e)
    #     writer.add_scalars('dice', {'train': H['train_dice'][-1], 'val': H['val_dice'][-1], 'test': H['test_dice'][-1]}, e)
    #     writer.add_scalar('epoch_time', time.time() - epoch_tic, e)

    #     # early stop
    #     if epochs_since_last_improvement > patience:
    #         logger.info(f'Early stop at epoch: {e}')
    #         break

    # utils.create_checkpoint(model, e, optimizer, H, osp.join(save_dir, 'checkpoint_final.pth'))
    # logger.info('\nTraining time: {:.3f} hrs.'.format((time.time() - tic) / 3600.0))
