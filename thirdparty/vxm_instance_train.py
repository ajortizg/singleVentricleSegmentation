import os
import os.path as osp
import sys
import configparser
from terminaltables import AsciiTable
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

# import voxelmorph with pytorch backend
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph.voxelmorph as vxm  # nopep8

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
from utilities import path_utils
from utilities import stuff
from datasets.flowunet_dataset import FlowUNetDataset
import segmentation.transforms as T
from thirdparty.vxm_train import test, create_checkpoint, train


if __name__ == '__main__':
    # Seeding for reproducible results
    stuff.seeding(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Read configuration options
    config = configparser.ConfigParser()
    config.read('parser/vxm_instance_train.ini')
    params = config['PARAMETERS']
    data = config['DATA']

    cudnn_nondet = params.getboolean('cudnn_nondet')
    model_weights = data['model']

    # Enabling cudnn determinism appears to speed up training by a lot
    torch.backends.cudnn.deterministic = not cudnn_nondet

    save_dir = path_utils.create_save_dir(data['output_dir'], data['patient'])
    stuff.save_config(config, save_dir)
    writer = SummaryWriter(log_dir=save_dir)
    logger = stuff.create_logger(save_dir)
    logger.info(f'Device: {device}')
    logger.info(f'Save dir: {save_dir}')
    logger.info(f'cudnn.deterministic: {(not cudnn_nondet)}')
    logger.info(f'Weights: {model_weights}')

    # Create dataloaders
    transforms = T.Compose([
        T.XYZT_To_TZYX(keys=['image', 'label']),
        T.AddDimAt(axis=1, keys=['image', 'label']),
        T.OneHotEncoding(config.getint('DATA', 'num_classes'), keys=['label']),
        T.ToTensor(keys=['image', 'label'])
    ])

    train_ds = FlowUNetDataset(data['root_dir'], 'test', transforms)
    train_ds.filter_patient(data['patient'])
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False, num_workers=params.getint('workers'))

    # load and set up model
    model = vxm.networks.VxmDense.load(model_weights, device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.getfloat('lr'))

    if params['img_loss'] == 'ncc':
        image_loss_func = vxm.losses.NCC().loss
    elif params['img_loss'] == 'mse':
        image_loss_func = vxm.losses.MSE().loss
    else:
        raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % params['img_loss'])

    losses = [image_loss_func, vxm.losses.Grad('l2', loss_mult=2).loss]
    weights = [1.0, params.getfloat('lambda')]
    H = {'train_loss': [], 'train_dice': [], 'val_loss': [], 'val_dice': [], 'test_dice': []}
    epochs_since_last_improvement = 0
    best_dice = 0.0
    patience = params.getint('patience')
    tic = time.time()

    for e in tqdm(range(1, params.getint('num_epochs') + 1)):
        epoch_tic = time.time()
        report = train(train_loader, model, optimizer, losses, weights, device, compute_hd=False, test=True)
        H['train_loss'].append(report['Loss'].mean())
        H['train_dice'].append(report['Dice'].mean())

        report = test(train_loader, model, device, logger, True)
        H['test_dice'].append(report['Dice'].mean())

        H['val_loss'].append(0)
        H['val_dice'].append(0)

        logger.info(AsciiTable([
            ['Split', 'Loss', 'Dice'],
            ['Train', '{:.6f}'.format(H['train_loss'][-1]), '{:.3f}'.format(H['train_dice'][-1])],
            ['Test', '-', '{:.3f}'.format(H['test_dice'][-1])],
            ['Epoch', e, epochs_since_last_improvement]
        ]).table)

        if H['train_dice'][-1] > best_dice:
            best_dice = H['train_dice'][-1]
            epochs_since_last_improvement = 0
            create_checkpoint(model, optimizer, H, e, osp.join(save_dir, 'checkpoint_best.pth'))
            model.save(osp.join(save_dir, 'model_best.pt'))
            logger.info(f'Checkpoint updated with dice: {best_dice:,.3f}')

        else:
            epochs_since_last_improvement += 1

        writer.add_scalars('loss', {'train': H['train_loss'][-1], 'val': H['val_loss'][-1]}, e)
        writer.add_scalars('dice', {'train': H['train_dice'][-1], 'val': H['val_dice'][-1], 'test': H['test_dice'][-1]}, e)
        writer.add_scalar('epoch_time', time.time() - epoch_tic, e)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], e)

        # early stop
        if epochs_since_last_improvement > patience:
            logger.info(f'Early stop at epoch: {e}')
            break

    create_checkpoint(model, optimizer, H, e, osp.join(save_dir, 'checkpoint_final.pth'))
    model.save(osp.join(save_dir, 'model_final.pt'))
    logger.info('\nTraining time: {:.3f} hrs.'.format((time.time() - tic) / 3600.0))

    # Plot loss history
    plt.figure()
    plt.plot(H['train_loss'], label='train')
    plt.plot(H['val_loss'], label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='lower left')
    plt.savefig(osp.join(save_dir, 'loss.png'))

    # Plot dice history
    plt.figure()
    plt.plot(H['train_dice'], label='train')
    plt.plot(H['val_dice'], label='val')
    plt.plot(H['test_dice'], label='test')
    plt.xlabel('Epoch')
    plt.ylabel('Dice')
    plt.legend(loc='lower right')
    plt.savefig(osp.join(save_dir, 'dice.png'))

    writer.close()
