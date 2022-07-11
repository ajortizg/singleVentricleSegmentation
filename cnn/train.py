import torch
import configparser
import sys
from torch.optim import Adam
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from unet_3d import UNet3D
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np
import os.path as osp
from torchsummary import summary
import cnn_utils
import os
# import torch.multiprocessing
import logging
from cnn_utils import seeding

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
from utils import plots
import utils.transforms as T
import cnn.dataset as ds
from cnn.trainer import Trainer


if __name__ == "__main__":
    # torch.multiprocessing.set_sharing_strategy('file_system')
    seeding(42)

    config = configparser.ConfigParser()
    config.read('parser/configCNNTrain.ini')

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = config.getint('PARAMETERS', 'BATCH_SIZE')
    LR = config.getfloat('PARAMETERS', 'LR')
    STEP_SIZE = config.getfloat('PARAMETERS', 'STEP_SIZE')
    GAMMA = config.getfloat('PARAMETERS', 'GAMMA')
    LR = config.getfloat('PARAMETERS', 'LR')
    WEIGHT_DECAY = config.getfloat('PARAMETERS', 'WEIGHT_DECAY')
    BETA1 = config.getfloat('PARAMETERS', 'BETA1')
    BETA2 = config.getfloat('PARAMETERS', 'BETA2')
    NUM_EPOCHS = config.getint('PARAMETERS', 'NUM_EPOCHS')
    SHUFFLE = config.getboolean('PARAMETERS', 'SHUFFLE')
    VERBOSE = config.getboolean('DEBUG', 'VERBOSE')
    DATA_NORM = config.get('PARAMETERS', 'DATA_NORM')
    LOSS_LAMBDA = config.getfloat('PARAMETERS', 'LOSS_LAMBDA')

    # Create train and validation datasets
    mean, std, min_obs, max_obs = ds.read_stats(config, 'stats.yaml')

    mask_transforms = T.ComposeUnary([T.Normalize(), T.ToTensor()])
    if DATA_NORM == 'MIN_MAX_LOCAL':
        data_transforms = T.ComposeUnary([T.Normalize(), T.ToTensor()])
    elif DATA_NORM == 'MIN_MAX_GLOBAL':
        data_transforms = T.ComposeUnary([T.Normalize(min=min_obs, max=max_obs), T.ToTensor()])
    elif DATA_NORM == 'STANDARIZATION':
        data_transforms = T.ComposeUnary([T.Standarize(mean=mean, std=std), T.ToTensor()])
    elif DATA_NORM == 'NONE':
        data_transforms = T.ComposeUnary([T.ToTensor()])
    else:
        data_transforms = None
        print(f'[ERROR]: invaldia DATA_NORM: {DATA_NORM}')
        sys.exit()

    train_ds = ds.SingleVentricleDataset(config, ds.DatasetMode.TRAIN, load_flow=True,
                                         data_transforms=data_transforms,
                                         mask_transforms=mask_transforms,
                                         data_mask_transforms=None,
                                         flow_transforms=None)
    val_ds = ds.SingleVentricleDataset(config, ds.DatasetMode.VAL, load_flow=True,
                                       data_transforms=data_transforms,
                                       mask_transforms=mask_transforms,
                                       data_mask_transforms=None,
                                       flow_transforms=None)

    # Create data loaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=os.cpu_count() // 2, collate_fn=cnn_utils.collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count() // 2, collate_fn=cnn_utils.collate_fn)

    save_dir = plots.createSaveDirectory(config.get('DATA', 'OUTPUT_PATH'), 'CNN')
    logger = plots.create_logger(save_dir)

    # UNet3D model
    net = UNet3D(config, logger).to(DEVICE)

    if VERBOSE:
        summary(net, input_size=(2, 16, 80, 80), batch_size=-1)
        logger.info("\n")
        logger.info("===========================================================")
        logger.info('CNN parameters')
        logger.info(f"\t* Patients for training: {len(train_ds)}")
        logger.info(f"\t* Patients for validation: {len(val_ds)}")
        logger.info(f'\t* Device: {DEVICE}')
        logger.info(f'\t* Batch size: {BATCH_SIZE}')
        logger.info(f'\t* Num epochs: {NUM_EPOCHS}')
        logger.info(f'\t* Learning rate: {LR}')
        logger.info(f'\t* Weight decay: {WEIGHT_DECAY}, betas: {(BETA1, BETA2)}')
        logger.info(f'\t* Step size: {STEP_SIZE}, gamma: {GAMMA}')
        logger.info(f'\t* Data normalization: {DATA_NORM}')
        logger.info(f'\t* Loss lambda: {LOSS_LAMBDA}')
        logger.info(f'\t* Num workers: {os.cpu_count()//2}')
        logger.info("===========================================================")
        logger.info("\n")

    net = torch.nn.DataParallel(net, device_ids=np.arange(BATCH_SIZE).tolist())
    opt = Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(BETA1, BETA2))
    schedule_lr = StepLR(opt, step_size=STEP_SIZE, gamma=GAMMA)

    writer = SummaryWriter(log_dir=save_dir)

    # save config file to save directory
    conifg_output = osp.join(save_dir, 'config.ini')
    with open(conifg_output, 'w') as config_file:
        config.write(config_file)

    cnn_utils.save_model(net, save_dir, 'net.txt')
    train_ds.save_patients(save_dir, 'train.txt')
    val_ds.save_patients(save_dir, 'val.txt')

    # Steps per epoch for training and evaluation set
    train_steps = len(train_ds) // BATCH_SIZE
    val_steps = len(val_ds) // BATCH_SIZE
    H = {"train_loss": [], "val_loss": []}

    logger.info('\n[INFO] Save directory: ' + save_dir)
    logger.info('\n[INFO]: Trainig CNN')

    pbar = tqdm(total=NUM_EPOCHS)
    tic = time.time()
    trainer = Trainer(net, pbar, config, DEVICE)
    for e in range(NUM_EPOCHS):
        total_train_loss = trainer.train_epoch(train_loader, opt)
        total_val_loss = trainer.val_epoch(val_loader)

        schedule_lr.step()
        avg_train_loss = total_train_loss / train_steps
        avg_val_loss = total_val_loss / val_steps
        logger.info(f'epoch: {e}\t train_loss: {avg_train_loss}\t val_loss: {avg_val_loss}')

        H['train_loss'].append(avg_train_loss)
        H['val_loss'].append(avg_val_loss)
        writer.add_scalars('loss', {'e_train_loss': avg_train_loss, 'e_val_loss:': avg_val_loss}, e)
        writer.add_scalar('lr', schedule_lr.get_last_lr()[0], e)
        cnn_utils.save_weights(net, e, 10, save_dir, 'model_e.pth')

        pbar.update(1)


toc = time.time()
logger.info('\nTotal time taken to train the model: {:.2f}s'.format(toc - tic))

plt.style.use('ggplot')
plt.figure()
plt.plot(H['train_loss'], label='train_loss')
plt.plot(H['val_loss'], label='val_loss')
plt.title('Training Loss on Dataset')
plt.xlabel('Epoch #')
plt.ylabel('Loss')
plt.legend(loc='lower left')
plt.savefig(osp.join(save_dir, 'loss.png'))
torch.save(net, osp.join(save_dir, 'model.pth'))
#net = torch.load('model.pth')
# torch.save(net.state_dict(), osp.join(save_dir, 'model_weights.pth'))
# net.load_state_dict(torch.load('model_weights.pth'))
pbar.close()
writer.close()
