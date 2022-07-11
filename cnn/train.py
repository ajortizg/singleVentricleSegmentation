import torch
import configparser
import sys
from torch.optim import Adam
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
# from unet_3d import UNet3D
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np
import os.path as osp
import cnn_utils
import os
from torchsummary import summary

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
from utils import plots
import utils.transforms as T
import cnn.dataset as ds
from cnn.trainer import Trainer


if __name__ == "__main__":
    cnn_utils.seeding(42)

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
    LOSS_LAMBDA = config.getfloat('PARAMETERS', 'LOSS_LAMBDA')
    NUM_GPUS = config.getint('PARAMETERS', 'NUM_GPUS')

    # Create train and validation datasets
    mask_transforms = T.ComposeUnary([T.ToTensor()])
    img4d_transforms = T.ComposeUnary([T.ToTensor()])

    train_ds = ds.SingleVentricleDataset(config, ds.DatasetMode.TRAIN, load_flow=True,
                                         img4d_transforms=img4d_transforms,
                                         mask_transforms=mask_transforms,
                                         img4d_mask_transforms=None,
                                         flow_transforms=None)
    val_ds = ds.SingleVentricleDataset(config, ds.DatasetMode.VAL, load_flow=True,
                                       img4d_transforms=img4d_transforms,
                                       mask_transforms=mask_transforms,
                                       img4d_mask_transforms=None,
                                       flow_transforms=None)

    # Create data loaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=16,
                              collate_fn=cnn_utils.collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=16,
                            collate_fn=cnn_utils.collate_fn)

    save_dir = plots.createSaveDirectory(config.get('DATA', 'OUTPUT_PATH'), 'CNN')
    logger = plots.create_logger(save_dir)

    # UNet3D model
    # net = UNet3D(config, logger).to(DEVICE)
    net = cnn_utils.create_model(config, logger).to(DEVICE)

    if VERBOSE:
        logger.info("===========================================================")
        summary(net, input_size=(2, 80, 80, 80), batch_size=-1)
        logger.info('CNN parameters')
        logger.info(f"\t* Patients for training: {len(train_ds)}")
        logger.info(f"\t* Patients for validation: {len(val_ds)}")
        logger.info(f'\t* Device: {DEVICE}')
        logger.info(f'\t* Batch size: {BATCH_SIZE}')
        logger.info(f'\t* Num epochs: {NUM_EPOCHS}')
        logger.info(f'\t* Learning rate: {LR}')
        logger.info(f'\t* Weight decay: {WEIGHT_DECAY}, betas: {(BETA1, BETA2)}')
        logger.info(f'\t* Step size: {STEP_SIZE}, gamma: {GAMMA}')
        logger.info(f'\t* Loss lambda: {LOSS_LAMBDA}')
        logger.info(f'\t* Num workers: {16}')
        logger.info(f'\t* Num GPUs: {NUM_GPUS}')
        logger.info("===========================================================")
        logger.info("\n")

    net = torch.nn.DataParallel(net, device_ids=np.arange(NUM_GPUS).tolist())
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
        total_train_loss, l1_train, l2_train, l3_train = trainer.train_epoch(train_loader, opt)
        total_val_loss, l1_val, l2_val, l3_val = trainer.val_epoch(val_loader)

        schedule_lr.step()

        avg_train_loss = total_train_loss / train_steps
        avg_l1_train_loss = l1_train / train_steps
        avg_l2_train_loss = l2_train / train_steps
        avg_l3_train_loss = l3_train / train_steps

        avg_val_loss = total_val_loss / val_steps
        avg_l1_val_loss = l1_val / val_steps
        avg_l2_val_loss = l2_val / val_steps
        avg_l3_val_loss = l3_val / val_steps

        logger.info(f'Epoch: {e}')
        logger.info(f'\t*Train:\tlt: {avg_train_loss:,.2f}\tl1: {avg_l1_train_loss:,.2f}\tl2: {avg_l2_train_loss:,.2f}\tl3: {avg_l3_train_loss:,.2f}')
        logger.info(f'\t*Val:\tlt: {avg_val_loss:,.2f}\tl1: {avg_l1_val_loss:,.2f}\tl2: {avg_l2_val_loss:,.2f}\tl3: {avg_l3_val_loss:,.2f}')

        H['train_loss'].append(avg_train_loss)
        H['val_loss'].append(avg_val_loss)

        writer.add_scalars('loss', {'e_train_loss': avg_train_loss, 'e_val_loss:': avg_val_loss}, e)
        # writer.add_scalars('l123', {'l123/train_l1': avg_l1_train_loss, 'l123/train_l2': avg_l2_train_loss, 'l123/train_l3': avg_l3_train_loss}, e)
        # writer.add_scalars('l123', {'l123/val_l1': avg_l1_val_loss, 'l123/val_l2': avg_l2_val_loss, 'l123/val_l3': avg_l3_val_loss}, e)
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
# net = torch.load('model.pth')
# torch.save(net.state_dict(), osp.join(save_dir, 'model_weights.pth'))
# net.load_state_dict(torch.load('model_weights.pth'))
pbar.close()
writer.close()
