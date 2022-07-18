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
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    config = configparser.ConfigParser()
    config.read('parser/configFineTuning.ini')

    PRETRAINED_MODEL_DIR = config.get('DATA', 'PRETRAINED_DIR')
    WEIGHTS_FILENAME = config.get('DATA', 'WEIGHTS_FILENAME')
    PATIENT_NAME = config.get('DATA', 'PATIENT_NAME')
    NUM_GPUS = config.getint('PARAMETERS', 'NUM_GPUS')
    NUM_WORKERS = config.getint('PARAMETERS', 'NUM_WORKERS')
    NUM_EPOCHS = config.getint('PARAMETERS', 'NUM_EPOCHS')
    BATCH_SIZE = 1
    LR = config.getfloat('PARAMETERS', 'LR')
    WEIGHT_DECAY = config.getfloat('PARAMETERS', 'WEIGHT_DECAY')
    STEP_SIZE = config.getfloat('PARAMETERS', 'STEP_SIZE')
    GAMMA = config.getfloat('PARAMETERS', 'GAMMA')
    BETA1 = config.getfloat('PARAMETERS', 'BETA1')
    BETA2 = config.getfloat('PARAMETERS', 'BETA2')

    config_train = configparser.ConfigParser()
    config_train.read(osp.join(PRETRAINED_MODEL_DIR, 'config.ini'))

    # Create validation dataset and loader
    img4d_transforms = T.ComposeUnary([T.ToTensor()])
    mask_transforms = T.ComposeUnary([T.Round(th=0.5), T.ToTensor()])
    train_ds = ds.SingleVentricleDataset(config_train, ds.DatasetMode.VAL, load_flow=True,
                                       img4d_transforms=img4d_transforms, mask_transforms=mask_transforms)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=cnn_utils.collate_fn_2)

    save_dir = plots.createSaveDirectory(config.get('DATA', 'OUTPUT_PATH'), 'TL')
    logger = plots.create_logger(save_dir)

    # UNet3D model
    net = UNet3D(config_train, logger).to(DEVICE)
    net = torch.nn.DataParallel(net, device_ids=np.arange(NUM_GPUS).tolist())

    PRETRAIED_WEIGHTS = osp.join(PRETRAINED_MODEL_DIR, WEIGHTS_FILENAME)
    net.load_state_dict(torch.load(PRETRAIED_WEIGHTS), strict=True)
    logger.info(f'Use pretrained model: {PRETRAIED_WEIGHTS}')

    opt = Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(BETA1, BETA2))
    schedule_lr = StepLR(opt, step_size=STEP_SIZE, gamma=GAMMA)

    writer = SummaryWriter(log_dir=save_dir)

    # save config file to save directory
    conifg_output = osp.join(save_dir, 'config.ini')
    with open(conifg_output, 'w') as config_file:
        config.write(config_file)

    cnn_utils.save_model(net, save_dir, 'net.txt')

    # Steps per epoch for training and evaluation set
    H = {"train_loss": []}
    train_steps = 1
    logger.info('Save directory: ' + save_dir)
    logger.info(f'Searching patient: {PATIENT_NAME}')

    pbar = tqdm(total=NUM_EPOCHS)
    tic = time.time()
    trainer = Trainer(net, pbar, config, DEVICE)
    best_train_loss = 1e10
    patient_found = False
    for (pnames, imgs4d, m0s, mks, list_times_fwd, list_times_bwd, ff, bf, offsets) in train_loader:
        if pnames[0] == PATIENT_NAME:
            patient_found = True
            imgs4d = imgs4d.to(DEVICE)
            m0s = m0s.to(DEVICE)
            mks = mks.to(DEVICE)
            ff = ff.to(DEVICE)
            bf = bf.to(DEVICE)
            logger.info(f'Patient found: {pnames[0]}')
            break

    if not patient_found:
        logger.error(f'No patient found: {PATIENT_NAME}')
        sys.exit()
    
    logger.info('Train CNN')
    for e in range(NUM_EPOCHS):
        pbar.set_postfix_str(f'Train: {pnames[0]}')
        total_train_loss, l1_train, l2_train, l3_train = trainer.train_patient(imgs4d, m0s, mks, list_times_fwd, list_times_bwd, ff, bf, offsets, opt)

        schedule_lr.step()

        avg_train_loss = total_train_loss / train_steps
        avg_l1_train_loss = l1_train / train_steps
        avg_l2_train_loss = l2_train / train_steps
        avg_l3_train_loss = l3_train / train_steps

        logger.info(f'Epoch: {e}')
        logger.info(f'\t*Train:\tlt: {avg_train_loss:,.2f}\tl1: {avg_l1_train_loss:,.2f}\tl2: {avg_l2_train_loss:,.2f}\tl3: {avg_l3_train_loss:,.2f}')

        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            torch.save(net, osp.join(save_dir, 'best_train_model.pth'))
            torch.save(net.state_dict(), osp.join(save_dir, 'best_train_weights.pth'))
            logger.info(f'\t*Best train model and weights updated with loss: {best_train_loss:,.2f}')

        H['train_loss'].append(avg_train_loss)

        writer.add_scalars('loss', {'e_train_loss': avg_train_loss}, e)
        writer.add_scalars('l123', {'l123/train_l1': avg_l1_train_loss, 'l123/train_l2': avg_l2_train_loss, 'l123/train_l3': avg_l3_train_loss}, e)
        writer.add_scalar('lr', schedule_lr.get_last_lr()[0], e)
        pbar.update(1)

toc = time.time()
logger.info('\nTotal time taken to train the model: {:.2f}s'.format(toc - tic))
logger.info('\nTotal time taken for loading data: {:.2f}s'.format(train_ds.total_time + train_ds.total_time))

plt.style.use('ggplot')
plt.figure()
plt.plot(H['train_loss'], label='train_loss')
plt.title('Training Loss on Dataset')
plt.xlabel('Epoch #')
plt.ylabel('Loss')
plt.legend(loc='lower left')
plt.savefig(osp.join(save_dir, 'loss.png'))
torch.save(net, osp.join(save_dir, 'model.pth'))
torch.save(net.state_dict(), osp.join(save_dir, 'weights.pth'))
pbar.close()
writer.close()
