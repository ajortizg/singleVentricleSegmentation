import torch
import configparser
import sys
from torch.optim import Adam
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
import time
import os.path as osp
from torchsummary import summary
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../..'))
sys.path.append(ROOT_DIR)
from cnn import cnn_utils
from cnn.unet_3d import UNet3D
from cnn.dataset import SingleVentricleDataset, DatasetMode
import utils.transforms as T
from utils import plots


if __name__ == "__main__":
    cnn_utils.seeding(42)
    plots.printConsoleOutput_Header('Train CNN')
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
    VERBOSE = config.getboolean('DEBUG', 'VERBOSE')
    SHUFFLE = config.getboolean('PARAMETERS', 'SHUFFLE')

    # Create train and validation datasets
    data_transf = T.ComposeUnary([T.ToTensor()])
    mask_transf = T.ComposeUnary([T.Round(th=0.5), T.ToTensor()])
    train_ds = SingleVentricleDataset(
        config, DatasetMode.TRAIN, load_flow=True, img4d_transforms=data_transf, mask_transforms=mask_transf)
    val_ds = SingleVentricleDataset(config, DatasetMode.VAL, load_flow=True,
                                    img4d_transforms=data_transf, mask_transforms=mask_transf)

    save_dir = plots.createSaveDirectory(config.get('DATA', 'OUTPUT_PATH'), 'CNN')
    logger = plots.create_logger(save_dir)

    # UNet3D model
    net = UNet3D(config, logger).to(DEVICE)
    opt = Adam(net.parameters(), lr=LR,
            weight_decay=WEIGHT_DECAY, betas=(BETA1, BETA2))
    schedule_lr = StepLR(opt, step_size=STEP_SIZE, gamma=GAMMA)


    writer = SummaryWriter(log_dir=save_dir)

    train_ds.save_patients(save_dir, 'train.txt')
    val_ds.save_patients(save_dir, 'val.txt')

    # save config file to save directory
    conifg_output = osp.join(save_dir, 'config.ini')
    with open(conifg_output, 'w') as config_file:
        config.write(config_file)

    cnn_utils.save_model(net, save_dir, 'net.txt')

    if VERBOSE:
        summary(net, input_size=(2, 80, 80, 80), batch_size=-1)
        logger.info("\n")
        logger.info("===========================================================")
        logger.info('CNN parameters')
        logger.info(f"\t* Patients for training: {len(train_ds)}")
        logger.info(f"\t* Patients for validation: {len(val_ds)}")
        logger.info(f'\t* Device: {DEVICE}')
        logger.info(f'\t* Learning rate: {LR}')
        logger.info(f'\t* Num epochs: {NUM_EPOCHS}')
        logger.info("===========================================================")
        logger.info("\n")

    # Steps per epoch for training and evaluation set
    train_steps = len(train_ds) // BATCH_SIZE
    val_steps = len(val_ds) // BATCH_SIZE
    H = {"train_loss": [], "val_loss": []}

    logger.info("[INFO]: Training the network")

    # seeding(42)

    avg_train_loss = 0
    avg_val_loss = 0
    pbar = tqdm(total=NUM_EPOCHS)
    train_idxs = np.arange(len(train_ds))
    tic = time.time()
    for e in range(NUM_EPOCHS):
        if SHUFFLE:
            np.random.shuffle(train_idxs)

        total_train_loss = cnn_utils.train(net, opt, train_idxs, train_ds, e, pbar, config, writer, DEVICE)
        total_val_loss = cnn_utils.validate(net, val_ds, e, pbar, config, writer, DEVICE)
        pbar.update(1)

        logger.info(f'Epoch: {e}')
        logger.info(f'\t*Train:\tlt: {avg_train_loss:,.2f}')
        logger.info(f'\t*Val:\tlt: {avg_val_loss:,.2f}')

        schedule_lr.step()
        avg_train_loss = total_train_loss / train_steps
        avg_val_loss = total_val_loss / val_steps
        H['train_loss'].append(avg_train_loss)
        H['val_loss'].append(avg_val_loss)
        writer.add_scalars('loss', {'e_train_loss': avg_train_loss, 'e_val_loss': avg_val_loss}, e)
        cnn_utils.save_weights(net, e, 10, save_dir, 'model_e.pth')


    toc = time.time()
    print('\nTotal time taken to train the model: {:.2f}s'.format(toc - tic))

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
