import torch
import configparser
import sys
from torch.optim import Adam
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np
import os.path as osp
import cnn_utils

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
from utils import plots
import utils.transforms as T
import cnn.dataset as ds
from cnn.trainer import Trainer


def search_patient(query, loader):
    found = False
    for data in loader:
        pnames = data[0]
        if pnames[0] == query:
            found = True
            logger.info(f'Patient found: {pnames[0]}')
            return found, data
    return found, None


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = configparser.ConfigParser()
    config.read('parser/configFineTuning.ini')
    P = cnn_utils.read_fine_tuning_params(config)

    config_train = configparser.ConfigParser()
    config_train.read(osp.join(P['pretrained_model_dir'], 'config.ini'))

    # Create dataset and loader
    transforms = T.ComposeFull([T.BinarizeMasks(th=0.5), T.ToTensorFull()])
    if P['dataset'] == 'train':
        train_ds = ds.SingleVentricleDataset(config_train, ds.DatasetMode.TRAIN, ds.LoadFlowMode.TRAIN_VAL_OF, full_transforms=transforms)
    elif P['dataset'] == 'val':
        train_ds = ds.SingleVentricleDataset(config_train, ds.DatasetMode.VAL, ds.LoadFlowMode.TRAIN_VAL_OF, full_transforms=transforms)
    elif P['dataset'] == 'test':
        test_transforms = T.ComposeUnary([T.Round(th=0.5), T.ToTensor()])
        train_ds = ds.SingleVentricleDataset(config_train, ds.DatasetMode.TEST, ds.LoadFlowMode.TRAIN_VAL_OF, full_transforms=transforms, test_masks_transforms=test_transforms)
        test_ds = ds.SingleVentricleDataset(config_train, ds.DatasetMode.TEST, ds.LoadFlowMode.TEST_OF, full_transforms=transforms, test_masks_transforms=test_transforms)
        test_loader = DataLoader(test_ds, batch_size=P['batch_size'], shuffle=False, num_workers=P['num_workers'], collate_fn=cnn_utils.collate_fn)
    
    train_loader = DataLoader(train_ds, batch_size=P['batch_size'], shuffle=False, num_workers=P['num_workers'], collate_fn=cnn_utils.collate_fn)

    save_dir = plots.createSaveDirectory(config.get('DATA', 'OUTPUT_PATH'), 'FT')
    logger = plots.create_logger(save_dir)
    writer = SummaryWriter(log_dir=save_dir)
    plots.save_config(config, save_dir, 'config.ini')

    # Create model and load weights
    net = cnn_utils.create_net(config_train, logger).to(device)
    net = torch.nn.DataParallel(net, device_ids=np.arange(P['num_gpus']).tolist())
    checkpoint = torch.load(osp.join(P['pretrained_model_dir'], P['weights_filename']))
    net.load_state_dict(checkpoint['model_state_dict'], strict=True)
    cnn_utils.save_model(net, save_dir, 'net.txt')

    opt = Adam(net.parameters(), lr=P['lr'], weight_decay=P['weight_decay'], betas=(P['beta1'], P['beta2']))
    scheduler = StepLR(opt, step_size=P['step_size'], gamma=P['gamma'])

    H = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'test_acc': []}
    logger.info('Save directory: ' + save_dir)
    logger.info('Searching patient: %s' % P['patient_name'])

    # Search retraning patient
    found, data = search_patient(P['patient_name'], train_loader)
    if not found:
        logger.error('Patient not found: %s ' % P['PATIENT_NAME'])
        sys.exit()
    pnames, imgs4d, m0s, mks, _, times_fwd, times_bwd, ff, bf, offsets = data
    imgs4d = imgs4d.to(device)
    m0s = m0s.to(device)
    mks = mks.to(device)
    ff = ff.to(device)
    bf = bf.to(device)

    if P['dataset'] == 'test':
        _, test_data = search_patient(P['patient_name'], test_loader)
        test_imgs4d = test_data[1].to(device)
        test_masks = test_data[4].to(device)
        test_ff = test_data[7].to(device)
        test_bf = test_data[8].to(device)
        test_times_fwd, test_times_bwd = test_ds.create_timeline(test_data[5][0], test_data[6][0], test_masks.shape[-1])

    pbar = tqdm(total=P['num_epochs'])
    tic = time.time()
    trainer = Trainer(net, pbar, config, device, writer)
    best_train_loss = 1e10

    df = pd.DataFrame({'Patient': pnames[0], 'Lambda': P['loss_lambda']}, index=[0])
    save_every = 100
    logger.info('Train CNN')
    
    for e in range(P['num_epochs']):
        pbar.set_postfix_str(f'Train: {pnames[0]}')
        train_res = trainer.train_patient(imgs4d, m0s, mks, times_fwd, times_bwd, ff, bf, offsets, opt)

        if P['dataset'] == 'test':
            pbar.set_postfix_str(f'Test: {test_data[0][0]}')
            test_acc, *_ = trainer.test_patient(test_imgs4d, test_masks, test_times_fwd, test_times_bwd, test_ff, test_bf, cnn=True)
        else:
            test_acc = 0.0

        best_train_loss, _ = cnn_utils.log(logger, writer, e, train_res, train_res, test_acc, net, opt,
                                           best_train_loss, 0, save_dir)
        H = cnn_utils.update_train_history(H, train_res, train_res, test_acc)

        if e % save_every == 0:
            df.insert(0, f'{e}', train_res[-1])

        scheduler.step()
        pbar.update(1)

    df.to_excel(osp.join(save_dir, 'accuracy.xlsx'), index=False)
    toc = time.time()
    logger.info('\nTotal time taken to train the model: {:.4f}s'.format(toc - tic))

    plots.save_loss(H, save_dir)
    plots.save_acc(H, save_dir)
    cnn_utils.checkpoint(e, net, opt, train_res, train_res, test_acc, save_dir, 'checkpoint.pth')
   
    pbar.close()
    writer.close()
   

   