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

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
from utilities import path_utils, stuff, plots
from utilities.collate import collate_fn_batch
from utilities import param_reader
import utilities.transforms.senary_transforms as T6
from cnn.dataset import *
# from cnn.trainer import Trainer
from cnn.trainer_multi_batch import Trainer
from cnn.models.model_factory import create_model, save_model

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
    P = param_reader.fine_tuning_params(config)

    # Create dataset and loader
    transforms = T6.Compose([
        # T6.Resize(1.0, (64, 64, 64)),  # for swin unet
        # T6.RoundMasks(), #  # for swin unet
        T6.ToTensor()
        ])

    if P['dataset'] == 'train':
        train_ds = SingleVentricleDataset(config, DatasetMode.TRAIN, LoadFlowMode.ED_ES, full_transforms=transforms)
    elif P['dataset'] == 'val':
        train_ds = SingleVentricleDataset(config, DatasetMode.VAL, LoadFlowMode.ED_ES, full_transforms=transforms)
    elif P['dataset'] == 'test':
        train_ds = SingleVentricleDataset(config, DatasetMode.TEST, LoadFlowMode.ED_ES, full_transforms=transforms)
        test_ds = SingleVentricleDataset(config, DatasetMode.TEST, LoadFlowMode.WHOLE_CYCLE, full_transforms=transforms)
        test_loader = DataLoader(test_ds, batch_size=P['batch_size'], shuffle=False, num_workers=P['num_workers'], collate_fn=collate_fn_batch)
    elif P['dataset'] == 'full':
        train_ds = SingleVentricleDataset(config, DatasetMode.FULL, LoadFlowMode.ED_ES, full_transforms=transforms)
        
    train_loader = DataLoader(train_ds, batch_size=P['batch_size'], shuffle=False, num_workers=P['num_workers'], collate_fn=collate_fn_batch)

    save_dir = path_utils.create_save_dir(config.get('DATA', 'OUTPUT_PATH'), 'FT')
    logger = stuff.create_logger(save_dir)
    writer = SummaryWriter(log_dir=save_dir)
    stuff.save_config(config, save_dir, 'config.ini')

    # Create model and load weights
    config_train = configparser.ConfigParser()
    config_train.read(osp.join(P['pretrained_model_dir'], 'config.ini'))
    net = create_model(config_train, logger).to(device)
    net = torch.nn.DataParallel(net, device_ids=np.arange(P['num_gpus']).tolist())
    checkpoint = torch.load(osp.join(P['pretrained_model_dir'], P['weights_filename']))
    net.load_state_dict(checkpoint['model_state_dict'], strict=True)
    save_model(net, save_dir, 'net.txt')

    opt = Adam(net.parameters(), lr=P['lr'], weight_decay=P['weight_decay'], betas=(P['beta1'], P['beta2']))
    scheduler = StepLR(opt, step_size=P['step_size'], gamma=P['gamma'])

    logger.info('Save directory: ' + save_dir)
    logger.info('Searching patient: %s' % P['patient_name'])

    # Search retraning patient
    found, data = search_patient(P['patient_name'], train_loader)
    if not found:
        logger.error('Patient not found: %s ' % P['PATIENT_NAME'])
        sys.exit()
    pnames, img4d, m0, mk, _, times_fwd, times_bwd, ff, bf, _ = data
    img4d = img4d.to(device)
    m0 = m0.to(device)
    mk = mk.to(device)
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
    trainer = Trainer(net, opt, pbar, config, device, writer, logger)
    logger.info('Train CNN')
    
    for e in range(P['num_epochs']):
        pbar.set_postfix_str(f'Train: {pnames[0]}')
        trainer.train_patient(img4d, m0, mk, times_fwd, times_bwd, ff, bf)

        if P['dataset'] == 'test':
            pbar.set_postfix_str(f'Test: {test_data[0][0]}')
            metrics, *_ = trainer.test_patient(test_imgs4d, test_masks, test_times_fwd, test_times_bwd, test_ff, test_bf, cnn=True, hd=True)
               
        trainer.log(e)
        logger.info('\ttest_hd:\t{:.3f}'.format(metrics['mean_hd']))
        scheduler.step()
        pbar.update(1)
        
        # trainer.create_checkpoint(e, save_dir, when_better=True, which='test', verbose=True)
        
        # early stop
        # patience = 50
        # if trainer.epochs_since_last_improvement > patience:
        #     logger.info(f'Early stop at epoch: {e}')
        #     break

    toc = time.time()
    logger.info('\nTotal time taken to train the model: {:.4f}s'.format(toc - tic))

    trainer.create_checkpoint(e, save_dir, when_better=False)
    trainer.save_stats(save_dir)
    trainer.plot_stats(save_dir, log_scale=False)

    # # Save nifti files
    # # Create save dirs
    # logger.info("\nSaving result nifti files")
    # patient_dir = path_utils.create_sub_dir(save_dir, 'nifti')
    # fwd_dir = path_utils.create_sub_dir(patient_dir, 'fwd')
    # bwd_dir = path_utils.create_sub_dir(patient_dir, 'bwd')
    # gt_dir = path_utils.create_sub_dir(patient_dir, 'gt')

    # *_, mts, mtts = trainer.test_patient(test_imgs4d, test_masks, test_times_fwd, test_times_bwd, test_ff, test_bf, cnn=True, hd=True)
    # ti = times_fwd[0]
    # tf = times_bwd[0]
    # times = np.arange(ti, tf+1)
    # for i, t in enumerate(times):
    #     hdr = test_loader.dataset.header(0)
    #     plots.save_nifti_mask(mts[i], hdr, fwd_dir, f'm_fwd_{t}.nii')
    #     plots.save_nifti_mask(mtts[i], hdr, bwd_dir, f'm_bwd_{t}.nii')
    
    # plots.save_nifti_mask(m0, hdr, gt_dir, "m0.nii")
    # plots.save_nifti_mask(mk, hdr, gt_dir, "mk.nii")
   
    pbar.close()
    writer.close()
   

   