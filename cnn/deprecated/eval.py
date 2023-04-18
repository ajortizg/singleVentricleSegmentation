import torch
import configparser
from tqdm import tqdm
import sys
import os.path as osp
import pandas as pd
import csv
from torch.utils.data import DataLoader
import nibabel as nib
import matplotlib.pyplot as plt
import os.path as osp
import numpy as np
from monai.metrics.meandice import compute_meandice


ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
from utilities import plots
from utilities.collate import collate_fn
from utilities import param_reader
import utilities.transforms.senary_transforms as T6
import utilities.transforms.unary_transforms as T1
from cnn.dataset import SingleVentricleDataset, DatasetMode, LoadFlowMode
from cnn.trainer import Trainer
from cnn.models.model_factory import create_model, save_model


def save_nifty(mask, hdr, save_dir, filename):
    mask = mask.squeeze()
    mask = torch.swapaxes(mask, 0, 2)           # xyz format
    mask = torch.where(mask > 0.5, 1.0, 0.0)    # binarize

    mt_nii = nib.Nifti1Image(T1.ToArray()(mask), affine=None, header=None)
    outputFile = osp.sep.join([save_dir, filename])
    nib.save(mt_nii, outputFile)


if __name__ == "__main__":
    config_eval = configparser.ConfigParser()
    config_eval.read('parser/configCNNEval.ini')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    P = param_reader.eval_params(config_eval)

    if P['fine_tuning']:
        config_tl = configparser.ConfigParser()
        config_tl.read(osp.join(P['trained_model_dir'], 'config.ini'))
        pretrained_dir = config_tl.get('DATA', 'PRETRAINED_DIR')
        patient_name = config_tl.get('DATA', 'PATIENT_NAME')

        config_train = configparser.ConfigParser()
        config_train.read(osp.join(pretrained_dir, 'config.ini'))
    else:
        config_train = configparser.ConfigParser()
        config_train.read(osp.join(P['trained_model_dir'], 'config.ini'))

    transforms = T6.Compose([T6.ToTensor()])
    if P['dataset'] == 'train':
        ds = SingleVentricleDataset(config_train, DatasetMode.TRAIN, LoadFlowMode.ED_ES, full_transforms=transforms)
    elif P['dataset'] == 'val':
        ds = SingleVentricleDataset(config_train, DatasetMode.VAL, LoadFlowMode.ED_ES, full_transforms=transforms)
    elif P['dataset'] == 'test':
        ds = SingleVentricleDataset(config_train, DatasetMode.TEST, LoadFlowMode.WHOLE_CYCLE, full_transforms=transforms)

    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=P['workers'], collate_fn=collate_fn)

    save_dir = plots.createSaveDirectory(P['out_path'], 'EVAL')
    param_reader.save_config(config_eval, save_dir, 'config.ini')

    logger = plots.create_logger(save_dir)
    logger.info(f'Device: {device}')
    logger.info('Dataset: %s' % P['dataset'])
    logger.info('Checkpoint: %s' % osp.join(P['trained_model_dir'], P['model_name']))

    # Create net and load parameters
    net = create_model(config_train, logger).to(device)
    net = torch.nn.DataParallel(net, device_ids=[0])
    checkpoint = torch.load(osp.join(P['trained_model_dir'], P['model_name']))
    net.load_state_dict(checkpoint['model_state_dict'], strict=True)

    save_size = (P['save_nz'], P['save_ny'], P['save_nx'])
    mask_posp = T1.Compose([T1.Resize(save_size), T1.Round(0.5), T1.Erode()])
    masks_gt_posp = T1.Compose([T1.Resize(save_size), T1.Round(0.5)])
    img_posp = T1.Compose([T1.Resize(save_size)])

    pbar = tqdm(total=len(ds))
    trainer = Trainer(net, None, pbar, config_train, device, None, None, display_prob=0)
    df = pd.DataFrame()

    for i, (pnames, imgs4d, m0s, mks, masks, times_fwd, times_bwd, ff, bf) in enumerate(loader):
        if P['fine_tuning']:
            if pnames[0] != patient_name:
                continue

        imgs4d = imgs4d.to(device)
        m0s = m0s.to(device)
        mks = mks.to(device)
        ff = ff.to(device)
        bf = bf.to(device)

        row = {'Patient': pnames[0]}

        if P['dataset'] == 'test':
            masks = masks.to(device)
            times_fwd, times_bwd = ds.create_timeline(times_fwd[0], times_bwd[0], masks.shape[-1])

            metrics_cnn, accs_fwd_cnn, accs_bwd_cnn, mts_cnn, mtts_cnn = trainer.test_patient(imgs4d, masks, times_fwd, times_bwd, ff, bf, cnn=True)
            metrics_flow, accs_fwd_flow, accs_bwd_flow, mts_flow, mtts_flow = trainer.test_patient(imgs4d, masks, times_fwd, times_bwd, ff, bf, cnn=False)
        else:
            metrics_cnn, mts_cnn, mtts_cnn = trainer.val_patient(imgs4d, m0s, mks, times_fwd, times_bwd, ff, bf, cnn=True)
            metrics_flow, mts_flow, mtts_flow = trainer.val_patient(imgs4d, m0s, mks, times_fwd, times_bwd, ff, bf, cnn=False)

        for k, v in metrics_cnn.items():
            row['cnn_' + k] = v
        for k, v in metrics_flow.items():
            row['flow_' + k] = v
        df_row = pd.DataFrame(row, index=[i])
        df = pd.concat([df_row, df])

        if P['save_imgs'] or P['save_nifti']:
            patient_dir = plots.createSubDirectory(save_dir, pnames[0])
            fwd_dir = plots.createSubDirectory(patient_dir, 'fwd')
            bwd_dir = plots.createSubDirectory(patient_dir, 'bwd')

            timesteps = mts_cnn.shape[0]
            for t in range(timesteps):
                if P['save_imgs']:
                    img3d = img_posp(imgs4d[..., times_fwd[t].item()].squeeze())

                    if P['dataset'] == 'test':
                        mtt_net = mask_posp(mtts_cnn[times_fwd[t]].squeeze())
                        mtt_of = mask_posp(mtts_flow[times_fwd[t]].squeeze())
                        mt_net = mask_posp(mts_cnn[times_fwd[t]].squeeze())
                        mt_of = mask_posp(mts_flow[times_fwd[t]].squeeze())

                        mgt = masks_gt_posp(masks[..., times_fwd[t]].squeeze())
                        plots.save_img_masks(img3d, [mgt, mtt_net, mtt_of], f'im_t_{times_fwd[t]}', fwd_dir, th=0.5,
                                             alphas=[0.3, 1.0, 1.0], colors=[[1, 0.7, 0], [0, 1, 0], [0, 0, 1]])

                        plots.save_img_masks(img3d, [mgt, mt_net, mt_of], f'im_t_{times_fwd[t]}', bwd_dir, th=0.5,
                                             alphas=[0.3, 1.0, 1.0], colors=[[1, 0.7, 0], [0, 1, 0], [0, 0, 1]])
                        if P['save_nifti']:
                            # save_nifty(mts_cnn[t], fwd_dir, f'mt_{t}.nii')
                            # save_nifty(mtts_cnn[t], bwd_dir, f'mtt_{t}.nii')
                            hdr = loader.dataset.header(i)
                            plots.save_nifti_mask(mts_cnn[t], hdr, fwd_dir, f'mt_{t}.nii')
                            plots.save_nifti_mask(mtts_cnn[t], hdr, bwd_dir, f'mtt_{t}.nii')

                    else:
                        mtt_net = mask_posp(mtts_cnn[t].squeeze())
                        mtt_of = mask_posp(mtts_flow[t].squeeze())
                        mt_net = mask_posp(mts_cnn[t].squeeze())
                        mt_of = mask_posp(mts_flow[t].squeeze())
                        if t == 0:
                            plots.save_img_masks(img3d, [masks_gt_posp(m0s.squeeze()), mtt_net, mtt_of], 'im_m0_m0tt', fwd_dir, th=0.5,
                                                 alphas=[0.3, 1.0, 1.0], colors=[[1, 0.7, 0], [0, 1, 0], [0, 0, 1]])

                        elif t == timesteps - 1:
                            plots.save_img_masks(img3d, [masks_gt_posp(mks.squeeze()), mt_net, mt_of], 'mk_mkt', bwd_dir, th=0.5,
                                                 alphas=[0.3, 1.0, 1.0], colors=[[1, 0.7, 0], [0, 1, 0], [0, 0, 1]])

                        plots.save_img_masks(img3d, [mt_net, mt_of], f'im_t_{times_fwd[t].item()}', fwd_dir, th=0.5,
                                             alphas=[1.0, 1.0], colors=[[0, 1, 0], [0, 0, 1]])

                        plots.save_img_masks(img3d, [mtt_net, mtt_of], f'im_tt_{times_fwd[t].item()}', bwd_dir, th=0.5,
                                             alphas=[1.0, 1.0], colors=[[0, 1, 0], [0, 0, 1]])

                        if P['save_nifti']:
                            hdr = loader.dataset.header(i)
                            plots.save_nifti_mask(mts_cnn[t], hdr, fwd_dir, f'mt_{times_fwd[t].item()}.nii')
                            plots.save_nifti_mask(mtts_cnn[t], hdr, bwd_dir, f'mtt_{times_fwd[t].item()}.nii')
        pbar.update(1)
    df = df.round(3)
    logger.info(df.mean(axis=0, numeric_only=True))
    df.to_excel(osp.join(save_dir, 'results.xlsx'), index=False)


# x = np.arange(abs(times_fwd[0] + 1 - times_bwd[0])).tolist()
# times = slice(times_fwd[0] + 1, times_bwd[0])
# # times_plot[0] = 'ES'
# # times_plot[-1] = 'ED'
# labels = [None] * len(x)
# labels[0] = 'ES'
# labels[-1] = 'ED'
# # labels[len(labels)//2] = 'Time'
# plt.figure(figsize=(3,1), dpi=100)
# plt.plot(x, accs_fwd_flow[times], marker='+', label='Forward Flow', linewidth=1.5, markersize=4)
# plt.plot(x, accs_bwd_flow[times], marker='*', label='Backward Flow', linewidth=1.5, markersize=4)
# plt.plot(x, accs_fwd_cnn[times], marker='o', label='Forward CNN', linewidth=1.5, markersize=4)
# plt.plot(x, accs_bwd_cnn[times], marker='x', label='Backward CNN', linewidth=1.5, markersize=4)
# plt.xticks(x, labels, fontsize=10)
# plt.yticks(fontsize=10)
# plt.xlabel('Time', fontsize=10, weight='bold')
# plt.ylabel('Dice', fontsize=10, weight='bold')
# # plt.legend(loc='lower right',frameon=False, fontsize=8)
# plt.savefig(osp.join(save_dir, pnames[0] + '_acc.pdf'))