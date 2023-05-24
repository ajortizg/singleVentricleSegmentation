import torch
import configparser
import os.path as osp
from torch.utils.data import DataLoader
import sys
from tqdm import tqdm
import pandas as pd
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import csv

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
from cnn.dataset import *
import utilities.transforms.senary_transforms as T6
import utilities.transforms.unary_transforms as T1
from utilities.collate import collate_fn_batch
from cnn.models.model_factory import create_model
from cnn.trainer_multi_batch import Trainer
from cnn.trainer_onehot import TrainerOneHot
from utilities import plots
from utilities import path_utils

__all__ = ['Evalautor']


class Evalautor:
    def __init__(self, config, params, device, logger, save_dir, verbose=False):
        self.P = params
        self.device = device
        self.logger = logger
        self.verbose = verbose
        self.save_dir = save_dir

        if self.verbose:
            self.logger.info(f'Device: {self.device}')

        self.read_train_config()
        self.load_data(config)
        self.load_model()

        # Transformations for saving
        save_size = (self.P['save_nz'], self.P['save_ny'], self.P['save_nx'])
        self.est_masks_transf = T1.Compose([T1.Resize(save_size), T1.Round(0.5), T1.Erode()])
        self.gt_masks_transf = T1.Compose([T1.Resize(save_size), T1.Round(0.5)])
        self.img_transf = T1.Compose([T1.Resize(save_size)])

    def read_train_config(self):
        if self.P['fine_tuning']:
            self.finetuning = True
            config_ft = configparser.ConfigParser()
            config_ft.read(osp.join(self.P['trained_model_dir'], 'config.ini'))
            pretrained_dir = config_ft.get('DATA', 'pretrained_dir')
            self.finetuned_patient = config_ft.get('DATA', 'patient_name')

            self.config_train = configparser.ConfigParser()
            self.config_train.read(osp.join(pretrained_dir, 'config.ini'))
        else:
            self.finetuned_patient = None
            self.finetuning = False
            self.config_train = configparser.ConfigParser()
            self.config_train.read(osp.join(self.P['trained_model_dir'], 'config.ini'))

    def load_data(self, config):
        transforms = T6.Compose([
            # T6.Resize(1.0, (64, 64, 64)),  # for swin unet
            # T6.RoundMasks(), #  # for swin unet
            T6.ToTensor()
        ])
        dsettype = self.P['dataset']
        self.is_testset = False
        if dsettype == 'train':
            self.dset = SingleVentricleDataset(config, DatasetMode.TRAIN,
                                               LoadFlowMode.ED_ES, full_transforms=transforms)
        elif dsettype == 'val':
            self.dset = SingleVentricleDataset(config, DatasetMode.VAL,
                                               LoadFlowMode.ED_ES, full_transforms=transforms)
        elif dsettype == 'test':
            self.is_testset = True
            self.dset = SingleVentricleDataset(config, DatasetMode.TEST,
                                               LoadFlowMode.WHOLE_CYCLE, full_transforms=transforms)
        elif dsettype == 'full':
            self.dset = SingleVentricleDataset(config, DatasetMode.FULL,
                                               LoadFlowMode.ED_ES, full_transforms=transforms)
        else:
            self.dset = None
            self.logger.info('Dataset not found: %s' % dsettype)
            sys.exit()

        self.loader = DataLoader(self.dset, batch_size=1, shuffle=False,
                                 num_workers=self.P['workers'], collate_fn=collate_fn_batch)
        if self.verbose:
            self.logger.info('Dataset: %s' % dsettype)

    def load_model(self):
        # Load model
        self.net = create_model(self.config_train, self.logger).to(self.device)
        self.net = torch.nn.DataParallel(self.net, device_ids=[0])
        checkpoint = torch.load(osp.join(self.P['trained_model_dir'], self.P['model_name']))
        self.net.load_state_dict(checkpoint['model_state_dict'], strict=True)

        if self.verbose:
            self.logger.info('Checkpoint: %s' % osp.join(self.P['trained_model_dir'], self.P['model_name']))

        # Create trainer for prediction
        self.pbar = tqdm(total=1) if self.finetuning else tqdm(total=len(self.dset))
        self.trainer = Trainer(self.net, None, self.pbar, self.config_train, self.device, None, self.logger)

    def evaluate(self):
        self.report = pd.DataFrame()

        if self.finetuning:
            self.finetuned_data()
            self.evaluate_patient(0)
        else:
            for i, data in enumerate(self.loader):
                self.prepare_data(data)
                self.evaluate_patient(i)

    def save_report(self, filename='report.xlsx', decimals=3):
        self.report = self.report.round(decimals)
        self.logger.info(self.report.mean(axis=0, numeric_only=True))
        self.report.to_excel(osp.join(self.save_dir, filename), index=False)

    def save_list_to_csv(self, accs, filename):
        with open(osp.join(self.save_dir, filename), 'w') as fp:
            writer = csv.writer(fp)
            writer.writerows(accs)

    def evaluate_patient(self, idx):
        if self.is_testset:
            self.cnn_res = self.trainer.test_patient(self.img4d, self.masks, self.timesfwd, self.timesbwd, self.ff, self.bf, cnn=True)
            self.flow_res = self.trainer.test_patient(self.img4d, self.masks, self.timesfwd, self.timesbwd, self.ff, self.bf, cnn=False)

            accs_fwd_cnn = self.cnn_res[1]
            accs_bwd_cnn = self.cnn_res[2]
            hd_fwd_cnn = self.cnn_res[5]
            hd_bwd_cnn = self.cnn_res[6]

            accs_fwd_flow = self.flow_res[1]
            accs_bwd_flow = self.flow_res[2]
            hd_fwd_flow = self.flow_res[5]
            hd_bwd_flow = self.flow_res[6]

            self.save_list_to_csv([accs_fwd_cnn, accs_bwd_cnn, accs_fwd_flow, accs_bwd_flow], f'{self.cur_patient}_acc.csv')
            self.save_list_to_csv([hd_fwd_cnn, hd_bwd_cnn, hd_fwd_flow, hd_bwd_flow], f'{self.cur_patient}_hd.csv')

            pd.DataFrame({'volgt': self.cnn_res[7], 'volfwd': self.cnn_res[8], 'volbwd': self.cnn_res[9]}
                         ).to_csv(osp.join(self.save_dir, f'{self.cur_patient}_vol.csv'), index=False)

            pd.DataFrame({'surgt': self.cnn_res[10], 'surfwd': self.cnn_res[11], 'surbwd': self.cnn_res[12]}
                         ).to_csv(osp.join(self.save_dir, f'{self.cur_patient}_sur.csv'), index=False)

            # x = np.arange(abs(self.timesfwd[0] + 1 - self.timesbwd[0])).tolist()
            # times = slice(self.timesfwd[0] + 1, self.timesbwd[0])
            # # # times_plot[0] = 'ES'
            # # # times_plot[-1] = 'ED'
            # labels = [None] * len(x)
            # labels[0] = 'ES'
            # labels[-1] = 'ED'
            # # # labels[len(labels)//2] = 'Time'
            # plt.figure(figsize=(3.1, 1.7), dpi=100)

            # # ft_fwd_color = 'mo-'
            # # ft_bwd_color = 'kx-'
            # cnn_fwd_color = 'gd-'
            # cnn_bwd_color = 'hc-'

            # plt.plot(x, accs_fwd_flow[times], 'r+-', label='Forward Flow', linewidth=1.5, markersize=2.5)
            # plt.plot(x, accs_bwd_flow[times], 'b*-', label='Backward Flow', linewidth=1.5, markersize=2.5)
            # plt.plot(x, accs_fwd_cnn[times], cnn_fwd_color, label='Forward CNN', linewidth=1.5, markersize=2.5)
            # plt.plot(x, accs_bwd_cnn[times], cnn_bwd_color, label='Backward CNN', linewidth=1.5, markersize=2.5)
            # plt.xticks(x, labels, fontsize=10)
            # plt.yticks(fontsize=10)
            # plt.xlabel('Time', fontsize=10, weight='bold')
            # plt.ylabel('Dice', fontsize=10, weight='bold')
            # # plt.legend(loc='lower right', frameon=False, fontsize=10)
            # plt.savefig(osp.join(self.save_dir, self.cur_patient + '_acc.pdf'))
        else:
            self.cnn_res = self.trainer.val_patient(self.img4d, self.m0, self.mk, self.timesfwd, self.timesbwd, self.ff, self.bf, cnn=True)
            self.flow_res = self.trainer.val_patient(self.img4d, self.m0, self.mk, self.timesfwd, self.timesbwd, self.ff, self.bf, cnn=False)

        self.update_report(idx)
        self.save_imgs()
        self.save_nifti()
        self.pbar.update(1)

    def update_report(self, idx):
        dictrow = {'Patient': self.cur_patient}
        for k, v in self.metrics(cnn=True).items():
            dictrow['cnn_' + k] = v

        for k, v in self.metrics(cnn=False).items():
            dictrow['flow_' + k] = v

        dfrow = pd.DataFrame(dictrow, index=[idx])
        self.report = pd.concat([dfrow, self.report])

    def finetuned_data(self):
        for data in self.loader:
            if data[0][0] == self.finetuned_patient:
                break
        self.prepare_data(data)

    def prepare_data(self, data):
        self.cur_patient = data[0][0]
        self.img4d = data[1].to(self.device)
        self.m0 = data[2].to(self.device)
        self.mk = data[3].to(self.device)
        self.ff = data[7].to(self.device)
        self.bf = data[8].to(self.device)
        self.timesfwd = data[5]
        self.timesbwd = data[6]

        self.pbar.set_postfix_str(f'P: {self.cur_patient}')

        # Load whole cardiac cycle masks for test dataset
        if self.is_testset:
            self.masks = data[4].to(self.device)
            self.timesfwd, self.timesbwd = self.dset.create_timeline(
                self.timesfwd[0], self.timesbwd[0], self.masks.shape[-1])
        else:
            self.masks = None

    def metrics(self, cnn):
        metrics = self.cnn_res[0] if cnn else self.flow_res[0]
        return metrics

    def estimated_masks(self, cnn, fwd):
        if self.is_testset:
            i = 3 if fwd else 4
            masks = self.cnn_res[i] if cnn else self.flow_res[i]
        else:
            i = 1 if fwd else 2
            masks = self.cnn_res[i] if cnn else self.flow_res[i]
        return masks

    def save_imgs(self):
        if not self.P['save_imgs']:
            return

        # Create save dirs
        patient_dir = path_utils.create_sub_dir(self.save_dir, self.cur_patient)
        fwd_dir = path_utils.create_sub_dir(patient_dir, 'fwd')
        bwd_dir = path_utils.create_sub_dir(patient_dir, 'bwd')

        # Get results
        cnn_fms = self.estimated_masks(cnn=True, fwd=True)
        flow_fms = self.estimated_masks(cnn=False, fwd=True)
        cnn_bms = self.estimated_masks(cnn=True, fwd=False)
        flow_bms = self.estimated_masks(cnn=False, fwd=False)

        timesteps = cnn_fms.shape[0]
        for t in range(timesteps):
            img3d = self.img_transf(self.img4d[..., self.timesfwd[t].item()].squeeze())
            mt = self.timesfwd[t].item() if self.is_testset else t

            cnn_fm = self.est_masks_transf(cnn_fms[mt].squeeze())
            flow_fm = self.est_masks_transf(flow_fms[mt].squeeze())
            cnn_bm = self.est_masks_transf(cnn_bms[mt].squeeze())
            flow_bm = self.est_masks_transf(flow_bms[mt].squeeze())

            # mask_cnn_fwd = self.img_transf(cnn_fms[mt].squeeze())
            # mask_cnn_bwd = self.img_transf(cnn_bms[mt].squeeze())
            # mask_flow_fwd = self.img_transf(flow_fms[mt].squeeze())
            # mask_flow_bwd = self.img_transf(flow_bms[mt].squeeze())

            gtm = self.label_mask(t, mt, timesteps)
            if gtm is not None:
                plots.save_img_masks(img3d, [gtm, cnn_fm, flow_fm], f'im_fwd_{self.timesfwd[t].item()}', fwd_dir, th=0.5,
                                     alphas=[0.3, 1.0, 1.0], colors=[[1, 0.7, 0], [0, 1, 0], [0, 0, 1]])

                plots.save_img_masks(img3d, [gtm, cnn_bm, flow_bm], f'im_bwd_{self.timesfwd[t].item()}', bwd_dir, th=0.5,
                                     alphas=[0.3, 1.0, 1.0], colors=[[1, 0.7, 0], [0, 1, 0], [0, 0, 1]])
            else:
                plots.save_img_masks(img3d, [cnn_fm, flow_fm], f'im_fwd_{self.timesfwd[t].item()}', fwd_dir, th=0.5,
                                     alphas=[1.0, 1.0], colors=[[0, 1, 0], [0, 0, 1]])
                plots.save_img_masks(img3d, [cnn_bm, flow_bm], f'im_bwd_{self.timesfwd[t].item()}', bwd_dir, th=0.5,
                                     alphas=[1.0, 1.0], colors=[[0, 1, 0], [0, 0, 1]])

            # # Save img
            # dir0 = plots.createSubDirectory(patient_dir, 'imgs')
            # plots.save_slices(img3d, f'im_{self.timesfwd[t].item()}', dir0)
            # plots.save_single_zslices(img3d, dir0, f'slices_{self.timesfwd[t].item()}')

            # # Save masks
            # dir1 = plots.createSubDirectory(patient_dir, 'masks_cnn_fwd')
            # plots.save_slices(mask_cnn_fwd, f'mask_cnn_fwd_{self.timesfwd[t].item()}', dir1)
            # plots.save_single_zslices(mask_cnn_fwd, dir1, f'slices_{self.timesfwd[t].item()}')

            # dir2 = plots.createSubDirectory(patient_dir, 'masks_cnn_bwd')
            # plots.save_slices(mask_cnn_bwd, f'mask_cnn_bwd_{self.timesfwd[t].item()}', dir2)
            # plots.save_single_zslices(mask_cnn_bwd, dir2, f'slices_{self.timesfwd[t].item()}')

            # dir3 = plots.createSubDirectory(patient_dir, 'masks_flow_fwd')
            # plots.save_slices(mask_flow_fwd, f'mask_flow_fwd_{self.timesfwd[t].item()}', dir3)
            # plots.save_single_zslices(mask_flow_fwd, dir3, f'slices_{self.timesfwd[t].item()}')

            # dir4 = plots.createSubDirectory(patient_dir, 'masks_flow_bwd')
            # plots.save_slices(mask_flow_bwd, f'mask_flow_bwd_{self.timesfwd[t].item()}', dir4)
            # plots.save_single_zslices(mask_flow_bwd, dir4, f'slices_{self.timesfwd[t].item()}')

    def label_mask(self, t, mt, timesteps):
        gtm = None
        if self.is_testset:
            gtm = self.gt_masks_transf(self.masks[..., mt].squeeze())
        else:
            if t == 0:
                gtm = self.gt_masks_transf(self.m0.squeeze())
            elif t == timesteps - 1:
                gtm = self.gt_masks_transf(self.mk.squeeze())
        return gtm

    def save_nifti(self):
        if not self.P['save_nifti']:
            return

        # Create save dirs
        patient_dir = path_utils.create_sub_dir(self.save_dir, self.cur_patient)
        fwd_dir = path_utils.create_sub_dir(patient_dir, 'fwd')
        bwd_dir = path_utils.create_sub_dir(patient_dir, 'bwd')

        # Get results
        cnn_fms = self.estimated_masks(cnn=True, fwd=True)
        cnn_bms = self.estimated_masks(cnn=True, fwd=False)

        timesteps = cnn_fms.shape[0]
        for t in range(timesteps):
            st = t if self.is_testset else self.timesfwd[t].item()

            hdr = self.loader.dataset.header(0)
            plots.save_nifti_mask(cnn_fms[t], hdr, fwd_dir, f'm_fwd_{st}.nii')
            plots.save_nifti_mask(cnn_bms[t], hdr, bwd_dir, f'm_bwd_{st}.nii')

    def complete_cardiac_cycle(self, config):
        transforms = T6.Compose([T6.ToTensor()])
        self.dset = SingleVentricleDataset(config, DatasetMode.FULL, LoadFlowMode.WHOLE_CYCLE, full_transforms=transforms)
        self.loader = DataLoader(self.dset, batch_size=1, shuffle=False, num_workers=self.P['workers'], collate_fn=collate_fn_batch)

        idx, found = self.dset.index_for_patient(self.P['patient_name'])
        if not found:
            self.logger.info('Patient not found %s' % self.P['patient_name'])
            sys.exit()

        mts, mtts = self.trainer.cardiac_cycle_propagation(self.dset, self.loader)
        mts = mts.squeeze(0)
        mts = torch.permute(mts, (4, 0, 1, 2, 3))
        mtts = mtts.squeeze(0)
        mtts = torch.permute(mtts, (4, 0, 1, 2, 3))

        cut_shape = self.dset.cutted_shape(0)
        orig_shape = self.dset.orig_shape(0)
        z_min, z_max, y_min, y_max, x_min, x_max = self.dset.cut_bounds(0)

        mts = F.interpolate(mts, size=cut_shape, align_corners=True, mode='trilinear')
        mtts = F.interpolate(mtts, size=cut_shape, align_corners=True, mode='trilinear')

        orig_mts = torch.zeros(size=(*mts.shape[:2], *orig_shape), dtype=mts.dtype, device=self.device)
        orig_mts[..., z_min:z_max + 1, y_min:y_max + 1, x_min:x_max + 1] = mts
        orig_mtts = torch.zeros_like(orig_mts)
        orig_mtts[..., z_min:z_max + 1, y_min:y_max + 1, x_min:x_max + 1] = mtts

        # Create save dirs
        patient_dir = path_utils.create_sub_dir(self.save_dir, self.cur_patient)
        fwd_dir = path_utils.create_sub_dir(patient_dir, 'ccc/fwd')
        bwd_dir = path_utils.create_sub_dir(patient_dir, 'ccc/bwd')

        ts = mts.shape[0]
        mts = mts.squeeze()
        mtts = mtts.squeeze()
        mts = torch.where(mts > 0.5, 1.0, 0.0)
        mtts = torch.where(mtts > 0.5, 1.0, 0.0)

        hdr = self.dset.header(0)
        for t in range(ts):
            plots.save_slices(mts[t], f'mask_t{t}.png', fwd_dir)
            plots.save_slices(mtts[t], f'mask_t{t}.png', bwd_dir)

            plots.save_nifti_mask(mts[t], hdr, fwd_dir, f'mask_{t}.nii')
            plots.save_nifti_mask(mtts[t], hdr, bwd_dir, f'mask_{t}.nii')
