import torch
from configparser import ConfigParser
import sys
from tqdm import tqdm
import os.path as osp
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pandas as pd
from monai.metrics.meandice import compute_dice
from monai.metrics.hausdorff_distance import compute_hausdorff_distance
from tabulate import tabulate

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
from cnn.dataset import *
from cnn.warp import WarpCNN
from utilities import path_utils, stuff
import utilities.transforms.senary_transforms as T6
import utilities.transforms.unary_transforms as T1
from utilities.plots import plots
from utilities.collate import collate_fn
from utilities.parser_conversions import str_to_tuple
from segmentation.transforms import one_hot
from datasets.preprocessing import viz


def propagate(m0, mk, ff, bf, n_classes, patient, config, pbar):
    nz, ny, nx = m0.shape[2:]
    timesteps = ff.shape[-1]
    warp = WarpCNN(config, nz, ny, nx)
    out = {'mt': [m0], 'mtt': [mk]}

    for t in range(timesteps):
        pbar.set_postfix_str(f'P: {patient}, S: {t+1}/{timesteps}')

        # Forward mask propagation m0 -> mk
        out['mt'].append(warp(out['mt'][-1], ff[..., t]))

        # Backward mask propagation mk -> m0
        out['mtt'].append(warp(out['mtt'][-1], bf[..., t]))

    assert(len(out['mt']) == len(out['mtt']))
    out['mtt'].reverse()
    mts = one_hot(torch.cat(out['mt'], dim=0), n_classes + 1, True)    # 0 -> k
    mtts = one_hot(torch.cat(out['mtt'], dim=0), n_classes + 1, True)  # 0 -> k
    return mts, mtts


def compute_metrics(mts, mtts, patient):
    dc_0 = compute_dice(mtts[0].unsqueeze(0), mts[0].unsqueeze(0), include_background=False).mean().item()
    dc_k = compute_dice(mts[-1].unsqueeze(0), mtts[-1].unsqueeze(0), include_background=False).mean().item()
    hd_0 = compute_hausdorff_distance(mtts[0].unsqueeze(0), mts[0].unsqueeze(0)).mean().item()
    hd_k = compute_hausdorff_distance(mts[-1].unsqueeze(0), mtts[-1].unsqueeze(0)).mean().item()
    return pd.DataFrame({'patient': patient, 'dc_0': dc_0, 'dc_k': dc_k, 'hd_0': hd_0, 'hd_k': hd_k}, index=[0])


def save_images(patient, imgs4d, mts, mtts, times_fwd, n_classes, save_dir):
    patient_dir = path_utils.create_sub_dir(save_dir, patient)
    stuff.seeding(280894)
    fwd_colors = plots.random_colors(n_classes)
    stuff.seeding(190319)
    bwd_colors = plots.random_colors(n_classes)

    imgs4d = torch.clip(imgs4d, 0.0, 1.0)
    for t in range(mts.shape[0]):
        img3d = imgs4d[times_fwd[t], 0]

        if t == 0:
            labels = [mts[t], T1.ErodeOneHot()(mtts[t]), T1.ErodeOneHot()(mts[t])]
            alphas = [0.2, 1.0, 1.0]
            colors = [fwd_colors, bwd_colors, fwd_colors]
        elif t == mts.shape[0] - 1:
            labels = [mtts[t], T1.ErodeOneHot()(mts[t]), T1.ErodeOneHot()(mtts[t])]
            alphas = [0.2, 1.0, 1.0]
            colors = [bwd_colors, fwd_colors, bwd_colors]
        else:
            labels = [T1.ErodeOneHot()(mts[t]), T1.ErodeOneHot()(mtts[t])]
            alphas = [1.0, 1.0]
            colors = [fwd_colors, bwd_colors]

        plots.save_img_masks_one_hot(img3d, labels, f'im_t_{times_fwd[t]}', patient_dir, alphas, colors)
        # if save_slices:
        #     plots.save_img_masks_slices(img3d, [mt, mtt], patient_dir, str(times_fwd[t].item()), 0.5,
        #                                 [1.0, 1.0], [blue, red])

        # if save_nifti:
        #     hdr = loader.dataset.header(i)
        #     plots.save_nifti_mask(out['mt'][t], hdr, patient_dir, f'mt_{times_fwd[t].item()}.nii')
        #     plots.save_nifti_mask(out['mtt'][t], hdr, patient_dir, f'mtt_{times_fwd[t].item()}.nii')

        # mt = mt_mtt_posp(out['mt'][t].squeeze())
        # mtt = mt_mtt_posp(out['mtt'][t].squeeze())

        # plots.save_slices(mt, f'mt_{times_fwd[t][batch_indices].item()}.png', mt_dir)
        # plots.save_single_zslices(mt, mt_slices_dir, str(times_fwd[t][batch_indices].item()))

        # plots.save_slices(mtt, f'mtt_{times_fwd[t][batch_indices].item()}.png', mtt_dir)
        # plots.save_single_zslices(mtt, mtt_slices_dir, str(times_fwd[t][batch_indices].item()))
    return patient_dir


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = ConfigParser()
    config.read('parser/flow_warping.ini')
    params = config['PARAMETERS']
    data = config['DATA']

    save_imgs = params.getboolean('save_imgs')
    save_gif = params.getboolean('save_gif')
    save_size = str_to_tuple(params['save_size'], int)
    save_nifti = params.getboolean('save_nifti')
    save_slices = params.getboolean('save_imgs_slices')

    transforms = T6.Compose([T6.ToTensor()])
    if params['dataset_mode'] == 'split':
        train_ds = SingleVentricleDataset(config, DatasetMode.TRAIN, LoadFlowMode.ED_ES, full_transforms=transforms)
        val_ds = SingleVentricleDataset(config, DatasetMode.VAL, LoadFlowMode.ED_ES, full_transforms=transforms)
        test_ds = SingleVentricleDataset(config, DatasetMode.TEST, LoadFlowMode.ED_ES, full_transforms=transforms)
        train_loader = DataLoader(train_ds, batch_size=1, shuffle=False, num_workers=8, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=8, collate_fn=collate_fn)
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=3, collate_fn=collate_fn)
        loaders = [train_loader, val_loader, test_loader]
        pbar = tqdm(total=len(train_ds) + len(val_ds) + len(test_ds))
        n_classes = train_ds.num_classes()
        dset_name = train_ds.name()
    elif params['dataset_mode'] == 'full':
        full_ds = SingleVentricleDataset(config, DatasetMode.FULL, LoadFlowMode.ED_ES, full_transforms=transforms)
        full_loader = DataLoader(full_ds, batch_size=1, shuffle=False, num_workers=8, collate_fn=collate_fn)
        loaders = [full_loader]
        pbar = tqdm(total=len(full_ds))
        n_classes = full_ds.num_classes()
        dset_name = full_ds.name()
    else:
        raise ValueError('Unknown dataset_mode')

    save_dir = path_utils.create_save_dir(data['output_path'], 'warp')
    stuff.save_config(config, save_dir, filename='config.ini')
    logger = stuff.create_logger(save_dir)
    logger.info('Save dir: ' + save_dir)
    logger.info('Dataset: ' + dset_name)
    logger.info('n_classes: {}'.format(n_classes))

    report = pd.DataFrame()

    for loader in loaders:
        for i, (pnames, imgs4d, m0, mk, _, times_fwd, times_bwd, ff, bf) in enumerate(loader):
            imgs4d = imgs4d.to(device)
            m0 = one_hot(m0, n_classes + 1).to(device)
            mk = one_hot(mk, n_classes + 1).to(device)
            ff = ff.to(device)
            bf = bf.to(device)

            mts, mtts = propagate(m0, mk, ff, bf, n_classes, pnames[0], config, pbar)
            report = pd.concat([report, compute_metrics(mts, mtts, pnames[0])], axis=0, ignore_index=True)

            if save_imgs:
                imgs4d = F.interpolate(imgs4d.squeeze(0).permute(4, 0, 1, 2, 3), size=save_size, mode='trilinear')
                mts = F.interpolate(mts, size=save_size, mode='nearest')
                mtts = F.interpolate(mtts, size=save_size, mode='nearest')
                patient_dir = save_images(pnames[0], imgs4d, mts, mtts, times_fwd, n_classes, save_dir)

                if save_gif:
                    viz.save_gif(patient_dir, 150)
            pbar.update(1)

    report.loc[len(report)] = ['mean',] + report.mean(numeric_only=True).to_list()
    report = report.round(3)
    logger.info(tabulate(report, headers='keys', tablefmt='psql'))
    report.to_excel(osp.join(save_dir, 'metrics.xlsx'), index=False)
