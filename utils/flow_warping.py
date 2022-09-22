import torch
import configparser
import sys
from tqdm import tqdm
import os.path as osp
import numpy as np
from torch.utils.data import DataLoader
import csv
from monai.metrics.meandice import compute_meandice
from monai.metrics.hausdorff_distance import compute_hausdorff_distance
import transforms.senary_transforms as T6
import transforms.unary_transforms as T1
import plots
from collate import collate_fn
import nibabel as nib

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
from cnn.dataset import *
from cnn.warp import WarpCNN


if __name__ == "__main__":
    plots.printConsoleOutput_Header('Identity warping')

    config = configparser.ConfigParser()
    config.read('parser/configFlowWarping.ini')
    save_imgs = config.getboolean('PARAMETERS', 'SAVE_IMGS')
    save_size = (config.getint('PARAMETERS', 'SAVE_NZ'),
                 config.getint('PARAMETERS', 'SAVE_NY'),
                 config.getint('PARAMETERS', 'SAVE_NX'))
    num_workers = config.getint('PARAMETERS', 'NUM_WORKERS')
    save_nifti = config.getboolean('PARAMETERS', 'SAVE_NIFTI')
    save_slices = config.getboolean('PARAMETERS', 'SAVE_IMGS_SLICES')
    hd_per = config.getint('PARAMETERS', 'HAUSDORFF_PERCENTILE')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transforms = T6.Compose([
        # T6.ElasticDeformation(1.0, (1, 5), 5, 'nearest', False, 'yx', order=1),
        # T6.RandomRotate(1.0, (0, 360), (0, 360), (0, 360), 'border'),
        # T6.OneOf([
        #     T6.RandomDepthFlip(1.0),
        #     T6.RandomHorizontalFlip(1.0),
        #     T6.RandomVerticalFlip(1.0)
        # ]),
        # T6.ElasticDeformation(1.0, (2.0, 2.0), 8, 'nearest', False, 'yx'),
        T6.ToTensor()
    ])

    try:
        train_ds = SingleVentricleDataset(config, DatasetMode.TRAIN, LoadFlowMode.ED_ES, full_transforms=transforms)
        val_ds = SingleVentricleDataset(config, DatasetMode.VAL, LoadFlowMode.ED_ES, full_transforms=transforms)
        test_ds = SingleVentricleDataset(config, DatasetMode.TEST, LoadFlowMode.ED_ES, full_transforms=transforms)

        train_loader = DataLoader(train_ds, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
        loaders = [train_loader, val_loader, test_loader]
        pbar = tqdm(total=len(train_ds) + len(val_ds) + len(test_ds))
    except FileNotFoundError:
        full_ds = SingleVentricleDataset(config, DatasetMode.FULL, LoadFlowMode.ED_ES, full_transforms=transforms)
        full_loader = DataLoader(full_ds, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
        loaders = [full_loader]
        pbar = tqdm(total=len(full_ds))

    save_dir = plots.createSaveDirectory(config.get('DATA', 'OUTPUT_PATH'), 'Warping')
    plots.save_config(config, save_dir, filename='config.ini')

    csv_file = open(osp.join(save_dir, 'metrics.csv'), 'w')
    writer = csv.writer(csv_file)
    writer.writerow(['Patient', 'dc_0', 'dc_k', 'hd_0', 'hd_k'])

    rre_transf = T1.Compose([T1.Resize(save_size), T1.Round(th=0.5), T1.Erode()])
    rr_transf = T1.Compose([T1.Resize(save_size), T1.Round(th=0.5)])
    r_transf = T1.Compose([T1.Resize(save_size)])

    metrics = {'dc_0': [], 'dc_k': [], 'hd_0': [], 'hd_k': []}

    for loader in loaders:
        for i, (pnames, imgs4d, m0, mk, _, times_fwd, times_bwd, ff, bf) in enumerate(loader):
            imgs4d = imgs4d.to(device)
            m0 = m0.to(device)
            mk = mk.to(device)
            ff = ff.to(device)
            bf = bf.to(device)

            BS, CH, NZ, NY, NX, NT = imgs4d.shape
            warp = WarpCNN(config, NZ, NY, NX)
            # batch_indices = torch.arange(BS)
            out = {'mt': [m0], 'mtt': [mk]}
            timesteps = ff.shape[-1]

            for t in range(timesteps):
                pbar.set_postfix_str(f'P: {pnames[0]}, S: {t+1}/{timesteps}')

                # Forward mask propagation m0 -> mk
                out['mt'].append(warp(out['mt'][-1], ff[..., t]))

                # Backward mask propagation mk -> m0
                out['mtt'].append(warp(out['mtt'][-1], bf[..., t]))

            assert(len(out['mt']) == len(out['mtt']))

            out['mtt'].reverse()
            row = [pnames[0]]

            m0tt = torch.where(out['mtt'][0] > 0.5, 1.0, 0.0)
            mkt = torch.where(out['mt'][-1] > 0.5, 1.0, 0.0)

            metrics['dc_0'].append(compute_meandice(m0tt, out['mt'][0]).mean().item())
            metrics['dc_k'].append(compute_meandice(mkt, out['mtt'][-1]).mean().item())
            metrics['hd_0'].append(compute_hausdorff_distance(m0tt, out['mt'][0], percentile=hd_per).mean().item())
            metrics['hd_k'].append(compute_hausdorff_distance(mkt, out['mtt'][-1], percentile=hd_per).mean().item())
            row.append('{:.3f}'.format(metrics['dc_0'][-1]))
            row.append('{:.3f}'.format(metrics['dc_k'][-1]))
            row.append('{:.3f}'.format(metrics['hd_0'][-1]))
            row.append('{:.3f}'.format(metrics['hd_k'][-1]))
            writer.writerow(row)

            if save_imgs:
                patient_dir = plots.createSubDirectory(save_dir, pnames[0])
                # mt_dir = plots.createSubDirectory(patient_dir, 'mt')
                # mt_slices_dir = plots.createSubDirectory(mt_dir, 'zslices')
                # mtt_dir = plots.createSubDirectory(patient_dir, 'mtt')
                # mtt_slices_dir = plots.createSubDirectory(mtt_dir, 'zslices')

                for t in range(len(out['mt'])):
                    img3d = r_transf(imgs4d[..., times_fwd[t]].squeeze())
                    mt = rre_transf(out['mt'][t].squeeze())
                    mtt = rre_transf(out['mtt'][t].squeeze())

                    if t == 0:
                        plots.save_img_masks(img3d, [rr_transf(m0.squeeze()), mtt], 'im_m0_m0tt', patient_dir,
                                             th=0.5, alphas=[0.2, 1.0], colors=[[1, 0.7, 0], [0, 0, 1]])
                        if save_slices:
                            plots.save_img_masks_slices(img3d, [rr_transf(m0.squeeze()), mtt], patient_dir, f'{times_fwd[t].item()}_0tt',
                                                        0.5, [0.2, 1.0], [[1, 0.7, 0], [0, 0, 1]])
                    elif t == len(out['mt']) - 1:
                        plots.save_img_masks(img3d, [rr_transf(mk.squeeze()), mt], 'mk_mkt', patient_dir,
                                             th=0.5, alphas=[0.2, 1.0], colors=[[0, 0.7, 1], [0, 1, 0]])
                        if save_slices:
                            plots.save_img_masks_slices(img3d, [rr_transf(mk.squeeze()), mt], patient_dir, f'{times_fwd[t].item()}_kt',
                                                        0.5, [0.2, 1.0], [[0, 0.7, 1], [0, 1, 0]])

                    plots.save_img_masks(img3d, [mt, mtt], f'im_t_{times_fwd[t]}', patient_dir,
                                         th=0.5, alphas=[1.0, 1.0], colors=[[0, 1, 0], [0, 0, 1]])
                    if save_slices:
                        plots.save_img_masks_slices(img3d, [mt, mtt], patient_dir, str(times_fwd[t].item()), 0.5,
                                                    [1.0, 1.0], [[0, 1, 0], [0, 0, 1]])

                    if save_nifti:
                        hdr = loader.dataset.header(i)
                        plots.save_nifti_mask(out['mt'][t], hdr, patient_dir, f'mt_{times_fwd[t].item()}.nii')
                        plots.save_nifti_mask(out['mtt'][t], hdr, patient_dir, f'mtt_{times_fwd[t].item()}.nii')

                    # mt = mt_mtt_posp(out['mt'][t].squeeze())
                    # mtt = mt_mtt_posp(out['mtt'][t].squeeze())

                    # plots.save_slices(mt, f'mt_{times_fwd[t][batch_indices].item()}.png', mt_dir)
                    # plots.save_single_zslices(mt, mt_slices_dir, str(times_fwd[t][batch_indices].item()))

                    # plots.save_slices(mtt, f'mtt_{times_fwd[t][batch_indices].item()}.png', mtt_dir)
                    # plots.save_single_zslices(mtt, mtt_slices_dir, str(times_fwd[t][batch_indices].item()))

            pbar.update(1)
    writer.writerow(['mean', '{:.3f}'.format(np.array(metrics['dc_0']).mean()),
                     '{:.3f}'.format(np.array(metrics['dc_k']).mean()),
                     '{:.3f}'.format(np.array(metrics['hd_0']).mean()),
                     '{:.3f}'.format(np.array(metrics['hd_k']).mean())])
    csv_file.close()
