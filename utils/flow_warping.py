import torch
import configparser
import sys
from tqdm import tqdm
import os.path as osp
import numpy as np
from torch.utils.data import DataLoader
import csv
from monai.metrics.meandice import compute_meandice
import transforms.senary_transforms as T6
import transforms.unary_transforms as T1
import plots
from collate import collate_fn

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
from cnn.dataset import *
from cnn.warp import WarpCNN


if __name__ == "__main__":
    save_imgs = True
    save_size = (16, 200, 200)

    plots.printConsoleOutput_Header('Identity warping')

    config = configparser.ConfigParser()
    config.read('parser/configCNNTrain.ini')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transforms = T6.Compose([T6.ToTensor()])
    train_ds = SingleVentricleDataset(config, DatasetMode.TRAIN, LoadFlowMode.ED_ES, full_transforms=transforms)
    val_ds = SingleVentricleDataset(config, DatasetMode.VAL, LoadFlowMode.ED_ES, full_transforms=transforms)
    test_ds = SingleVentricleDataset(config, DatasetMode.TEST, LoadFlowMode.ED_ES, full_transforms=transforms)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False, num_workers=8, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=8, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=8, collate_fn=collate_fn)

    save_dir = plots.createSaveDirectory(config.get('DATA', 'OUTPUT_PATH'), 'Warping')
    plots.save_config(config, save_dir, filename='config.ini')

    csv_file = open(osp.join(save_dir, 'accuracy.csv'), 'w')
    writer = csv.writer(csv_file)
    writer.writerow(['Patient', 'dc_0', 'dc_k'])
    pbar = tqdm(total=len(train_ds) + len(val_ds) + len(test_ds))

    rre_transf = T1.Compose([T1.Resize(save_size), T1.Round(th=0.5), T1.Erode()])
    rr_transf = T1.Compose([T1.Resize(save_size), T1.Round(th=0.5)])
    r_transf = T1.Compose([T1.Resize(save_size)])

    acc = {'0': [], 'k': []}

    for loader in [train_loader, val_loader, test_loader]:
        for (pnames, imgs4d, m0s, mks, _, times_fwd, times_bwd, ff, bf, _) in loader:
            imgs4d = imgs4d.to(device)
            m0s = m0s.to(device)
            mks = mks.to(device)
            ff = ff.to(device)
            bf = bf.to(device)

            BS, CH, NZ, NY, NX, NT = imgs4d.shape
            warp = WarpCNN(config, NZ, NY, NX)
            # batch_indices = torch.arange(BS)
            out = {'mt': [m0s], 'mtt': [mks]}
            timesteps = ff.shape[-1]

            for t in range(timesteps):
                pbar.set_postfix_str(f'P: {pnames[0]}, S: {t+1}/{timesteps}')

                # Forward mask propagation mi -> mf
                out['mt'].append(warp(out['mt'][-1], ff[..., t]))

                # Backward mask propagation mf -> mi
                out['mtt'].append(warp(out['mtt'][-1], bf[..., t]))

            assert(len(out['mt']) == len(out['mtt']))

            out['mtt'].reverse()
            row = [pnames[0]]
            acc['0'].append(compute_meandice(out['mtt'][0], out['mt'][0]).mean().item())
            acc['k'].append(compute_meandice(out['mt'][-1], out['mtt'][-1]).mean().item())
            row.append('{:.3f}'.format(acc['0'][-1]))
            row.append('{:.3f}'.format(acc['k'][-1]))
            writer.writerow(row)

            if save_imgs:
                patient_dir = plots.createSubDirectory(save_dir, pnames[0])
                # mt_dir = plots.createSubDirectory(patient_dir, 'mt')
                # mt_slices_dir = plots.createSubDirectory(mt_dir, 'zslices')
                # mtt_dir = plots.createSubDirectory(patient_dir, 'mtt')
                # mtt_slices_dir = plots.createSubDirectory(mtt_dir, 'zslices')

                for t in range(len(out['mt'])):
                    img3d = r_transf(imgs4d[..., times_fwd[t].item()].squeeze())
                    mt = rre_transf(out['mt'][t].squeeze())
                    mtt = rre_transf(out['mtt'][t].squeeze())

                    if t == 0:
                        plots.save_img_masks(img3d, [rr_transf(m0s.squeeze()), mtt], 'im_m0_m0tt', patient_dir,
                                             th=0.5, alphas=[0.3, 1.0], colors=[[1, 0.7, 0], [0, 0, 1]])
                    elif t == len(out['mt']) - 1:
                        plots.save_img_masks(img3d, [rr_transf(mks.squeeze()), mt], 'mk_mkt', patient_dir,
                                             th=0.5, alphas=[0.3, 1.0], colors=[[1, 0.7, 0], [0, 1, 0]])

                    plots.save_img_masks(img3d,
                                         [mt, mtt],
                                         f'im_t_{times_fwd[t].item()}', patient_dir,
                                         th=0.5, alphas=[1.0, 1.0], colors=[[0, 1, 0], [0, 0, 1]])

                    # mt = mt_mtt_posp(out['mt'][t].squeeze())
                    # mtt = mt_mtt_posp(out['mtt'][t].squeeze())

                    # plots.save_slices(mt, f'mt_{times_fwd[t][batch_indices].item()}.png', mt_dir)
                    # plots.save_single_zslices(mt, mt_slices_dir, str(times_fwd[t][batch_indices].item()))

                    # plots.save_slices(mtt, f'mtt_{times_fwd[t][batch_indices].item()}.png', mtt_dir)
                    # plots.save_single_zslices(mtt, mtt_slices_dir, str(times_fwd[t][batch_indices].item()))

            pbar.update(1)
    writer.writerow(['mean', '{:.3f}'.format(np.array(acc['0']).mean()), '{:.3f}'.format(np.array(acc['k']).mean())])
    csv_file.close()
