import torch
import configparser
import sys
from tqdm import tqdm
import os.path as osp
import csv

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
from utils import plots
from cnn.dataset import SingleVentricleDataset, DatasetMode
import utils.transforms as T
from cnn.loss import loss_func_three
from cnn.warp import Warp
from cnn.metrics import dc


def compute_dc(x: torch.Tensor, y: torch.Tensor):
    transf = T.ComposeUnary([T.ToArray(), T.Round(th=0.5)])
    x = transf(x)
    y = transf(y)
    return dc(x, y)


if __name__ == "__main__":
    plots.printConsoleOutput_Header('Identity warping')

    config = configparser.ConfigParser()
    config.read('parser/configCNNTrain.ini')
    cuda_availabe = config.get('DEVICE', 'CUDA_AVAILABLE')
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_transforms = T.ComposeUnary([T.ToTensor()])
    mask_transforms = T.ComposeUnary([T.ToTensor()])

    train_ds = SingleVentricleDataset(config, DatasetMode.TRAIN, load_flow=True,
                                      img4d_transforms=data_transforms,
                                      mask_transforms=mask_transforms,
                                      img4d_mask_transforms=None,
                                      flow_transforms=None)

    val_ds = SingleVentricleDataset(config, DatasetMode.VAL, load_flow=True,
                                    img4d_transforms=data_transforms,
                                    mask_transforms=mask_transforms,
                                    img4d_mask_transforms=None,
                                    flow_transforms=None)

    save_dir = plots.createSaveDirectory(config.get('DATA', 'OUTPUT_PATH'), 'Warping')

    # save config file to save directory
    conifg_output = osp.join(save_dir, 'config.ini')
    with open(conifg_output, 'w') as config_file:
        config.write(config_file)

    csv_file = open(osp.join(save_dir, 'diff.csv'), 'w')
    writer = csv.writer(csv_file)
    writer.writerow(['Patient', 'L1-OF', 'L2-OF', 'L3-OF', 'LT-OF', 'dc_0', 'dc_k'])
    pbar = tqdm(total=len(train_ds) + len(val_ds))
    mask_posp = T.ComposeUnary([T.ToArray(), T.Normalize(), T.Round(th=0.5), T.Erode(), T.ToTensor()])

    for ds in [train_ds, val_ds]:
        for (pname, img4d, m0, mk, init_ts, final_ts, ff, bf) in ds:
            ff = ff.to(DEVICE)
            bf = bf.to(DEVICE)
            NZ, NY, NX, NT = img4d.shape
            warp = Warp(config, NZ, NY, NX)
            mts = [m0.to(DEVICE)]
            mtts = [mk.to(DEVICE)]
            ts = ff.shape[0]
            for t in range(ts):
                pbar.set_postfix_str(f'P: {pname}, S: {t+1}/{ts}')

                # Forward mask propagation m0 -> mk
                mt = warp(mts[-1], ff[t, :, :, :, :])
                # print(torch.allclose(mts[-1], mt))
                mts.append(mt)

                # Backward mask propagation mk -> m0
                mtt = warp(mtts[-1], bf[t, :, :, :, :])
                mtts.append(mtt)

            assert(len(mts) == len(mtts))

            mtts.reverse()
            loss, l1, l2, l3 = loss_func_three(mts, mtts)

            row = [pname]
            row.append('{:.2f}'.format(l1.item()))
            row.append('{:.2f}'.format(l2.item()))
            row.append('{:.2f}'.format(l3.item()))
            row.append('{:.2f}'.format(loss.item()))
            row.append('{:.3f}'.format(compute_dc(m0, mtts[0])))
            row.append('{:.3f}'.format(compute_dc(mk, mts[-1])))
            writer.writerow(row)

            if True:
                patient_dir = plots.createSubDirectory(save_dir, pname)
                fwd_dir = plots.createSubDirectory(patient_dir, 'fwd')
                bwd_dir = plots.createSubDirectory(patient_dir, 'bwd')

                for i in range(len(mts)):
                    img3d = img4d[:, :, :, init_ts + i]
                    mt = mask_posp(mts[i])
                    mtt = mask_posp(mtts[i])

                    if i == 0:
                        plots.save_img_masks(img3d, [m0, mtt], 'im_m0_m0tt', fwd_dir, th=0.5, alphas=[0.3, 1.0], colors=[[1, 0.7, 0], [0, 1, 0]])
                        plots.save_img_masks_slices(img3d, [m0, mtt], fwd_dir, 'im_m0_m0tt_slices', th=0.5,
                                                    alphas=[0.3, 1.0], colors=[[1, 0.7, 0], [0, 1, 0]])
                    elif i == len(mtts) - 1:
                        plots.save_img_masks(img3d, [mk, mt], 'mk_mkt', bwd_dir, th=0.5, alphas=[0.3, 1.0], colors=[[1, 0.7, 0], [0, 1, 0]])
                        plots.save_img_masks_slices(img3d, [mk, mt], bwd_dir, 'mk_mkt_slices', th=0.5,
                                                    alphas=[0.3, 1.0], colors=[[1, 0.7, 0], [0, 1, 0]])

                    plots.save_img_masks(img3d, [mt], f'im_t_{init_ts + i}', fwd_dir, th=0.5, alphas=[1.0], colors=[[0, 1, 0]])
                    plots.save_img_masks_slices(img3d, [mt], fwd_dir, f'im_t_{init_ts + i}', th=0.5, alphas=[1.0], colors=[[0, 1, 0]])

                    plots.save_img_masks(img3d, [mtt], f'im_tt_{init_ts + i}', bwd_dir, th=0.5, alphas=[1.0], colors=[[0, 1, 0]])
                    plots.save_img_masks_slices(img3d, [mtt], bwd_dir, f'im_tt_{init_ts + i}', th=0.5, alphas=[1.0], colors=[[0, 1, 0]])

            pbar.update(1)
    csv_file.close()
