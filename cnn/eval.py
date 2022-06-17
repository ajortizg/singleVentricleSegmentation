import torch
import configparser
from tqdm import tqdm
from dataset import SingleVentricleDataset, DatasetMode
from warp import Warp
from cnn_utils import warp_forward
import sys
import os.path as osp
from loss import loss_func_three
import csv
import transforms as T

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
from utils import plots


if __name__ == "__main__":
    config_eval = configparser.ConfigParser()
    config_eval.read('parser/configCNNEval.ini')

    TRAINED_MODEL_DIR = config_eval.get('DATA', 'TRAINED_MODEL_DIR')
    MODEL_NAME = config_eval.get('DATA', 'MODEL_NAME')
    USE_CUDA = config_eval.get('DEVICE', 'CUDA_AVAILABLE')
    VERBOSE = config_eval.getboolean('DEBUG', 'VERBOSE')
    if USE_CUDA and torch.cuda.is_available():
        DEVICE = 'cuda'
        CUDA_DEVICE = config_eval.getint('DEVICE', 'cuda_device')
        torch.cuda.set_device(CUDA_DEVICE)
    else:
        DEVICE = 'cpu'

    config_train = configparser.ConfigParser()
    config_train.read(TRAINED_MODEL_DIR + 'config.ini')

    # transf = [transforms.Normalize(mean=0.1478, std=0.1385)]
    transf = T.ComposeUnary([T.Normalize()])
    ds = SingleVentricleDataset(config_train, DatasetMode.VAL, load_flow=True, data_transforms=transf)
    save_dir = plots.createSaveDirectory(config_eval.get('DATA', 'OUTPUT_PATH'), 'CNN_EVAL')

    # save config file to save directory
    conifg_output = osp.join(save_dir, 'config.ini')
    with open(conifg_output, 'w') as config_file:
        config_eval.write(config_file)

    csv_file = open(osp.join(save_dir, 'loss.csv'), 'w')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Patient', 'L1-CNN', 'L2-CNN', 'L3-CNN', 'LT-CNN', 'L1-OF', 'L2-OF', 'L3-OF', 'LT-OF'])

    net = torch.load(TRAINED_MODEL_DIR + MODEL_NAME).to(DEVICE)
    pbar = tqdm(total=len(ds))

    mask_transf = T.ComposeUnary([T.Normalize(), T.Erode()])
    net.eval()
    with torch.no_grad():
        for (pname, vol, m0, mk, init_ts, final_ts, ff, bf) in ds:
            NZ, NY, NX, NT = vol.shape
            nsize = (1, 1, NZ, NY, NX)

            warp = Warp(config_train, NZ, NY, NX)
            mts_cnn = [m0.reshape(nsize).to(DEVICE)]
            mtts_cnn = [mk.reshape(nsize).to(DEVICE)]

            mts_iw = [m0.to(DEVICE)]
            mtts_iw = [mk.to(DEVICE)]

            ts = ff.shape[0]
            for t in range(ts):
                pbar.set_postfix_str(f'P: {pname}, S: {t+1}/{ts}')

                # Forward mask propagation m0 -> mk
                fwd_time = init_ts + t + 1
                data_t = vol[:, :, :, fwd_time].to(DEVICE)
                u = ff[t, :, :, :, :].to(DEVICE)
                mt_cnn = warp_forward(net, warp, data_t, nsize, u, mts_cnn[-1])
                mts_cnn.append(mt_cnn)
                mts_iw.append(warp(mts_iw[-1], u))

                # Backward mask propagation mk -> m0
                bwd_time = final_ts - t - 1
                data_t = vol[:, :, :, bwd_time].to(DEVICE)
                u = bf[t, :, :, :, :].to(DEVICE)
                mtt_cnn = warp_forward(net, warp, data_t, nsize, u, mtts_cnn[-1])
                mtts_cnn.append(mtt_cnn)
                mtts_iw.append(warp(mtts_iw[-1], u))

            mtts_cnn.reverse()
            mtts_iw.reverse()
            loss_cnn, l1_cnn, l2_cnn, l3_cnn = loss_func_three(mts_cnn, mtts_cnn)
            loss_iw, l1_iw, l2_iw, l3_iw = loss_func_three(mts_iw, mtts_iw)
            row = [pname]
            row.append('{:.2f}'.format(l1_cnn.item()))
            row.append('{:.2f}'.format(l2_cnn.item()))
            row.append('{:.2f}'.format(l3_cnn.item()))
            row.append('{:.2f}'.format(loss_cnn.item()))
            row.append('{:.2f}'.format(l1_iw.item()))
            row.append('{:.2f}'.format(l2_iw.item()))
            row.append('{:.2f}'.format(l3_iw.item()))
            row.append('{:.2f}'.format(loss_iw.item()))
            csv_writer.writerow(row)

            if VERBOSE:
                patient_dir = plots.createSubDirectory(save_dir, pname)
                fwd_dir = plots.createSubDirectory(patient_dir, 'fwd')
                bwd_dir = plots.createSubDirectory(patient_dir, 'bwd')

                for i in range(len(mts_cnn)):
                    data_t = vol[:, :, :, init_ts + i]
                    if i == 0:
                        plots.save_img_masks(data_t, [m0, mask_transf(mtts_cnn[i].squeeze().cpu()), mask_transf(mtts_iw[i].squeeze().cpu())],
                                             'im_m0_m0tt', fwd_dir, th=0.5, alphas=[0.3, 1.0, 1.0], colors=[[1, 0.7, 0], [0, 1, 0], [0, 0, 1]])
                    elif i == len(mts_cnn) - 1:
                        plots.save_img_masks(data_t,
                                             [mk, mask_transf(mts_cnn[i].squeeze().cpu()), mask_transf(mts_iw[i].squeeze().cpu())],
                                             'mk_mkt', bwd_dir, th=0.5, alphas=[0.3, 1.0, 1.0], colors=[[1, 0.7, 0], [0, 1, 0], [0, 0, 1]])

                    plots.save_img_masks(data_t, [mask_transf(mts_cnn[i].squeeze().cpu()), mask_transf(mts_iw[i].cpu())],
                                         f'im_t_{init_ts + i}', fwd_dir, th=0.5, alphas=[1.0, 1.0], colors=[[0, 1, 0], [0, 0, 1]])

                    plots.save_img_masks(data_t, [mask_transf(mtts_cnn[i].squeeze().cpu()), mask_transf(mtts_iw[i].cpu())],
                                         f'im_tt_{init_ts + i}', bwd_dir, th=0.5, alphas=[1.0, 1.0], colors=[[0, 1, 0], [0, 0, 1]])
            pbar.update(1)
    csv_file.close()
