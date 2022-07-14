import torch
import configparser
from tqdm import tqdm
from warp import Warp
import sys
import os.path as osp
from loss import loss_func_three
import csv
from metrics import dc

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
from utils import plots
import utils.transforms as T
from cnn.dataset import SingleVentricleDataset, DatasetMode


def compute_dc(x: torch.Tensor, y: torch.Tensor):
    transf = T.ComposeUnary([T.ToArray(), T.Round(th=0.5)])
    x = transf(x)
    y = transf(y)
    return dc(x, y)


def compute_hd(x: torch.Tensor, y: torch.Tensor):
    transf = T.ComposeUnary([T.ToArray(), T.Round(th=0.5)])
    x = transf(x)
    y = transf(y)
    return 1.0


def warp_forward(net, warp, data, u, mt, use_sigmoid):
    mt = warp(mt.squeeze(), u).unsqueeze(0).unsqueeze(0)
    x = torch.cat((data, mt), dim=1)
    if use_sigmoid:
        x = torch.sigmoid(net(x))
    else:
        x = net(x)
    return x


if __name__ == "__main__":
    config_eval = configparser.ConfigParser()
    config_eval.read('parser/configCNNEval.ini')

    TRAINED_MODEL_DIR = config_eval.get('DATA', 'TRAINED_MODEL_DIR')
    MODEL_NAME = config_eval.get('DATA', 'MODEL_NAME')
    VERBOSE = config_eval.getboolean('DEBUG', 'VERBOSE')
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    config_train = configparser.ConfigParser()
    config_train.read(osp.join(TRAINED_MODEL_DIR, 'config.ini'))

    DATASET = config_eval.get('DATA', 'DATASET')

    data_transf = T.ComposeUnary([T.ToTensor()])
    mask_transf = T.ComposeUnary([T.ToTensor()])

    if DATASET == 'train':
        ds = SingleVentricleDataset(config_train, DatasetMode.TRAIN, load_flow=True, img4d_transforms=data_transf, mask_transforms=mask_transf)
    else:
        ds = SingleVentricleDataset(config_train, DatasetMode.VAL, load_flow=True, img4d_transforms=data_transf, mask_transforms=mask_transf)

    save_dir = plots.createSaveDirectory(config_eval.get('DATA', 'OUTPUT_PATH'), 'CNN_EVAL')

    # save config file to save directory
    conifg_output = osp.join(save_dir, 'config.ini')
    with open(conifg_output, 'w') as config_file:
        config_eval.write(config_file)

    csv_file = open(osp.join(save_dir, 'loss.csv'), 'w')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Patient', 'L1-CNN', 'L2-CNN', 'L3-CNN', 'LT-CNN', 'L1-OF', 'L2-OF', 'L3-OF', 'LT-OF', 'dc_0', 'dc_k', 'hd_0', 'hd_k'])

    net = torch.load(osp.join(TRAINED_MODEL_DIR, MODEL_NAME), map_location='cpu').to(DEVICE)
    pbar = tqdm(total=len(ds))

    save_size = (config_eval.getint('PARAMETERS', 'save_NZ'),
                 config_eval.getint('PARAMETERS', 'save_NY'),
                 config_eval.getint('PARAMETERS', 'save_NX'))
    mask_posp = T.ComposeUnary([T.ToArray(), T.Resize(size=save_size), T.Round(th=0.5), T.Erode(), T.ToTensor()])
    m0_mk_posp = T.ComposeUnary([T.ToArray(), T.Resize(size=save_size), T.Round(th=0.5), T.ToTensor()])
    img_posp = T.ComposeUnary([T.ToArray(), T.Resize(size=save_size), T.Normalize(), T.ToTensor()])

    net.eval()
    with torch.no_grad():
        for (pname, img4d, m0, mk, init_ts, final_ts, ff, bf) in ds:
            img4d = img4d.to(DEVICE)
            m0 = m0.to(DEVICE)
            mk = mk.to(DEVICE)
            ff = ff.to(DEVICE)
            bf = bf.to(DEVICE)

            NZ, NY, NX, _ = img4d.shape
            nsize = (1, 1, NZ, NY, NX)
            warp = Warp(config_train, NZ, NY, NX)

            mts_cnn_list = [m0.reshape(nsize)]
            mtts_cnn_list = [mk.reshape(nsize)]
            mts_iw_list = [m0]
            mtts_iw_list = [mk]

            ts = ff.shape[0]
            for t in range(ts):
                pbar.set_postfix_str(f'P: {pname}, S: {t+1}/{ts}')

                # Forward mask propagation m0 -> mk
                img3d = img4d[:, :, :, init_ts + t + 1].reshape(nsize)
                mt_cnn = warp_forward(net, warp, img3d, ff[t, :, :, :, :], mts_cnn_list[-1], use_sigmoid=False)
                mts_cnn_list.append(mt_cnn)

                mt_iw = warp(mts_iw_list[-1], ff[t, :, :, :, :])
                mts_iw_list.append(mt_iw)

                # Backward mask propagation mk -> m0
                img3d = img4d[:, :, :, final_ts - t - 1].reshape(nsize)
                mtt_cnn = warp_forward(net, warp, img3d, bf[t, :, :, :, :], mtts_cnn_list[-1], use_sigmoid=False)
                mtts_cnn_list.append(mtt_cnn)

                mtt_iw = warp(mtts_iw_list[-1], bf[t, :, :, :, :])
                mtts_iw_list.append(mtt_iw)

            assert(len(mtts_cnn_list) == len(mts_cnn_list) and len(mtts_iw_list) == len(mts_iw_list))

            mtts_cnn_list.reverse()
            mtts_iw_list.reverse()
            loss_cnn, l1_cnn, l2_cnn, l3_cnn = loss_func_three(mts_cnn_list, mtts_cnn_list)
            loss_iw, l1_iw, l2_iw, l3_iw = loss_func_three(mts_iw_list, mtts_iw_list)
            row = [pname]
            row.append('{:.2f}'.format(l1_cnn.item()))
            row.append('{:.2f}'.format(l2_cnn.item()))
            row.append('{:.2f}'.format(l3_cnn.item()))
            row.append('{:.2f}'.format(loss_cnn.item()))
            row.append('{:.2f}'.format(l1_iw.item()))
            row.append('{:.2f}'.format(l2_iw.item()))
            row.append('{:.2f}'.format(l3_iw.item()))
            row.append('{:.2f}'.format(loss_iw.item()))
            row.append('{:.3f}'.format(compute_dc(m0, mtts_cnn_list[0].squeeze())))
            row.append('{:.3f}'.format(compute_dc(mk, mts_cnn_list[-1].squeeze())))
            row.append('{:.3f}'.format(compute_hd(m0, mtts_cnn_list[0].squeeze())))
            row.append('{:.3f}'.format(compute_hd(mk, mts_cnn_list[-1].squeeze())))
            csv_writer.writerow(row)

            if VERBOSE:
                patient_dir = plots.createSubDirectory(save_dir, pname)
                fwd_dir = plots.createSubDirectory(patient_dir, 'fwd')
                bwd_dir = plots.createSubDirectory(patient_dir, 'bwd')

                for i in range(len(mts_cnn_list)):
                    img3d = img4d[:, :, :, init_ts + i]
                    img3d = img_posp(img3d.squeeze())

                    mtt_cnn = mask_posp(mtts_cnn_list[i].squeeze())
                    mtt_iw = mask_posp(mtts_iw_list[i])

                    mt_cnn = mask_posp(mts_cnn_list[i].squeeze())
                    mt_iw = mask_posp(mts_iw_list[i])

                    if i == 0:
                        plots.save_img_masks(img3d, [m0_mk_posp(m0), mtt_cnn, mtt_iw], 'im_m0_m0tt', fwd_dir, th=0.5,
                                             alphas=[0.3, 1.0, 1.0], colors=[[1, 0.7, 0], [0, 1, 0], [0, 0, 1]])
                        # plots.save_img_masks_slices(data_t, [m0_mk_posp(m0), mtt_cnn, mtt_iw], fwd_dir, 'im_m0_m0tt_slices', th=0.5,
                        #                             alphas=[0.3, 1.0, 1.0], colors=[[1, 0.7, 0], [0, 1, 0], [0, 0, 1]])
                    elif i == len(mts_cnn_list) - 1:
                        plots.save_img_masks(img3d, [m0_mk_posp(mk), mt_cnn, mt_iw], 'mk_mkt', bwd_dir, th=0.5,
                                             alphas=[0.3, 1.0, 1.0], colors=[[1, 0.7, 0], [0, 1, 0], [0, 0, 1]])
                        # plots.save_img_masks_slices(data_t, [m0_mk_posp(mk), mt_cnn, mt_iw], bwd_dir, 'mk_mkt_slices', th=0.5,
                        #                             alphas=[0.3, 1.0, 1.0], colors=[[1, 0.7, 0], [0, 1, 0], [0, 0, 1]])

                    plots.save_img_masks(img3d, [mt_cnn, mt_iw], f'im_t_{init_ts + i}', fwd_dir, th=0.5,
                                         alphas=[1.0, 1.0], colors=[[0, 1, 0], [0, 0, 1]])
                    # plots.save_img_masks_slices(data_t, [mt_cnn, mt_iw], fwd_dir, f'im_t_{init_ts + i}', th=0.5,
                    #                             alphas=[1.0, 1.0], colors=[[0, 1, 0], [0, 0, 1]])

                    plots.save_img_masks(img3d, [mtt_cnn, mtt_iw], f'im_tt_{init_ts + i}', bwd_dir, th=0.5,
                                         alphas=[1.0, 1.0], colors=[[0, 1, 0], [0, 0, 1]])
                    # plots.save_img_masks_slices(data_t, [mtt_cnn, mtt_iw], bwd_dir, f'im_tt_{init_ts + i}', th=0.5,
                    #                             alphas=[1.0, 1.0], colors=[[0, 1, 0], [0, 0, 1]])
            pbar.update(1)
    csv_file.close()
