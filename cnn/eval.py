import torch
import configparser
from tqdm import tqdm
from warp import WarpCNN
import sys
import os.path as osp
from loss import loss_func_three
import csv
from torch.utils.data import DataLoader
import nibabel as nib
from monai.metrics.meandice import compute_meandice

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
from utils import plots
import utils.transforms as T
from cnn.dataset import SingleVentricleDataset, DatasetMode, LoadFlowMode
import cnn.cnn_utils as utils


def save_nifty(mask, save_dir, filename):
    mask = mask.squeeze()
    mask = torch.swapaxes(mask, 0, 2)           # xyz format
    mask = torch.where(mask > 0.5, 1.0, 0.0)    # binarize

    mt_nii = nib.Nifti1Image(T.ToArray()(mask), affine=None, header=None)
    outputFile = osp.sep.join([save_dir, filename])
    nib.save(mt_nii, outputFile)


def create_row(pname, mts_cnn_list, mtts_cnn_list, mts_iw_list, mtts_iw_list):
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
    row.append('{:.3f}'.format(compute_meandice(mtts_cnn_list[0], mts_cnn_list[0]).mean().item()))
    row.append('{:.3f}'.format(compute_meandice(mts_cnn_list[-1], mtts_cnn_list[-1]).mean().item()))
    return row


if __name__ == "__main__":
    config_eval = configparser.ConfigParser()
    config_eval.read('parser/configCNNEval.ini')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    P = utils.read_eval_params(config_eval)

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

    data_transf = T.ComposeUnary([T.ToTensor()])
    mask_transf = T.ComposeUnary([T.Round(th=0.5), T.ToTensor()])
    if P['dataset'] == 'train':
        ds = SingleVentricleDataset(config_train, DatasetMode.TRAIN, LoadFlowMode.TRAIN_OF, data_transf, mask_transf)
    else:
        ds = SingleVentricleDataset(config_train, DatasetMode.VAL, LoadFlowMode.TRAIN_OF, data_transf, mask_transf)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=P['workers'], collate_fn=utils.collate_fn)

    save_dir = plots.createSaveDirectory(config_eval.get('DATA', 'OUTPUT_PATH'), 'EVAL')
    utils.save_config(config_eval, save_dir, 'config.ini')

    csv_file = open(osp.join(save_dir, 'loss.csv'), 'w')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Patient', 'L1-CNN', 'L2-CNN', 'L3-CNN', 'LT-CNN', 'L1-OF', 'L2-OF', 'L3-OF', 'LT-OF', 'dc_0', 'dc_k'])
    logger = plots.create_logger(save_dir)

    net = utils.create_net(config_train, logger).to(device)
    net = torch.nn.DataParallel(net, device_ids=[0])
    checkpoint = torch.load(osp.join(P['trained_model_dir'], P['model_name']))
    net.load_state_dict(checkpoint['model_state_dict'], strict=True)

    save_size = (P['save_nz'], P['save_ny'], P['save_nx'])
    mask_posp = T.ComposeUnary([T.ToArray(), T.Resize(size=save_size), T.Round(th=0.5), T.Erode(), T.ToTensor()])
    m0_mk_posp = T.ComposeUnary([T.ToArray(), T.Resize(size=save_size), T.Round(th=0.5), T.ToTensor()])
    img_posp = T.ComposeUnary([T.ToArray(), T.Resize(size=save_size), T.Normalize(), T.ToTensor()])

    pbar = tqdm(total=len(ds))
    net.eval()
    with torch.no_grad():
        for (pnames, imgs4d, m0s, mks, list_times_fwd, list_times_bwd, ff, bf, offsets) in loader:
            if P['fine_tuning']:
                if pnames[0] != patient_name:
                    continue

            imgs4d = imgs4d.to(device)
            m0s = m0s.to(device)
            mks = mks.to(device)
            ff = ff.to(device)
            bf = bf.to(device)

            BS, CH, NZ, NY, NX, NT = imgs4d.shape
            warp = WarpCNN(config_train, NZ, NY, NX)
            mts_cnn_list = [m0s]
            mts_iw_list = [m0s]
            mtts_cnn_list = [mks]
            mtts_iw_list = [mks]
            batch_indices = torch.arange(BS)
            timesteps = ff.shape[1]
            for t in range(timesteps):
                pbar.set_postfix_str(f'P: {pnames[0]}, S: {t+1}/{timesteps}')

                # Forward mask propagation m0 -> mk
                mt = warp(mts_cnn_list[-1], ff[:, t, :, :, :, :])
                mt, mh = net(torch.cat((imgs4d[batch_indices, :, :, :, :, list_times_fwd[t + 1][batch_indices]], mt), dim=1))
                mts_cnn_list.append(mt)

                mt_iw = warp(mts_iw_list[-1], ff[:, t, :, :, :, :])
                mts_iw_list.append(mt_iw)

                # Backward mask propagation mk -> m0
                mtt = warp(mtts_cnn_list[-1], bf[:, t, :, :, :, :])
                mtt, mhh = net(torch.cat((imgs4d[batch_indices, :, :, :, :, list_times_bwd[t + 1][batch_indices]], mtt), dim=1))
                mtts_cnn_list.append(mtt)

                mtt_iw = warp(mtts_iw_list[-1], bf[:, t, :, :, :, :])
                mtts_iw_list.append(mtt_iw)

            assert(len(mtts_cnn_list) == len(mts_cnn_list) and len(mtts_iw_list) == len(mts_iw_list))

            mtts_cnn_list.reverse()
            mtts_iw_list.reverse()
            csv_writer.writerow(create_row(pnames[0], mts_cnn_list, mtts_cnn_list, mts_iw_list, mtts_iw_list))

            if P['save_imgs'] or P['save_nifti']:
                patient_dir = plots.createSubDirectory(save_dir, pnames[0])
                fwd_dir = plots.createSubDirectory(patient_dir, 'fwd')
                bwd_dir = plots.createSubDirectory(patient_dir, 'bwd')

                for t in range(len(mts_cnn_list)):
                    if P['save_imgs']:
                        img3d = img_posp(imgs4d[batch_indices, :, :, :, :, list_times_fwd[t][batch_indices]].squeeze())
                        mtt_cnn = mask_posp(mtts_cnn_list[t].squeeze())
                        mtt_iw = mask_posp(mtts_iw_list[t].squeeze())
                        mt_cnn = mask_posp(mts_cnn_list[t].squeeze())
                        mt_iw = mask_posp(mts_iw_list[t].squeeze())

                        if t == 0:
                            plots.save_img_masks(img3d, [m0_mk_posp(m0s.squeeze()), mtt_cnn, mtt_iw], 'im_m0_m0tt', fwd_dir, th=0.5,
                                                 alphas=[0.3, 1.0, 1.0], colors=[[1, 0.7, 0], [0, 1, 0], [0, 0, 1]])
                            # plots.save_img_masks_slices(data_t, [m0_mk_posp(m0), mtt_cnn, mtt_iw], fwd_dir, 'im_m0_m0tt_slices', th=0.5,
                            #                             alphas=[0.3, 1.0, 1.0], colors=[[1, 0.7, 0], [0, 1, 0], [0, 0, 1]])
                        elif t == len(mts_cnn_list) - 1:
                            plots.save_img_masks(img3d, [m0_mk_posp(mks.squeeze()), mt_cnn, mt_iw], 'mk_mkt', bwd_dir, th=0.5,
                                                 alphas=[0.3, 1.0, 1.0], colors=[[1, 0.7, 0], [0, 1, 0], [0, 0, 1]])
                            # plots.save_img_masks_slices(data_t, [m0_mk_posp(mk), mt_cnn, mt_iw], bwd_dir, 'mk_mkt_slices', th=0.5,
                            #                             alphas=[0.3, 1.0, 1.0], colors=[[1, 0.7, 0], [0, 1, 0], [0, 0, 1]])

                        plots.save_img_masks(img3d, [mt_cnn, mt_iw], f'im_t_{list_times_fwd[t][batch_indices].item()}', fwd_dir, th=0.5,
                                             alphas=[1.0, 1.0], colors=[[0, 1, 0], [0, 0, 1]])
                        # plots.save_img_masks_slices(data_t, [mt_cnn, mt_iw], fwd_dir, f'im_t_{init_ts + i}', th=0.5,
                        #                             alphas=[1.0, 1.0], colors=[[0, 1, 0], [0, 0, 1]])

                        plots.save_img_masks(img3d, [mtt_cnn, mtt_iw], f'im_tt_{list_times_fwd[t][batch_indices].item()}', bwd_dir, th=0.5,
                                             alphas=[1.0, 1.0], colors=[[0, 1, 0], [0, 0, 1]])
                        # plots.save_img_masks_slices(data_t, [mtt_cnn, mtt_iw], bwd_dir, f'im_tt_{init_ts + i}', th=0.5,
                        #                             alphas=[1.0, 1.0], colors=[[0, 1, 0], [0, 0, 1]])

                    if P['save_nifti']:
                        save_nifty(mts_cnn_list[t], fwd_dir, f'mt_{list_times_fwd[t][batch_indices].item()}.nii')
                        save_nifty(mtts_cnn_list[t], bwd_dir, f'mtt_{list_times_fwd[t][batch_indices].item()}.nii')
            pbar.update(1)
    csv_file.close()
