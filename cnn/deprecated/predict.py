import torch
import configparser
from tqdm import tqdm
from warp import WarpCNN
import sys
import os.path as osp
from torch.utils.data import DataLoader
import nibabel as nib

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
from utilities import plots
import utilities.transforms.senary_transforms as T6
from cnn.dataset import SingleVentricleDataset, DatasetMode, LoadFlowMode
from utilities import cnn_utils


def save_nifty(mask, save_dir, filename):
    mask = mask.squeeze()
    mask = torch.swapaxes(mask, 0, 2)           # xyz format
    mask = torch.where(mask > 0.5, 1.0, 0.0)    # binarize

    mt_nii = nib.Nifti1Image(T.ToArray()(mask), affine=None, header=None)
    outputFile = osp.sep.join([save_dir, filename])
    nib.save(mt_nii, outputFile)


if __name__ == "__main__":
    config_eval = configparser.ConfigParser()
    config_eval.read('parser/configCNNEval.ini')

    TRAINED_MODEL_DIR = config_eval.get('DATA', 'TRAINED_MODEL_DIR')
    MODEL_NAME = config_eval.get('DATA', 'MODEL_NAME')
    SAVE_IMGS = config_eval.getboolean('DEBUG', 'SAVE_IMGS')
    SAVE_NIFTI = config_eval.getboolean('DEBUG', 'SAVE_NIFTI')
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_WORKERS = config_eval.getint('PARAMETERS', 'NUM_WORKERS')

    FINE_TUNING = config_eval.getboolean('DATA', 'FINE_TUNING')
    if FINE_TUNING:
        config_tl = configparser.ConfigParser()
        config_tl.read(osp.join(TRAINED_MODEL_DIR, 'config.ini'))
        pretrained_dir = config_tl.get('DATA', 'PRETRAINED_DIR')
        PATIENT_NAME = config_tl.get('DATA', 'PATIENT_NAME')

        config_train = configparser.ConfigParser()
        config_train.read(osp.join(pretrained_dir, 'config.ini'))
    else:
        config_train = configparser.ConfigParser()
        config_train.read(osp.join(TRAINED_MODEL_DIR, 'config.ini'))

    DATASET = config_eval.get('DATA', 'DATASET')

    transforms = T6.Compose([T6.ToTensor()])

    transforms = T6.Compose([T6.ToTensor()])
    if DATASET == 'train':
        ds = SingleVentricleDataset(config_train, DatasetMode.TRAIN, LoadFlowMode.PREDICT_OF, transf)
    else:
        ds = SingleVentricleDataset(config_train, DatasetMode.VAL, LoadFlowMode.PREDICT_OF, data_transf, mask_transf)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=NUM_WORKERS, collate_fn=cnn_utils.collate_fn)

    save_dir = plots.createSaveDirectory(config_eval.get('DATA', 'OUTPUT_PATH'), 'PREDICT')

    # save config file to save directory
    conifg_output = osp.join(save_dir, 'config.ini')
    with open(conifg_output, 'w') as config_file:
        config_eval.write(config_file)

    net = torch.load(osp.join(TRAINED_MODEL_DIR, MODEL_NAME), map_location='cpu').to(DEVICE)
    pbar = tqdm(total=len(ds))

    out_size = (config_eval.getint('PARAMETERS', 'save_NZ'),
                 config_eval.getint('PARAMETERS', 'save_NY'),
                 config_eval.getint('PARAMETERS', 'save_NX'))
    mask_posp = T.ComposeUnary([T.ToArray(), T.Resize(size=out_size), T.Round(th=0.5), T.Erode(), T.ToTensor()])
    img_posp = T.ComposeUnary([T.ToArray(), T.Resize(size=out_size), T.Normalize(), T.ToTensor()])

    net.eval()
    with torch.no_grad():
        for (pnames, imgs4d, m0s, mks, list_times_fwd, list_times_bwd, ff, bf, offsets) in loader:
            if FINE_TUNING:
                if pnames[0] != PATIENT_NAME:
                    continue

            init_ts = list_times_fwd[0]
            final_ts = list_times_bwd[0]
            original_NT = ff.shape[1]
            times_fwd, times_bwd = ds.create_timeline(init_ts, final_ts, original_NT)

            imgs4d = imgs4d.to(DEVICE)
            m0s = m0s.to(DEVICE)
            ff = ff.to(DEVICE)
            bf = bf.to(DEVICE)
            BS, CH, NZ, NY, NX, NT = imgs4d.shape
            warp = WarpCNN(config_train, NZ, NY, NX)
            mts_cnn_list = [m0s]
            batch_indices = torch.arange(BS)
            print('patient: ', pnames)
            print('init_ts: ', init_ts)
            print('final_ts: ', final_ts)
            print('original_NT: ', original_NT)
            print(times_fwd)

            for t in range(original_NT - 1):
                pbar.set_postfix_str(f'P: {pnames[0]}, S: {t+1}/{original_NT}')

                # Forward mask propagation m0 -> mk
                mt = warp(mts_cnn_list[-1], ff[:, t, :, :, :, :])
                mt_cnn = torch.cat((imgs4d[batch_indices, :, :, :, :, times_fwd[t + 1]], mt), dim=1)
                mts_cnn_list.append(net(mt_cnn))

            if SAVE_IMGS or SAVE_NIFTI:
                patient_dir = plots.createSubDirectory(save_dir, pnames[0])
                fwd_dir = plots.createSubDirectory(patient_dir, 'fwd')

                for t in range(len(mts_cnn_list)):
                    if SAVE_IMGS:
                        img3d = img_posp(imgs4d[batch_indices, :, :, :, :, times_fwd[t]].squeeze())
                        mt_cnn = mask_posp(mts_cnn_list[t].squeeze())

                        plots.save_img_masks(img3d, [mt_cnn], f'im_t_{times_fwd[t]}', fwd_dir, th=0.5,
                                             alphas=[1.0], colors=[[0, 1, 0]])
                    if SAVE_NIFTI:
                        save_nifty(mts_cnn_list[t], fwd_dir, f'mt_{times_fwd[t]}.nii')

            pbar.update(1)
