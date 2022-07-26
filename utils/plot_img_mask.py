import configparser
import os
import os.path as osp
import torch
from tqdm import tqdm
import sys

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
import utils.plots as plots
import utils.transforms as T
from cnn.dataset import SingleVentricleDataset, DatasetMode, LoadFlowMode

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('parser/configPreprocessing.ini')

    img4d_transf = T.ComposeUnary([T.Normalize(), T.ToTensor()])
    mask_transf = T.ComposeUnary([T.Round(th=0.5), T.ToTensor()])
    train_ds = SingleVentricleDataset(config, DatasetMode.TRAIN, LoadFlowMode.NO_LOAD_OF, img4d_transf, mask_transf)
    val_ds = SingleVentricleDataset(config, DatasetMode.VAL, LoadFlowMode.NO_LOAD_OF, img4d_transf, mask_transf)

    save_dir = plots.createSaveDirectory(config.get('DATA', 'OUTPUT_PATH'), 'ImageMask')

    conifg_output = os.path.sep.join([save_dir, "config.ini"])
    with open(conifg_output, 'w') as configfile:
        config.write(configfile)

    erode = T.ComposeUnary([T.ToArray(), T.Erode(th=0.5), T.ToTensor()])
    pbar = tqdm(total=len(train_ds) + len(val_ds))
    for ds in [train_ds, val_ds]:
        for (pname, data, m0, mk, init_ts, final_ts, _, _) in ds:
            pbar.set_postfix_str(f'{pname}')

            timesteps = data.shape[3]
            for t in range(timesteps):
                u = data[:, :, :, t]
                # uk= data[:, :, :, final_ts]

                patient_dir = plots.createSubDirectory(save_dir, pname)
                if t == init_ts:
                    plots.save_img_masks(u, [m0, erode(m0)], f'{pname}_u{t}.png', patient_dir,
                                         th=0.5, alphas=[0.2, 1.0], colors=[[1, 0.75, 0], [0, 1, 0]])
                elif t == final_ts:
                    plots.save_img_masks(u, [mk, erode(mk)], f'{pname}_u{t}.png', patient_dir,
                                         th=0.5, alphas=[0.2, 1.0], colors=[[0, 0.75, 1], [1, 0, 0]])
                else:
                    plots.save_img_masks(u, [], f'{pname}_u{t}.png', patient_dir, th=None, alphas=None, colors=None)

                # plots.save_img_masks_slices(u0, [m0, erode(m0)], patient_dir, 'm0', 0.5, alphas=[0.2, 1.0], colors=[[1, 0.75, 0], [0, 1, 0]])
                # plots.save_img_masks_slices(uk, [mk, erode(mk)], patient_dir, 'mk', 0.5, alphas=[0.2, 1.0], colors=[[0, 0.75, 1], [1, 0, 0]])

            pbar.update(1)
