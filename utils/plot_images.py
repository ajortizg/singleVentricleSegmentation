import configparser
import os
import os.path as osp
import torch
from tqdm import tqdm
import sys

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
import utils.plots as plots
import utils.transforms.unary_transforms as T1
from cnn.dataset import SingleVentricleDataset, DatasetMode, LoadFlowMode

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('parser/configFlowWarping.ini')

    save_size = (config.getint('PARAMETERS', 'SAVE_NZ'),
                 config.getint('PARAMETERS', 'SAVE_NY'),
                 config.getint('PARAMETERS', 'SAVE_NX'))

    img4d_transf = T1.Compose([T1.ToTensor()])
    mask_transf = T1.Compose([T1.ToTensor()])

    # train_ds = SingleVentricleDataset(config, DatasetMode.TRAIN, LoadFlowMode.NO_LOAD, img4d_transf, mask_transf)
    # val_ds = SingleVentricleDataset(config, DatasetMode.VAL, LoadFlowMode.NO_LOAD, img4d_transf, mask_transf)
    # test_ds = SingleVentricleDataset(config, DatasetMode.TEST, LoadFlowMode.NO_LOAD, img4d_transf, mask_transf, test_masks_transforms=mask_transf)

    dset = SingleVentricleDataset(config, DatasetMode.FULL, LoadFlowMode.NO_LOAD, img4d_transf, mask_transf)

    save_dir = plots.createSaveDirectory(config.get('DATA', 'OUTPUT_PATH'), 'ImageMask')
    plots.save_config(config, save_dir, 'config.ini')

    mask_posp = T1.Compose([T1.Resize(save_size), T1.Round(0.5), T1.Erode()])
    m0_mk_posp = T1.Compose([T1.Resize(save_size), T1.Round(0.5)])
    img_posp = T1.Compose([T1.Resize(save_size), T1.Normalize()])

    # pbar = tqdm(total=len(train_ds) + len(val_ds) + len(test_ds))
    pbar = tqdm(total=len(dset))
    for ds in [dset]:  # [test_ds, train_ds, val_ds]:
        for (pname, data, m0, mk, masks, init_ts, final_ts, _, _) in ds:
            if masks is not None:
                print(pname, 'full_cycle')
                timesteps = masks.shape[-1]
            else:
                timesteps = data.shape[-1]

            for t in range(timesteps):
                u = img_posp(data[:, :, :, t])
                # uk= data[:, :, :, final_ts]

                patient_dir = plots.createSubDirectory(save_dir, pname)

                if t == init_ts:
                    mask = masks[..., t] if masks is not None else m0
                    plots.save_img_masks(u, [m0_mk_posp(m0), mask_posp(mask)], f'{pname}_t{t}.png', patient_dir,
                                         th=0.5, alphas=[0.2, 1.0, 1.0], colors=[[1, 0.75, 0], [0, 1, 0]])
                elif t == final_ts:
                    mask = masks[..., t] if masks is not None else mk
                    plots.save_img_masks(u, [m0_mk_posp(mk), mask_posp(mask)], f'{pname}_t{t}.png', patient_dir,
                                         th=0.5, alphas=[0.2, 1.0, 1.0], colors=[[0, 0.75, 1], [1, 0, 0]])
                else:
                    if masks is not None:
                        plots.save_img_masks(u, [mask_posp(masks[..., t])], f'{pname}_t{t}.png',
                                             patient_dir, th=0.5, alphas=[1.0], colors=[[0, 0, 1]])
                    else:
                        plots.save_img_masks(u, [], f'{pname}_t{t}.png', patient_dir, th=None, alphas=None, colors=None)

            pbar.update(1)
