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
    save_size = (16, 200, 200)
    config = configparser.ConfigParser()
    config.read('parser/configPreprocessing.ini')

    img4d_transf = T.ComposeUnary([T.ToTensor()])
    mask_transf = T.ComposeUnary([T.ToTensor()])
    train_ds = SingleVentricleDataset(config, DatasetMode.TRAIN, LoadFlowMode.NO_LOAD_OF, img4d_transf, mask_transf)
    # val_ds = SingleVentricleDataset(config, DatasetMode.VAL, LoadFlowMode.NO_LOAD_OF, img4d_transf, mask_transf)

    save_dir = plots.createSaveDirectory(config.get('DATA', 'OUTPUT_PATH'), 'ImageMask')
    plots.save_config(config, save_dir, 'config.ini')
    
    mask_posp = T.ComposeUnary([T.ToArray(), T.Resize(size=save_size), T.Round(th=0.5), T.Erode(), T.ToTensor()])
    m0_mk_posp = T.ComposeUnary([T.ToArray(), T.Resize(size=save_size), T.Round(th=0.5), T.ToTensor()])
    img_posp = T.ComposeUnary([T.ToArray(), T.Resize(size=save_size), T.Normalize(), T.ToTensor()])

    pbar = tqdm(total=len(train_ds))
    for ds in [train_ds]:
        for (pname, data, m0, mk, masks, init_ts, final_ts, _, _) in ds:
            if masks is not None:
                print(pname, 'full_cycle')

            timesteps = data.shape[3]
            for t in range(timesteps):
                u = img_posp(data[:, :, :, t])
                # uk= data[:, :, :, final_ts]

                patient_dir = plots.createSubDirectory(save_dir, pname)

                if t == init_ts:
                    plots.save_img_masks(u, [m0_mk_posp(m0), mask_posp(m0)], f'{pname}_t{t}.png', patient_dir,
                                         th=0.5, alphas=[0.2, 1.0, 1.0], colors=[[1, 0.75, 0], [0, 1, 0]])
                    plots.save_img_masks_slices(u,[m0_mk_posp(m0), mask_posp(m0)],patient_dir,'slices',th=0.5,alphas=[0.2,1.0], colors=[[1, 0.75, 0], [0, 1, 0]])
                elif t == final_ts:
                    plots.save_img_masks(u, [m0_mk_posp(mk), mask_posp(mk)], f'{pname}_t{t}.png', patient_dir,
                                         th=0.5, alphas=[0.2, 1.0, 1.0], colors=[[0, 0.75, 1], [1, 0, 0]])
                else:
                    if masks is not None:
                        plots.save_img_masks(u, [mask_posp(masks[t])], f'{pname}_t{t}.png', patient_dir, th=0.5, alphas=[1.0], colors=[[0, 0, 1]])
                    else:
                        plots.save_img_masks(u, [], f'{pname}_t{t}.png', patient_dir, th=None, alphas=None, colors=None)

            pbar.update(1)
