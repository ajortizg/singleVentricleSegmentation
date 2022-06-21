import configparser
import os
import os.path as osp
from tqdm import tqdm
import sys

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
import utils.plots as plots
import utils.transforms as T
from cnn.dataset import SingleVentricleDataset, DatasetMode

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('parser/configPreprocessing.ini')

    transf = T.ComposeUnary([T.Normalize()])
    train_ds = SingleVentricleDataset(config, DatasetMode.TRAIN, load_flow=False, data_transforms=transf)
    val_ds = SingleVentricleDataset(config, DatasetMode.VAL, load_flow=False, data_transforms=transf)

    save_dir = plots.createSaveDirectory(config.get('DATA', 'OUTPUT_PATH'), 'ImageMask')

    conifg_output = os.path.sep.join([save_dir, "config.ini"])
    with open(conifg_output, 'w') as configfile:
        config.write(configfile)

    erode = T.Erode(0.5)
    pbar = tqdm(total=len(train_ds) + len(val_ds))
    for ds in [train_ds, val_ds]:
        for (pname, data, m0, mk, init_ts, final_ts, _, _) in ds:
            pbar.set_postfix_str(f'P: {pname}')

            u0 = data[:, :, :, init_ts]
            uk = data[:, :, :, final_ts]

            patient_dir = plots.createSubDirectory(save_dir, pname)
            plots.save_img_masks(u0, [m0, erode(m0)], f'{pname}_u0_m0.png', patient_dir, th=0.5, alphas=[0.2, 1.0], colors=[[1, 0.75, 0], [0, 1, 0]])
            plots.save_img_masks(uk, [mk, erode(mk)], f'{pname}_uk_mk.png', patient_dir, th=0.5, alphas=[0.2, 1.0], colors=[[0, 0.75, 1], [1, 0, 0]])

            plots.save_img_masks_slices(u0, [m0, erode(m0)], patient_dir, 'm0', 0.5, alphas=[0.2, 1.0], colors=[[1, 0.75, 0], [0, 1, 0]])
            plots.save_img_masks_slices(uk, [mk, erode(mk)], patient_dir, 'mk', 0.5, alphas=[0.2, 1.0], colors=[[0, 0.75, 1], [1, 0, 0]])

            pbar.update(1)
