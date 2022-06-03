import torch
from dataset import SingleVentricleDataset, DatasetMode
import configparser
import os
import os.path as osp
import transforms as T
from tqdm import tqdm
import sys


ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
import utils.plots as plots


config = configparser.ConfigParser()
config.read('parser/configCNN.ini')

ds = SingleVentricleDataset(config, DatasetMode.FULL, [T.Normalize()], load_flow=False)
save_dir = plots.createSaveDirectory(config.get('DATA', 'OUTPUT_PATH'), 'ImageMasks')

conifg_output = os.path.sep.join([save_dir, "config.ini"])
with open(conifg_output, 'w') as configfile:
    config.write(configfile)

pbar = tqdm(total=len(ds))
sum = 0
squared_sum = 0
for (pname, data, m0, mk, init_ts, final_ts, _, _) in ds:
    pbar.set_postfix_str(f'P: {pname}')
    u0 = data[:, :, :, init_ts]
    uk = data[:, :, :, final_ts]

    patient_dir = plots.createSubDirectory(save_dir, pname)
    # save_slices(u0, f'{pname}_u0.png', patient_dir)
    # save_slices(uk, f'{pname}_uk.png', patient_dir)
    # save_slices(m0, f'{pname}_m0.png', patient_dir)
    # save_slices(mk, f'{pname}_mk.png', patient_dir)
    # plots.save_img_mask_slices(u0, m0, f'{pname}_u0_m0.png', patient_dir)
    # plots.save_img_mask_slices(uk, mk, f'{pname}_uk_mk.png', patient_dir)

    plots.save_img_mask_slices(u0, plots.erode_mask(m0), f'{pname}_border_u0_m0.png', patient_dir, color=[0, 0, 1], alpha=0.5)
    plots.save_img_mask_slices(uk, plots.erode_mask(mk), f'{pname}_border_uk_mk.png', patient_dir, color=[0, 0, 1], alpha=0.5)
    # plots.save_img_mask_single_zslices(u0, m0, patient_dir, 'u0_m0_slices')
    # plots.save_img_mask_single_zslices(uk, mk, patient_dir, 'uk_mk_slices')
    pbar.update(1)
