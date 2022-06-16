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

normalize = T.Normalize()
transf = T.ComposeUnary([T.Normalize()])
ds = SingleVentricleDataset(config, DatasetMode.FULL, load_flow=False, data_transforms=transf)
save_dir = plots.createSaveDirectory(config.get('DATA', 'OUTPUT_PATH'), 'ImageMasks')

conifg_output = os.path.sep.join([save_dir, "config.ini"])
with open(conifg_output, 'w') as configfile:
    config.write(configfile)

print(ds.validation_patients)

pbar = tqdm(total=len(ds))
for (pname, data, m0, mk, init_ts, final_ts, _, _) in ds:
    pbar.set_postfix_str(f'P: {pname}')
    u0 = data[:, :, :, init_ts]
    uk = data[:, :, :, final_ts]

    patient_dir = plots.createSubDirectory(save_dir, pname)
    plots.save_img_masks(u0, [m0], f'{pname}_u0_m0.png', patient_dir, th=0.5, alphas=[0.3], colors=[[1, 0.75, 0]])
    plots.save_img_masks(uk, [mk], f'{pname}_uk_mk.png', patient_dir, th=0.5, alphas=[0.3], colors=[[0, 0.75, 1]])
   
    pbar.update(1)
