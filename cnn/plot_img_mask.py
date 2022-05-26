import torch
from dataset import SingleVentricleDataset
import configparser
import os
import os.path as osp
from tqdm import tqdm
import sys
import cv2
import math
import matplotlib.pyplot as plt

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
from utils.plots import save_img_mask_slices, createSaveDirectory, createSubDirectory, save_slices
from utils.torch_utils import normalize


config = configparser.ConfigParser()
config.read('parser/configCNN.ini')

ds = SingleVentricleDataset(config, load_flow=False)

# PATIENT_NAME = config.get('DATA', 'PATIENT_NAME')
# idx, found = ds.index_for_patient(PATIENT_NAME)
# if not found:
#     print(PATIENT_NAME + " not found!")
#     sys.exit()

save_dir = createSaveDirectory(config.get('DATA', 'OUTPUT_PATH'), 'ImageMasks')

conifg_output = os.path.sep.join([save_dir, "config.ini"])
with open(conifg_output, 'w') as configfile:
    config.write(configfile)

pbar = tqdm(total=len(ds))
for idx in range(len(ds)):
    (pname, data, mask_systole, mask_diastole, systole_time, diastole_time, _, _) = ds[idx]
    pbar.set_postfix_str(f'P: {pname}')
    data = normalize(data)
    init_ts = min(diastole_time, systole_time)
    final_ts = max(diastole_time, systole_time)

    m0 = mk = None
    if init_ts == systole_time:
        m0 = mask_systole
        mk = mask_diastole
    else:
        m0 = mask_diastole
        mk = mask_systole
    # print(pname, systole_time, diastole_time, ' <-> ', init_ts, final_ts)

    u0 = data[:, :, :, init_ts]
    uk = data[:, :, :, final_ts]

    patient_dir = createSubDirectory(save_dir, pname)
    save_slices(u0, 'u0.png', patient_dir)
    save_slices(uk, 'uk.png', patient_dir)
    save_slices(m0, 'm0.png', patient_dir)
    save_slices(mk, 'mk.png', patient_dir)
    save_img_mask_slices(u0, m0, 'u0_m0.png', patient_dir)
    save_img_mask_slices(uk, mk, 'uk_mk.png', patient_dir)
    pbar.update(1)
