import numpy as np
from tqdm import tqdm
from ml_collections import config_dict
import os.path as osp
import math
import yaml
from monai.data import MetaTensor
import torch

from svs.utils import dirs
from svs.modules.datasets import MRIBaseDataset


def normalize(x: MetaTensor, seg_dia: MetaTensor, seg_sys: MetaTensor, tdia: int, tsys: int) -> MetaTensor:
    """
    Normalize the intensity values of a 4D image array using a custom normalization function.

    Args:
        x (MetaTensor): 4D tensor with shape [t, x, y, z], representing the image.
        seg_dia (MetaTensor): 3D tensor with shape [c, x, y, z], representing the diastole segmentation mask.
        seg_sys (MetaTensor): 3D tensor with shape [c, x, y, z], representing the systole segmentation mask.
        tdia (int): Time index for the diastole phase.
        tsys (int): Time index for the systole phase.

    Returns:
        torch.Tensor: The normalized 4D image MetaTensor.
    """
    p95 = np.quantile(x.array, 0.95).item()
    x = torch.clamp(x, 0, p95)
    avg_dia = torch.mean(x[tdia][seg_dia.squeeze(0).bool()]).item()
    avg_sys = torch.mean(x[tsys][seg_sys.squeeze(0).bool()]).item()
    avg = 0.5 * (avg_dia + avg_sys)

    # n(I) = a I / sqrt(1 + beta I^2)
    norm_a = math.sqrt(p95 * p95 - avg * avg) / (math.sqrt(3) * p95 * avg)
    norm_b = (p95 * p95 - 4. * avg * avg) / (3. * p95 * p95 * avg * avg)
    x = norm_a * x / torch.sqrt(1 + norm_b * x**2)

    # print("norm(per95) = ", norm_a * p95 / math.sqrt(1 + norm_b * p95 * p95))
    # print("norm(avg) = ", norm_a * avg / math.sqrt(1 + norm_b * avg * avg))
    # print(torch.min(x), torch.max(x))

    return x


def run(config_dir='conf', config_name='preprocessing.yaml', base_dir=None):
    print('Normalization')

    # Load configuration file
    cfg = config_dict.ConfigDict(yaml.load(open(osp.join(config_dir, config_name), 'r'), Loader=yaml.FullLoader))

    # Create output dirs
    save_dir = dirs.create_timestamped_dir(cfg.data.out_dir, 'preprocessing_norm')
    out_img_dir = dirs.create_subdir(save_dir, cfg.data.imgs_dir)
    out_seg_dir = dirs.create_subdir(save_dir, cfg.data.segs_dir)

    if base_dir is not None:
        cfg.data.base_dir = base_dir
    ds = MRIBaseDataset(cfg.data.base_dir, cfg.data.imgs_dir, cfg.data.segs_dir, cfg.data.metadata_file)

    for i in tqdm(range(len(ds))):
        patient = ds[i]

        patient.img = normalize(patient.img, patient.seg_dia, patient.seg_sys, patient.tdia, patient.tsys)

        patient.write_nifti(out_img_dir, out_seg_dir)

        if cfg.debug.viz:
            patient.viz_data(osp.join(save_dir, "images"), cfg.debug.gif, cfg.debug.dur, cfg.debug.alpha, cfg.debug.color, 5.)

    # Save metadata to Excel file
    ds.df.to_excel(osp.join(save_dir, cfg.data.metadata_file), index=False)

    # Save configuration to JSON file
    with open(osp.join(save_dir, 'config.json'), 'w') as f:
        f.write(cfg.to_json(indent=4))

    return save_dir


if __name__ == "__main__":
    run()
