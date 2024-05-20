import numpy as np
from tqdm import tqdm
from ml_collections import config_dict
import os.path as osp
import math
import yaml

from svs.utils import dirs
from svs.modules.datasets import MRIBaseDataset


def normalize(x: np.ndarray, seg_dia: np.ndarray, seg_sys: np.ndarray, tdia: int, tsys: int) -> np.ndarray:
    """
    Normalize the intensity values of a 4D image array using a custom normalization function.

    This function performs intensity normalization on a 4D medical image array. It first clips the image 
    intensities at the 95th percentile to reduce the impact of outliers. Then, it computes the average 
    intensities within the diastole and systole segmentation masks. These averages are used to define 
    normalization parameters that scale the image intensity values according to the formula:

    n(I) = a * I / sqrt(1 + beta * I^2)

    Args:
        x (np.ndarray): 4D array with shape [x, y, z, t], representing the image.
        seg_dia (np.ndarray): 3D array with shape [x, y, z], representing the diastole segmentation mask.
        seg_sys (np.ndarray): 3D array with shape [x, y, z], representing the systole segmentation mask.
        tdia (int): Time index for the diastole phase.
        tsys (int): Time index for the systole phase.

    Returns:
        np.ndarray: The normalized 4D image array.
    """
    p95 = np.percentile(x, 95)
    x = np.clip(x, 0, p95)

    avg_dia = np.mean(x[..., tdia], where=seg_dia.astype('bool'))
    avg_sys = np.mean(x[..., tsys], where=seg_sys.astype('bool'))
    avg = 0.5 * (avg_dia + avg_sys)

    # n(I) = a I/sqrt(1+beta I**2)
    norm_a = math.sqrt(p95 * p95 - avg * avg) / (math.sqrt(3) * p95 * avg)
    norm_b = (p95 * p95 - 4. * avg * avg) / (3. * p95 * p95 * avg * avg)
    x = norm_a * x / np.sqrt(1 + norm_b * x**2)
    # print("norm(per95) = ", norm_a * p95 / math.sqrt(1 + norm_b * p95 * p95))
    # print("norm(avg) = ", norm_a * avg / math.sqrt(1 + norm_b * avg * avg))
    # print(np.min(x), np.max(x))
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

        # Normalize image intensity values
        norm_img_array = normalize(patient.img_array(), patient.seg_dia_array(), patient.seg_sys_array(), patient.tdia, patient.tsys)

        # Update patient image with normalized values
        patient.img_from_array(norm_img_array, patient.img.affine.copy(), patient.img.header.copy(), update_shape=False)

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
