import numpy as np
from tqdm import tqdm
import os.path as osp
import pandas as pd
from ml_collections import config_dict
import yaml

from svs.utils import dirs
from svs.modules.datasets import MRIBaseDataset, Patient


def get_range_mask(mask, print_range=False, name=""):
    """
    Compute the bounding box of non-zero values in a 3D mask.

    Args:
        mask (np.ndarray): The 3D mask array.
        print_range (bool): Whether to print the range of the mask.
        name (str): Name of the mask for printing purposes.

    Returns:
        Tuple[int, int, int, int, int, int]: The bounding box (zmin, zmax, ymin, ymax, xmin, xmax).
    """
    x, y, z = np.nonzero(mask)
    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(y)
    ymax = np.max(y)
    zmin = np.min(z)
    zmax = np.max(z)
    if print_range:
        print("\nrange of mask", name, ":")
        print("(xmin, xmax) = ", xmin, ",", xmax)
        print("(ymin, ymax) = ", ymin, ",", ymax)
        print("(zmin, zmax) = ", zmin, ",", zmax)
    return zmin, zmax, ymin, ymax, xmin, xmax


def run(config_dir='conf', config_name='preprocessing.yaml', base_dir=None):
    print('Cutting')
    # Load configuration
    cfg = config_dict.ConfigDict(yaml.load(open(osp.join(config_dir, config_name), 'r'), Loader=yaml.FullLoader))
    save_dir = dirs.create_timestamped_dir(cfg.data.out_dir, "preprocessing_cut")

    if base_dir is not None:
        cfg.data.base_dir = base_dir
    ds = MRIBaseDataset(cfg.data.base_dir, cfg.data.imgs_dir, cfg.data.segs_dir, cfg.data.metadata_file)

    # Generate columns for (x,y,z)-shifts and original dimensions
    xshifts = np.zeros(len(ds))
    yshifts = np.zeros(len(ds))
    zshifts = np.zeros(len(ds))
    original_NX = np.zeros(len(ds))
    original_NY = np.zeros(len(ds))
    original_NZ = np.zeros(len(ds))
    original_NT = np.zeros(len(ds))

    out_img_dir = dirs.create_subdir(save_dir, cfg.data.imgs_dir)
    out_seg_dir = dirs.create_subdir(save_dir, cfg.data.segs_dir)

    for i in tqdm(range(len(ds))):
        patient = ds[i]

        # Get bounding boxes of diastole and systole masks
        zmin_dia, zmax_dia, ymin_dia, ymax_dia, xmin_dia, xmax_dia = get_range_mask(patient.seg_dia.get_fdata())
        zmin_sys, zmax_sys, ymin_sys, ymax_sys, xmin_sys, xmax_sys = get_range_mask(patient.seg_sys.get_fdata())

        # Original shape of the 4D image
        NX, NY, NZ, NT = patient.img_array().shape

        # Extend range by tolerance values from the configuration
        x_tol = cfg.cutting.x_tol
        y_tol = cfg.cutting.y_tol
        z_tol = cfg.cutting.z_tol
        xmin_total = max(0, min(xmin_dia, xmin_sys) - x_tol)
        xmax_total = min(NX - 1, max(xmax_dia, xmax_sys) + x_tol)
        ymin_total = max(0, min(ymin_dia, ymin_sys) - y_tol)
        ymax_total = min(NY - 1, max(ymax_dia, ymax_sys) + y_tol)
        zmin_total = max(0, min(zmin_dia, zmin_sys) - z_tol)
        zmax_total = min(NZ - 1, max(zmax_dia, zmax_sys) + z_tol)

        xshifts[i] = xmin_total
        yshifts[i] = ymin_total
        zshifts[i] = zmin_total
        original_NX[i] = NX
        original_NY[i] = NY
        original_NZ[i] = NZ
        original_NT[i] = NT

        # Create a new patient with cropped data
        cut_patient = Patient(
            name=patient.name,
            tsys=patient.tsys,
            tdia=patient.tdia,
            init_ts=patient.init_ts,
            final_ts=patient.final_ts
        )

        # Crop and save the image and segmentation masks
        cut_patient.img_from_array(
            x=patient.img_array()[xmin_total:xmax_total + 1, ymin_total:ymax_total + 1, zmin_total:zmax_total + 1, :],
            affine=patient.img.affine.copy(),
            header=patient.img.header.copy(),
            update_shape=True
        )
        cut_patient.seg_dia_from_array(
            x=patient.seg_dia_array()[xmin_total:xmax_total + 1, ymin_total:ymax_total + 1, zmin_total:zmax_total + 1],
            affine=patient.seg_dia.affine.copy(),
            header=patient.seg_dia.header.copy(),
            update_shape=True
        )
        cut_patient.seg_sys_from_array(
            x=patient.seg_sys_array()[xmin_total:xmax_total + 1, ymin_total:ymax_total + 1, zmin_total:zmax_total + 1],
            affine=patient.seg_sys.affine.copy(),
            header=patient.seg_sys.header.copy(),
            update_shape=True
        )
        cut_patient.write_nifti(out_img_dir, out_seg_dir)

        if cfg.debug.viz:
            cut_patient.viz_data(osp.join(save_dir, "images"), cfg.debug.gif, cfg.debug.dur, cfg.debug.alpha, cfg.debug.color, cfg.debug.aspect_ratio)

   # Save metadata with shifts and original dimensions
    output_df = ds.df.copy()
    output_df['xshift'] = xshifts
    output_df['yshift'] = yshifts
    output_df['zshift'] = zshifts
    output_df['original_NX'] = original_NX
    output_df['original_NY'] = original_NY
    output_df['original_NZ'] = original_NZ
    output_df['original_NT'] = original_NT
    output_df_file = osp.join(save_dir, cfg.data.metadata_file)
    output_df.to_excel(output_df_file, index=False)

    with open(osp.join(save_dir, "config.json"), "w") as f:
        f.write(cfg.to_json(indent=4))

    return save_dir


if __name__ == "__main__":
    run()
