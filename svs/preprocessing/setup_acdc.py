import numpy as np
import nibabel as nib
import os.path as osp
from tqdm import tqdm
import pandas as pd
import shutil
import yaml
from ml_collections import config_dict

from svs.utils import dirs
from svs.modules.datasets import RawACDCDadataset


def extract_label(seg: nib.Nifti1Image, label: int) -> nib.Nifti1Image:
    """
    Extracts the specified label from the segmentation mask.

    Args:
        seg (Nifti1Image): Input segmentation mask image.
        label (int): Label value to extract from the segmentation masks.

    Returns:
        Nifti1Image: Mask image with the specified label extracted.
    """
    seg_array_res = np.where(seg.get_fdata() == label, 1.0, 0.0)
    seg_res = nib.Nifti1Image(seg_array_res, affine=seg.affine, header=seg.header)
    return seg_res


def run(config_dir='conf', config_name='setup_acdc.yaml'):
    # Load configuration
    cfg = config_dict.ConfigDict(yaml.load(open(osp.join(config_dir, config_name), 'r'), Loader=yaml.FullLoader))

    roi = cfg.label.roi
    tag = cfg.label.tag
    save_dir = dirs.create_timestamped_dir(cfg.data.save_dir, f'acdc_{roi}')
    out_imgs_dir = dirs.create_subdir(save_dir, cfg.data.out_imgs_dir)
    out_segs_dir = dirs.create_subdir(save_dir, cfg.data.out_segs_dir)

    print(f'Extracting: {roi}, wit Tag: {tag}')

    # Initialize DataFrame for dataset metadata
    df_dset = pd.DataFrame(columns=['Name', 'Systole', 'Diastole'])
    ds = RawACDCDadataset(cfg.data.raw_dir)

    for i in tqdm(range(len(ds))):
        patient = ds[i]
        patient.seg_dia = extract_label(patient.seg_dia, tag)
        patient.seg_sys = extract_label(patient.seg_sys, tag)
        patient.save_nifti(out_imgs_dir, out_segs_dir)

        if cfg.debug.viz:
            patient.viz_data(osp.join(save_dir, "images"), cfg.debug.gif, cfg.debug.dur)

        df_dset = pd.concat([df_dset,  pd.DataFrame({'Name': patient.name, 'Systole': patient.tsys,
                            'Diastole': patient.tdia}, index=[0])], ignore_index=True)

    # Save metadata and configuration
    df_dset.to_excel(osp.join(save_dir, cfg.data.out_metadata_file), index=False)
    shutil.copyfile(osp.join(config_dir, config_name), osp.join(save_dir, config_name))
    return save_dir


if __name__ == "__main__":
    save_dir = run()
