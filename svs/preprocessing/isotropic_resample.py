from tqdm import tqdm
from ml_collections import config_dict
import os.path as osp
import yaml
from monai.transforms import Spacing

from svs.utils import dirs
from svs.modules.datasets import MRIBaseDataset


def run(config_dir='conf', config_name='preprocessing.yaml', base_dir=None):
    print('Isotropic resampling')

    # Load configuration file
    cfg = config_dict.ConfigDict(yaml.load(open(osp.join(config_dir, config_name), 'r'), Loader=yaml.FullLoader))

    # Create output dirs
    save_dir = dirs.create_timestamped_dir(cfg.data.out_dir, 'preprocessing_isores')
    out_img_dir = dirs.create_subdir(save_dir, cfg.data.imgs_dir)
    out_seg_dir = dirs.create_subdir(save_dir, cfg.data.segs_dir)

    if base_dir is not None:
        cfg.data.base_dir = base_dir
    ds = MRIBaseDataset(cfg.data.base_dir, cfg.data.imgs_dir, cfg.data.segs_dir, cfg.data.metadata_file)

    rescaler_img = Spacing(
        pixdim=cfg.isotropic_resample.pixdim,
        mode=cfg.isotropic_resample.mode,
        padding_mode=cfg.isotropic_resample.padding_mode,
        align_corners=cfg.isotropic_resample.align_corners,
        recompute_affine=True
    )

    rescaler_segs = Spacing(
        pixdim=cfg.isotropic_resample.pixdim,
        mode='nearest',
        padding_mode=cfg.isotropic_resample.padding_mode,
        align_corners=cfg.isotropic_resample.align_corners,
        recompute_affine=True
    )

    for i in tqdm(range(len(ds))):
        patient = ds[i]
        patient.img.affine = patient.seg_dia.affine

        patient.img = rescaler_img(patient.img)
        patient.seg_dia = rescaler_segs(patient.seg_dia)
        patient.seg_sys = rescaler_segs(patient.seg_sys)

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
