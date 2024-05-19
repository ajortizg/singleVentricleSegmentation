from tqdm import tqdm
from ml_collections import config_dict
import os.path as osp
import yaml
from monai.transforms import Spacing, Resize, Compose
from monai.data import MetaTensor
import torch.nn.functional as F

from svs.utils import dirs
from svs.modules.datasets import MRIBaseDataset


def time_pading(maxt: int, img: MetaTensor) -> MetaTensor:
    # img meta tensor with shape t,x,y,z
    nt = img.shape[0]
    diff_t = maxt - nt
    return F.pad(img, [0, 0,
                       0, 0,
                       0, 0,
                       0, diff_t])


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

    img_transforms = Compose([
        Spacing(pixdim=cfg.isotropic_resample.pixdim,
                mode=cfg.isotropic_resample.mode,
                padding_mode=cfg.isotropic_resample.padding_mode,
                align_corners=cfg.isotropic_resample.align_corners,
                recompute_affine=True),
        Resize(spatial_size=cfg.isotropic_resample.out_size[1:],
               mode=cfg.isotropic_resample.mode)
    ])
    segs_transforms = Compose([
        Spacing(pixdim=cfg.isotropic_resample.pixdim,
                mode='nearest',
                padding_mode=cfg.isotropic_resample.padding_mode,
                align_corners=cfg.isotropic_resample.align_corners,
                recompute_affine=True),
        Resize(spatial_size=cfg.isotropic_resample.out_size[1:],
               mode='nearest')
    ])

    for i in tqdm(range(len(ds))):
        patient = ds[i]
        patient.img.affine = patient.seg_dia.affine

        patient.img = img_transforms(patient.img)
        patient.img = time_pading(cfg.isotropic_resample.out_size[0], patient.img)
        patient.seg_dia = segs_transforms(patient.seg_dia)
        patient.seg_sys = segs_transforms(patient.seg_sys)

        patient.img.meta['spatial_shape'] = patient.img.shape[1:]
        patient.seg_dia.meta['spatial_shape'] = patient.seg_dia.shape[1:]
        patient.seg_sys.meta['spatial_shape'] = patient.seg_sys.shape[1:]

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
