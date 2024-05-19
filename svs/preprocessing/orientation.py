from tqdm import tqdm
from ml_collections import config_dict
import os.path as osp
import yaml
from monai.transforms import Orientation

from svs.utils import dirs
from svs.modules.datasets import MRIBaseDataset, Patient


def run(config_dir='conf', config_name='preprocessing.yaml', base_dir=None):
    print('Orientation')

    # Load configuration file
    cfg = config_dict.ConfigDict(yaml.load(open(osp.join(config_dir, config_name), 'r'), Loader=yaml.FullLoader))

    # Create output dirs
    save_dir = dirs.create_timestamped_dir(cfg.data.out_dir, 'preprocessing_orient')
    out_img_dir = dirs.create_subdir(save_dir, cfg.data.imgs_dir)
    out_seg_dir = dirs.create_subdir(save_dir, cfg.data.segs_dir)

    if base_dir is not None:
        cfg.data.base_dir = base_dir
    ds = MRIBaseDataset(cfg.data.base_dir, cfg.data.imgs_dir, cfg.data.segs_dir, cfg.data.metadata_file)

    reorient = Orientation(axcodes=cfg.orientation.axcodes)

    for i in tqdm(range(len(ds))):
        patient = ds[i]
        patient.img.affine = patient.seg_dia.affine

        patient.img = reorient(patient.img)
        patient.seg_dia = reorient(patient.seg_dia)
        patient.seg_sys = reorient(patient.seg_sys)

        patient.write_nifti(out_img_dir, out_seg_dir)

        if cfg.debug.viz:
            patient.viz_data(osp.join(save_dir, "images"), cfg.debug.gif, cfg.debug.dur, cfg.debug.alpha, cfg.debug.color, cfg.debug.aspect_ratio)

    # Save metadata to Excel file
    ds.df.to_excel(osp.join(save_dir, cfg.data.metadata_file), index=False)

    # Save configuration to JSON file
    with open(osp.join(save_dir, 'config.json'), 'w') as f:
        f.write(cfg.to_json(indent=4))

    return save_dir


if __name__ == "__main__":
    run()
