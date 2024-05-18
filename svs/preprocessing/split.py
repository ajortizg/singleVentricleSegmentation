import yaml
from ml_collections import config_dict
import shutil
import os.path as osp
import numpy as np
import shutil

from svs.modules.datasets import MRIBaseDataset
from svs.utils import dirs


def run(config_dir='conf', config_name='preprocessing.yaml', base_dir=None):
    print("Split")

    # Load configuration file
    cfg = config_dict.ConfigDict(yaml.load(open(osp.join(config_dir, config_name), 'r'), Loader=yaml.FullLoader))
    np.random.seed(cfg.split.seed)

    # Create output dirs
    save_dir = dirs.create_timestamped_dir(cfg.data.out_dir, 'preprocessing_split')

    if base_dir is not None:
        cfg.data.base_dir = base_dir
    ds = MRIBaseDataset(cfg.data.base_dir, cfg.data.imgs_dir, cfg.data.segs_dir, cfg.data.metadata_file)
    df = ds.df.copy()

    val_idxs = np.random.choice(np.arange(len(ds)), cfg.split.val_size, replace=False)
    df['Split'] = 'train'
    df.loc[val_idxs, 'Split'] = 'val'

    shutil.copytree(cfg.data.base_dir, save_dir, dirs_exist_ok=True)
    df.to_excel(osp.join(save_dir, cfg.data.metadata_file), index=False)

    # Save the configuration used for preprocessing
    with open(osp.join(save_dir, 'config.json'), 'w') as f:
        f.write(cfg.to_json(indent=4))

    return save_dir


if __name__ == "__main__":
    run()
