import numpy as np
import os
import os.path as osp
import torch
import logging
import sys
import json


def seeding(seed):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def create_logger(save_dir: str) -> logging.Logger:
    logging.basicConfig(filename=os.path.join(save_dir, "console.log"),
                        format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO, force=True)
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger


def save_model(net, save_dir, filename):
    filepath = osp.join(save_dir, filename)
    total_params = 0

    modules = [module for module in net.modules()]
    params = [param for param in net.parameters()]
    with open(filepath, 'w') as mfile:
        for idx, m in enumerate(modules):
            mfile.write(f'{idx} -> {m}\n')
        for p in params:
            total_params += p.numel()
        mfile.write(f'\nTotal parameters: {total_params}')
    mfile.close()


def save_transforms_to_json(transforms, file) -> None:
    json_list = []
    for t in transforms.transforms:
        json_list.append({t.class_name(): t.items()})
    save_json(json_list, file, default=str)


def save_json(obj, file: str, indent: int = 4, sort_keys: bool = True, default=None) -> None:
    with open(file, 'w') as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent, default=default)


def save_config(config, save_dir, filename='config.ini'):
    # save config file to save directory
    fout = osp.join(save_dir, filename)
    with open(fout, 'w') as config_file:
        config.write(config_file)
