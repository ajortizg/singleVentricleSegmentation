import json
import os.path as osp
import time
import os
import logging
import csv


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


def create_save_dir(OUTPUT_PATH, name):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    saveDir = os.path.sep.join([OUTPUT_PATH, name + "_" + timestr])
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    # print("save results to directory: ", saveDir, "\n")
    return saveDir


def create_sub_dir(saveDir, SUBDIR_PATH):
    subDir = os.path.sep.join([saveDir, SUBDIR_PATH])
    if not os.path.exists(subDir):
        os.makedirs(subDir)
    return subDir


def save_config(config, save_dir, filename='config.ini'):
    # save config file to save directory
    conifg_output = osp.join(save_dir, filename)
    with open(conifg_output, 'w') as config_file:
        config.write(config_file)


def write_list(save_dir, filename, lst):
    with open(osp.join(save_dir, filename), 'w') as fp:
        writer = csv.writer(fp)
        for elem in lst:
            writer.writerow(elem)
