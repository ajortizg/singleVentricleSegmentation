import os.path as osp
import time
import os


def create_save_dir(out_dir, name):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    save_dir = osp.join(out_dir, name + "_" + timestr)
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir


def create_sub_dir(save_dir, subdir):
    subdir = osp.join(save_dir, subdir)
    if not osp.exists(subdir):
        os.makedirs(subdir)
    return subdir
