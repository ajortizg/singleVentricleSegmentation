import json
import os.path as osp
import time
import os
import logging
import csv


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


def write_list(save_dir, filename, lst):
    with open(osp.join(save_dir, filename), 'w') as fp:
        writer = csv.writer(fp)
        for elem in lst:
            writer.writerow(elem)
