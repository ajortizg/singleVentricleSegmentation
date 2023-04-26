import os.path as osp
import sys

import torch

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
from utilities.fold import create_5fold
from datasets.segmentation_dataset import SegmentationDataset
import segmentation.transforms as T


def get_fold(config):
    root_dir = config['root_dir']
    fold = config.getint('fold')
    dataset = SegmentationDataset(root_dir, mode='train')
    return create_5fold(len(dataset))[fold], fold


def norm_fn(norm_type):
    if norm_type == 'qn':
        norm = T.QuadraticNormalization(q2=95, use_label=True, keys=['image'], label_key='label')
    elif norm_type == 'zscore':
        norm = T.ZScoreNormalization(keys=['image'])
    elif norm_type == 'minmax':
        norm = T.MinMaxNormalization(q1=0.5, q2=99.5, keys=['image'])
    else:
        norm = None
        raise ValueError('Unsoported normalization.')
    return norm


# def create_checkpoint(net, e, opt, H, filepath):
#     torch.save({'epoch': e,
#                 'model_state_dict': net.state_dict(),
#                 'optimizer_state_dict': opt.state_dict(),
#                 'train_loss': H['train_loss'][-1],
#                 'train_acc': H['train_dice'][-1],
#                 'val_loss': H['val_loss'][-1],
#                 'val_acc': H['val_dice'][-1],
#                 'test_acc': H['test_dice'][-1]
#                 }, filepath)
