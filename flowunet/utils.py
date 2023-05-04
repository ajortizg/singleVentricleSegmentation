import os.path as osp
import sys


ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
from datasets.flowunet_dataset import FlowUNetDataset
from utilities.fold import create_5fold


def get_fold(config):
    """
    Returns fold indices, fold number and number of classes
    """
    root_dir = config['root_dir']
    fold = config.getint('fold')
    dataset = FlowUNetDataset(root_dir, mode='train')
    return create_5fold(len(dataset))[fold], fold, dataset.num_classes()
