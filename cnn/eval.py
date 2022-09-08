import torch
import configparser
import sys
import os.path as osp
import os.path as osp
from evaluator import *


ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
from utils import plots
from utils import param_reader


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = configparser.ConfigParser()
    config.read('parser/configCNNEval.ini')

    P = param_reader.eval_params(config)

    save_dir = plots.createSaveDirectory(P['out_path'], 'EVAL')
    param_reader.save_config(config, save_dir, 'config.ini')
    logger = plots.create_logger(save_dir)

    eval = Evalautor(P, device, logger, save_dir, verbose=True)
    eval.evaluate(save_report=True)
