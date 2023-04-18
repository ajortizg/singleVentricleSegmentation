import torch
import configparser
import sys
import os.path as osp
import os.path as osp
from evaluator import *


ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
import utilities.file_paths_utils as fpu
from utilities import param_reader


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = configparser.ConfigParser()
    config.read('parser/configCNNEval.ini')
    P = param_reader.eval_params(config)

    save_dir = fpu.create_save_dir(P['out_path'], 'EVAL')
    param_reader.save_config(config, save_dir, 'config.ini')
    logger = fpu.create_logger(save_dir)
    logger.info('save_dir: '+ save_dir)

    eval = Evalautor(config, P, device, logger, save_dir, verbose=True)
    eval.evaluate()
    eval.save_report()

    if P['ccc']:
        eval.complete_cardiac_cycle(config)
