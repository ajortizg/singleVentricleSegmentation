import numpy as np
import os
import torch
import logging
import sys


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
