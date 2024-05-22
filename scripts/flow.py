"""
This script computes 3D optical flow using the TVL1-3D formulation. It reads data from a
MRIBaseDataset, processes it with a TVL13DOpticalFlow object, and saves the results.

**Configuration:**
- The script loads the default configuration from a YAML file ('conf/flow.yaml').
- Command-line arguments can be used to modify specific configuration parameters.
"""
import time
from absl import flags, app
from ml_collections import config_dict, config_flags
import yaml
import os.path as osp
from tqdm import tqdm
import torch
import numpy as np

from svs.modules.flow.processor import OpticalFlowProcessor

# Load default configuration from config file. However, the config parameters can be modified
# via command line args, for example: python scripts/flow.py --config.flow.mode=backward
_CONFIG = config_flags.DEFINE_config_dict('config', config_dict.ConfigDict(
    yaml.load(open(osp.join('conf', 'flow.yaml'), 'r'), Loader=yaml.FullLoader)))


def main(_):
    cfg = _CONFIG.value
    processor = OpticalFlowProcessor(cfg)
    processor.process()


if __name__ == '__main__':
    app.run(main)
