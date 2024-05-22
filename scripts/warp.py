from absl import flags, app
from ml_collections import config_dict, config_flags
import yaml
import os.path as osp


from svs.modules.flow.warping import OpticalFlowWarper


# Load default configuration from config file. However, the config parameters can be modified
# via command line args, for example: python scripts/flow.py --config.debug.viz=false
_CONFIG = config_flags.DEFINE_config_dict('config', config_dict.ConfigDict(
    yaml.load(open(osp.join('conf', 'warping.yaml'), 'r'), Loader=yaml.FullLoader)))


def main(_):
    cfg = _CONFIG.value
    processor = OpticalFlowWarper(cfg)
    processor.process()

if __name__ == '__main__':
    app.run(main)
