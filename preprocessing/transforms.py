import os.path as osp
import sys

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
from utils import plots
import TVL1OF.transforms as T


def get_transforms(img_size):
    transforms = T.Compose([
        T.XYZT_To_TZYX(keys=['image', 'label']),
        T.ToRAS(keys=['image', 'label']),
        T.CropForeground(tol=10, keys=['image', 'label'], label_key='label'),
        T.QuadraticNormalization(q2=95, keys=['image'], label_key='label'),
        T.Resize(p=1.0, size=(img_size, img_size, img_size), keys=['image', 'label'], label_key='label'),
        T.ToTensor()
    ])
    return transforms
