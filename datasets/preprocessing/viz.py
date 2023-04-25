from PIL import Image
import glob
import os.path as osp
import sys

from natsort import natsorted

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
import utilities.plots as plots
from preprocessing.transforms import get_viz_transforms


def save_gif(imgs_dir, dur):
    frames = [Image.open(image) for image in natsorted(glob.glob(f"{imgs_dir}/*.png"))]
    frame_one = frames[0]
    frame_one.save(osp.join(imgs_dir, "animation.gif"), format="GIF", append_images=frames,
                   save_all=True, duration=dur, loop=0)


def save_png(data, viz_cfg, save_dir):
    transforms = get_viz_transforms(viz_cfg)
    data = transforms(data)
    is_test = data['label'].shape[0] > 2

    for t in range(data['image'].shape[0]):
        if is_test:
            label = data['label'][t]
        else:
            if t == data['es']:
                label = data['label'][0]
            elif t == data['ed']:
                label = data['label'][1]
            else:
                label = None
        plots.save_overlaped_img_mask(data['image'][t], label, '{}_{}.png'.format(data['patient'], t), save_dir, 0.3)
