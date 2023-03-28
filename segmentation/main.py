from dataset import SVDSegmentation
import transforms as T
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import os.path as osp
import sys

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
from utils import plots
from cnn.models import model_factory

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    train_transforms = T.Compose([
        T.CropForeground(p=1.0, tol=10),
        T.Resize(p=1.0, size=(96, 96, 96)),
        T.MinMaxNormalization(p=1.0),
        T.RandomRotate(p=1.0, range_z=(0, 360), boundary='border'),
        T.BinarizeMasks(th=0.5),
        T.ToTensor(add_ch_dim=True)
        # T6.ElasticDeformation(P['ed_prob'], P['ed_sigma_range'], P['ed_grid'], P['ed_boundary'], P['ed_prefilter'], P['ed_axis'], P['ed_order'], P['clip_interval']),
        # T6.RandomVerticalFlip(P['vflip_prob']),
        # T6.RandomHorizontalFlip(P['hflip_prob']),
        # T6.RandomDepthFlip(P['dflip_prob']),
        # T6.GammaScaling(P['gamma_scaling_prob'], P['gamma_scaling_range']),
        # T6.MutiplicativeScaling(P['mult_scaling_prob'], P['gamma_scaling_range'], P['clip_interval']),
        # T6.AdditiveScaling(P['add_scaling_prob'], P['add_scaling_mean'], P['add_scaling_std'], P['clip_interval']),
        # T6.AdditiveGaussianNoise(P['noise_prob'], P['noise_mu'], P['noise_std'], P['clip_interval']),
    ])

    ds = SVDSegmentation('data/svd_segmentation', mode='train', transforms=train_transforms)
    loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=8, collate_fn=ds.collate_fn)

    save_dir = plots.createSaveDirectory('results_2023', 'SEG')

    # blue = [1, 0.7, 0]
    # red = [0, 0, 1]
    # tr = T.Compose([T.ToArray(), T.Resize(p=1.0, size=(16, 96, 96)), T.ToTensor()])
    for i, data in enumerate(loader):
        img = data['img'].to(device)
        mask = data['mask'].to(device)
        # patient = data['patient'].to(d)

        print(img[..., 0].shape, mask[..., 0].shape)

        # plots.save_img_masks(img[..., 0], [mask[..., 0]], f'{patient}_es.png', save_dir, th=0.5, alphas=[0.25], colors=[blue])
        # plots.save_img_masks(img[..., 1], [mask[..., 1]], f'{patient}_ed.png', save_dir, th=0.5, alphas=[0.25], colors=[red])
