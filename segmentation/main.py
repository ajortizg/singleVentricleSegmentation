from dataset import SVDataset, ACDCDataset
import transforms as T
import torch
import os.path as osp
import sys
from tqdm import tqdm
import configparser


ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
from utils import plots


if __name__ == '__main__':
    save_dir = plots.createSaveDirectory('results', 'IMGS')
    transforms = T.Compose([
        T.ToRAS(),
        T.ToTensor()
    ])
    ds = SVDataset('data/singleVentricleData', 'train', transforms)
    for data in tqdm(ds):
        patient = data['patient']
        ts = data['mask'].shape[-1]
        for i in range(ts):
            plots.save_overlaped_img_mask(data['img'][..., i],
                                          data['mask'][..., i],
                                          '{}_t{}'.format(patient, i),
                                          save_dir,
                                          0.5,
                                          0.3)


    # save_dir = plots.createSaveDirectory('results', 'TESTS')

    # transforms = T.Compose([
    #     T.CropForeground(p=1.0, tol=10),
    #     T.Resize(p=1.0, size=(96, 96, 96)),
    #     # T.ZScoreNormalization(p=1.0),
    #     # T.QuadraticNormalization(p=1.0),
    #     T.MinMaxNormalization(p=1.0),
    #     # T.RandomRotate(p=1.0, range_z=(0, 360), boundary='border'),
    #     # T.RandomVerticalFlip(p=1.0),
    #     # T.RandomHorizontalFlip(p=1.0),
    #     # T.RandomDepthFlip(p=0.5),
    #     T.ElasticDeformation(1.0, (1.4, 1.5), 8, 'nearest', False, 'yx', 1),
    #     T.BinarizeMasks(th=0.5),
    #     T.ToTensor(add_ch_dim=False)
    #     # T.ElasticDeformati),
    #     # T6.GammaScaling(P['gamma_scaling_prob'], P['gamma_scaling_range']),
    #     # T6.MutiplicativeScaling(P['mult_scaling_prob'], P['gamma_scaling_range'], P['clip_interval']),
    #     # T6.AdditiveScaling(P['add_scaling_prob'], P['add_scaling_mean'], P['add_scaling_std'], P['clip_interval']),
    #     # T6.AdditiveGaussianNoise(P['noise_prob'], P['noise_mu'], P['noise_std'], P['clip_interval']),
    # ])

    # ds = SVDSegmentation('data/svd_segmentation', mode='test', transforms=transforms)
    # red = [0, 0, 1]
    # saving_transforms = T.Compose([T.ToArray(), T.Resize(p=1.0, size=(16, 96, 96)), T.ToTensor(add_ch_dim=False)])
    # for data in ds:
    #     data = saving_transforms(data)

    #     NT = data['img'].shape[-1]
    #     patient_dir = plots.createSubDirectory(save_dir, data['patient'])
    #     for t in range(NT):
    #         img = data['img'][..., t]
    #         mask = data['mask'][..., t]

    #         plots.save_img_masks(img, [mask], f'img_{t}.png', patient_dir, 0.5, [0.2], [red])
