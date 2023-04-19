import os.path as osp
import sys

from monai.losses import DiceCELoss, DiceLoss
import torch.nn as nn
import torch

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
from utilities.fold import create_5fold
from segmentation.dataset import SegmentationDataset
import segmentation.transforms as T
from utilities.parser_conversions import str_to_tuple


def get_fold(config):
    root_dir = config['root_dir']
    fold = config.getint('fold')
    dataset = SegmentationDataset(root_dir, mode='train')
    return create_5fold(len(dataset))[fold], fold


def get_transforms(config):
    params = config['PARAMETERS']
    DA = config['DATA_AUGMENTATION']
    img_sz = params.getint('img_size')
    img_shape = (img_sz, img_sz, img_sz)
    num_classes = params.getint('num_classes')

    norm_type = params['norm']
    if norm_type == 'qn':
        norm_fns = [T.QuadraticNormalization(q2=95, use_label=True, keys=['image'], label_key='label')] * 3
    elif norm_type == 'zscore':
        norm_fns = [T.ZScoreNormalization(keys=['image'])] * 3
    elif norm_type == 'minmax':
        norm_fns = [T.MinMaxNormalization(q1=0.5, q2=99.5, keys=['image'])] * 3
    else:
        norm_fns = None
        raise ValueError('Unsoported normalization.')

    train_transforms = T.Compose([
        T.AddNLeadingDims(
            n=2,
            keys=['image', 'label']
        ),
        T.BCXYZ_To_BCZYX(
            keys=['image', 'label']
        ),
        T.ToRAS(
            keys=['image', 'label']
        ),
        T.CropForeground(
            keys=['image', 'label'],
            label_key='label'
        ),
        norm_fns[0],
        T.Resize(
            1.0,
            img_shape,
            keys=['image', 'label'],
            label_key='label'
        ),

        # Geometric transformations
        T.RandomRotate(
            DA.getfloat('rot_prob'),
            str_to_tuple(DA['rot_x_range'], float),
            str_to_tuple(DA['rot_y_range'], float),
            str_to_tuple(DA['rot_z_range'], float),
            DA['rot_boundary'],
            keys=['image', 'label'],
            label_key='label'
        ),
        T.RandomScale(
            DA.getfloat('scaling_prob'),
            str_to_tuple(DA['scaling_range'], float),
            DA['scaling_boundary'],
            keys=['image', 'label'],
            label_key='label'
        ),
        T.RandomFlip(
            DA.getfloat('depth_flip_prob'),
            2,
            keys=['image', 'label']
        ),
        T.RandomFlip(
            DA.getfloat('vertical_flip_prob'),
            3,
            keys=['image', 'label']
        ),
        T.RandomFlip(
            DA.getfloat('horizontal_flip_prob'),
            4,
            keys=['image', 'label']
        ),
        T.ElasticDeformation(
            DA.getfloat('ed_prob'),
            str_to_tuple(DA['ed_sigma_range'], float),
            DA.getint('ed_grid'),
            DA['ed_boundary'],
            DA['ed_axis'],
            keys=['image', 'label'],
            label_key='label'
        ),
        T.SimulateLowResolution(
            DA.getfloat('lowres_prob'),
            str_to_tuple(DA['lowres_zoom_range'], float),
            keys=['image', 'label'],
            label_key='label'
        ),

        # Intensity transformations
        T.AdditiveGaussianNoise(
            DA.getfloat('noise_prob'),
            str_to_tuple(DA['noise_std_range'], float),
            DA.getfloat('noise_mu'),
            keys=['image']
        ),
        T.GaussialBlur(
            DA.getfloat('blur_prob'),
            str_to_tuple(DA['blur_sigma_range'], float),
            keys=['image']
        ),
        T.MultiplicativeScaling(
            DA.getfloat('mult_scaling_prob'),
            str_to_tuple(DA['mult_scaling_range']),
            keys=['image']
        ),
        T.ContrastAugmentation(
            DA.getfloat('contrast_prob'),
            str_to_tuple(DA['contrast_range']),
            DA.getboolean('contrast_preserve_range'),
            keys=['image']
        ),
        T.GammaCorrection(
            DA.getfloat('gamma_scaling_prob'),
            str_to_tuple(DA['gamma_scaling_range'], float),
            invert_image=False,
            retain_stats=DA.getboolean('gamma_retain_stats'),
            keys=['image']
        ),
        T.GammaCorrection(
            DA.getfloat('gamma_scaling_prob') / 2.0,
            str_to_tuple(DA['gamma_scaling_range'], float),
            invert_image=True,
            retain_stats=DA.getboolean('gamma_retain_stats'),
            keys=['image']
        ),

        T.OneHotEncoding(
            num_classes,
            keys=['label']
        ),
        T.RemoveNLeadingDims(
            n=1,
            keys=['image', 'label']
        ),
        T.ToTensor(
            keys=['image', 'label']
        )
    ])

    val_transforms = T.Compose([
        T.AddNLeadingDims(
            n=2,
            keys=['image', 'label']
        ),
        T.BCXYZ_To_BCZYX(
            keys=['image', 'label']
        ),
        T.ToRAS(
            keys=['image', 'label']
        ),
        T.CropForeground(
            keys=['image', 'label'],
            label_key='label'
        ),
        norm_fns[1],
        T.Resize(
            1.0,
            img_shape,
            keys=['image', 'label']
        ),
        T.OneHotEncoding(
            num_classes,
            keys=['label']
        ),
        T.RemoveNLeadingDims(
            n=1,
            keys=['image', 'label']
        ),
        T.ToTensor(
            keys=['image', 'label']
        )
    ])

    test_transforms = T.Compose([
        T.XYZT_To_TZYX(
            keys=['image', 'label']
        ),
        T.AddDimAt(
            axis=1,
            keys=['image', 'label']
        ),
        T.ToRAS(
            keys=['image', 'label']
        ),
        T.CropForeground(
            keys=['image', 'label'],
            label_key='label'
        ),
        norm_fns[2],
        T.Resize(
            1.0,
            img_shape,
            keys=['image', 'label']
        ),
        T.OneHotEncoding(
            num_classes,
            keys=['label']
        ),
        T.ToTensor(
            keys=['image', 'label']
        )
    ])
    return train_transforms, val_transforms, test_transforms


def get_loss_fn(config):
    params = config['PARAMETERS']
    loss_fn_type = params['loss_fn']
    if loss_fn_type == 'ce':
        loss_fn = nn.CrossEntropyLoss()
    elif loss_fn_type == 'dice':
        loss_fn = DiceLoss(softmax=True)
    elif loss_fn_type == 'dice_ce':
        lambda_dice = params.getfloat('lambda_dice')
        lambda_ce = params.getfloat('lambda_ce')
        loss_fn = DiceCELoss(softmax=True, lambda_dice=lambda_dice, lambda_ce=lambda_ce)
    else:
        loss_fn = None
        raise ValueError('Unsoported loss function.')
    return loss_fn


def create_checkpoint(net, e, opt, H, filepath):
    torch.save({'epoch': e,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'train_loss': H['train_loss'][-1],
                'train_acc': H['train_dice'][-1],
                'val_loss': H['val_loss'][-1],
                'val_acc': H['val_dice'][-1],
                'test_acc': H['test_dice'][-1]
                }, filepath)
