import os.path as osp
import sys


ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
from utilities.fold import create_5fold
from segmentation.dataset import SegmentationDataset
import segmentation.transforms as T


def get_fold(config):
    root_dir = config['root_dir']
    fold = config.getint('fold')
    dataset = SegmentationDataset(root_dir, mode='train')
    return create_5fold(len(dataset))[fold], fold


def get_transforms(config):
    img_sz = config.getint('PARAMETERS', 'img_size')
    num_classes = config.getint('PARAMETERS', 'num_classes')
    data_aug = config['DATA_AUGMENTATION']

    train_transforms = T.Compose([
        T.AddNLeadingDims(n=2, keys=['image', 'label']),
        T.BCXYZ_To_BCZYX(keys=['image', 'label']),
        T.ToRAS(keys=['image', 'label']),
        T.CropForeground(keys=['image', 'label'], label_key='label'),
        T.QuadraticNormalization(q2=99, use_label=True, keys=['image'], label_key='label'),
        T.Resize(1.0, (img_sz, img_sz, img_sz), keys=['image', 'label']),
        # Geometric transformations
        T.RandomRotate(0.2, (-30, 30), (-30, 30), (-30, 30), keys=['image', 'label'], label_key='label'),
        T.RandomScale(0.2, (0.7, 1.4), keys=['image', 'label'], label_key='label'),
        T.RandomFlip(0.5, 2, keys=['image', 'label']),
        T.RandomFlip(0.5, 3, keys=['image', 'label']),
        T.RandomFlip(0.5, 4, keys=['image', 'label']),
        T.ElasticDeformation(0.1, (0.5, 2.0), 8, 'constant', 'zyx', keys=['image', 'label'], label_key='label'),
        # Intensity transformations
        T.AdditiveGaussianNoise(0.1, sigma_range=(0.0, 0.1), mu=0.0, keys=['image']),
        T.GaussialBlur(0.2, sigma_range=(0.5, 1.), keys=['image']),
        T.MultiplicativeScaling(0.15, (0.75, 1.25), keys=['image']),
        T.ContrastAugmentation(0.15, (0.75, 1.25), keys=['image']),
        T.GammaCorrection(0.1, (0.7, 1.5), True, True, keys=['image']),
        T.GammaCorrection(0.3, (0.7, 1.5), False, True, keys=['image']),

        T.OneHotEncoding(num_classes, keys=['label']),
        T.RemoveNLeadingDims(n=1, keys=['image', 'label']),
        T.ToTensor(keys=['image', 'label'])
    ])

    val_transforms = T.Compose([
        T.AddNLeadingDims(n=2, keys=['image', 'label']),
        T.BCXYZ_To_BCZYX(keys=['image', 'label']),
        T.ToRAS(keys=['image', 'label']),
        T.CropForeground(keys=['image', 'label'], label_key='label'),
        T.QuadraticNormalization(q2=99, use_label=True, keys=['image'], label_key='label'),
        T.Resize(1.0, (img_sz, img_sz, img_sz), keys=['image', 'label']),
        T.OneHotEncoding(num_classes, keys=['label']),
        T.RemoveNLeadingDims(n=1, keys=['image', 'label']),
        T.ToTensor(keys=['image', 'label'])
    ])

    test_transforms = T.Compose([
        T.XYZT_To_TZYX(keys=['image', 'label']),
        T.AddDimAt(axis=1, keys=['image', 'label']),
        T.ToRAS(keys=['image', 'label']),
        T.CropForeground(keys=['image', 'label'], label_key='label'),
        T.QuadraticNormalization(q2=99, use_label=True, keys=['image'], label_key='label'),
        T.Resize(1.0, (img_sz, img_sz, img_sz), keys=['image', 'label']),
        T.OneHotEncoding(num_classes, keys=['label']),
        T.ToTensor(keys=['image', 'label'])
    ])

    return train_transforms, val_transforms, test_transforms


#   train_transforms = T.Compose([
#         T.AddChannelDim(keys=['image', 'label']),
#         T.CXYZ_To_CZYX(keys=['image', 'label']),
#         T.ToRAS(),
#         T.CropForeground(p=1.0, tol=10),
#         T.Resize(p=1.0, size=(img_sz, img_sz, img_sz)),
#         T.RandomRotate(p=data_aug.getfloat('rot_prob'),
#                        range_z=tuple(map(float, data_aug['ROT_Z_RANGE'].split(','))),
#                        range_y=tuple(map(float, data_aug['ROT_Y_RANGE'].split(','))),
#                        range_x=tuple(map(float, data_aug['ROT_X_RANGE'].split(','))),
#                        boundary=data_aug['ROT_BOUNDARY']),
#         T.RandomVerticalFlip(data_aug.getfloat('VERTICAL_FLIP_PROB')),
#         T.RandomHorizontalFlip(data_aug.getfloat('HORIZONTAL_FLIP_PROB')),
#         T.RandomDepthFlip(data_aug.getfloat('DEPTH_FLIP_PROB')),
#         T.ElasticDeformation(p=data_aug.getfloat('ED_PROB'),
#                              sigma_range=tuple(map(float, data_aug['ED_SIGMA_RANGE'].split(','))),
#                              points=data_aug.getint('ED_GRID'),
#                              boundary=data_aug['ED_BOUNDARY'],
#                              prefilter=data_aug.getboolean('ED_USE_PREFILTER'),
#                              axis=data_aug['ED_AXIS'],
#                              order=data_aug.getint('ED_ORDER')),
#         # T.GammaScaling(data_aug.getfloat('GAMMA_SCALING_PROB'), tuple(map(float, data_aug['GAMMA_SCALING_RANGE'].split(',')))),
#         T.GammaTransform(0.1, (0.7, 1.5), True, False),
#         T.GammaTransform(0.3, (0.7, 1.5), False, False),
#         T.MutiplicativeScaling(data_aug.getfloat('MULT_SCALING_PROB'),
#                                tuple(map(float, data_aug['MULT_SCALING_RANGE'].split(',')))),
#         T.AdditiveScaling(data_aug.getfloat('ADD_SCALING_PROB'),
#                           data_aug.getfloat('ADD_SCALING_MEAN'),
#                           data_aug.getfloat('ADD_SCALING_STD')),
#         T.AdditiveGaussianNoise(data_aug.getfloat('NOISE_PROB'),
#                                 data_aug.getfloat('NOISE_MU'),
#                                 data_aug.getfloat('NOISE_STD')),
#         T.QuadraticNormalization(mean_inside_mask=True),
#         T.Discretize(th=0.5),
#         T.AddChannelDim(),
#         T.OneHotEncoding(num_classes),
#         T.ToTensor()
#     ])

#     val_transforms = T.Compose([
#         T.ToRAS(),
#         T.CropForeground(p=1.0, tol=10),
#         T.Resize(p=1.0, size=(img_sz, img_sz, img_sz)),
#         T.QuadraticNormalization(mean_inside_mask=True),
#         T.Discretize(th=0.5),
#         T.AddChannelDim(),
#         T.OneHotEncoding(num_classes),
#         T.ToTensor()
#     ])
