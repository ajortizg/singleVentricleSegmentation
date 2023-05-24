import configparser
import os
import os.path as osp

__all__ = ['save_config', 'train_params', 'eval_params', 'fine_tuning_params']


def save_config(config, save_dir, filename='config.ini'):
    # save config file to save directory
    conifg_output = osp.join(save_dir, filename)
    with open(conifg_output, 'w') as config_file:
        config.write(config_file)


def train_params(config):
    rot_range_x = tuple(map(float, config.get('DATA_AUGMENTATION', 'ROT_X_RANGE').split(',')))
    rot_range_y = tuple(map(float, config.get('DATA_AUGMENTATION', 'ROT_Y_RANGE').split(',')))
    rot_range_z = tuple(map(float, config.get('DATA_AUGMENTATION', 'ROT_Z_RANGE').split(',')))
    mult_scaling_range = tuple(map(float, config.get('DATA_AUGMENTATION', 'MULT_SCALING_RANGE').split(',')))
    gamma_scaling_range = tuple(map(float, config.get('DATA_AUGMENTATION', 'GAMMA_SCALING_RANGE').split(',')))
    ed_sigma_range = tuple(map(float, config.get('DATA_AUGMENTATION', 'ED_SIGMA_RANGE').split(',')))
    noise_std_range = tuple(map(float, config.get('DATA_AUGMENTATION', 'NOISE_STD_RANGE').split(',')))
    contrast_range = tuple(map(float, config.get('DATA_AUGMENTATION', 'contrast_range').split(',')))
    blur_sigma_range = tuple(map(float, config.get('DATA_AUGMENTATION', 'blur_sigma_range').split(',')))

    params = {
        'batch_size': config.getint('PARAMETERS', 'BATCH_SIZE'),
        'lr': config.getfloat('PARAMETERS', 'LR'),
        'step_size': config.getfloat('PARAMETERS', 'STEP_SIZE'),
        'gamma': config.getfloat('PARAMETERS', 'GAMMA'),
        'weight_decay': config.getfloat('PARAMETERS', 'WEIGHT_DECAY'),
        'beta1': config.getfloat('PARAMETERS', 'BETA1'),
        'beta2': config.getfloat('PARAMETERS', 'BETA2'),
        'epochs': config.getint('PARAMETERS', 'NUM_EPOCHS'),
        'loss_lambda': config.getfloat('PARAMETERS', 'LOSS_LAMBDA'),
        'loss_gamma': config.getfloat('PARAMETERS', 'LOSS_PENALIZATION_GAMMA'),
        'gpus': config.getint('PARAMETERS', 'NUM_GPUS'),
        'workers': config.getint('PARAMETERS', 'NUM_WORKERS'),
        'pretrained': config.getboolean('PARAMETERS', 'PRETRAINED'),
        'checkpoint_file': config.get('PARAMETERS', 'CHECKPOINT_FILE'),
        'patience': config.getint('PARAMETERS', 'PATIENCE'),
        'which': config.get('PARAMETERS', 'WHICH'),

        # Flip
        'vflip_prob': config.getfloat('DATA_AUGMENTATION', 'VERTICAL_FLIP_PROB'),
        'hflip_prob': config.getfloat('DATA_AUGMENTATION', 'HORIZONTAL_FLIP_PROB'),
        'dflip_prob': config.getfloat('DATA_AUGMENTATION', 'DEPTH_FLIP_PROB'),

        # Rotation
        'rot_prob': config.getfloat('DATA_AUGMENTATION', 'ROT_PROB'),
        'rot_boundary': config.get('DATA_AUGMENTATION', 'ROT_BOUNDARY'),
        'rot_range_x': rot_range_x,
        'rot_range_y': rot_range_y,
        'rot_range_z': rot_range_z,

        # Multiplicative scaling
        'mult_scaling_prob': config.getfloat('DATA_AUGMENTATION', 'MULT_SCALING_PROB'),
        'mult_scaling_range': mult_scaling_range,

        # Additive scaling
        'add_scaling_prob': config.getfloat('DATA_AUGMENTATION', 'ADD_SCALING_PROB'),
        'add_scaling_mean': config.getfloat('DATA_AUGMENTATION', 'ADD_SCALING_MEAN'),
        'add_scaling_std': config.getfloat('DATA_AUGMENTATION', 'ADD_SCALING_STD'),

        # Gamma scaling
        'gamma_scaling_prob': config.getfloat('DATA_AUGMENTATION', 'GAMMA_SCALING_PROB'),
        'gamma_scaling_range': gamma_scaling_range,
        'gamma_retain_stats': config.getboolean('DATA_AUGMENTATION', 'gamma_retain_stats'),
        'gamma_invert_image': config.getboolean('DATA_AUGMENTATION', 'gamma_invert_image'),

        # Gaussian noise
        'noise_prob': config.getfloat('DATA_AUGMENTATION', 'NOISE_PROB'),
        'noise_mu': config.getfloat('DATA_AUGMENTATION', 'NOISE_MU'),
        'noise_std_range': noise_std_range,

        # Elastic deformation
        'ed_prob': config.getfloat('DATA_AUGMENTATION', 'ED_PROB'),
        'ed_grid': config.getint('DATA_AUGMENTATION', 'ED_GRID'),
        'ed_sigma_range': ed_sigma_range,
        'ed_boundary': config.get('DATA_AUGMENTATION', 'ED_BOUNDARY'),
        'ed_prefilter': config.getboolean('DATA_AUGMENTATION', 'ED_USE_PREFILTER'),
        'ed_axis': config.get('DATA_AUGMENTATION', 'ED_AXIS'),
        'ed_order': config.getint('DATA_AUGMENTATION', 'ED_ORDER'),

        # Contrast augmentation
        'contrast_prob': config.getfloat('DATA_AUGMENTATION', 'contrast_prob'),
        'contrast_range': contrast_range,
        'contrast_preserve_range': config.getboolean('DATA_AUGMENTATION', 'contrast_preserve_range'),

        # Gaussian blur
        'blur_prob': config.getfloat('DATA_AUGMENTATION', 'blur_prob'),
        'blur_sigma_range': blur_sigma_range
    }
    return params


def eval_params(config):
    params = {
        'trained_model_dir': config.get('DATA', 'TRAINED_MODEL_DIR'),
        'model_name': config.get('DATA', 'MODEL_NAME'),
        'out_path': config.get('DATA', 'OUTPUT_PATH'),
        'save_imgs': config.getboolean('DEBUG', 'SAVE_IMGS'),
        'save_nifti': config.getboolean('DEBUG', 'SAVE_NIFTI'),
        'workers': config.getint('PARAMETERS', 'NUM_WORKERS'),
        'dataset': config.get('DATA', 'DATASET'),
        'fine_tuning': config.getboolean('DATA', 'FINE_TUNING'),
        'save_nz': config.getint('PARAMETERS', 'save_NZ'),
        'save_ny': config.getint('PARAMETERS', 'save_NY'),
        'save_nx': config.getint('PARAMETERS', 'save_NX'),
        'ccc': config.getboolean('PARAMETERS', 'PREDICT_CARDIAC_CYCLE'),
        'patient_name': config.get('DATA', 'PATIENT_NAME')
    }
    return params


def fine_tuning_params(config):
    params = {
        'pretrained_model_dir': config.get('DATA', 'PRETRAINED_DIR'),
        'weights_filename': config.get('DATA', 'WEIGHTS_FILENAME'),
        'patient_name': config.get('DATA', 'PATIENT_NAME'),
        'num_gpus': config.getint('PARAMETERS', 'NUM_GPUS'),
        'num_workers': config.getint('PARAMETERS', 'NUM_WORKERS'),
        'num_epochs': config.getint('PARAMETERS', 'NUM_EPOCHS'),
        'batch_size': 1,
        'lr': config.getfloat('PARAMETERS', 'LR'),
        'weight_decay': config.getfloat('PARAMETERS', 'WEIGHT_DECAY'),
        'step_size': config.getfloat('PARAMETERS', 'STEP_SIZE'),
        'gamma': config.getfloat('PARAMETERS', 'GAMMA'),
        'beta1': config.getfloat('PARAMETERS', 'BETA1'),
        'beta2': config.getfloat('PARAMETERS', 'BETA2'),
        'loss_lambda': config.getfloat('PARAMETERS', 'LOSS_LAMBDA'),
        'dataset': config.get('DATA', 'DATASET')
    }
    return params
