import configparser
import os
import os.path as osp

__all__ = ['ParamReader']


class ParamReader:
    def __init__(self, config):
        self.config = config

    def save(self, save_dir, filename='config.ini'):
        # save config file to save directory
        conifg_output = osp.join(save_dir, filename)
        with open(conifg_output, 'w') as config_file:
            self.config.write(config_file)

    def read_train(self):
        rot_range_x = tuple(map(float, self.config.get('DATA_AUGMENTATION', 'ROT_X_RANGE').split(',')))
        rot_range_y = tuple(map(float, self.config.get('DATA_AUGMENTATION', 'ROT_Y_RANGE').split(',')))
        rot_range_z = tuple(map(float, self.config.get('DATA_AUGMENTATION', 'ROT_Z_RANGE').split(',')))
        mult_scaling_range = tuple(map(float, self.config.get('DATA_AUGMENTATION', 'MULT_SCALING_RANGE').split(',')))
        clip_interval = tuple(map(float, self.config.get('DATA_AUGMENTATION', 'CLIP_INTERVAL').split(',')))
        gamma_scaling_range = tuple(map(float, self.config.get('DATA_AUGMENTATION', 'GAMMA_SCALING_RANGE').split(',')))
        ed_sigma_range = tuple(map(float, self.config.get('DATA_AUGMENTATION', 'ED_SIGMA_RANGE').split(',')))

        params = {
            'batch_size': self.config.getint('PARAMETERS', 'BATCH_SIZE'),
            'lr': self.config.getfloat('PARAMETERS', 'LR'),
            'step_size': self.config.getfloat('PARAMETERS', 'STEP_SIZE'),
            'gamma': self.config.getfloat('PARAMETERS', 'GAMMA'),
            'weight_decay': self.config.getfloat('PARAMETERS', 'WEIGHT_DECAY'),
            'beta1': self.config.getfloat('PARAMETERS', 'BETA1'),
            'beta2': self.config.getfloat('PARAMETERS', 'BETA2'),
            'epochs': self.config.getint('PARAMETERS', 'NUM_EPOCHS'),
            'loss_lambda': self.config.getfloat('PARAMETERS', 'LOSS_LAMBDA'),
            'gpus': self.config.getint('PARAMETERS', 'NUM_GPUS'),
            'workers': self.config.getint('PARAMETERS', 'NUM_WORKERS'),
            'pretrained': self.config.getboolean('PARAMETERS', 'PRETRAINED'),
            'checkpoint_file': self.config.get('PARAMETERS', 'CHECKPOINT_FILE'),

            # Flip
            'vflip_prob': self.config.getfloat('DATA_AUGMENTATION', 'VERTICAL_FLIP_PROB'),
            'hflip_prob': self.config.getfloat('DATA_AUGMENTATION', 'HORIZONTAL_FLIP_PROB'),
            'dflip_prob': self.config.getfloat('DATA_AUGMENTATION', 'DEPTH_FLIP_PROB'),

            # Rotation
            'rot_prob': self.config.getfloat('DATA_AUGMENTATION', 'ROT_PROB'),
            'rot_boundary': self.config.get('DATA_AUGMENTATION', 'ROT_BOUNDARY'),
            'rot_range_x': rot_range_x,
            'rot_range_y': rot_range_y,
            'rot_range_z': rot_range_z,

            # Multiplicative scaling
            'mult_scaling_prob': self.config.getfloat('DATA_AUGMENTATION', 'MULT_SCALING_PROB'),
            'mult_scaling_range': mult_scaling_range,

            # Additive scaling
            'add_scaling_prob': self.config.getfloat('DATA_AUGMENTATION', 'ADD_SCALING_PROB'),
            'add_scaling_mean': self.config.getfloat('DATA_AUGMENTATION', 'ADD_SCALING_MEAN'),
            'add_scaling_std': self.config.getfloat('DATA_AUGMENTATION', 'ADD_SCALING_STD'),

            # Gamma scaling
            'gamma_scaling_prob': self.config.getfloat('DATA_AUGMENTATION', 'GAMMA_SCALING_PROB'),
            'gamma_scaling_range': gamma_scaling_range,

            # Gaussian noise
            'noise_prob': self.config.getfloat('DATA_AUGMENTATION', 'NOISE_PROB'),
            'noise_mu': self.config.getfloat('DATA_AUGMENTATION', 'NOISE_MU'),
            'noise_std': self.config.getfloat('DATA_AUGMENTATION', 'NOISE_STD'),

            # Elastic deformation
            'ed_prob': self.config.getfloat('DATA_AUGMENTATION', 'ED_PROB'),
            'ed_grid': self.config.getint('DATA_AUGMENTATION', 'ED_GRID'),
            'ed_sigma_range': ed_sigma_range,
            'ed_boundary': self.config.get('DATA_AUGMENTATION', 'ED_BOUNDARY'),
            'ed_prefilter': self.config.getboolean('DATA_AUGMENTATION', 'ED_USE_PREFILTER'),
            'ed_axis': self.config.get('DATA_AUGMENTATION', 'ED_AXIS'),

            # Clip
            'clip_interval': clip_interval
        }
        return params

    def read_eval(self):
        params = {
            'trained_model_dir': self.config.get('DATA', 'TRAINED_MODEL_DIR'),
            'model_name': self.config.get('DATA', 'MODEL_NAME'),
            'save_imgs': self.config.getboolean('DEBUG', 'SAVE_IMGS'),
            'save_nifti': self.config.getboolean('DEBUG', 'SAVE_NIFTI'),
            'workers': self.config.getint('PARAMETERS', 'NUM_WORKERS'),
            'dataset': self.config.get('DATA', 'DATASET'),
            'fine_tuning': self.config.getboolean('DATA', 'FINE_TUNING'),
            'save_nz': self.config.getint('PARAMETERS', 'save_NZ'),
            'save_ny': self.config.getint('PARAMETERS', 'save_NY'),
            'save_nx': self.config.getint('PARAMETERS', 'save_NX')
        }
        return params

    def read_fine_tuning(self):
        params = {
            'pretrained_model_dir': self.config.get('DATA', 'PRETRAINED_DIR'),
            'weights_filename': self.config.get('DATA', 'WEIGHTS_FILENAME'),
            'patient_name': self.config.get('DATA', 'PATIENT_NAME'),
            'num_gpus': self.config.getint('PARAMETERS', 'NUM_GPUS'),
            'num_workers': self.config.getint('PARAMETERS', 'NUM_WORKERS'),
            'num_epochs': self.config.getint('PARAMETERS', 'NUM_EPOCHS'),
            'batch_size': 1,
            'lr': self.config.getfloat('PARAMETERS', 'LR'),
            'weight_decay': self.config.getfloat('PARAMETERS', 'WEIGHT_DECAY'),
            'step_size': self.config.getfloat('PARAMETERS', 'STEP_SIZE'),
            'gamma': self.config.getfloat('PARAMETERS', 'GAMMA'),
            'beta1': self.config.getfloat('PARAMETERS', 'BETA1'),
            'beta2': self.config.getfloat('PARAMETERS', 'BETA2'),
            'loss_lambda': self.config.getfloat('PARAMETERS', 'LOSS_LAMBDA'),
            'dataset': self.config.get('DATA', 'DATASET')
        }
        return params
