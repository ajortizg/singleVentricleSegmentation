import torch
import numpy as np
from torch.nn import Module
import sys
import os.path as osp
import torch.nn as nn
import os


ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
import utils.transforms as T
from cnn.unet_3d import UNet3d, BasicUnet3d, ResUnet3d


def create_net(config, logger):
    net_type = config.get('PARAMETERS', 'NET')
    net = nn.Module
    if net_type == 'unet3d':
        net = UNet3d(config, logger)
    elif net_type == 'basic_unet3d':
        net = BasicUnet3d(config, logger)
    elif net_type == 'res_unet3d':
        net = ResUnet3d(config, logger)
    else:
        print('Unknown network: ' + net_type)
        sys.exit()
    return net


def save_config(config, save_dir, filename='config.ini'):
    # save config file to save directory
    conifg_output = osp.join(save_dir, filename)
    with open(conifg_output, 'w') as config_file:
        config.write(config_file)


def read_train_params(config):
    rot_range_x = tuple(map(float, config.get('DATA_AUGMENTATION', 'ROT_X_RANGE').split(',')))
    rot_range_z = tuple(map(float, config.get('DATA_AUGMENTATION', 'ROT_Z_RANGE').split(',')))
    rot_range_y = tuple(map(float, config.get('DATA_AUGMENTATION', 'ROT_Y_RANGE').split(',')))
    mult_scaling_range = tuple(map(float, config.get('DATA_AUGMENTATION', 'MULT_SCALING_RANGE').split(',')))
    clip_interval = tuple(map(float, config.get('DATA_AUGMENTATION', 'CLIP_INTERVAL').split(',')))
    gamma_scaling_range = tuple(map(float, config.get('DATA_AUGMENTATION', 'GAMMA_SCALING_RANGE').split(',')))
    ed_sigma_range = gamma_scaling_range = tuple(map(float, config.get('DATA_AUGMENTATION', 'ED_SIGMA_RANGE').split(',')))

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
        'gpus': config.getint('PARAMETERS', 'NUM_GPUS'),
        'workers': config.getint('PARAMETERS', 'NUM_WORKERS'),
        'pretrained': config.getboolean('PARAMETERS', 'PRETRAINED'),
        'checkpoint_file': config.get('PARAMETERS', 'CHECKPOINT_FILE'),

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

        # Gaussian noise
        'noise_prob': config.getfloat('DATA_AUGMENTATION', 'NOISE_PROB'),
        'noise_mu': config.getfloat('DATA_AUGMENTATION', 'NOISE_MU'),
        'noise_std': config.getfloat('DATA_AUGMENTATION', 'NOISE_STD'),

        # Elastic deformation
        'ed_prob': config.getfloat('DATA_AUGMENTATION', 'ED_PROB'),
        'ed_grid': config.getint('DATA_AUGMENTATION', 'ED_GRID'),
        'ed_sigma_range': ed_sigma_range,
        'ed_boundary': config.get('DATA_AUGMENTATION', 'ED_BOUNDARY'),
        'ed_prefilter': config.getboolean('DATA_AUGMENTATION', 'ED_USE_PREFILTER'),
        'ed_axis': config.get('DATA_AUGMENTATION', 'ED_AXIS'),

        # Clip
        'clip_interval': clip_interval
    }

    return params


def read_eval_params(config):
    params = {
        'trained_model_dir': config.get('DATA', 'TRAINED_MODEL_DIR'),
        'model_name': config.get('DATA', 'MODEL_NAME'),
        'save_imgs': config.getboolean('DEBUG', 'SAVE_IMGS'),
        'save_nifti': config.getboolean('DEBUG', 'SAVE_NIFTI'),
        'workers': config.getint('PARAMETERS', 'NUM_WORKERS'),
        'dataset': config.get('DATA', 'DATASET'),
        'fine_tuning': config.getboolean('DATA', 'FINE_TUNING'),
        'save_nz': config.getint('PARAMETERS', 'save_NZ'),
        'save_ny': config.getint('PARAMETERS', 'save_NY'),
        'save_nx': config.getint('PARAMETERS', 'save_NX')
    }
    return params


def checkpoint(e, net, opt, loss, acc, save_dir, filename):
    torch.save({
        'epoch': e,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'loss': loss,
        'acc': acc
    }, osp.join(save_dir, filename))


def update_train_history(H, train_avg, val_avg):
    H['train_loss'].append(train_avg[0])
    H['val_loss'].append(val_avg[0])
    H['train_acc'].append(train_avg[-1])
    H['val_acc'].append(val_avg[-1])
    return H


def log(logger, writer, e, train_avg, val_avg, net, opt, schedule_lr, best_train_loss, best_val_loss, save_dir):
    log_train = '\t*Train:'
    log_val = '\t*Val:'
    for i in range(len(train_avg) - 1):
        writer.add_scalars(f'loss/l{i}', {'train': train_avg[i], 'val': val_avg[i]}, e)
        log_train += f'\tl{i}: {train_avg[i]:,.3f}'
        log_val += f'\tl{i}: {val_avg[i]:,.3f}'
    log_train += f'\tacc: {train_avg[-1]:,.3f}'
    log_val += f'\tacc: {val_avg[-1]:,.3f}'
    logger.info(f'Epoch: {e}')
    logger.info(log_train)
    logger.info(log_val)

    # writer.add_scalar('lr', schedule_lr.get_last_lr()[0], e)
    writer.add_scalar('lr', opt.param_groups[0]['lr'], e)
    writer.add_scalars('acc', {'train': train_avg[-1], 'val': val_avg[-1]}, e)

    if train_avg[0] < best_train_loss:
        best_train_loss = train_avg[0]
        checkpoint(e, net, opt, train_avg[0], train_avg[-1], save_dir, 'best_train_checkpoint.pth')
        logger.info(f'\t*Best train checkpoint updated with loss: {best_train_loss:,.3f}')
    if val_avg[0] < best_val_loss:
        best_val_loss = val_avg[0]
        checkpoint(e, net, opt, val_avg[0], val_avg[-1], save_dir, 'best_val_checkpoint.pth')
        logger.info(f'\t*Best val checkpoint updated with loss: {best_val_loss:,.3f}')

    return best_train_loss, best_val_loss


def seeding(seed):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def save_model(net: Module, save_dir: str, filename: str):
    filepath = osp.join(save_dir, filename)
    total_params = 0

    modules = [module for module in net.modules()]
    params = [param for param in net.parameters()]
    with open(filepath, 'w') as mfile:
        for idx, m in enumerate(modules):
            mfile.write(f'{idx} -> {m}\n')
        for p in params:
            total_params += p.numel()
        mfile.write(f'\nTotal parameters: {total_params}')
    mfile.close()


def save_weights(net: Module, epoch: int, every: int, save_dir: str, filename: str):
    if epoch % every == 0:
        torch.save(net, osp.join(save_dir, filename))


def collate_fn(data):
    pnames, imgs4d, m0s, mks, masks, init_ts, final_ts, ff, bf = zip(*data)

    list_times_fwd, list_times_bwd = collate_times(init_ts, final_ts)
    ff, bf, offsets = collate_optical_flow(ff, bf)

    to_tensor = T.ListToTensor()
    imgs4d = to_tensor(imgs4d)
    m0s = to_tensor(m0s)
    mks = to_tensor(mks)
    imgs4d.unsqueeze_(1)
    m0s.unsqueeze_(1)
    mks.unsqueeze_(1)

    # print(masks)

    # if masks is not None:
    #     masks = to_tensor(masks)
    #     masks.unsqueeze_(1)

    return (pnames, imgs4d, m0s, mks, masks, list_times_fwd, list_times_bwd, ff, bf, offsets)


def collate_times(init_ts, final_ts):
    init_ts = torch.tensor(init_ts)
    final_ts = torch.tensor(final_ts)
    num_ts = (final_ts - init_ts).max().item()

    list_times_fwd = [init_ts]
    list_times_bwd = [final_ts]

    for _ in range(num_ts):
        next_time = list_times_fwd[-1] + 1
        list_times_fwd.append(torch.where(next_time > final_ts, final_ts, next_time))

        prev_time = list_times_bwd[-1] - 1
        list_times_bwd.append(torch.where(prev_time < init_ts, init_ts, prev_time))

    return(list_times_fwd, list_times_bwd)


def collate_optical_flow(off, ofb):
    BS = len(off)
    maxts_flow = max_ts(off)
    NT, NZ, NY, NX, CH = off[0].shape

    off_t = torch.zeros(size=(BS, maxts_flow, NZ, NY, NX, CH))
    ofb_t = torch.zeros(size=(BS, maxts_flow, NZ, NY, NX, CH))
    offsets = torch.zeros(BS, dtype=torch.int)

    for b in range(BS):
        # Prepare optical flow
        diff_of_ts = (int)(maxts_flow - off[b].shape[0])
        offsets[b] = diff_of_ts
        if diff_of_ts != 0:
            zeros = torch.zeros(size=(diff_of_ts, NZ, NY, NX, 3))
            off_t[b, :, :, :, :, :] = torch.cat((off[b], zeros), dim=0)
            ofb_t[b, :, :, :, :, :] = torch.cat((ofb[b], zeros), dim=0)
        else:
            off_t[b, :, :, :, :, :] = off[b]
            ofb_t[b, :, :, :, :, :] = ofb[b]

    return (off_t, ofb_t, offsets)


def max_ts(ff):
    BS = len(ff)
    maxts = 0
    for b in range(BS):
        t = ff[b].shape[0]
        if t > maxts:
            maxts = t
    return maxts


# def collate_fn(data):
#     pnames, vols, m0s, mks, init_ts, final_ts, ff, bf = zip(*data)
#     to_tensor = T.ListToTensor()
#     vols = to_tensor(vols)
#     m0s = to_tensor(m0s)
#     mks = to_tensor(mks)
#     init_ts = torch.tensor(init_ts)
#     final_ts = torch.tensor(final_ts)

#     maxts = max_ts(ff)
#     BS, NZ, NY, NX = m0s.shape
#     fwd_t = torch.zeros(size=(BS, maxts, NZ, NY, NX, 3))
#     bwd_t = torch.zeros(size=(BS, maxts, NZ, NY, NX, 3))
#     offsets = torch.zeros(BS, dtype=torch.int)
#     for b in range(BS):
#         diff_t = (int)(maxts - ff[b].shape[0])
#         offsets[b] = diff_t
#         if diff_t != 0:
#             zeros = torch.zeros(size=(diff_t, NZ, NY, NX, 3))
#             fwd_t[b, :, :, :, :, :] = torch.cat((ff[b], zeros), dim=0)
#             bwd_t[b, :, :, :, :, :] = torch.cat((bf[b], zeros), dim=0)
#         else:
#             fwd_t[b, :, :, :, :, :] = ff[b]
#             bwd_t[b, :, :, :, :, :] = bf[b]

#     return (pnames, vols.unsqueeze_(1), m0s.unsqueeze_(1), mks.unsqueeze_(1), init_ts, final_ts, fwd_t, bwd_t, offsets)
