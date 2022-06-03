# import torch
# import configparser
# import sys
# from dataset import SingleVentricleDatasetBatch, DatasetMode
# from torch.optim import Adam
# import torch.nn as nn
# from tqdm import tqdm
# from torch.optim.lr_scheduler import StepLR
# from unet_3d import UNet3D
# import torch.nn.functional as F
# import csv
# from torch.utils.data import DataLoader

# from torch.utils.tensorboard import SummaryWriter
# import time
# import os.path as osp
# from torchsummary import summary
# from loss import loss_func_complete
# import numpy as np
# import matplotlib.pyplot as plt
# from cnn_utils import *
# import os
# from warp import Warp

# ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
# sys.path.append(ROOT_DIR)
# from utils import plots
# from utils import torch_utils


# print("\n\n")
# print("===========================================================")
# print("===========================================================")
# print("                      Train CNN:")
# print("===========================================================")
# print("===========================================================")
# print("\n\n")


# config = configparser.ConfigParser()
# config.read('parser/configCNN.ini')
# cuda_availabe = config.get('DEVICE', 'CUDA_AVAILABLE')
# if cuda_availabe and torch.cuda.is_available():
#     DEVICE = 'cuda'
#     # CUDA_DEVICE = config.getint('DEVICE', 'CUDA_DEVICE')
#     # torch.cuda.set_device(CUDA_DEVICE)
# else:
#     DEVICE = 'cpu'


# BATCH_SIZE = config.getint('PARAMETERS', 'BATCH_SIZE')
# LR = config.getfloat('PARAMETERS', 'LR')
# STEP_SIZE = config.getfloat('PARAMETERS', 'STEP_SIZE')
# GAMMA = config.getfloat('PARAMETERS', 'GAMMA')
# LR = config.getfloat('PARAMETERS', 'LR')
# WEIGHT_DECAY = config.getfloat('PARAMETERS', 'WEIGHT_DECAY')
# BETA1 = config.getfloat('PARAMETERS', 'BETA1')
# BETA2 = config.getfloat('PARAMETERS', 'BETA2')
# NUM_EPOCHS = config.getint('PARAMETERS', 'NUM_EPOCHS')
# VERBOSE = config.getboolean('DEBUG', 'VERBOSE')
# SHUFFLE = config.getboolean('PARAMETERS', 'SHUFFLE')

# # Create train and validation datasets
# train_ds = SingleVentricleDatasetBatch(config, DatasetMode.TRAIN, load_flow=True)
# val_ds = SingleVentricleDatasetBatch(config, DatasetMode.VAL, load_flow=True)

# # Create data loaders
# train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=os.cpu_count())
# val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count())

# # UNet3D model
# net = UNet3D(config).to(DEVICE)

# if VERBOSE:
#     summary(net, input_size=(2, 16, 100, 100), batch_size=BATCH_SIZE)
#     print("\n")
#     print("===========================================================")
#     print('CNN parameters')
#     print(f"\t* Patients for training: {len(train_ds)}")
#     print(f"\t* Patients for validation: {len(val_ds)}")
#     print(f'\t* Device: {DEVICE}')
#     print(f'\t* Learning rate: {LR}')
#     print(f'\t* Num epochs: {NUM_EPOCHS}')
#     print("===========================================================")
#     print("\n")

# net = torch.nn.DataParallel(net, device_ids=[0, 1, 2])
# opt = Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(BETA1, BETA2))
# # wd = StepLR(opt, step_size=STEP_SIZE, gamma=GAMMA)

# save_dir = plots.createSaveDirectory(config.get('DATA', 'OUTPUT_PATH'), 'CNN')
# writer = SummaryWriter(log_dir=save_dir)

# # save config file to save directory
# conifg_output = osp.join(save_dir, 'config.ini')
# with open(conifg_output, 'w') as config_file:
#     config.write(config_file)


# # Steps per epoch for training and evaluation set
# train_steps = len(train_ds) // BATCH_SIZE
# val_steps = len(val_ds) // BATCH_SIZE
# H = {"train_loss": [], "val_loss": []}


# print("\n")
# print("===========================================================")
# print("Training the network")
# print("===========================================================")
# print("\n")

# avg_train_loss = 0
# avg_val_loss = 0
# pbar = tqdm(total=NUM_EPOCHS)
# tic = time.time()
# for e in range(NUM_EPOCHS):
#     # csv_file = open(osp.join(save_dir, f'diff_{e}.csv'), 'w')
#     # csv_writer = csv.writer(csv_file)
#     net.train()

#     # Initialize the total training and validation loss
#     total_train_loss = 0
#     total_val_loss = 0
#     for i, (pname, vol, m0, mk, init_ts, final_ts, ff, bf) in enumerate(train_loader):
#         pbar.set_postfix_str(f'Train S:{i+1}/{train_steps} E: {e+1}/{NUM_EPOCHS}, L: {avg_train_loss:.2f}')
#         BS, NZ, NY, NX, NT = vol.shape
#         nsize = (BS, 1, NZ, NY, NX)
#         warp = Warp(config, NZ, NY, NX)
#         mts = [m0.reshape(nsize).to(DEVICE)]
#         mtts = [mk.reshape(nsize).to(DEVICE)]
#         print(ff.shape)
#         for t in range(len(ff)):
#             # Forward mask propagation m0 -> mk
#             mt = propagate_batch(net, warp, vol[:, :, :, :, init_ts + t + 1], ff[:, :, :, :, :, t].to(DEVICE), mts[-1])
#             mts.append(mt)

#             # Backward mask propagation mk -> m0
#             mtt = propagate_batch(net, warp, vol[:, :, :, :, final_ts - t - 1], bf[:, :, :, :, :, t].to(DEVICE), mtts[-1])
#             mtts.append(mtt)

#         mtts.reverse()
#         train_loss, l1, l2, l3 = loss_func_complete(mts, mtts)
#         opt.zero_grad()
#         train_loss.backward()
#         opt.step()

#         with torch.no_grad():
#             total_train_loss += train_loss.item()
#             writer.add_image(f'train_{pname}_mkt', torch.swapaxes(torch_utils.normalize(mt).squeeze(1), 0, 1), dataformats='NCHW')
#             writer.add_image(f'train_{pname}_m0tt', torch.swapaxes(torch_utils.normalize(mtt).squeeze(1), 0, 1), dataformats='NCHW')
#             # row = [pname]
#             # row.append('{:.2f}'.format(l1.item()))
#             # row.append('{:.2f}'.format(l2.item()))
#             # row.append('{:.2f}'.format(l3.item()))
#             # row.append('{:.2f}'.format(train_loss.item()))
#             # csv_writer.writerow(row)
#         pbar.update(1)

#         #     # Loop over validation set
#         #     with torch.no_grad():
#         #         net.eval()
#         #         for idx in val_idxs:
#         #             (pname, vol, m0, mk, init_ts, final_ts, ff, bf) = train_ds[idx]
#         #             pbar.set_postfix_str(f'Val P: {pname}, E: {e}, L: {avg_val_loss:.2f}')
#         #             NZ, NY, NX, NT = vol.shape
#         #             nsize = (1, 1, NZ, NY, NX)
#         #             vol = torch_utils.normalize(vol)
#         #             warp = Warp(config, NZ, NY, NX)
#         #             mts = [m0.reshape(nsize).to(DEVICE)]
#         #             mtts = [mk.reshape(nsize).to(DEVICE)]
#         #             bf.reverse()
#         #             for t in range(len(ff)):
#         #                 # Forward mask propagation m0 -> mk
#         #                 mt = propagate(net, warp, vol[:, :, :, init_ts + t + 1].to(DEVICE), nsize, ff[t].to(DEVICE), mts[-1])
#         #                 mts.append(mt)

#         #                 # Backward mask propagation mk -> m0
#         #                 mtt = propagate(net, warp, vol[:, :, :, final_ts - t - 1].to(DEVICE), nsize, bf[t].to(DEVICE), mtts[-1])
#         #                 mtts.append(mtt)

#         #             mtts.reverse()
#         #             val_loss, _, _, _ = loss_func_complete(mts, mtts)
#         #             total_val_loss += val_loss.item()
#         #             writer.add_image(f'val_{pname}_mkt', torch.swapaxes(torch_utils.normalize(mt).squeeze(1), 0, 1), dataformats='NCHW')
#         #             writer.add_image(f'val_{pname}_m0tt', torch.swapaxes(torch_utils.normalize(mtt).squeeze(1), 0, 1), dataformats='NCHW')
#         #             pbar.update(1)

#         #     # wd.step()
#         #     avg_train_loss = total_train_loss / train_steps
#         #     avg_val_loss = total_val_loss / val_steps
#         #     H['train_loss'].append(avg_train_loss)
#         #     H['val_loss'].append(avg_val_loss)
#         #     # csv_file.close()
#         #     writer.add_scalars('loss', {'e_train_loss': avg_train_loss, 'e_val_loss': avg_val_loss}, e)

#         # toc = time.time()
#         # print('\nTotal time taken to train the model: {:.2f}s'.format(toc - tic))

#         # plt.style.use('ggplot')
#         # plt.figure()
#         # plt.plot(H['train_loss'], label='train_loss')
#         # plt.plot(H['val_loss'], label='val_loss')
#         # plt.title('Training Loss on Dataset')
#         # plt.xlabel('Epoch #')
#         # plt.ylabel('Loss')
#         # plt.legend(loc='lower left')
#         # plt.savefig(osp.join(save_dir, 'loss.png'))
#         # torch.save(net, osp.join(save_dir, 'model.pth'))
#         # #net = torch.load('model.pth')
#         # # torch.save(net.state_dict(), osp.join(save_dir, 'model_weights.pth'))
#         # # net.load_state_dict(torch.load('model_weights.pth'))
#         # pbar.close()
#         # writer.close()
