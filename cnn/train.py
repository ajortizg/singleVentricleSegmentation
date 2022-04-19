import torch
import configparser
import sys
from dataset import SingleVentricleDataset
from torch.optim import Adam
from tqdm import tqdm
from unet_3d import UNet3D
from torch.utils.data import DataLoader
from torch.nn import MSELoss
import os
from torch.utils.tensorboard import SummaryWriter
import time
import os.path as osp
from torchsummary import summary
import matplotlib.pyplot as plt
from warp import Warp

sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '../utils')))
import plots
import torch_utils

print("\n\n")
print("===========================================================")
print("===========================================================")
print("                      Train CNN:")
print("===========================================================")
print("===========================================================")
print("\n\n")


config = configparser.ConfigParser()
config.read('parser/configCNN.ini')
cuda_availabe = config.get('DEVICE', 'CUDA_AVAILABLE')
DEVICE = 'cuda' if cuda_availabe and torch.cuda.is_available() else 'cpu'
BATCH_SIZE = config.getint('PARAMETERS', 'BATCH_SIZE')
LR = config.getfloat('PARAMETERS', 'LR')
NUM_EPOCHS = config.getint('PARAMETERS', 'NUM_EPOCHS')
VERBOSE = config.getboolean('DEBUG', 'VERBOSE')
SHUFFLE = config.getboolean('PARAMETERS', 'SHUFFLE')

train_ds = SingleVentricleDataset(config, load_flow=True)

# UNet3D model
net = UNet3D(config).to(DEVICE)
# net = torch.nn.DataParallel(net, device_ids=[2])
opt = Adam(net.parameters(), lr=LR)

if VERBOSE:
    summary(net, input_size=(2, 16, 300, 300), batch_size=-1)
    print("\n")
    print("===========================================================")
    print('CNN parameters')
    print(f"\t* Found {len(train_ds)} examples in the training set")
    print(f'\t* Device: {DEVICE}')
    print(f'\t* Learning rate: {LR}')
    print(f'\t* Num epochs: {NUM_EPOCHS}')
    print("===========================================================")
    print("\n")

save_dir = plots.createSaveDirectory(config.get('DATA', 'OUTPUT_PATH'), 'CNN')
writer = SummaryWriter(log_dir=save_dir)

# save config file to save directory
conifg_output = osp.join(save_dir, 'config.ini')
with open(conifg_output, 'w') as config_file:
    config.write(config_file)

# Steps per epoch for training set
train_steps = len(train_ds) // BATCH_SIZE
H = {"train_loss": [], "test_loss": []}

print("\n")
print("===========================================================")
print("training the network")
print("===========================================================")
print("\n")

iter = 0
for e in tqdm(range(NUM_EPOCHS)):
    net.train()
    total_train_loss = 0

    tic = time.time()
    # Loop over the training set
    idxs = torch.randperm(len(train_ds)) if SHUFFLE else torch.arange(len(train_ds))
    for idx in idxs:
        (pname, vol, mask_syst, mask_diast, tsyst, tdias, ff, bf) = train_ds[idx]
        vol = torch_utils.normalize(vol)
        NZ, NY, NX, NT = vol.shape

        if VERBOSE:
            print("\n")
            print("===========================================================")
            print(f'Load data for patient: {pname}')
            print(f'\t* (NZ, NY, NX, NT) = ({NZ}, {NY}, {NX}, {NT})')
            print(f'\t* masks: {mask_syst.shape}, { mask_diast.shape}')
            print(f'\t* Systole at time: {tsyst}')
            print(f'\t* Diastole at time: {tdias}')
            # print(f'\t* Optflows: {len(ff)}, with shape: {ff[0].shape}')
            print("===========================================================")

        init_ts = min(tdias, tsyst)
        final_ts = max(tdias, tsyst)
        steps = abs(init_ts - final_ts)

        # Mask initialization
        m0, mk = None, None
        if init_ts == tsyst:
            m0 = mask_syst.unsqueeze(dim=0).unsqueeze(dim=0).to(DEVICE)
            mk = mask_diast.unsqueeze(dim=0).unsqueeze(dim=0).to(DEVICE)
            print('m0 = mask_systole', '\tmk = mask_diastole')
        else:
            m0 = mask_diast.unsqueeze(dim=0).unsqueeze(dim=0).to(DEVICE)
            mk = mask_syst.unsqueeze(dim=0).unsqueeze(dim=0).to(DEVICE)
            print('m0 = mask_diastole', '\tmk = mask_systole')

        patient_dir = plots.createSubDirectory(save_dir, pname)
        save_epoch_dir = plots.createSubDirectory(patient_dir, 'm0tt')
        plots.save_slices(m0.squeeze(), 'm0.png', patient_dir)
        plots.save_slices(mk.squeeze(), 'mk.png', patient_dir)

        warp = Warp(config, NZ, NY, NX)
        mts = [m0]
        mtts = [mk]
        bf.reverse()
        for t in range(steps):
            # Forward mask propagation m0 -> mk
            fwd_time = init_ts + t
            data_t = vol[:, :, :, fwd_time].unsqueeze(dim=0).unsqueeze(dim=0).to(DEVICE)
            u = ff[t].to(DEVICE)
            mt = warp(mts[-1].squeeze(), u).unsqueeze(dim=0).unsqueeze(dim=0)
            # x = torch.cat((data_t, mt), dim=1)
            # mt = net(x)
            mts.append(mt)

            # Backward mask propagation mk -> m0
            bwd_time = final_ts - t
            data_t = vol[:, :, :, bwd_time].unsqueeze(dim=0).unsqueeze(dim=0).to(DEVICE)
            u = bf[t].to(DEVICE)
            mtt = warp(mtts[-1].squeeze(), u).unsqueeze(dim=0).unsqueeze(dim=0)
            x = torch.cat((data_t, mtt), dim=1)
            mtt = net(x)
            mtts.append(mtt)

        mtts.reverse()
        plots.save_slices(mtts[5].squeeze(), f"m0tt_{e}", save_epoch_dir)
        total_loss = 0.0
        for k in range(len(mtts)):
            loss = 0.5 * (mts[k] - mtts[k]).pow(2).sum()
            total_loss += loss

        opt.zero_grad()
        total_loss.backward()
        opt.step()

        with torch.no_grad():
            total_train_loss += total_loss
            writer.add_scalar('train_loss', total_loss.item(), iter)
            iter += 1

    avg_train_loss = total_train_loss / train_steps
    H["train_loss"].append(avg_train_loss.cpu().detach().numpy())
    print("EPOCH: {}/{}".format(e + 1, NUM_EPOCHS))
    print("Train loss: {:.6f}".format(avg_train_loss))

    # total_diff = 0.0
    # for k in range(len(mtts)):
    #     diff = torch.abs(mts[k] - mtts[k]).squeeze()
    #     ndiff = diff.norm().item()
    #     total_diff += ndiff
    #     print(f"|diff| = {ndiff}")

    #     plots.save_slices(diff, f'Diff_mt_mtt_t{k}.png', patient_dir)
    #     plots.save_single_zslices(diff, patient_dir, f'Diff_{k}', 1., 0)
    # print(f'mean |diff| =  {total_diff / len(mtts)}')

toc = time.time()
print("\nTotal time taken to train the model: {:.2f}s".format(toc - tic))


# plt.style.use("ggplot")
# plt.figure()
# plt.plot(H["train_loss"], label="train_loss")
# plt.title("Training Loss on Dataset")
# plt.xlabel("Epoch #")
# plt.ylabel("Loss")
# plt.legend(loc="lower left")
# plt.savefig(osp.join(save_dir, 'plot.png'))

# torch.save(net, osp.join(save_dir, 'model.pth'))
