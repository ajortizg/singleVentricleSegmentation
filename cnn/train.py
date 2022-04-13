import torch
import configparser
import sys
from dataset import SingleVentricleDataset
from torch.optim import Adam
from tqdm import tqdm
from unet_3d import UNet3D
from torch.utils.data import DataLoader
import os
import time
import os.path as osp
from torchsummary import summary
import matplotlib.pyplot as plt

sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '../utils')))
from torch_warping import create_grid, warp
import plots

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
DEVICE = 'cuda:2' if cuda_availabe and torch.cuda.is_available() else 'cpu'
PIN_MEMORY = False if DEVICE == 'cpu' else True
BATCH_SIZE = config.getint('PARAMETERS', 'BATCH_SIZE')
LR = config.getfloat('PARAMETERS', 'LR')
NUM_EPOCHS = config.getint('PARAMETERS', 'NUM_EPOCHS')

train_ds = SingleVentricleDataset(config)
train_loader = DataLoader(train_ds, shuffle=True, batch_size=BATCH_SIZE,
                          pin_memory=PIN_MEMORY, num_workers=os.cpu_count())

# UNet3D model
net = UNet3D(config).to(DEVICE)
net = torch.nn.DataParallel(net, device_ids=[2])
opt = Adam(net.parameters(), lr=LR)
summary(net, input_size=(2, 100, 100, 100), batch_size=-1)

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

# Steps per epoch for training set
train_steps = len(train_ds) // BATCH_SIZE
H = {"train_loss": [], "test_loss": []}

print("\n")
print("===========================================================")
print("training the network")
print("===========================================================")
print("\n")
tic = time.time()
for e in tqdm(range(NUM_EPOCHS)):
    net.train()
    total_train_loss = 0

    # Loop over the training set
    for i, (vol, mask_syst, mask_diast, tsyst, tdias, ff, bf, pname) in enumerate(train_loader):
        _, _, NZ, NY, NX, NT = vol.shape
        grid = create_grid(NZ, NY, NX).unsqueeze(dim=0).to(DEVICE)

        print("\n")
        print("===========================================================")
        print(f'Load data for patient: {pname}')
        print(f'\t* (NZ, NY, NX, NT) = ({NZ}, {NY}, {NX}, {NT})')
        print(f'\t* {mask_syst.shape}, { mask_diast.shape}')
        print(f'\t* Systole at time: {tsyst}')
        print(f'\t* Diastole at time: {tdias}')
        print(f'\t* Optflows: {len(ff)}, with shape: {ff[0].shape}')
        print("===========================================================")

        init_ts = min(tdias, tsyst)
        final_ts = max(tdias, tsyst)

        # Mask initialization
        m0, mk = None, None
        if init_ts == tsyst:
            m0 = mask_syst.clone().to(DEVICE)
            mk = mask_diast.clone().to(DEVICE)
        else:
            m0 = mask_diast.clone().to(DEVICE)
            mk = mask_syst.clone().to(DEVICE)

        mts = [m0]
        k = 0
        for t in range(init_ts, final_ts, 1):
            img = vol[:, :, :, :, :, t + 1].to(DEVICE)
            u = bf[k].to(DEVICE)
            mt = warp(mts[-1], grid - u, mode="bilinear")
            x = torch.cat((img, mt), dim=1)
            mt = net(x)
            mts.append(mt)
            k += 1

        mtts = [mk]
        ff.reverse()
        k = 0
        for t in range(final_ts, init_ts, -1):
            img = vol[:, :, :, :, :, t - 1].to(DEVICE)
            u = ff[k].to(DEVICE)
            mtt = warp(mtts[-1], grid - u, mode="bilinear")
            x = torch.cat((img, mtt), dim=1)
            mtt = net(x)
            mtts.append(mtt)
            k += 1

        mtts.reverse()
        total_loss = 0
        for k in range(len(mtts)):
            loss = 0.5 * (mts[k] - mtts[k]).pow(2).sum()
            total_loss += loss

        opt.zero_grad()
        total_loss.backward()
        opt.step()
        total_train_loss += total_loss

    # Average training loss
    avg_train_loss = total_train_loss / train_steps
    # Update training history
    H["train_loss"].append(avg_train_loss.cpu().detach().numpy())
    print("EPOCH: {}/{}".format(e + 1, NUM_EPOCHS))
    print("Train loss: {:.6f}".format(avg_train_loss))

toc = time.time()
print("\nTotal time taken to train the model: {:.2f}s".format(toc - tic))

plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_loss"], label="train_loss")
plt.title("Training Loss on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(osp.join(save_dir, 'plot.png'))

torch.save(net, osp.join(save_dir, 'model.pth'))
