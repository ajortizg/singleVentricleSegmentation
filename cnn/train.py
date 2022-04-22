import torch
import configparser
import sys
from dataset import SingleVentricleDataset
from torch.optim import Adam
from tqdm import tqdm
from unet_3d import UNet3D
from torch.utils.tensorboard import SummaryWriter
import time
import os.path as osp
from torchsummary import summary
from loss import L2LossReduced
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
if cuda_availabe and torch.cuda.is_available():
    DEVICE = 'cuda'
    torch.cuda.set_device(3)
else:
    DEVICE = 'cpu'
BATCH_SIZE = config.getint('PARAMETERS', 'BATCH_SIZE')
LR = config.getfloat('PARAMETERS', 'LR')
NUM_EPOCHS = config.getint('PARAMETERS', 'NUM_EPOCHS')
VERBOSE = config.getboolean('DEBUG', 'VERBOSE')
SHUFFLE = config.getboolean('PARAMETERS', 'SHUFFLE')

train_ds = SingleVentricleDataset(config, load_flow=True)

# UNet3D model
net = UNet3D(config).to(DEVICE)
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

avg_train_loss = 0
pbar = tqdm(total=NUM_EPOCHS * train_steps)
tic = time.time()
for e in range(NUM_EPOCHS):
    net.train()
    total_train_loss = 0

    # Loop over the training set
    idxs = torch.randperm(len(train_ds)) if SHUFFLE else torch.arange(len(train_ds))
    for idx in idxs:
        (pname, vol, mask_syst, mask_diast, tsyst, tdias, ff, bf) = train_ds[idx]
        pbar.set_postfix_str(f'P: {pname}, E: {e}, L: {avg_train_loss:.2f}')
        vol = torch_utils.normalize(vol)
        NZ, NY, NX, NT = vol.shape

        # if VERBOSE:
        #     print("\n")
        #     print("===========================================================")
        #     print(f'{e} Load data for patient: {pname}')
        #     print(f'\t* (NZ, NY, NX, NT) = ({NZ}, {NY}, {NX}, {NT})')
        #     print(f'\t* masks: {mask_syst.shape}, { mask_diast.shape}')
        #     print(f'\t* Systole at time: {tsyst}')
        #     print(f'\t* Diastole at time: {tdias}')
        #     print(f'\t* Optflows: {len(ff)}, with shape: {ff[0].shape}')
        #     print("===========================================================")

        init_ts = min(tdias, tsyst)
        final_ts = max(tdias, tsyst)
        steps = abs(init_ts - final_ts)

        # Mask initialization
        m0, mk = None, None
        if init_ts == tsyst:
            m0 = mask_syst.unsqueeze(dim=0).unsqueeze(dim=0).to(DEVICE)
            mk = mask_diast.unsqueeze(dim=0).unsqueeze(dim=0).to(DEVICE)
            # print('m0 = mask_systole', '\tmk = mask_diastole')
        else:
            m0 = mask_diast.unsqueeze(dim=0).unsqueeze(dim=0).to(DEVICE)
            mk = mask_syst.unsqueeze(dim=0).unsqueeze(dim=0).to(DEVICE)
            # print('m0 = mask_diastole', '\tmk = mask_systole')

        patient_dir = plots.createSubDirectory(save_dir, pname)
        save_m0tt = plots.createSubDirectory(patient_dir, 'm0tt')
        save_mkt = plots.createSubDirectory(patient_dir, 'mkt')
        plots.save_slices(m0.squeeze(), 'm0.png', patient_dir)
        plots.save_slices(mk.squeeze(), 'mk.png', patient_dir)

        warp = Warp(config, NZ, NY, NX)
        # mts = [m0]
        # mtts = [mk]
        mt = m0.clone()
        mtt = mk.clone()
        # mts_tensor = torch.zeros((steps + 1,) + m0.shape)
        # mts_tensor[0, :, :, :, :, :] = m0
        bf.reverse()
        for t in range(steps):
            # Forward mask propagation m0 -> mk
            fwd_time = init_ts + t + 1
            data_t = vol[:, :, :, fwd_time].unsqueeze(dim=0).unsqueeze(dim=0).to(DEVICE)
            u = ff[t].to(DEVICE)
            mt = warp(mt.squeeze(), u).unsqueeze(dim=0).unsqueeze(dim=0)
            # mt = warp(mts[-1].squeeze(), u).unsqueeze(dim=0).unsqueeze(dim=0)
            x = torch.cat((data_t, mt), dim=1)
            mt = net(x)
            # mts.append(mt)

            # Backward mask propagation mk -> m0
            bwd_time = final_ts - t - 1
            data_t = vol[:, :, :, bwd_time].unsqueeze(dim=0).unsqueeze(dim=0).to(DEVICE)
            u = bf[t].to(DEVICE)
            mtt = warp(mtt.squeeze(), u).unsqueeze(dim=0).unsqueeze(dim=0)
            # mtt = warp(mtts[-1].squeeze(), u).unsqueeze(dim=0).unsqueeze(dim=0)
            x = torch.cat((data_t, mtt), dim=1)
            mtt = net(x)
            # mtts.append(mtt)

        # mtts.reverse()
        # plots.save_slices(mtts[0].squeeze(), f"m0tt_{e}", save_m0tt)
        # plots.save_slices(mts[-1].squeeze(), f"mtk_{e}", save_mkt)
        plots.save_slices(mtt.squeeze(), f"m0tt_{e}", save_m0tt)
        plots.save_slices(mt.squeeze(), f'mkt_{e}', save_mkt)

        total_loss = (0.5 * (mtt - m0).pow(2).sum()) + (0.5 * (mt - mk).pow(2).sum())
        # total_loss = 0
        # for k in range(len(mtts)):
        #     loss = 0.5 * (mts[k] - mtts[k]).pow(2).sum()
        #     total_loss += loss
        # Loss = L2LossReduced.apply
        # total_loss = Loss(mtt, mt, m0, mk)

        opt.zero_grad()
        total_loss.backward()
        opt.step()

        # with torch.no_grad():
        total_train_loss += total_loss.item()
        pbar.update(1)

    avg_train_loss = total_train_loss / train_steps
    H['train_loss'].append(avg_train_loss)
    # print('EPOCH: {}/{}'.format(e + 1, NUM_EPOCHS))
    # print('Train loss: {:.6f}'.format(avg_train_loss))
    writer.add_scalar('train_loss', avg_train_loss, e)

toc = time.time()
print('\nTotal time taken to train the model: {:.2f}s'.format(toc - tic))

plt.style.use('ggplot')
plt.figure()
plt.plot(H['train_loss'], label='train_loss')
plt.title('Training Loss on Dataset')
plt.xlabel('Epoch #')
plt.ylabel('Loss')
plt.legend(loc='lower left')
plt.savefig(osp.join(save_dir, 'loss.png'))
torch.save(net, osp.join(save_dir, 'model.pth'))
#net = torch.load('model.pth')
# torch.save(net.state_dict(), osp.join(save_dir, 'model_weights.pth'))
# net.load_state_dict(torch.load('model_weights.pth'))
pbar.close()
