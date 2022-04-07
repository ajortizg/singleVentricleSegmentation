import torch
import nibabel as nib
import numpy as np
import configparser
import os.path as osp
import pandas
from tqdm import tqdm
from unet_3d import UNet3D
import time
from torchsummary import summary
from dataset import SingleVentricleDataset

config = configparser.ConfigParser()
config.read("parser/configCNN.ini")

ds = SingleVentricleDataset(config)

for i, (v, ms, md, ts, td) in enumerate(ds):
    print(ds.get_patient_name(i))
    print(v.shape)
    print(ms.shape)
    print(md.shape)
    print(ts, td)
    print()


cuda_availabe = config.get('DEVICE', 'cuda_availabe')
DEVICE = "cuda" if cuda_availabe and torch.cuda.is_available() else "cpu"

net = UNet3D(config).to(DEVICE)
net = torch.nn.DataParallel(net, device_ids=[0, 1])

x = torch.rand(1, 1, 14, 352, 352).to(DEVICE)
y = net(x)

summary(net, input_size=x[-4:], batch_size=1)
