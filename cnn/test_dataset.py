import torch
import nibabel as nib
import numpy as np
import configparser
import os.path as osp
import pandas
from tqdm import tqdm
import sys
from unet_3d import UNet3D
import time
from torchsummary import summary
from dataset import SingleVentricleDataset
from torchvision.transforms import Compose
import custom_transforms as ct


config = configparser.ConfigParser()
config.read("parser/configCNN.ini")
cuda_availabe = config.get('DEVICE', 'CUDA_AVAILABLE')
DEVICE = "cuda" if cuda_availabe and torch.cuda.is_available() else "cpu"
PATIENT_NAME = config.get('DATA', 'PATIENT_NAME')

transforms = Compose([ct.ToTensor()])

ds = SingleVentricleDataset(config, transforms)

# for i, (v, ms, md, ts, td, pn) in enumerate(ds):
#     print(ds.get_patient_name(i))
#     print(v.shape)
#     print(ms.shape)
#     print(md.shape)
#     print(ts, td)
#     print()


idx, found = ds.index_for_patient(PATIENT_NAME)
if not found:
    print(PATIENT_NAME + " not found!")
    sys.exit()

(data, mask_systole, mask_diastole, systole_time, diastole_time, pn) = ds[idx]
NZ, NY, NX, NT = data.shape
print("================================================")
print("Load data for patient: " + PATIENT_NAME)
print(f"\t* (NZ, NY, NX, NT) = ({NZ}, {NY}, {NX}, {NT})")
print(f"\t* Systole at time: {systole_time}")
print(f"\t* Diastole at time: {diastole_time}")
print("================================================")

flows = ds.optflow_results_for_patient(PATIENT_NAME)

# net = UNet3D(config).to(DEVICE)
# net = torch.nn.DataParallel(net, device_ids=[0, 1])

# x = torch.rand(1, 1, 14, 352, 352).to(DEVICE)
# y = net(x)

# summary(net, input_size=x[-4:], batch_size=1)
