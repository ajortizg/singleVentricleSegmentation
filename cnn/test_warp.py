import torch
import numpy as np
import os
from dataset import SingleVentricleDataset
from torchvision.transforms import Compose
import configparser
import custom_transforms as ct
import sys


config = configparser.ConfigParser()
config.read('parser/configCNN.ini')
cuda_availabe = config.get('DEVICE', 'CUDA_AVAILABLE')
DEVICE = 'cuda' if cuda_availabe and torch.cuda.is_available() else 'cpu'
PATIENT_NAME = config.get('DATA', 'PATIENT_NAME')

transforms = Compose([ct.ToTensor()])

ds = SingleVentricleDataset(config, transforms)
idx, found = ds.index_for_patient(PATIENT_NAME)
if not found:
    print(PATIENT_NAME + ' not found!')
    sys.exit()