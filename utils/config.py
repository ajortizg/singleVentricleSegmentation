import torch
import os

BASE_PATH = "/home/antonio/Documents/datasets/mri/singleVentricleDataLeonAnon/"
VOLUMES_PATH = os.path.sep.join([BASE_PATH, "NIFTI_4D_Datasets"])
SEGMENTATIONS_PATH = os.path.sep.join(
    [BASE_PATH, "NIFTI Single Ventricle Segmentations"])

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False

# TVL1 algorithm parameters
SIGMA = 0.8
ZOOM_FACTOR = 0.5
NUM_SCALES = 3
WARPS = 3
LAMBDA = 0.15
THETA = 0.3

