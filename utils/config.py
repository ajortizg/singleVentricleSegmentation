import torch
import os

BASE_PATH = "singleVentricleData"
VOLUMES_PATH = os.path.sep.join([BASE_PATH, "NIFTI_4D_Datasets"])
SEGMENTATIONS_PATH = os.path.sep.join([BASE_PATH, "NIFTI Single Ventricle Segmentations"])
OUTPUT_PATH = "/home/stefan/results/opticalFlow"
PATIENT_FILE = "Adult_30.nii.gz"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# PIN_MEMORY = True if DEVICE == "cuda" else False

# TVL1 algorithm parameters
SIGMA = 0.8
ZOOM_FACTOR = 0.5
INV_ZOOM_FACTOR = 1.0 / ZOOM_FACTOR
DOWN_FACTOR = (ZOOM_FACTOR, ZOOM_FACTOR, ZOOM_FACTOR) # z,y,x
UP_FACTOR = (INV_ZOOM_FACTOR, INV_ZOOM_FACTOR, INV_ZOOM_FACTOR) # z,y,x,d
# NUM_SCALES = 3
NUM_SCALES = 1
MAX_WARPS = 25
# MAX_OUTER_ITERATIONS = 1
MAX_OUTER_ITERATIONS = 100
MAX_INNER_ITERATIOS = 5
LAMBDA = 25
THETA = 0.01
LT = LAMBDA * THETA
TAU = 0.25
