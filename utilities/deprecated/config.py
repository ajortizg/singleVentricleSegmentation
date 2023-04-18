import torch
import os

BASE_PATH = "singleVentricleData"
VOLUMES_PATH = os.path.sep.join([BASE_PATH, "NIFTI_4D_Datasets"])
SEGMENTATIONS_PATH = os.path.sep.join([BASE_PATH, "NIFTI_Single_Ventricle_Segmentations"])
SEGMENTATIONS_FILE = os.path.sep.join([BASE_PATH, "Segmentation_volumes.xlsx"])
PATIENT_NAME = "Adult_30"
# PATIENT_NAME = "Adolescent_87"
#PATIENT_NAME = "Child_10"
OUTPUT_PATH = "results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# PIN_MEMORY = True if DEVICE == "cuda" else False

############################
# TVL1 algorithm parameters
############################


## new
NUM_SCALES = 3
MAX_WARPS = 25
MAX_OUTER_ITERATIONS = 1000

#primalFctWeight_Matching = 1.
#dualFctWeight_TV = 25.
primalFctWeight_Matching = 100.
dualFctWeight_TV = 0.

ChambollePockType = 2
sigma = 0.5
tau = 0.5
theta = 0.25
gamma = 1.


##2D##
BASE_PATH_DATA_2D = "MiddleburyOpticalFlow"
Image0_PATH = os.path.sep.join([BASE_PATH_DATA_2D, "other-data-gray/Beanbags/frame07.png"])
Image1_PATH = os.path.sep.join([BASE_PATH_DATA_2D, "other-data-gray/Beanbags/frame08.png"])


