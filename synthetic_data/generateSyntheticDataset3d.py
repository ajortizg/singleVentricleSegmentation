import matplotlib.pyplot as plt
import numpy as np
from ellipse import Ellipsoid
from tqdm import trange
import sys
import os
import configparser
# import nibabel as nib
import torch
import pandas
import utils

utils_lib_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../utils'))
sys.path.append(utils_lib_path)
import plots


def combine_voxels(ellipsoids):
    NZ, NY, NX = ellipsoids[0].mask.shape
    img = np.zeros((NZ, NY, NX))

    e1 = ellipsoids[0]
    e2 = ellipsoids[1]
    e3 = ellipsoids[2]

    for z in range(NZ):
        for y in range(NY):
            for x in range(NX):
                if e3.mask[z, y, x]:
                    img[z, y, x] = e3.voxels[z, y, x]
                elif e2.mask[z, y, x]:
                    img[z, y, x] = e2.voxels[z, y, x]
                elif e1.mask[z, y, x]:
                    img[z, y, x] = e1.voxels[z, y, x]
    return img


if __name__ == "__main__":

    print("\n\n")
    print("==================================================")
    print("==================================================")
    print("        generate synthetic data set:")
    print("==================================================")
    print("==================================================")
    print("\n\n")

    # load config parser
    config = configparser.ConfigParser()
    config.read('parser/configSynthetic3D.ini')
    cuda_availabe = config.get('DEVICE', 'cuda_availabe')
    DEVICE = "cuda" if cuda_availabe else "cpu"
    # TODO include check from torch
    #DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # create save directories
    saveDir = config.get('DATA', 'OUTPUT_PATH')
    saveDir = plots.createSaveDirectory(saveDir, "Synthetic3D")
    VOLUMES_PATH = plots.createSubDirectory(saveDir, config.get('DATA', 'VOLUMES_SUBDIR_PATH'))
    SEGMENTATIONS_PATH = plots.createSubDirectory(saveDir, config.get('DATA', 'SEGMENTATIONS_SUBDIR_PATH'))
    SEGMENTATIONS_FILE = os.path.sep.join([saveDir, config.get('DATA', 'SEGMENTATIONS_FILE_NAME')])

    df = pandas.DataFrame(columns=['Name', 'Systole', 'Diastole'])

    # save config file to save directory
    conifgOutput = os.path.sep.join([saveDir, "config.ini"])
    with open(conifgOutput, 'w') as configfile:
        config.write(configfile)

    # possibly iterate over patientis
    PATIENT_NAME = config.get('DATA', 'PATIENT_NAME')
    saveDirPatient = plots.createSubDirectory(SEGMENTATIONS_PATH, PATIENT_NAME)

    # Size of voxel map
    NZ, NY, NX = 16, 352, 352
    grid = utils.create_grid(NZ, NY, NX)
    e1A = Ellipsoid(cx=0, cy=0, cz=0, rx=90, ry=150,
                    rz=9, angx=0, angy=0, angz=0)
    e1A.create_voxels(grid, False, value1=0.1, value2=0.3)
    e2A = Ellipsoid(cx=0, cy=0, cz=0, rx=50, ry=100,
                    rz=9, angx=0, angy=0, angz=0)
    e2A.create_voxels(grid, False, value1=0.3, value2=0.6)
    e3A = Ellipsoid(cx=0, cy=0, cz=0, rx=30, ry=50,
                    rz=9, angx=0, angy=0, angz=0)
    e3A.create_voxels(grid, False, value1=0.6, value2=0.9)
    eAs = [e1A, e2A, e3A]

    e1B = Ellipsoid(cx=10, cy=-10, cz=0, rx=90, ry=150,
                    rz=9, angx=0, angy=0, angz=-10)
    e1B.create_voxels(grid, False, value1=0.1, value2=0.3)
    e2B = Ellipsoid(cx=10, cy=-18, cz=0, rx=50, ry=100,
                    rz=9, angx=0, angy=0, angz=15)
    e2B.create_voxels(grid, False, value1=0.3, value2=0.6)
    e3B = Ellipsoid(cx=15, cy=-10, cz=0, rx=30, ry=50,
                    rz=9, angx=0, angy=0, angz=12)
    e3B.create_voxels(grid, False, value1=0.6, value2=0.9)
    eBs = [e1B, e2B, e3B]

    imgA = combine_voxels(eAs)
    imgB = combine_voxels(eBs)

    # Compute intermediate steps
    ts = 10
    tSystole = 0
    tDiastole = 9
    alpha = np.linspace(0.0, 1.0, ts)

    data4d = torch.zeros([NZ, NY, NX, ts]).float().to(DEVICE)

    for t in trange(ts):
        es = []
        for eA, eB in zip(eAs, eBs):
            ei = eA*(1-alpha[t]) + eB*(alpha[t])  # fwd (A -> B)
            ei.create_voxels(grid, eA.constant, eA.value1, eA.value2)
            # ei = eA*(alpha[t]) + eB*(1.0-alpha[t])  # bwd (B -> A)
            es.append(ei)

        img_t = combine_voxels(es)

        mask = torch.from_numpy(es[-1].mask).float().to(DEVICE)
        plots.save3D_torch_to_nifty(mask, saveDirPatient, f"mask_time{t}.nii")

        if t == tSystole:
            plots.save3D_torch_to_nifty(
                mask, saveDirPatient, PATIENT_NAME + "_Systole_Labelmap")
        if t == tDiastole:
            plots.save3D_torch_to_nifty(
                mask, saveDirPatient, PATIENT_NAME + "_Diastole_Labelmap")

        data4d[:, :, :, t] = torch.from_numpy(img_t).float().to(DEVICE)

    plots.save4D_torch_to_nifty(data4d, VOLUMES_PATH, PATIENT_NAME + ".nii.gz")

    patient_row = {'Name': PATIENT_NAME,
                   'Systole': tSystole, 'Diastole': tDiastole}
    df_patient = pandas.DataFrame(patient_row, index=[0])
    df = pandas.concat([df, df_patient], ignore_index=True)
    df.to_excel(SEGMENTATIONS_FILE)
