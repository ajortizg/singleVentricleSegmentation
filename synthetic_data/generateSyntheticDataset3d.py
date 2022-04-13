import numpy as np
from ellipse import Ellipsoid
import sys
import os
from tqdm import trange
import configparser
# import nibabel as nib
import torch
import pandas
import os.path as osp

utils_lib_path = osp.abspath(osp.join(osp.dirname(__file__), '../utils'))
sys.path.append(utils_lib_path)
import plots
import torch_warping


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


def create_ellipsoid(config, name, grid):
    params_str = config.get('PARAMETERS', name).replace(" ", "")
    params = list(map(float, params_str.split(',')))
    constant = config.getboolean('PARAMETERS', 'CONSTANT_GRAY')
    e = Ellipsoid(cx=params[0],
                  cy=params[1],
                  cz=params[2],
                  rx=params[3],
                  ry=params[4],
                  rz=params[5],
                  angx=params[6],
                  angy=params[7],
                  angz=params[8]
                  )
    e.create_voxels(grid, constant=constant, value1=params[9], value2=params[10])
    return e


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
    # use_cuda = config.get('DEVICE', 'USE_CUDA')
    # DEVICE = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'

    # create save directories
    saveDir = config.get('DATA', 'OUTPUT_PATH')
    saveDir = plots.createSaveDirectory(saveDir, 'Synthetic3D')
    VOLUMES_PATH = plots.createSubDirectory(saveDir, config.get('DATA', 'VOLUMES_SUBDIR_PATH'))
    SEGMENTATIONS_PATH = plots.createSubDirectory(saveDir, config.get('DATA', 'SEGMENTATIONS_SUBDIR_PATH'))
    SEGMENTATIONS_FILE = os.path.sep.join([saveDir, config.get('DATA', 'SEGMENTATIONS_FILE_NAME')])

    df = pandas.DataFrame(columns=['Name', 'Systole', 'Diastole'])

    # save config file to save directory
    conifgOutput = os.path.sep.join([saveDir, 'config.ini'])
    with open(conifgOutput, 'w') as configfile:
        config.write(configfile)

    # possibly iterate over patientis
    PATIENT_NAME = config.get('DATA', 'PATIENT_NAME')
    saveDirPatient = plots.createSubDirectory(SEGMENTATIONS_PATH, PATIENT_NAME)

    # Size of voxel map
    NZ = config.getint('PARAMETERS', 'NZ')
    NY = config.getint('PARAMETERS', 'NY')
    NX = config.getint('PARAMETERS', 'NX')

    grid = torch_warping.create_grid(NZ, NY, NX).numpy()

    eA1 = create_ellipsoid(config, 'eA1', grid)
    eA2 = create_ellipsoid(config, 'eA2', grid)
    eA3 = create_ellipsoid(config, 'eA3', grid)
    eAs = [eA1, eA2, eA3]

    eB1 = create_ellipsoid(config, 'eB1', grid)
    eB2 = create_ellipsoid(config, 'eB2', grid)
    eB3 = create_ellipsoid(config, 'eB3', grid)
    eBs = [eB1, eB2, eB3]

    imgA = combine_voxels(eAs)
    imgB = combine_voxels(eBs)

    plots.save_slices(torch.from_numpy(imgA), 'A.png', saveDir)
    plots.save_slices(torch.from_numpy(imgB), 'B.png', saveDir)

    # Compute intermediate steps
    ts = 10
    tSystole = 0
    tDiastole = 9
    alpha = np.linspace(0.0, 1.0, ts)

    data4d = torch.zeros([NZ, NY, NX, ts]).float()

    save_steps_dir = plots.createSubDirectory(saveDir, 'Steps')

    for t in trange(ts):
        es = []
        for eA, eB in zip(eAs, eBs):
            ei = eA * (1 - alpha[t]) + eB * (alpha[t])  # fwd (A -> B)
            ei.create_voxels(grid, eA.constant, eA.value1, eA.value2)
            # ei = eA*(alpha[t]) + eB*(1.0-alpha[t])  # bwd (B -> A)
            es.append(ei)

        img_t = combine_voxels(es)
        plots.save_slices(torch.from_numpy(img_t), f'step_{t}.png', save_steps_dir)

        mask = torch.from_numpy(es[-1].mask).float()
        plots.save3D_torch_to_nifty(mask, saveDirPatient, f'mask_time{t}.nii')

        if t == tSystole:
            plots.save3D_torch_to_nifty(mask, saveDirPatient, PATIENT_NAME + '_Systole_Labelmap')
        if t == tDiastole:
            plots.save3D_torch_to_nifty(mask, saveDirPatient, PATIENT_NAME + '_Diastole_Labelmap')

        data4d[:, :, :, t] = torch.from_numpy(img_t).float()

    plots.save4D_torch_to_nifty(data4d, VOLUMES_PATH, PATIENT_NAME + '.nii.gz')

    patient_row = {'Name': PATIENT_NAME, 'Systole': tSystole, 'Diastole': tDiastole}
    df_patient = pandas.DataFrame(patient_row, index=[0])
    df = pandas.concat([df, df_patient], ignore_index=True)
    df.to_excel(SEGMENTATIONS_FILE)
