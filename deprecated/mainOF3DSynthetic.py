# import sys
# import nibabel as nib
# from cnn.dataset import SingleVentricleDataset
# import numpy as np
# import os.path as osp
# import os
# from TVL1OF.TVL1OF3D import *

# utils_lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'utils'))
# sys.path.append(utils_lib_path)
# import plots

# from opticalFlow_cuda_ext import opticalFlow


# def iou_score(mask1, mask2):
#     intersection = torch.logical_and(mask1, mask2)
#     union = torch.logical_or(mask1, mask2)
#     return torch.sum(intersection) / torch.sum(union)


# if __name__ == "__main__":

#     print("\n\n")
#     print("==================================================")
#     print("==================================================")
#     print("        compute TV-L1 optical flow:")
#     print("==================================================")
#     print("==================================================")
#     print("\n\n")

#     # load config parser
#     config = configparser.ConfigParser()
#     config.read('parser/configTVL1OF3D.ini')
#     cuda_availabe = config.get('DEVICE', 'cuda_availabe')
#     DEVICE = "cuda" if cuda_availabe and torch.cuda.is_available() else "cpu"
#     PATIENT_NAME = config.get('DATA', 'PATIENT_NAME')

#     # create save directory
#     saveDir = plots.createSaveDirectory(config.get('DATA', 'OUTPUT_PATH'), "TVL1OF3DBackward")

#     # save config file to save directory
#     conifgOutput = os.path.sep.join([saveDir, "config.ini"])
#     with open(conifgOutput, 'w') as configfile:
#         config.write(configfile)

#     ds = SingleVentricleDataset(config, load_flow=False)

#     idx, found = ds.index_for_patient(PATIENT_NAME)
#     if not found:
#         print(PATIENT_NAME + " not found!")
#         sys.exit()

#     pname, data, mask_systole, mask_diastole, systole_time, diastole_time, _, _ = ds[idx]
#     data = data.squeeze().to(DEVICE)
#     mask_systole = mask_systole.squeeze().to(DEVICE)
#     mask_diastole = mask_diastole.squeeze().to(DEVICE)
#     NZ, NY, NX, NT = data.shape
#     print("=======================================")
#     print("load data for patient: ", pname)
#     print(f"   * dimensions: (Z,Y,X,T) = {data.shape}")
#     print("   * systole at time:  ", systole_time)
#     print("   * diastole at time: ", diastole_time)
#     numTimeSteps = abs(diastole_time - systole_time)
#     initTimeStep = min(diastole_time, systole_time)
#     finalTimeStep = max(diastole_time, systole_time)
#     print("=======================================")
#     print("\n")

#     saveDirInitTime = plots.createSubDirectory(saveDir, f"time{initTimeStep}")

#     # initialization of optical flow and mask
#     u = torch.zeros([NZ, NY, NX, 3]).float().to(DEVICE)
#     p = torch.zeros([NZ, NY, NX, 3, 3]).float().to(DEVICE)
#     mask = None
#     if initTimeStep == systole_time:
#         mask = mask_systole.clone().detach()
#         print('mask=mask_systole')
#     else:
#         mask = mask_diastole.clone().detach()
#         print('mask=mask_diastole')

#     # accum_iou = 0
#     # accum_diff = 0
#     for t in range(initTimeStep, finalTimeStep):
#         saveDirTimeStep = plots.createSubDirectory(saveDir, f"time{t}")

#         # convert to torch for given time steps
#         t0 = t + 1
#         t1 = t
#         I0 = data[:, :, :, t0]
#         I1 = data[:, :, :, t1]

#         # Compute the optical flow
#         alg = TVL1OpticalFlow3D(saveDirTimeStep, config)
#         u, p = alg.computeOnPyramid(I0, I1, u, p)
#         # flowName = "flow_it0.pt"
#         # fileNameFlow = os.path.join(saveDirStep, flowName)
#         # u = torch.load(fileNameFlow, map_location=torch.device(DEVICE))

#         # read gt mask
#         # gt_file = osp.sep.join([SEGMENTATIONS_PATH, PATIENT_NAME, f'mask_time{t}.nii'])
#         # gt = nib.load(gt_file)
#         # gt = torch.from_numpy(np.swapaxes(gt.get_fdata(), 0, 2)).float().to(DEVICE)
#         # iou = iou_score(gt, mask)
#         # accum_iou += iou
#         # print(f'iou {t} -> {iou}')
#         # diff = torch.abs(mask - gt)
#         # accum_diff += diff.mean()
#         # print(f'diff {t} -> {diff.mean()}')

#         print(torch.unique(mask))
#         # save the old mask
#         save3D_torch_to_nifty(mask, saveDirTimeStep, f"mask_time{t1}.nii")
#         save_slices(mask, f"mask.png", saveDirTimeStep)
#         save_single_zslices(mask, saveDirTimeStep, "mask_slices", 1., 2)

#         # warp mask with the computed optical flow
#         mask = alg.warpMask(mask, u, t0, saveDirTimeStep)
#         # print(f'mean {torch.mean(mask)}')
#         # mask = torch.sigmoid(mask)
#         # mask = normalize(mask)
#         # mask = torch.where(mask > 0.5, 1.0, 0.0)
#         # save_slices(mask, f"mask_warped_norm_time{t0}.png", saveDir)

#     # print(accum_iou / numTimeSteps)
#     # print(accum_diff / numTimeSteps)


# import numpy as np
# import os.path as osp
# import torch
# import configparser
# import sys
# from cnn.dataset import SingleVentricleDataset
# from TVL1OF.TVL1OF3D import *


# utils_lib_path = osp.abspath(osp.join(osp.dirname(__file__), 'utils'))
# sys.path.append(utils_lib_path)
# import plots


# if __name__ == "__main__":
#     print("\n\n")
#     print("==================================================")
#     print("==================================================")
#     print("        compute TV-L1 optical flow:")
#     print("==================================================")
#     print("==================================================")
#     print("\n\n")

#     config = configparser.ConfigParser()
#     config.read('parser/configTVL1OF3D.ini')
#     cuda_availabe = config.get('DEVICE', 'cuda_availabe')
#     DEVICE = 'cuda' if cuda_availabe and torch.cuda.is_available() else 'cpu'
#     # PATIENT_NAME = config.get('DATA', 'PATIENT_NAME')

#     # create save directory
#     save_dir = plots.createSaveDirectory(config.get('DATA', 'OUTPUT_PATH'), 'TVL1OF3DBackward')

#     # save config file to save directory
#     conifg_output = os.path.sep.join([save_dir, 'config.ini'])
#     with open(conifg_output, 'w') as config_file:
#         config.write(config_file)

#     ds = SingleVentricleDataset(config, load_flow=False)

#     # idx, found = ds.index_for_patient(PATIENT_NAME)
#     # if not found:
#     #     print(PATIENT_NAME + " not found!")
#     #     sys.exit()

#     for idx in range(len(ds)):
#         (pname, data, mask_systole, mask_diastole, systole_time, diastole_time, _, _) = ds[idx]
#         NZ, NY, NX, NT = data.shape
#         print('\n====================================')
#         print('Load data for patient: ' + pname)
#         print(f'\t* (NZ, NY, NX, NT) = ({NZ}, {NY}, {NX}, {NT})')
#         print(f'\t* Systole at time: {systole_time}')
#         print(f'\t* Diastole at time: {diastole_time}')
#         print('====================================\n')

#         # num_timesteps = abs(diastole_time - systole_time)
#         init_timestep = min(diastole_time, systole_time)
#         final_timestep = max(diastole_time, systole_time)

#         patient_dir = osp.join(save_dir, pname)

#         saveDirInitTime = plots.createSubDirectory(patient_dir, f"time{init_timestep}")

#         # initialization of optical flow and mask
#         u = torch.zeros([NZ, NY, NX, 3]).float().to(DEVICE)
#         p = torch.zeros([NZ, NY, NX, 3, 3]).float().to(DEVICE)

#         mask = None
#         if init_timestep == systole_time:
#             mask = mask_systole.clone().detach().to(DEVICE)
#         else:
#             mask = mask_diastole.clone().detach().to(DEVICE)

#         for t in range(init_timestep, final_timestep):
#             save_timestep_dir = plots.createSubDirectory(patient_dir, f"time{t}")

#             # Compute the optical flow for given time steps
#             # t0, t1 = t + 1, t  # bwd
#             t0, t1 = t, t + 1  # fwd
#             print(f"{t1} -> {t0}")
#             I0 = data[:, :, :, t0].to(DEVICE)
#             I1 = data[:, :, :, t1].to(DEVICE)
#             alg = TVL1OpticalFlow3D(save_timestep_dir, config)
#             u, p = alg.computeOnPyramid(I0, I1, u, p)

#             # save the old mask
#             save3D_torch_to_nifty(mask, save_timestep_dir, f"mask_time{t1}.nii")
#             save_slices(mask, f"mask.png", save_timestep_dir)
#             save_single_zslices(mask, save_timestep_dir, "mask_slices", 1., 2)

#             # warp mask with the computed optical flow
#             mask = alg.warpMask(mask, u, t0, save_timestep_dir)
