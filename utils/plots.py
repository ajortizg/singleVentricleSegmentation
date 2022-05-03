# ==================================
import sys
sys.path.append('core')

# ==================================
import os
import nibabel as nib
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import numpy as np
import math
import time
from termcolor import colored

# scipy
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# from skimage import measure
# from mpl_toolkits.mplot3d import axes3d

##########################
# general helper functions
#########################


def createSaveDirectory(OUTPUT_PATH, name):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    saveDir = os.path.sep.join([OUTPUT_PATH, name + "_" + timestr])
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    print("save results to directory: ", saveDir, "\n")
    return saveDir


def createSubDirectory(saveDir, SUBDIR_PATH):
    subDir = os.path.sep.join([saveDir, SUBDIR_PATH])
    if not os.path.exists(subDir):
        os.makedirs(subDir)
    return subDir


def printColoredError(diff, tol=1.e-5, accTol=1.e-2):
    if(diff < tol):
        print(colored(diff, 'green'))
    elif(diff < accTol):
        print(colored(diff, 'yellow'))
    else:
        print(colored(diff, 'red'))


##########################
# pytorch input
#########################

def saveCurve1D(input, LX1D, saveDir, name, type="plot"):
    NX1D = input.shape[0]
    grid1D = torch.linspace(0, LX1D, steps=NX1D).cuda()
    if type == "plot":
        plt.plot(grid1D.cpu().detach().numpy(), input.cpu().detach().numpy())
    elif type == "loglog":
        plt.loglog(grid1D.cpu().detach().numpy(), input.cpu().detach().numpy())
    else:
        print("wrong type for saveCurve1D")
    plt.ylabel(name)
    pathName = os.path.join(saveDir, name)
    plt.savefig(pathName, dpi=100)
    plt.close('all')


def saveImage(input, saveDir, name, max_gray_value=1.):
    factor_gray_value = 255. / max_gray_value
    # NY2D = input.shape[0]
    # NX2D = input.shape[1]
    pathNameA = os.path.join(saveDir, name)
    img = input.cpu().detach().numpy()
    # img = np.swapaxes(img, 0, 1)
    cv2.imwrite(pathNameA, factor_gray_value * img)


def save3D_torch_to_nifty(data, saveDir, fileName, affine=None):
    # convert
    nii_data_zyx = data.cpu().detach().numpy()
    nii_data_xyz = np.swapaxes(nii_data_zyx, 0, 2)
    # img
    nii_img = nib.Nifti1Image(nii_data_xyz, affine=affine)
    # save
    outputFile = os.path.sep.join([saveDir, fileName])
    nib.save(nii_img, outputFile)


def save4D_torch_to_nifty(data, saveDir, fileName, affine=None):
    # convert
    nii_data_zyxt = data.cpu().detach().numpy()
    nii_data_xyzt = np.swapaxes(nii_data_zyxt, 0, 2)
    # img
    nii_img = nib.Nifti1Image(nii_data_xyzt, affine=affine)
    # save
    outputFile = os.path.sep.join([saveDir, fileName])
    nib.save(nii_img, outputFile)

    # #convert
    # prolongation_4d_np = prolongation_4d.cpu().detach().numpy()
    # prolongation_4d_xyzt = np.swapaxes(prolongation_4d_np, 0, 2)
    # #header
    # ni_img_4d_hdr = nib.nifti1.Nifti1Header()
    # ni_img_4d_hdr.set_data_shape((NX_prolong,NY_prolong,NZ_prolong,NT))
    # ni_img_4d_hdr.set_zooms( vol_hdr.get_zooms()  )
    # #img
    # ni_img_4d = nib.Nifti1Image(prolongation_4d_xyzt, affine=vol_affine, header=ni_img_4d_hdr)
    # #save
    # outputFile_4d = os.path.sep.join([saveDir4D, PATIENT_NAME + ".nii.gz"])
    # nib.save(ni_img_4d, outputFile_4d)


def save_single_zslices(image3D, saveDir, subdir, max_gray_value=1., color_channel=-1):

    saveDirSlices = os.path.sep.join([saveDir, subdir])
    if not os.path.exists(saveDirSlices):
        os.makedirs(saveDirSlices)
    factor_gray_value = 255. / max_gray_value
    numZSlices = image3D.shape[0]
    for z in range(numZSlices):
        img = image3D[z, :, :].cpu().detach().numpy()
        imgName = f"img_z{z}.png"
        pathName = os.path.join(saveDirSlices, imgName)
        # matplotlib.image.imsave(pathName,img,cmap='gray')
        cv2.imwrite(pathName, factor_gray_value * img)

        if color_channel in range(0, 3):
            imgColor = np.zeros((img.shape[0], img.shape[1], 3))
            imgColor[:, :, color_channel] = img[:, :]
            imgNameColor = f"colorimg_z{z}.png"
            pathNameColor = os.path.join(saveDirSlices, imgNameColor)
            cv2.imwrite(pathNameColor, factor_gray_value * imgColor)


def save_slices(image3D, fileName, saveDir, max_gray_value=1):
    """
    image3D:  pytorch array with shape [Z,Y,X]
    """
    numZSlices = image3D.shape[0]
    aspect_ratio = 16. / 9.
    numCols = int(numZSlices / aspect_ratio)
    if(numZSlices % numCols > 0):
        numCols += 1
    numRows = math.ceil(numZSlices / numCols)

    fig, axs = plt.subplots(numRows, numCols, constrained_layout=True, figsize=(16., 9.), dpi=4)
    #fig.canvas.manager.set_window_title('4D Nifti Image')
    #fig.suptitle('4D_Nifti file: {} \n with {} slices in z-direction'.format(os.path.basename(fileName),numZSlices), fontsize=16)
    fig.suptitle('file: {}'.format(os.path.basename(fileName)), fontsize=16)
    for z, ax in enumerate(axs.flat):
        if z < numZSlices:
            ax.imshow(image3D[z, :, :].cpu().detach().numpy(), cmap='gray', vmin=0, vmax=max_gray_value, interpolation=None)
            ax.set_title("layer {}".format(z))
            ax.axis('off')
        else:
            ax.axis('off')
    pathName = os.path.join(saveDir, fileName)
    plt.savefig(pathName, dpi=100)
    plt.close('all')


def save_color_slices(image3D, mask3D, saveDir, subdir, alpha=0.3, max_gray_value=1):
    """Apply the given mask to the image.
    """
    color = [1, 1, 0]
    saveDirSlices = os.path.sep.join([saveDir, subdir])
    if not os.path.exists(saveDirSlices):
        os.makedirs(saveDirSlices)

    factor_gray_value = 255. / max_gray_value
    numZSlices = image3D.shape[0]

    for z in range(numZSlices):
        img = image3D[z, :, :].cpu().detach().numpy()
        mask = mask3D[z, :, :].cpu().detach().numpy()

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        for c in range(3):
            img[:, :, c] = np.where(mask == 1,
                                    img[:, :, c] *
                                    (1 - alpha) + alpha * color[c],
                                    img[:, :, c])

        imgNameColor = f"colorimg_z{z}.png"
        pathNameColor = os.path.join(saveDirSlices, imgNameColor)
        cv2.imwrite(pathNameColor, factor_gray_value * img)
