#==================================
import sys
sys.path.append('core')

#==================================
import os
import torch
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import math

##########################
# pytorch input
#########################

def saveCurve1D(input, LX1D, saveDir, name):
    NX1D = input.shape[0]
    grid1D = torch.linspace(0, LX1D, steps=NX1D).cuda()
    plt.plot(grid1D.cpu().detach().numpy(), input.cpu().detach().numpy())
    plt.ylabel(name)
    pathName = os.path.join(saveDir, name)
    plt.savefig(pathName,dpi=100)
    plt.close('all')

def saveImage(input,saveDir,name,max_gray_value=1.):
    factor_gray_value = 255. / max_gray_value
    NY2D = input.shape[0]
    NX2D = input.shape[1]
    pathNameA = os.path.join(saveDir, name) 
    img_yx = input.cpu().detach().numpy()
    img = np.swapaxes(img_yx, 0, 1)
    cv2.imwrite(pathNameA,factor_gray_value * img)


def save_single_zslices(image3D, saveDir, subdir, max_gray_value=1., color_channel = -1):

    saveDirSlices = os.path.sep.join([saveDir, subdir])
    if not os.path.exists(saveDirSlices):
        os.makedirs(saveDirSlices)
    factor_gray_value = 255. / max_gray_value
    numZSlices = image3D.shape[0]
    for z in range(numZSlices):
        img = image3D[z,:,:].cpu().detach().numpy()
        imgName = f"img_z{z}.png"
        pathName = os.path.join(saveDirSlices, imgName) 
        #matplotlib.image.imsave(pathName,img,cmap='gray')
        cv2.imwrite(pathName,factor_gray_value * img)

        if color_channel in range(0,3):
            imgColor = np.zeros((img.shape[0], img.shape[1], 3))
            imgColor[:,:,color_channel] = img[:,:]
            imgNameColor = f"colorimg_z{z}.png"
            pathNameColor = os.path.join(saveDirSlices, imgNameColor) 
            cv2.imwrite(pathNameColor,factor_gray_value * imgColor)


def save_slices(image3D, fileName, saveDir, max_gray_value=1):
    """
    image3D:  pytorch array with shape [Z,Y,X]
    """
    numZSlices = image3D.shape[0]
    aspect_ratio = 16./9.
    numCols = int(numZSlices / aspect_ratio)
    if( numZSlices % numCols > 0):
        numCols += 1
    numRows = math.ceil(numZSlices / numCols)

    fig, axs = plt.subplots(numRows, numCols,constrained_layout=True,figsize=(16.,9.),dpi=4)
    #fig.canvas.manager.set_window_title('4D Nifti Image')
    fig.suptitle('4D_Nifti file: {} \n with {} slices in z-direction'.format(os.path.basename(fileName),numZSlices), fontsize=16)
    for z, ax in enumerate(axs.flat):
        if z < numZSlices:
            ax.imshow(image3D[z,:,:].cpu().detach().numpy(), cmap='gray', vmin=0, vmax=max_gray_value, interpolation=None)
            ax.set_title("layer {}".format(z))
            ax.axis('off')
        else:
            ax.axis('off')
    pathName = os.path.join(saveDir, fileName)
    plt.savefig(pathName,dpi=100)
    plt.close('all')

##########################
# numpy input
#########################
def plot_slices(X, str="", block=True):
    """
    X:  numpy array with shape [Z,Y,X]
    """
    fig, _ = plt.subplots(3, X.shape[0] // 3)
    plt.suptitle(f"{str} {X.shape}")
    for i, ax in enumerate(fig.get_axes()):
        ax.imshow(X[i, :, :], cmap="gray", origin="lower")
        # ax.imshow(X[:, :, i], cmap="gray")
    plt.show(block=block)


def plot_slice(slice, block=True):
    plt.figure()
    # plt.imshow(slice.T, cmap="gray", origin="lower")
    plt.imshow(slice, cmap="gray")
    plt.title(f"{slice.shape}")
    plt.show(block=block)


def quiver3():
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    x, y, z = np.meshgrid(np.arange(-0.8, 1, 0.2),
                          np.arange(-0.8, 1, 0.2),
                          np.arange(-0.8, 1, 0.8))

    u = np.sin(np.pi * x) * np.cos(np.pi * y) * np.cos(np.pi * z)
    v = -np.cos(np.pi * x) * np.sin(np.pi * y) * np.cos(np.pi * z)
    w = (np.sqrt(2.0 / 3.0) * np.cos(np.pi * x) * np.cos(np.pi * y) *
         np.sin(np.pi * z))

    ax.quiver(x, y, z, u, v, w, length=0.1, color='black')
    plt.show()
