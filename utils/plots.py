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


def saveCurve1D(input, LX1D, saveDir, name):
    NX1D = input.shape[0]
    grid1D = torch.linspace(0, LX1D, steps=NX1D).cuda()
    plt.plot(grid1D.cpu().detach().numpy(), input.cpu().detach().numpy())
    plt.ylabel(name)
    pathName = os.path.join(saveDir, name)
    plt.savefig(pathName,dpi=100)
    plt.close('all')

def saveImage(input,saveDir,name):
    NY2D = input.shape[0]
    NX2D = input.shape[1]
    pathNameA = os.path.join(saveDir, name) 
    cv2.imwrite(pathNameA,input.cpu().detach().numpy())

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
