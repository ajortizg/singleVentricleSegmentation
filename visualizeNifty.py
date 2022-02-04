# plot images from 4D Nifti file #

import sys
sys.path.append('core')

#==================================
import argparse
import os
import cv2
import nibabel as nib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tikzplotlib #to save from matplotlib to tikz
from math import ceil


def saveSingleZSlice(nii_data, z, t, saveDir):
    img = nii_data[:,:,z,t]
    ################
    #test
    #print(np.amin(img))
    #print(np.amax(img))
    #print(img.shape)
    #print(img.dtype)
    ################
    imgName = f"img_t{t}_z{z}.png"
    pathName = os.path.join(saveDir, imgName) 
    #matplotlib.image.imsave(pathName,img,cmap='gray')
    cv2.imwrite(pathName,img)
    

def saveVideo(nii_data, fileName, saveDirVideo):
    numZSlices = nii_data.shape[2]
    numTimeSteps = nii_data.shape[3]
     
    aspect_ratio = 16./9.
    numCols = int(numZSlices / aspect_ratio)
    if( numZSlices % numCols > 0):
        numCols += 1
    numRows = ceil(numZSlices / numCols)

    fig, axs = plt.subplots(numRows, numCols,constrained_layout=True,figsize=(16.,9.),dpi=4)
    #fig.canvas.manager.set_window_title('4D Nifti Image')
    fig.suptitle('4D_Nifti file: {} \n with {} slices in z-direction and {} frames in time'.format(os.path.basename(fileName),numZSlices, numTimeSteps), fontsize=16)
    for t in range(numTimeSteps):
        for z, ax in enumerate(axs.flat):
            if z < numZSlices:
                ax.imshow(nii_data[:,:,z,t],cmap='gray', interpolation=None)
                ax.set_title("layer {} / frame {}".format(z, t))
                ax.axis('off')
            else:
                ax.axis('off')
        imgName = f"video{t}.png"
        pathName = os.path.join(saveDirVideo, imgName)
        plt.savefig(pathName,dpi=100)
    plt.close('all')


def saveVideoSeg(nii_data, fileName, fileNameSeg, timeSeg, saveDirVideoSeg):

    print( timeSeg )

    nii_segimg = nib.load(fileNameSeg)
    nii_segdata = nii_segimg.get_fdata()

    numZSlices = nii_data.shape[2]
    numTimeSteps = nii_data.shape[3]
     
    tikzplotlib.Flavors.latex.preamble() 
    saveDirVideoSegTex = saveDirVideoSeg + '/tex/'
    if not os.path.exists(saveDirVideoSegTex):
        os.makedirs(saveDirVideoSegTex)

    aspect_ratio = 16./9.
    numCols = int(numZSlices / aspect_ratio)
    if( numZSlices % numCols > 0):
        numCols += 1
    numRows = ceil(numZSlices / numCols)

    for t in range(numTimeSteps):
        fig = plt.figure( facecolor='orange', figsize=(16,9), dpi=8 )
        fig.suptitle('4D_Nifti file: {} \n with {} slices in z-direction and {} frames in time'.format(os.path.basename(fileName),numZSlices, numTimeSteps), fontsize=16)
        #axs = fig.add_subplot(numRows, numCols)

        if t == timeSeg:
            for z in range(numZSlices):
                ax = fig.add_subplot(2*numRows, numCols, z+1)
                ax.imshow(nii_data[:,:,z,t],cmap='gray', interpolation=None)
                ax.set_title("layer {} / frame {}".format(z, t))
                ax.axis('off')
            
                axSeg = fig.add_subplot(2*numRows, numCols, z+1+numZSlices)
                axSeg.imshow(nii_segdata[:,:,z],cmap='gray', interpolation=None)
                #axSeg.set_title("layer {} / frame {}".format(z, t))
                axSeg.axis('off')
        else:
            for z in range(numZSlices):
                ax = fig.add_subplot(numRows, numCols, z+1)
                ax.imshow(nii_data[:,:,z,t],cmap='gray', interpolation=None)
                ax.set_title("layer {} / frame {}".format(z, t))
                ax.axis('off')

        imgName = f"video{t}.png"
        pathName = os.path.join(saveDirVideoSeg, imgName)
        plt.savefig(pathName,dpi=100)
        imgNameTex = f"video{t}.tex"
        pathNameTex = os.path.join(saveDirVideoSegTex, imgNameTex)
        tikzplotlib.save(pathNameTex)
        plt.close('all')



def plotNifty(args):

    #==================================
    # Load 4D nifty [x,y,z,t]
    if not (args.fileName):
       parser.error('add -fileName')
    fileName = args.fileName

    nii_img = nib.load(fileName)
    nii_data = nii_img.get_fdata()

    numZSlices = nii_data.shape[2]
    numTimeSteps = nii_data.shape[3]


    #==================================
    #scaling of data 
    print("scaling of data")
    totalMinValue = np.amin(nii_data)
    totalMaxValue = np.amax(nii_data)
    print(f"min value = {totalMinValue}")
    print(f"max value = {totalMaxValue}")
    scaleMaxValue = 255.
    nii_data *= scaleMaxValue / totalMaxValue
    

    #==================================
    # save directory
    saveDir = os.path.dirname("results") 
    if args.saveDir is not None:
        saveDir = os.path.dirname(args.saveDir) 
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    #===================================================
    # save all z slices for all time steps as png
    if args.useSaveAllZSlices:
        saveDirSlices = saveDir + '/all/'
        print(f"save all z slices for all time steps as png to directory {saveDirSlices}")
        if not os.path.exists(saveDirSlices):
           os.makedirs(saveDirSlices)
        for t in range(numTimeSteps):
           for z in range(numZSlices):
               saveSingleZSlice( nii_data, z, t, saveDirSlices )
    else:
        print("skip to save all slices (optionally set useSaveAllZSlices)")

    #===================================================
    # save video
    if args.useSaveVideo:
        print("save video")
        saveDirVideo = saveDir + '/video/'
        if not os.path.exists(saveDirVideo):
           os.makedirs(saveDirVideo)
        saveVideo( nii_data, fileName, saveDirVideo )
    else:
        print("skip to save video (optionally set useSaveVideo)")



    #===================================================
    # save video with segmentation
    if args.useSaveVideo:
        print("save video with segmenation")
    if args.fileNameSeg is not None:
        fileNameSeg = args.fileNameSeg
        if not (args.timeSeg):
            parser.error('add -timeSeg')
        timeSeg = args.timeSeg
        saveDirVideoSeg = saveDir + '/videoSeg/'
        if not os.path.exists(saveDirVideoSeg):
            os.makedirs(saveDirVideoSeg)
        saveVideoSeg( nii_data, fileName, fileNameSeg, timeSeg, saveDirVideoSeg )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fileName', help="file name of nifty")
    parser.add_argument('--saveDir', help="directory for saving")
    
    #parser.add_argument('--useSaveAllZSlices', default=True, action='store_false', help='save every slice as image')
    parser.add_argument('--useSaveAllZSlices', dest='useSaveAllZSlices', action='store_true')
    parser.add_argument('--no-useSaveAllZSlices', dest='useSaveAllZSlices', action='store_false')
    parser.set_defaults(useSaveAllZSlices=True)

    parser.add_argument('--useSaveVideo', dest='useSaveVideo', action='store_true')
    parser.add_argument('--no-useSaveVideo', dest='useSaveVideo', action='store_false')
    parser.set_defaults(useSaveVideo=False)

    parser.add_argument('--fileNameSeg', help="file name of segmentatioin")
    parser.add_argument('--timeSeg', type=int, help="time of segmentation")

    args = parser.parse_args()

    plotNifty(args)