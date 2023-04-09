# import nibabel as nib
import numpy as np
import os
# import pandas
import configparser
from PIL import Image

from TVL1OF.deprecated.TVL1OF2D import *

from opticalFlow_cuda_ext import opticalFlow


if __name__ == "__main__":

    print("\n\n")
    print("=======================================================")
    print("=======================================================")
    print("        compute TV-L1 optical flow in 2D:")
    print("=======================================================")
    print("=======================================================")
    print("\n\n")

    # load config parser
    config = configparser.ConfigParser()
    config.read('parser/configTVL1OF2D.ini')
    cuda_availabe = config.get('DEVICE', 'cuda_availabe')
    DEVICE = "cuda" if cuda_availabe else "cpu"
    #TODO include check from torch
    #DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # create save directory
    timestr = time.strftime("%Y%m%d-%H%M%S")
    OUTPUT_PATH = config.get('DATA', 'OUTPUT_PATH')
    saveDir = os.path.sep.join([OUTPUT_PATH, "TVL1OF2D_" + timestr])
    if not os.path.exists(saveDir):
      os.makedirs(saveDir)
    print("save results to directory: ", saveDir, "\n")

    #save config file to save directory
    conifgOutput = os.path.sep.join([saveDir, "config.ini"])
    with open(conifgOutput, 'w') as configfile:
      config.write(configfile)

    # Load 2D images [x,y]
    print("============================================")
    print("   * load data")
    BASE_PATH_2D = config.get('DATA', 'BASE_PATH_2D')
    Image0_SUB_PATH = config.get('DATA', 'Image0_SUB_PATH')
    Image0_PATH = os.path.sep.join([BASE_PATH_2D, Image0_SUB_PATH])
    image0_PIL = Image.open(Image0_PATH)
    image0 = np.array(image0_PIL,dtype='f')
    Image1_SUB_PATH = config.get('DATA', 'Image1_SUB_PATH')
    Image1_PATH = os.path.sep.join([BASE_PATH_2D, Image1_SUB_PATH])
    image1_PIL = Image.open(Image1_PATH)
    image1 = np.array(image1_PIL,dtype='f')
    
    # summarize some details about the image
    #print(image0.format)
    #print(image0.size)
    #print(image0.mode)

    NX = image0.shape[1]
    NY = image0.shape[0]
    print( f"   * dimensions I0 input: (Y,X) = ({NY},{NX})" )

    # #==================================
    # #scaling of data 
    totalMinValue = min(np.amin(image0),np.amin(image1))
    totalMaxValue = max(np.amax(image0),np.amax(image1))
    scaleMinValue = 0.
    scaleMaxValue = 1.
    # scaleMaxValue = 255.
    print(f"   * scaling of data in range {totalMinValue,totalMaxValue} to {scaleMinValue,scaleMaxValue}")
    image0 *= scaleMaxValue / totalMaxValue
    image1 *= scaleMaxValue / totalMaxValue

    #convert to torch for given time steps
    I0 = torch.from_numpy(image0).float().to(DEVICE)
    I1 = torch.from_numpy(image1).float().to(DEVICE)

    #initialization of optical flow 
    u = torch.zeros([NY,NX,2]).float().to(DEVICE)
    p = torch.zeros([NY,NX,2,2]).float().to(DEVICE)

    # Compute the optical flow
    alg = TVL1OpticalFlow2D(saveDir,config)
    alg.computeOnPyramid(I0, I1, u, p)


