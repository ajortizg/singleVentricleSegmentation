
import numpy as np
import os
import configparser
from PIL import Image

from ROF.ROF2D import *

from opticalFlow_cuda_ext import opticalFlow


if __name__ == "__main__":

    print("\n\n")
    print("=======================================================")
    print("=======================================================")
    print("        compute ROF in 2D:")
    print("=======================================================")
    print("=======================================================")
    print("\n\n")

    # load config parser
    config = configparser.ConfigParser()
    config.read('parser/configROF2D.ini')
    cuda_availabe = config.get('DEVICE', 'cuda_availabe')
    DEVICE = "cuda" if cuda_availabe else "cpu"

    # create save directory
    timestr = time.strftime("%Y%m%d-%H%M%S")
    OUTPUT_PATH = config.get('DATA', 'OUTPUT_PATH')
    saveDir = os.path.sep.join([OUTPUT_PATH, "ROF2D_" + timestr])
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
    NX = image0.shape[1]
    NY = image0.shape[0]
    print( f"   * dimensions I0 input: (Y,X) = ({NY},{NX})" )

    # #==================================
    # #scaling of data 
    totalMinValue = np.amin(image0)
    totalMaxValue = np.amax(image0)
    scaleMinValue = 0.
    scaleMaxValue = 1.
    # scaleMaxValue = 255.
    print(f"   * scaling of data in range {totalMinValue,totalMaxValue} to {scaleMinValue,scaleMaxValue}")
    image0 *= scaleMaxValue / totalMaxValue

    ## swap from (X,Y) to cuda-compatible (Y,X):
    # I0np = np.swapaxes(image0, 0, 1)
    # print( f"   * dimensions I0 after swap: (Y,X) = {I0np.shape}")

    # add noise
    print(f"   * add noise to input image")
    noiseParam = config.getfloat('DEBUG', 'addNoiseToInput')
    if noiseParam > 0.:
      image0 += noiseParam*np.random.randn(*image0.shape)

    #convert to torch for given time steps
    I0 = torch.from_numpy(image0).float().to(DEVICE)
    print( f"   * dimensions I0 torch: (Y,X) = {I0.shape}")

    #initialization of optical flow 
    I = torch.zeros([NY,NX]).float().to(DEVICE)
    p = torch.zeros([NY,NX,2]).float().to(DEVICE)

    # Compute the optical flow
    alg = ROF2D(saveDir,config)
    alg.computeOnPyramid(I0, I, p)
    #alg.computeOnPyramid(I1, I0, u, p)


