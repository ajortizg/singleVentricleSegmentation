
import sys
import numpy as np
import nibabel as nib
import os
import pandas
import configparser
import time

utils_lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils'))
sys.path.append(utils_lib_path)
import plots

def getRangeOfMask_xyz(mask,printRange=False,name="" ):
    x,y,z = np.nonzero(mask)
    xmin = np.min(x) 
    xmax = np.max(x)
    ymin = np.min(y) 
    ymax = np.max(y)
    zmin = np.min(z) 
    zmax = np.max(z)
    if printRange:
        print("\nrange of mask", name, ":")
        print("(xmin, xmax) = ", xmin, ",", xmax)
        print("(ymin, ymax) = ", ymin, ",", ymax)
        print("(zmin, zmax) = ", zmin, ",", zmax)
    return zmin, zmax, ymin, ymax, xmin, xmax 

def save_np_to_nifty(file,saveDir,fileName,hdr_old):
    #header 
    hdr = nib.nifti1.Nifti1Header()
    hdr.set_data_shape(file.shape)
    hdr.set_qform( hdr_old.get_qform() )
    hdr.set_sform( hdr_old.get_sform() )
    hdr.set_zooms( hdr_old.get_zooms() )
    #img
    ni_img = nib.Nifti1Image(file, affine=None, header=hdr)
    #save
    outputFile = os.path.sep.join([saveDir, fileName])
    nib.save(ni_img, outputFile)
    # print("old header:")
    # print(hdr_old) 
    # print("new header:")
    # print(hdr) 

if __name__ == "__main__":

    print("\n\n")
    print("==================================================")
    print("==================================================")
    print("  preprocessing data: cutting out heart region    ")
    print("==================================================")
    print("==================================================")
    print("\n\n")

    # load config parser
    config = configparser.ConfigParser()
    config.read('parser/configTVL1OF3D.ini')

    # create save directory
    saveDir = plots.createSaveDirectory(config.get('DATA', 'OUTPUT_PATH'), "preprocessing3D_cut" )

    #save config file to save directory
    conifgOutput = os.path.sep.join([saveDir, "config.ini"])
    with open(conifgOutput, 'w') as configfile:
      config.write(configfile)

    # load data base
    BASE_PATH_3D = config.get('DATA', 'BASE_PATH_3D')
    VOLUMES_SUBDIR_PATH = config.get('DATA', 'VOLUMES_SUBDIR_PATH')
    VOLUMES_PATH = os.path.sep.join([BASE_PATH_3D, VOLUMES_SUBDIR_PATH])
    SEGMENTATIONS_FILE_NAME = config.get('DATA', 'SEGMENTATIONS_FILE_NAME')
    SEGMENTATIONS_FILE = os.path.sep.join([BASE_PATH_3D, SEGMENTATIONS_FILE_NAME])
    df = pandas.read_excel(SEGMENTATIONS_FILE)
    numDataFiles = df.shape[0]
    print("number of data files = ", numDataFiles)
    SEGMENTATIONS_SUBDIR_PATH = config.get('DATA', 'SEGMENTATIONS_SUBDIR_PATH')
    SEGMENTATIONS_PATH = os.path.sep.join([BASE_PATH_3D, SEGMENTATIONS_SUBDIR_PATH])

    #generate columns for (x,y,z)-shifts
    xshifts = np.zeros(numDataFiles)
    yshifts = np.zeros(numDataFiles)
    zshifts = np.zeros(numDataFiles)

    #
    saveDir4D = plots.createSubDirectory(saveDir, VOLUMES_SUBDIR_PATH)
    saveDirSegmentations = plots.createSubDirectory(saveDir, SEGMENTATIONS_SUBDIR_PATH)

    for index, row in df.iterrows():

        # Load 4D nifty [x,y,z,t]
        print("=======================================")
        PATIENT_NAME = row['Name']
        print("load data for patient: ", PATIENT_NAME)
        vol = nib.load(os.path.sep.join([VOLUMES_PATH, PATIENT_NAME + ".nii.gz"]))
        vol_hdr = vol.header
        #vol_affine = vol.affine
        nii_data_xyzt = vol.get_fdata()
        NX = nii_data_xyzt.shape[0]
        NY = nii_data_xyzt.shape[1]
        NZ = nii_data_xyzt.shape[2]
        NT = nii_data_xyzt.shape[3]
        print("   * (NX,NY,NZ,NT) = ", NX, NY, NZ, NT )
        #
        saveDirPatient = plots.createSubDirectory(saveDirSegmentations, PATIENT_NAME)
        #read time steps for diastole and systole
        tDiastole = row["Diastole"]
        tSystole = row["Systole"]
        print("   * systole at time:  ", tSystole)
        print("   * diastole at time: ", tDiastole)
        print("=======================================")

        # get input masks for diastole
        nii_mask_diastole_load = nib.load(os.path.sep.join([SEGMENTATIONS_PATH, PATIENT_NAME, PATIENT_NAME + "_Diastole_Labelmap.nii"]))
        hdr_mask_diastole = nii_mask_diastole_load.header
        #affine_mask_diastole = nii_mask_diastole_load.affine
        nii_mask_diastole_xyz = nii_mask_diastole_load.get_fdata()
        zmin_dia, zmax_dia, ymin_dia, ymax_dia, xmin_dia, xmax_dia = getRangeOfMask_xyz(nii_mask_diastole_xyz,printRange=True,name="diastole")

        # get input masks for systole
        nii_mask_systole_load = nib.load(os.path.sep.join([SEGMENTATIONS_PATH, PATIENT_NAME, PATIENT_NAME + "_Systole_Labelmap.nii"]))
        hdr_mask_systole = nii_mask_systole_load.header
        #affine_mask_systole = nii_mask_systole_load.affine
        nii_mask_systole_xyz = nii_mask_systole_load.get_fdata()
        zmin_sys, zmax_sys, ymin_sys, ymax_sys, xmin_sys, xmax_sys = getRangeOfMask_xyz(nii_mask_systole_xyz,printRange=True,name="systole")


        xmin_total = max(0, min(xmin_dia,xmin_sys) - 10)
        xmax_total = min(NX-1, max(xmax_dia,xmax_sys) + 10)
        ymin_total = max(0, min(ymin_dia,ymin_sys) - 10)
        ymax_total = min(NY-1, max(ymax_dia,ymax_sys) + 10)
        zmin_total = max(0, min(zmin_dia,zmin_sys) - 10 )
        zmax_total = min(NZ-1, max(zmax_dia,zmax_sys) + 10)
        print("\ntotal range:")
        print("(xmin, xmax) = ", xmin_total, ",", xmax_total)
        print("(ymin, ymax) = ", ymin_total, ",", ymax_total)
        print("(zmin, zmax) = ", zmin_total, ",", zmax_total)

        xshifts[index] = xmin_total 
        yshifts[index] = ymin_total
        zshifts[index] = zmin_total

        NX_cut = xmax_total - xmin_total + 1
        NY_cut = ymax_total - ymin_total + 1
        NZ_cut = zmax_total - zmin_total + 1

        cutting_4d = nii_data_xyzt[xmin_total:xmax_total+1,ymin_total:ymax_total+1,zmin_total:zmax_total+1,:]
        cutting_diastole = nii_mask_diastole_xyz[xmin_total:xmax_total+1,ymin_total:ymax_total+1,zmin_total:zmax_total+1]
        cutting_systole = nii_mask_systole_xyz[xmin_total:xmax_total+1,ymin_total:ymax_total+1,zmin_total:zmax_total+1]

        #save to nifty
        save_np_to_nifty(cutting_4d, saveDir4D, PATIENT_NAME + ".nii.gz", vol_hdr)
        save_np_to_nifty(cutting_diastole, saveDirPatient, PATIENT_NAME + "_Diastole_Labelmap.nii", hdr_mask_diastole)
        save_np_to_nifty(cutting_systole, saveDirPatient, PATIENT_NAME + "_Systole_Labelmap.nii", hdr_mask_systole)


    #save data base with shifts
    df['xshifts'] = xshifts
    df['yshifts'] = yshifts
    df['zshifts'] = zshifts
    output_df = os.path.sep.join([saveDir, SEGMENTATIONS_FILE_NAME])
    df.to_excel(output_df)

