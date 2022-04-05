
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

def getRangeOfMask_xyz(mask):
    x,y,z = np.nonzero(mask)
    return np.min(z), np.max(z), np.min(y), np.max(y), np.min(x), np.max(x) 

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
        nii_data_xyzt = vol.get_fdata()
        NX = nii_data_xyzt.shape[0]
        NY = nii_data_xyzt.shape[1]
        NZ = nii_data_xyzt.shape[2]
        NT = nii_data_xyzt.shape[3]
        print("   * (NX,NY,NZ,NT) = ", NX, NY, NZ, NT )

        #
        saveDirPatient = os.path.sep.join([saveDirSegmentations, PATIENT_NAME])
        if not os.path.exists(saveDirPatient):
            os.makedirs(saveDirPatient)

        # hdr = vol.header
        # print(hdr.get_xyzt_units())

        #read time steps for diastole and systole
        # rowPatient = df[df['Name'] == PATIENT_NAME]
        # indexPatient = rowPatient.index[0]
        # tDiastole = rowPatient.loc[indexPatient, "Diastole"]
        # tSystole = rowPatient.loc[indexPatient, "Systole"]
        tDiastole = row["Diastole"]
        tSystole = row["Systole"]
        print("   * systole at time:  ", tSystole)
        print("   * diastole at time: ", tDiastole)
        numTimeSteps = tDiastole - tSystole
        print("=======================================")


        # get input masks for diastole and systole 
        nii_mask_diastole_load = nib.load(os.path.sep.join([SEGMENTATIONS_PATH, PATIENT_NAME, PATIENT_NAME + "_Diastole_Labelmap.nii"]))
        nii_mask_diastole_xyz = nii_mask_diastole_load.get_fdata()
        zmin_dia, zmax_dia, ymin_dia, ymax_dia, xmin_dia, xmax_dia = getRangeOfMask_xyz(nii_mask_diastole_xyz)
        print("\nrange of diastole:")
        print("(xmin, xmax) = ", xmin_dia, ",", xmax_dia)
        print("(ymin, ymax) = ", ymin_dia, ",", ymax_dia)
        print("(zmin, zmax) = ", zmin_dia, ",", zmax_dia)

        nii_mask_systole_load = nib.load(os.path.sep.join([SEGMENTATIONS_PATH, PATIENT_NAME, PATIENT_NAME + "_Systole_Labelmap.nii"]))
        nii_mask_systole_xyz = nii_mask_systole_load.get_fdata()
        zmin_sys, zmax_sys, ymin_sys, ymax_sys, xmin_sys, xmax_sys = getRangeOfMask_xyz(nii_mask_systole_xyz)
        print("\nrange of systole:")
        print("(xmin, xmax) = ", xmin_sys, ",", xmax_sys)
        print("(ymin, ymax) = ", ymin_sys, ",", ymax_sys)
        print("(zmin, zmax) = ", zmin_sys, ",", zmax_sys)


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


        # save as nifty
        ni_img_4d = nib.Nifti1Image(cutting_4d, affine=np.eye(4))
        #TODO?
        #ni_img.get_data_dtype() == np.dtype(np.int16)
        #ni_img.header.get_xyzt_units()
        outputFile_4d = os.path.sep.join([saveDir4D, PATIENT_NAME + ".nii.gz"])
        nib.save(ni_img_4d, outputFile_4d)

        cutting_diastole = nii_mask_systole_xyz[xmin_total:xmax_total+1,ymin_total:ymax_total+1,zmin_total:zmax_total+1]
        ni_img_diastole = nib.Nifti1Image(cutting_diastole, affine=np.eye(4))
        outputFile_diastole = os.path.sep.join([saveDirPatient, PATIENT_NAME + "_Diastole_Labelmap.nii"])
        nib.save(ni_img_diastole, outputFile_diastole)

        cutting_systole = nii_mask_systole_xyz[xmin_total:xmax_total+1,ymin_total:ymax_total+1,zmin_total:zmax_total+1]
        ni_img_systole = nib.Nifti1Image(cutting_systole, affine=np.eye(4))
        outputFile_systole = os.path.sep.join([saveDirPatient, PATIENT_NAME + "_Systole_Labelmap.nii"])
        nib.save(ni_img_systole, outputFile_systole)

    #save data base with shifts
    df['xshifts'] = xshifts
    df['yshifts'] = yshifts
    df['zshifts'] = zshifts
    output_df = os.path.sep.join([saveDir, SEGMENTATIONS_FILE_NAME])
    df.to_excel(output_df)  