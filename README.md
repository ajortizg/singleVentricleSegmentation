# singleVentricleSegmentation

## install
* conda env create -f environment.yaml
<!-- * conda create -n svs-env python=3.10
* conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
* pip install opencv-python
* pip install nibabel
* pip install matplotlib
* pip install tikzplotlib
* pip install termcolor
* pip install scipy
* conda install scikit-image
* pip install tqdm
* pip install pandas
* pip install openpyxl
* pip install torchsummary
* pip install elasticdeform
* pip install monai
* pip install intensity-normalization
* pip install natsort
* pip install batchgenerators
* pip install tabulate -->

to install setup file use one of the following:
* pip install . 
* pip install . --use-feature=in-tree-build

to use config file link data and result folder: 

on linux:
* ln -s /home/.../singleVentricleData/ /home/.../singleVentricleSegmentation/
* ln -s /home/.../results /home/.../singleVentricleSegmentation/

on windows:
* New-Item -ItemType SymbolicLink -Target "C:\Users\...\data\" -Path "C:\Users\...\singleVentricleSegmentation\data"
* New-Item -ItemType SymbolicLink -Target "C:\Users\...\results\" -Path "C:\Users\...\singleVentricleSegmentation\results"

## Instructions for the ACDC dataset
* First you need to format the acdc dataset to separately extract the 3 different ROIs contained in the dataset:
    ```
    python deprecated/dataset/format_acdc.py
    ```


## 1. Preprocessing 
*   First preprocess the raw data to store it in a specific format:
    ```
    python datasets/preprocessing/preprocess_{dataset_name}.py
    ```
    This will create a new folder *{dataset_name}_{date-time}* inside *{output_dir}*. 
*   Use *parser/preprocessing.ini* to modify the configuration parameters.

## 2. Optical flow
*   After preprocess the data, the optical flow can be computed:
    ```
    python TVL1OF/compute_flow.py
    ```
*   This will create a folder *optical_flow* inside the *{root_dir}*. Use *parser/flow_compute.ini* to set the correct paths and desired configuration.

## 3. Train CNN

<!-- 1. possibly flip the original data
   in parser set BASE_PATH_3D to data/singleVentricleData
    ```
    python ./dataset/preprocessing/preprocessing_flipping.py
    ```
    save resulting folder to /data/singleVentricleData_flip

2. cut the (possibly flipped) original data
    in parser set BASE_PATH_3D to data/singleVentricleData_norm
    ```
    python ./dataset/preprocessing/preprocessing_cutting.py
    ```
    save resulting folder to /data/singleVentricleData_cut

3. resize the (flipped and cutted) data
    in parser set BASE_PATH_3D to data/singleVentricleData_cut
    ```
    python ./dataset/preprocessing/preprocessing_prolongation.py
    ```
    Save resulting folder to /data/singleVentricleData_prol.
    Note that this depends on the Interpolationtype and Bondarytype.

4. Data normalization
    in parset set BASE_PATH_3D to /data/singleVentricleData_prol
    ```
    python ./dataset/preprocessing/preprocessing_normalization.py
    ```
    Save resulting folder to /data/singleVentricleData_norm

5. split the dataset for training the CNN in training, validation and testing sets, 
    in parser set BASE_PATH_3D to data/singleVentricleData_norm
    ```
    python ./dataset/preprocessing/preprocessing_split.py
    ```
    Save resulting folder to /data/singleVentricleData_split. -->

