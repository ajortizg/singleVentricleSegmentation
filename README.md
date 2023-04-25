# singleVentricleSegmentation

## install
* conda create -n singleVentricleSegmentationEnv
* conda install python=3.9
* conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
* conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia (for CUDA 11.6)
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

