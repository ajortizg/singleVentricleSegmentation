# singleVentricleSegmentation

## install
* conda create -n singleVentricleSegmentationEnv
* conda install python=3.9
* conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
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


## preprocessing 
always use /parser/configPreprocessing.ini

1. possibly flip the original data
    ```
    python ./dataset/preprocessing/preprocessing_flipping.py
    ```
    save resulting folder to /data/singleVentricleData_flip

2. cut the (possibly flipped) original data
    in parser set BASE_PATH_3D to data/singleVentricleData_flip
    ```
    python ./dataset/preprocessing/preprocessing_cutting.py
    ```
    save resulting folder to /data/singleVentricleData_cut

3. resize the (flipped and cutted) data
    in parser set BASE_PATH_3D to data/singleVentricleData_cut
    ```
    python ./dataset/preprocessing/preprocessing_prolongation.py
    ```
    Save resulting folder to /data/singleVentricleData_prolong.
    Note that this depends on the Interpolationtype and Bondarytype.

4. split the dataset for training the CNN in validation and training sets, 
    in parser set BASE_PATH_3D to data/singleVentricleData_prolongLinear
    ```
    python ./dataset/preprocessing/preprocessing_split.py
    ```
    Save resulting folder to /data/singleVentricleData_prolongLinear_split.

5. Data augmentation only for training the CNN.
    in parset set BASE_PATH_3D to /data/singleVentricleData_prolongLinear_split
    ```
    python ./dataset/preprocessing/preprocessing_data_augmentation.py
    ```
    Save resulting folder to /data/singleVentricleData_prolongLinear_split_xN,
    where N is the number of new patients created per patient.

## TODOs

### saving
* nifty header
* cutting/prolongation: variable tolerance for cutting, prolongation of 4d data and mask: so far both by nearest neighbor

### tests
* compare u,p on different levels, scaling correct?
* convergence in number of warpings?
* influence of smoothing the input images or using ROF-output as input
* check Moreau identity u = prox_{tau F}(u) + tau prox_{F^{*}/tau}(u/tau) = u
