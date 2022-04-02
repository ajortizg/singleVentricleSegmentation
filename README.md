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

to install setup file:
* pip install . 

to use config file link data and result folder: 

on linux:
* ln -s /home/.../singleVentricleData/ /home/.../singleVentricleSegmentation/
* ln -s /home/.../results /home/.../singleVentricleSegmentation/

on windows:
* New-Item -ItemType SymbolicLink -Target "C:\Users\...\data\" -Path "C:\Users\...\singleVentricleSegmentation\data"
* New-Item -ItemType SymbolicLink -Target "C:\Users\...\results\" -Path "C:\Users\...\singleVentricleSegmentation\results"


## TODOs

* distinguish cases where diastole > systole or vice-versa

### Regularizer
* directional TV

### preprocessing
* cut out heart region (based on the two segmentations), save shifts to recover the object in the original files
* refine in z-direction

### tests
* compare u,p on different levels, scaling correct?
* convergence in number of warpings?
* influence of smoothing the input images or using ROF-output as input
* check Moreau identity u = prox_{tau F}(u) + tau prox_{F^{*}/tau}(u/tau) = u