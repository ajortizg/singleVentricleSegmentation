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

### saving
* nifty header
* cutting/prolongation: variable tolerance for cutting, prolongation of 4d data and mask: so far both by nearest neighbor

### tests
* compare u,p on different levels, scaling correct?
* convergence in number of warpings?
* influence of smoothing the input images or using ROF-output as input
* check Moreau identity u = prox_{tau F}(u) + tau prox_{F^{*}/tau}(u/tau) = u
