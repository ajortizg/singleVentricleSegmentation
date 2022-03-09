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

to install setup file:
* pip install . 

to use config file link data and result folder: 
* ln -s /home/.../singleVentricleData/ /home/.../singleVentricleSegmentation/
* ln -s /home/.../results /home/.../singleVentricleSegmentation/
