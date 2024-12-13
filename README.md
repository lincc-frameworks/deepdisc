# DeepDISC
Using deep learning for Detection, Instance Segmentation, and Classification on astronomical survey images.

*Reference Paper:* [Merz et al. 2023](https://academic.oup.com/mnras/article/526/1/1122/7273850)

*Corresponding Author:* 
[Grant Merz](gmerz3@illinois.edu), University of Illinois at Urbana-Champaign

*Contributors (in alphabetical order):* Patrick D. Aleo, Colin J. Burke, Yichen Liu, Xin Liu, Grant Merz, Anshul Shah, .

This is an updated repo of the original implementation (https://github.com/burke86/astro_rcnn)

## Description:

DeepDISC is a deep learning framework for efficiently performing source detection, classification, and segmnetation (deblending) on astronomical images.  We have built the code using detectron2 https://detectron2.readthedocs.io/en/latest/ for a modular design an access to state-of-the-art models. 

## Installation:

1) Create a conda environment.  We recommend using python 3.9.  You can use the environment.yml file provided and run  
   `conda env create -f environment.yml`  

or create an environment from scratch and install by hand the packages listed in the environment.yml file  

2) Install deepdisc with  
   `pip install deepdisc`  
   You can also install by cloning this repo and running `pip install [e].`  [e] is optional and will install in editable mode.  Use if you are going to change the source code  

Usage:
```
demo_decam.ipynb
```
This notebook demonstrates how to set up, train and evaluate a model using the detectron2 API. It requires the user to have downloaded the PhoSim simulated DECam data used in [Burke et al. 2019, MNRAS, 490 3952.](http://adsabs.harvard.edu/doi/10.1093/mnras/stz2845).   The data can be found here: [training set (1,000 images)](https://uofi.box.com/s/svlkblkh5o4a3q3qwu7iks6r21cmmu64) [validation set (250 images)](https://uofi.box.com/s/m22q747nawtxq8e5iihjulpapwlvucr5) [test set (50 images)](https://uofi.box.com/s/bmtkjrj9g832w9qybjd1yc4l6cyqx6cs).

```
demo_hsc.ipynb
```
This notebook follows a very similar procedure to ```demo_decam.ipynb```, but for real HSC data.  The ground truth object locations and masks are constructed following ```training_data.ipynb``` and classes are constructed with external catalog matching following ```hsc_class_assign.ipynb``` It is largely for demo purposes, so will not reflect the results of the paper.  The training scripts we used to recreate the paper results are in ```train_decam.py``` and ```train_hsc_primary.py```  


