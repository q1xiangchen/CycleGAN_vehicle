# Image-to-Image translation from Vehicle-X to VeRi
## Overview
This repository contains the implementation code for performing image-to-image translation between the Vehicle-X and VeRi datasets using [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). Our approach aims to transfer the style of images from Vehicle-X to match the appearance of vehicles in the VeRi dataset.
## Getting started
### Prerequisites
- Before you begin, ensure you have the following requirements installed: Python 3.x, pip3.
- To install the required libraries, execute the command in terminal:
    ```
    pip3 install -r requirements.txt
    ```
### Dataset Preparation
- To download the same dataset, please access: [Vehicle-X](https://github.com/yorkeyao/VehicleX) and [VeRi](https://github.com/JDAI-CV/VeRidataset).
- Extract the datasets and place them into a `data` folder within the project directory.

### Training the Model
- To start training the CycleGAN model, use the command below. Additional arguments can be appended as needed:
    ```
    python cyclegan.py
    ```

- To view training progress, run 
    `tensorboard --logdir=ckpt/cyclegan` and click the URL http://localhost:6006/
- Training checkpoints will be saved in the `ckpt/cyclegan`directory. 
- During training, sample images will be stored in the `sample_images_while_training` folder for visual progress inspection.

### Test the Model
 Follow the common practice, we use the test set of VehicleX as the source domain, and the test set of VeRi as the target domain. 
 - To generate fake VeRi images (50 samples by default), run:
    ```
    python cyclegan_test.py
    ```

- To calculate the Fréchet Inception Distance (FID) score between real and generated images, execute::

    ```
    $ python -m pytorch_fid [path/to/real_images] [path/to fake_images]
    ```
    Replace `path/to/real_images` and `path/to/fake_images` with the actual paths to your datasets.

## Supplementary Files
Supplementary materials and additional resources are provided in the `supp` folder.