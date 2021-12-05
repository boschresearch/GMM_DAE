# Shape your Space: A Gaussian Mixture Regularization Approach to Deterministic Autoencoders PyTorch 

PyTorch implementation of the NeurIPS 2021 paper "Shape your Space: A Gaussian Mixture Regularization Approach to Deterministic Autoencoders". The paper can be found 
[here](https://proceedings.neurips.cc/paper/2021/hash/3c057cb2b41f22c0e740974d7a428918-Abstract.html). The code allows the users to
reproduce and extend the results reported in the paper. Please cite the
above paper when reporting, reproducing or extending the results.

## Purpose of the project

This software is a research prototype, solely developed for and published as
part of the publication. It will neither be
maintained nor monitored in any way.

## Setup.

1. Create a conda virtual environment
2. Clone the repository
3. Activate the environment and run 
 ```bash
cd GMM_DAE
pip install requirements.txt
```
## Dataset

The provided implementation is tested on MNIST, FASHION MNIST, SVHN and CELEBA images. 

### MNIST, FASHION MNIST
Resize the original images to size 32x32. Provide the processed images path in the data_dir, line 4 in config.ini for MNIST and line 22 for FASHION MNIST.

### SVHN
Download the SVHN cropped images of size 32x32 and provide the path to the images in line 40 in config.ini

### CELEBA
Download the images and pre-process the image as follows. Center crop the images to size 140x140 and then resize to 64x64. Provide the dataset path to the line 59 in config.ini.
  
### Usage

To run the code clone the repository and then run

```bash
python train.py <DATASETNAME> eg: MNIST, FASHIONMNIST, SVHN or CELEB
```
For FID computation we used the github repo [pytorch-fid](https://github.com/mseitzer/pytorch-fid)
## License

GMM_DAE is open-sourced under the AGPL-3.0 license. See the
[LICENSE](LICENSE) file for details.

