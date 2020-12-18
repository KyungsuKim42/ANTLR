# ANTLR

This repository is the official implementation of [Unifying Activation- and Timing-based Learning Rules for Spiking Neural Networks](https://papers.nips.cc/paper/2020/file/e2e5096d574976e8f115a8f1e0ffb52b-Paper.pdf) (NeurIPS 2020)


## Requirements


### Install python packages :

with anaconda :
```setup
git clone https://github.com/KyungsuKim42/ANTLR.git
cd ANTLR
conda env create -f requirements.yml
conda activate antlr
```

### MNIST dataset

Will be automatically downloaded when needed.

### N-MNIST dataset


1. Download [N-MNIST dataset](https://www.garrickorchard.com/datasets/n-mnist) and put it in  ANTLR/dataset/N-MNIST
2. unzip `Train.zip`, `Test.zip`. The path should be `ANTLR/dataset/N-MNIST/Train` and `ANTLR/dataset/N-MNIST/Test`.
3. Run `python preprocess_nmnist.py` (This process may take a while.)




## Training

To train the model, run `main.py`. For example, to train the network with mnist dataset and learning rate of 0.0001, run following command.  

```train
python main.py --task <mnist or nmnist> --tag <tag for logging> --learning-rate 0.0001
```
Default values of each arguments are specified in `main.py`


## Evaluation

To evaluate the trained model on the test dataset, enable evaluation mode as follows:

```eval
python main.py --tag <tag of the model you want to evaluate> --eval-mode
```
