## Table of contents

* [Introduction](#introduction)
* [Requirements](#requirements)
* [How to use](#how-to-use)
   * [Cifar10](#cifar10)
   * [Cifar100](#cifar100)
* [Resources](#resources)
   * [Data](#data)
   * [Hardware](#hardware)
* [Full documentation](#full-documentation)
* [Reference](#reference)
* [License](#license)

## Introduction

This repository contains an implementation of the Kernelized Wasserstein Natural Gradient estimator and provides scripts to reproduce the results of its [eponymous paper](https://arxiv.org/abs/1910.09652) published at ICLR 2020.


## Requirements


This a Pytorch implementation which requires the follwoing packages:

```
python==3.6.2 or newer
torch==1.2.0 or newer
torchvision==0.4.0 or newer
numpy==1.17.2  or newer
```

All dependencies can be installed using:

```
pip install -r requirements.txt
```




## How to use


### Cifar10
```
python train.py --device=-1 --config='configs/cifar10_KWNG.yml' 
```

### Cifar100

```
python train.py --device=-1 --config='configs/cifar100_KWNG.yml'
```




## Resources

### Data

To be able to reproduce the results of the paper on Cifar10 and Cifar100 using the prodivided scripts, both datasets need to be downloaded. This is automatically done by the script. By default a directory named 'data' containing both datasets is created in the working directory. 


### Hardware

To use a particular GPU, set —device=#gpu_id
To use GPU without specifying a particular one, set —device=-1
To use CPU set —device=-2


## Full documentation

```
--resume resume             from checkpoint [False]
--log_name                  log name ['']
--log_dir log directory for summaries and checkpoints ['']
--dataset                   name of the dataset to use cifar10 or cifar100 ['cifar10']
--data_dir                  directory to the dataset ['data']
--log_in_file               log output in a file [False]

--device                    gpu device [0]
--seed                      seed for randomness [0]
--dtype                     32 for float32 and 64 for float64 ['32']
--total_epochs              total number of epochs [350]

--network                   classifier network: [Ill-conditioned case: 'ResNet18IllCond',  well-conditioned case: 'ResNet18']
--num_classes               number of classes [10]
--criterion                 top level loss ['cross_entropy']

# Optimizer parameters
--optimizer                 Inner optimizer to compute the euclidean gradient['sgd']
--b_size                    batch size [128]
--lr                        learning rate [.1]
--momentum                  momentum [0.]
--weight_decay              weight decay [0.]

--lr_decay                  decay factor for lr [0.1]
--clip_grad                 clip the gradient by norm ['store_true']

# Scheduler parameters 
--use_scheduler             schedule the lr ['store_true']
--scheduler                 scheduler ['MultiStepLR']
--milestone                 help='decrease schedule for lr  ['100,200,300']

# estimator of the natural gradient
--estimator                 proposed estimator ['KWNG']
--kernel                    the kernel used in the estimator  ['gaussian']
--log_bandwidth             log bandwidth of the kernel [0.]
--epsilon                   Initial value for damping [1e-5]
--num_basis                 Number of basis for KWNG [5]

# Dumping parameters
--dumping_freq              update epsilon each dumping_freq iterations [5]
--reduction_coeff           increase or descrease epsilon by  reduction_coeff factor [0.85]
--min_red                   min threshold for reduction factor [0.25]
--max_red                   max threshold for reduction factor [0.75]
--with_diag_mat             Use the norm of the jacobian for non isotropic damping [1]

--configs                   config file for the run ['']
--with_sacred               disabled by default, can only work if sacred is installed [False]

```

## Reference

If using this code for research purposes, please cite:

[1] M. Arbel, A. Gretton, W. Li, G. Montufar [*Kernelized Wasserstein Natural Gradient*](https://arxiv.org/abs/1910.09652)

```
@article{Arbel:2018,
        author  = {Michael Arbel, Arthur Gretton, Wuchen Li, Guido Montufar},
        title   = {Kernelized Wasserstein Natural Gradient},
        journal = {ICLR},
        year    = {2020},
        url     = {https://arxiv.org/abs/1910.09652},
}                            }
```


## License 

This code is under a BSD license.
