
Code to reproduce results of Kernelized Wasserstein Natural Gradient: https://openreview.net/forum?id=Hklz71rYvS&noteId=Hklz71rYvS



Requirements:

python=3.6.2
torch=1.2.0
torchvision=0.4.0
tensorboardX=1.8


To reproduce the results:

For Cifar10:

python train.py --device=-1 --config='configs/cifar10_KWNG.yml' 

For Cifar100:

python train.py --device=-1 --config='configs/cifar100_KWNG.yml' 



To use a particular GPU, set —device=#gpu_id
To use GPU without specifying a particular one, set —device=-1
To use CPU set —device=-2


