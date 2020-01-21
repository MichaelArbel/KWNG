from __future__ import print_function

import torch
import argparse
import yaml

from trainer import Trainer
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark=False



def make_flags(args,config_file):
    if config_file:
        config = yaml.load(open(config_file))
        dic = vars(args)
        all(map(dic.pop, config))
        dic.update(config)
    return args

parser = argparse.ArgumentParser(description='KWNG')

parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--log_name', default = '',type= str,  help='log name')
parser.add_argument('--log_dir', default = '',type= str,  help='log directory for summaries and checkpoints')
parser.add_argument('--dataset', default = 'cifar10',type= str,  help='name of the dataset to use cifar10 or cifar100')
parser.add_argument('--data_dir', default = 'data',type= str,  help='directory to the dataset')
parser.add_argument('--log_in_file', action = 'store_true' ,  help='log output in a file')

parser.add_argument('--device', default = 0 ,type= int,  help='gpu device')
parser.add_argument('--seed', default = 0 ,type= int ,  help='seed for randomness')
parser.add_argument('--dtype',  default = '32' ,type= str ,   help='32 for float32 and 64 for float64')
parser.add_argument('--total_epochs', default=350, type=int, help='total number of epochs')

parser.add_argument('--network', default = 'ResNet18IllCond' ,type= str,  help='classifier network: Ill-conditioned case:ResNet18IllCond and well-conditioned case:ResNet18 ')
parser.add_argument('--num_classes', default = 10 ,type= int ,  help='number of classes')
parser.add_argument('--criterion', default = 'cross_entropy' ,type= str ,  help='top level loss')

# Optimizer parameters
parser.add_argument('--optimizer', default = 'sgd',type= str,  help='sgd')
parser.add_argument('--b_size', default = 128 ,type= int,  help='batch size')
parser.add_argument('--lr', default=.1, type=float, help='learning rate')
parser.add_argument('--momentum', default=0., type=float, help='momentum')
parser.add_argument('--weight_decay', default=0., type=float, help='weight decay')

parser.add_argument('--lr_decay',  default = 0.1 ,type= float ,  help='decay factor for lr')
parser.add_argument('--clip_grad', action = 'store_true',  help=' clip the gradient by norm ')

# Scheduler parameters 
parser.add_argument('--use_scheduler', default = 'store_true' ,  help='schedule the lr')
parser.add_argument('--scheduler',  default ='MultiStepLR' ,type= str ,  help=' scheduler ')
parser.add_argument('--milestone',  default = '100,200,300' ,type= str ,  help='decrease schedule for lr  ')

# estimator of the natural gradient
parser.add_argument('--estimator', default = 'KWNG',type= str,  help='proposed estimator')
parser.add_argument('--kernel', default = 'gaussian' ,type= str,  help=' the kernel used in the estimator  ')
parser.add_argument('--log_bandwidth', default = 0. ,type= float ,  help=' log bandwidth of the kernel ')
parser.add_argument('--epsilon', default = 1e-5 ,type= float, help=' Initial value for damping ')
parser.add_argument('--num_basis', default = 5 ,type= int ,  help='number of basis for KWNG ')

# Dumping parameters
parser.add_argument('--dumping_freq', default = 5 ,type= int ,  help=' update epsilon each dumping_freq iterations ')
parser.add_argument('--reduction_coeff', default = 0.85 ,type= float ,  help=' increase or descrease epsilon by  reduction_coeff factor')
parser.add_argument('--min_red', default = 0.25 ,type= float ,  help=' min threshold for reduction factor')
parser.add_argument('--max_red', default = 0.75 ,type= float ,  help=' max threshold for reduction factor')
parser.add_argument('--with_diag_mat', default=1, type=int,  help=' 1: Use the norm of the jacobian for non isotropic damping')

parser.add_argument('--config',  default ='' ,type= str ,  help='config file for the run ')
parser.add_argument('--with_sacred',  default =False ,type= bool ,  help=' disabled by default, can only work if sacred is installed')



args = parser.parse_args()
args = make_flags(args,args.config)
exp = Trainer(args)


train_acc,val_acc = exp.train()
test_acc = exp.test()
print('Training completed!')







