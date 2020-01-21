from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import os, sys
from tensorboardX import SummaryWriter

import time
import numpy as np
import pprint
import socket
import pickle

from resnet import *
from kwng import *
from gaussian import *
from data_loader import *

class Trainer(object):
	def __init__(self,args):
		torch.manual_seed(args.seed)
		self.args = args
		self.device = assign_device(args.device)	
		self.log_dir = make_log_dir(args)
		
		if args.log_in_file:
			self.log_file = open(os.path.join(self.log_dir, 'log.txt'), 'w', buffering=1)
			sys.stdout = self.log_file
			sys.stderr = self.log_file
		print("Process id: " + str(os.getpid()) + " | hostname: " + socket.gethostname())
		pp = pprint.PrettyPrinter(indent=4)
		pp.pprint(vars(args))

		print('Creating writer')
		self.writer = SummaryWriter(self.log_dir)

		print('Loading data')
		self.data_loaders = get_data_loader(args)
		self.total_epochs = self.args.total_epochs
		print('==> Building model..')
		self.build_model()


	def build_model(self):
		self.net = get_network(self.args)
		self.net = self.net.to(self.device)
		if self.args.dtype=='64':
			self.net = self.net.double()
		if self.device == 'cuda':
			self.net = torch.nn.DataParallel(self.net)
			cudnn.benchmark = True
		self.init_train_values()
		self.criterion = get_criterion(self.args)
		self.optimizer = get_optimizer(self.args,self.net.parameters(),self.net)
		self.scheduler = get_scheduler(self.args,self.optimizer)
		self.wrapped_optimizer = get_wrapped_optimizer(self.args,self.optimizer,self.criterion,self.net, device=self.device)
	

	def train(self):

		print(' Starting training')

		self.init_train_values()

		for epoch in range(self.start_epoch, self.start_epoch+self.total_epochs):
			
			train_acc =  self.epoch_pass(epoch,'train')
			val_acc =  self.epoch_pass(epoch,'val')
			if self.args.use_scheduler:
				self.scheduler.step()

		return train_acc,val_acc
	def test(self):
		print('Starting test')
		test_acc =  self.epoch_pass(0,'test')
		return test_acc

	def init_train_values(self):
		if self.args.resume:
			# Load checkpoint.
			print('==> Resuming from checkpoint..')
			assert os.path.isdir(self.log_dir+'/checkpoint'), 'Error: no checkpoint directory found!'
			checkpoint = torch.load(self.log_dir+'/checkpoint/ckpt.t7')
			self.net.load_state_dict(checkpoint['net'])
			self.best_acc = checkpoint['acc']
			self.best_loss = checkpoint['loss']
			self.start_epoch = checkpoint['epoch']
			self.total_iters = checkpoint['total_iters']
		else:
			self.best_acc = 0  # best test accuracy
			self.start_epoch = 0  # start from epoch 0 or last checkpoint epoch
			self.total_iters = 0
			self.best_loss = torch.tensor(np.inf)

	def epoch_pass(self,epoch,phase):
		print('Epoch: '+ str(epoch) + ' | ' + phase + ' phase')
		if phase == 'train':
			self.net.train(True)  # Set model to training mode
		else:
			self.net.train(False)  # Set model to evaluate mode

		self.net.train()
		loss = 0
		correct = 0
		total = 0
		counts = 0
		for batch_idx, (inputs, targets) in enumerate(self.data_loaders[phase]):
			tic = time.time()
			inputs, targets = inputs.to(self.device), targets.to(self.device)
			if self.args.dtype=='64':
				inputs=inputs.double()
			if phase=="train":
				self.total_iters+=1

				loss_step, predicted = self.wrapped_optimizer.step(inputs,targets)
			loss_step, predicted = self.wrapped_optimizer.eval(inputs,targets)
			loss += loss_step
			running_loss = loss/(batch_idx+1)
			total += targets.size(0)
			correct += predicted.eq(targets).sum().item()
			acc= 100.*correct/total
			if phase=="train":
				self.writer.add_scalars('data/train_loss_step',{"loss_step":loss_step,"loss_averaged":running_loss},self.total_iters)
			toc = time.time()
			print(' Loss: ' + str(round(running_loss,3))+ ' | Acc: '+ str(acc) + '  ' +'('+str(correct) +'/'+str(total)+')' + ' time: ' + str(toc-tic) + ' iter: '+ str(batch_idx))
			counts += 1

		self.writer.add_scalars('data/total_stats_'+phase, {"loss":loss/(batch_idx+1), "correct":acc},epoch)

		# Save checkpoint.
		if phase == 'val':
			avg_loss = loss/(batch_idx+1)
			if avg_loss < self.best_loss:
				save_checkpoint(self.writer.logdir,acc,avg_loss,epoch,self.total_iters,self.wrapped_optimizer.net)
				self.best_loss = avg_loss

		return acc

def save_checkpoint(checkpoint_dir,acc,loss,epoch,total_iters,net):

	print('Saving..')
	state = {
		'net': net.state_dict(),
		'acc': acc,
		'loss':loss,
		'epoch': epoch,
		'total_iters':total_iters,
	}
	if not os.path.isdir(checkpoint_dir +'/checkpoint'):
		os.mkdir(checkpoint_dir + '/checkpoint')
	torch.save(state,checkpoint_dir +'/checkpoint/ckpt.t7')

def assign_device(device):
	if device >-1:
		device = 'cuda:'+str(device) if torch.cuda.is_available() and device>-1 else 'cpu'
	elif device==-1:
		device = 'cuda'
	elif device==-2:
		device = 'cpu'
	return device
def make_log_dir(args):
	if args.with_sacred:
		log_dir = args.log_dir + '_' + args.log_name
	else:
		log_dir = os.path.join(args.log_dir,args.log_name)
	if not os.path.isdir(log_dir):
		os.mkdir(log_dir)
	return log_dir

def get_dtype(args):
	if args.dtype=='32':
		return torch.float32
	elif args.dtype=='64':
		return torch.float64


def get_network(args):
	if args.network=='ResNet18':
		return ResNet18(num_classes = args.num_classes)
	elif args.network=='ResNet18IllCond':
		return ResNet18IllCond(num_classes = args.num_classes)
def get_kernel(args,device = 'cuda'):
		dtype = get_dtype(args)
		if args.kernel=='gaussian':
			return  Gaussian(1,args.log_bandwidth,dtype=dtype, device = device)

def get_wrapped_optimizer(args,optimizer,criterion,net,device = 'cuda'):
	if args.estimator=='EuclideanGradient':
		return  OptimizerWrapper(optimizer,criterion,net,args.clip_grad)
	elif args.estimator=='KWNG':
		kernel  = get_kernel(args, device=device)
		estimator = KWNG(kernel,eps=args.epsilon, num_basis = args.num_basis,with_diag_mat = args.with_diag_mat)
		return KWNGWrapper(optimizer,criterion,net,args.clip_grad,estimator,args.dumping_freq,args.reduction_coeff,args.min_red,args.max_red)

def get_data_loader(args):
	if args.dataset=='cifar10':
		args.num_classes = 10
		return CIFARLoader(args.data_dir,args.b_size)
	elif args.dataset=='cifar100':
		args.num_classes = 100
		return CIFAR100Loader(args.data_dir,args.b_size)

def get_optimizer(args,params,net):
	if args.optimizer=='sgd':
		return optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

def get_scheduler(args,optimizer):
	if args.scheduler=='MultiStepLR':
		if args.milestone is None:
			lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.total_epochs*0.5), int(args.total_epochs*0.75)], gamma=args.lr_decay)
		else:
			milestone = [int(_) for _ in args.milestone.split(',')]
			lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestone, gamma=args.lr_decay)
		return lr_scheduler
def get_criterion(args):
	if args.criterion=='cross_entropy':
		return nn.CrossEntropyLoss()




