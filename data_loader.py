import torch
from torchvision import datasets
import torchvision
import torchvision.transforms as transforms
import os


def CIFARLoader( data_path, train_batch = 128, test_batch = 100):

	data_path = os.path.join(data_path,'CIFAR10')

	transform_train = transforms.Compose([
	    transforms.RandomCrop(32, padding=4),
	    transforms.RandomHorizontalFlip(),
	    transforms.ToTensor(),
	    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])

	transform_test = transforms.Compose([
	    transforms.ToTensor(),
	    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])

	trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_train)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch, shuffle=True, num_workers=2)

	testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_test)
	testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

	data_loaders = {"train": trainloader, "val": testloader, "test":testloader}
	return data_loaders




def CIFAR100Loader( data_path, train_batch = 128, test_batch = 100):
	data_path = os.path.join(data_path,'CIFAR100')

	transform_train = transforms.Compose([
	    transforms.RandomCrop(32, padding=4),
	    transforms.RandomHorizontalFlip(),
	    transforms.ToTensor(),
	    transforms.Normalize(mean=[n/255.
                        for n in [129.3, 124.1, 112.4]], std=[n/255. for n in [68.2,  65.4,  70.4]]),
	])

	transform_test = transforms.Compose([
	    transforms.ToTensor(),
	    transforms.Normalize(mean=[n/255.
                        for n in [129.3, 124.1, 112.4]], std=[n/255. for n in [68.2,  65.4,  70.4]]),
	])

	trainset = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True, transform=transform_train)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch, shuffle=True, num_workers=2)

	testset = torchvision.datasets.CIFAR100(root=data_path, train=False, download=True, transform=transform_test)
	testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

	data_loaders = {"train": trainloader, "val": testloader, "test":testloader}
	return data_loaders


