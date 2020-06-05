import torch
from torchvision import datasets
from torch.utils.data import Subset
import torchvision.transforms as transforms
import random
from .custom_transforms import *
from .Datasets import *

__all__ = ['DataSetter']


class DataSetter():
    def __init__(self):
        
        # datasets
        self.datasets = {'cifar10': datasets.CIFAR10, 
                         'cifar100': datasets.CIFAR100, 
                         'tiny-imagenet': TinyImageNet}
        # mean and std
        self.mean = {'cifar10': [0.4914, 0.4822, 0.4465],
                     'cifar100': [0.5071, 0.4867, 0.4408],
                     'tiny-imagenet': [0.485, 0.456, 0.406]}
        self.std = {'cifar10': [0.2023, 0.1994, 0.2010],
                    'cifar100': [0.2675, 0.2565, 0.2761],
                    'tiny-imagenet': [0.229, 0.224, 0.225]}
        
        # image sizes
        self.image_size = {'cifar10': 32, 'cifar100': 32, 'tiny-imagenet': 64}
        self.image_num = {'cifar10': 50000, 'cifar100': 50000, 'tiny-imagenet': 100000}
        
        
    def data_loader(self, datasets, root, rep_augment=None, batch_size=128, 
                    valid_ratio=0.1, pin_memory=False, num_workers=5, drop_last=True):
        if rep_augment is None:
            train_transforms = transforms.Compose([
                transforms.RandomCrop(self.image_size[datasets], 
                                      padding=int(self.image_size[datasets]/8)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean[datasets], std=self.std[datasets])])

        elif rep_augment == 'simclr':
            rep_augment = simclr_transform(self.image_size[datasets], 1)
            train_transforms = RepLearnTransform(rep_augment)
            
        else:
            train_transforms = RepLearnTransform(rep_augment)
            
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean[datasets], std=self.std[datasets])])        
        
        valid_transforms = test_transforms if rep_augment is None else train_transforms
        
        train_set = self.datasets[datasets](root, train=True, transform=train_transforms, download=True)
        valid_set = self.datasets[datasets](root, train=True, transform=valid_transforms, download=True)
        
        train_list, valid_list = self._valid_sampler(datasets, valid_ratio)
        
        train_set = Subset(train_set, train_list)
        valid_set = Subset(valid_set, valid_list)
        test_set = self.datasets[datasets](root, train=False, transform=test_transforms, download=True)
        
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=pin_memory, 
                                                   num_workers=num_workers, drop_last=drop_last)
        valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, 
                                                   num_workers=num_workers, drop_last=drop_last)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, 
                                                   num_workers=num_workers, drop_last=drop_last)
        
        dataloaders = {'train' : train_loader, 'valid' : valid_loader, 'test' : test_loader,}
        dataset_sizes = {'train': len(train_set), 'valid' : len(valid_set), 'test' : len(test_set)}
        
        return dataloaders, dataset_sizes
    
    
    def _valid_sampler(self, datasets, valid_ratio):
        random.seed(0)

        valid_size = int(self.image_num[datasets] * valid_ratio)
        valid_list = random.sample(range(0, self.image_num[datasets]), valid_size)
        
        train_list = [x for x in range(self.image_num[datasets])]
        train_list = list(set(train_list)-set(valid_list))
        
        return train_list, valid_list
