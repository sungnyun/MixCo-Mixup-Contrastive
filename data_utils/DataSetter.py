import torch
from torchvision import datasets
from torch.utils.data import Subset
import torchvision.transforms as transforms
import random
from .custom_transforms import *
from .Datasets import *

__all__ = ['data_loader']

        
# datasets
DataSet = {'cifar10': datasets.CIFAR10, 
            'cifar100': datasets.CIFAR100, 
            'tiny-imagenet': TinyImageNet,
            'imagenet': ImageNet}
# mean and std
mean = {'cifar10': [0.4914, 0.4822, 0.4465],
        'cifar100': [0.5071, 0.4867, 0.4408],
        'tiny-imagenet': [0.485, 0.456, 0.406],
        'imagenet': [0.485, 0.456, 0.406]}

std = {'cifar10': [0.2023, 0.1994, 0.2010],
        'cifar100': [0.2675, 0.2565, 0.2761],
        'tiny-imagenet': [0.229, 0.224, 0.225],
        'imagenet': [0.229, 0.224, 0.225]}
        
# image sizes
image_size = {'cifar10': 32, 'cifar100': 32, 'tiny-imagenet': 64, 'imagenet': 224}
image_num = {'cifar10': 50000, 'cifar100': 50000, 'tiny-imagenet': 100000, 'imagenet': None}



def data_loader(datasets, root, rep_augment=None, batch_size=128, 
                valid_ratio=0.1, pin_memory=False, num_workers=5, drop_last=True):
        
    if rep_augment == 'simclr':
        rep_augment = simclr_transform(mean[datasets], std[datasets], image_size[datasets], 1)
        
    if rep_augment is None:
        train_transforms = transforms.Compose([
            transforms.RandomCrop(image_size[datasets], 
                                  padding=int(image_size[datasets]/8)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean[datasets], std=std[datasets])])
        
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean[datasets], std=std[datasets])])
    else:
        train_transforms = RepLearnTransform(rep_augment)
        test_transforms = RepLearnTransform(rep_augment)

    if datasets == 'imagenet':
        transforms = {'pretrain': RepLearnTransform(simclr_transform(mean[datasets], 
                                                                     std[datasets], 
                                                                     image_size[datasets], 1)), 
                      'train': transforms.Compose([transforms.RandomCrop(image_size[datasets], 
                                                                          padding=int(image_size[datasets]/8)),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean[datasets], std[datasets])]),
                      'valid': transforms.Compose([transforms.CenterCrop(image_size[datasets] * 0.875),
                                                   transforms.Resize(image_size[datasets]),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(mean[datasets], std[datasets])])
                      'test':  transforms.Compose([transforms.CenterCrop(image_size[datasets] * 0.875),
                                                   transforms.Resize(image_size[datasets]),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(mean[datasets], std[datasets])])
        dataset_paths = {'train': os.path.join(root, 'train'),
                         'test': os.path.join(root, 'val')}
        dataloaders, dataset_sizes = imagenet_dataloader(dataset_paths, transforms, batch_size, pin_memory, num_workers)

        return dataloaders, dataset_sizes
    
    else:
        valid_transforms = test_transforms if rep_augment is None else train_transforms

        train_set = DataSet[datasets](root, train=True, transform=train_transforms, download=True)
        valid_set = DataSet[datasets](root, train=True, transform=valid_transforms, download=True)

        train_list, valid_list = _valid_sampler(datasets, valid_ratio)

        train_set = Subset(train_set, train_list)
        valid_set = Subset(valid_set, valid_list)
        test_set = DataSet[datasets](root, train=False, transform=test_transforms, download=True)

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=pin_memory, 
                                                   num_workers=num_workers, drop_last=drop_last)
        valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, 
                                                   num_workers=num_workers, drop_last=drop_last)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, 
                                                   num_workers=num_workers, drop_last=drop_last)

        dataloaders = {'train' : train_loader, 'valid' : valid_loader, 'test' : test_loader}
        dataset_sizes = {'train': len(train_set), 'valid' : len(valid_set), 'test' : len(test_set)}

        return dataloaders, dataset_sizes


def _valid_sampler(datasets, valid_ratio):
    random.seed(0)

    valid_size = int(image_num[datasets] * valid_ratio)
    valid_list = random.sample(range(0, image_num[datasets]), valid_size)

    train_list = [x for x in range(image_num[datasets])]
    train_list = list(set(train_list)-set(valid_list))

    return train_list, valid_list
