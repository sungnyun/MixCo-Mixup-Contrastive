import torch
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import Subset
import torchvision.transforms as transforms
import random, os
from .utils import *
from .TinyImageNet import TinyImageNet

__all__ = ['cifar_10_setter', 'cifar_100_setter', 'imagenet_setter', 'tiny_imagenet_setter']

def cifar_10_setter(batch_size=128, valid_size=5000, pin_memory=False, num_workers=4, add_transforms=None, \
                    root='./data/cifar10', download=True, fixed_valid=True, drop_last=False):
    if fixed_valid:
        random.seed(10000)
    # Data augmentation and normalization for training
    # Just normalization for validation
    mean = [0.4914, 0.4822, 0.4465]
    stdv = [0.2023, 0.1994, 0.2010]
    
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv),
    ])
    
    if add_transforms is None:
        base_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()])
        
        train_transforms = join_transform(train_transforms, base_transforms)
    
    else:
        train_transforms = join_transform(train_transforms, add_transforms)
        
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv),
    ])
    
    batch_size = batch_size
    valid_size = valid_size
    
    # Datasets
    cifar10_train_set = datasets.CIFAR10(root, train=True, transform=train_transforms, download=download) # train transform applied
    cifar10_valid_set = datasets.CIFAR10(root, train=True, transform=test_transforms, download=download) # test transform applied
    valid_list = random.sample(range(0, len(cifar10_train_set)), valid_size)
    train_list = [x for x in range(len(cifar10_valid_set))]
    train_list = list(set(train_list)-set(valid_list))

    cifar10_train_set = Subset(cifar10_train_set, train_list)
    cifar10_valid_set = Subset(cifar10_valid_set, valid_list)
    cifar10_test_set = datasets.CIFAR10(root, train=False, transform=test_transforms, download=download)

    train_loader = torch.utils.data.DataLoader(cifar10_train_set, batch_size=batch_size, shuffle=True, \
                                               pin_memory=pin_memory, num_workers=num_workers, drop_last=drop_last)
    valid_loader = torch.utils.data.DataLoader(cifar10_valid_set, batch_size=batch_size, \
                                               pin_memory=pin_memory, num_workers=num_workers, drop_last=drop_last)
    test_loader = torch.utils.data.DataLoader(cifar10_test_set, batch_size=batch_size, \
                                              pin_memory=pin_memory, num_workers=num_workers, drop_last=drop_last)

    dataloaders = {'train' : train_loader,
                   'valid' : valid_loader,
                   'test' : test_loader,}

    dataset_sizes = {'train': len(cifar10_train_set), 'valid' : len(cifar10_valid_set), 'test' : len(cifar10_test_set)}
    
    return dataloaders, dataset_sizes


def cifar_100_setter(batch_size=128, valid_size=5000, pin_memory=False, num_workers=4, add_transforms=None, \
                     root='./data/cifar100', download=True, fixed_valid=True, drop_last=False):
    if fixed_valid:
        random.seed(10000)
    # Data augmentation and normalization for training
    # Just normalization for validation
    mean = [0.4914, 0.4822, 0.4465]
    stdv = [0.2023, 0.1994, 0.2010]
    
    if add_transforms is None:
        base_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()])
        
        train_transforms = join_transform(train_transforms, base_transforms)
    
    else:
        train_transforms = join_transform(train_transforms, add_transforms)
        
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv),
    ])
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv),
    ])
    
    batch_size = batch_size
    valid_size = valid_size
    
    # Datasets
    cifar100_train_set = datasets.CIFAR100(root, train=True, transform=train_transforms, download=download)
    cifar100_valid_set = datasets.CIFAR100(root, train=True, transform=test_transforms, download=download)

    valid_list = random.sample(range(0, len(cifar100_train_set)), valid_size)
    train_list = [x for x in range(len(cifar100_valid_set))]
    train_list = list(set(train_list)-set(valid_list))

    cifar100_train_set = Subset(cifar100_train_set, train_list)
    cifar100_valid_set = Subset(cifar100_valid_set, valid_list)
    cifar100_test_set = datasets.CIFAR100(root, train=False, transform=test_transforms, download=download)

    train_loader = torch.utils.data.DataLoader(cifar100_train_set, batch_size=batch_size, \
                                               shuffle=True, pin_memory=pin_memory, num_workers=num_workers, drop_last=drop_last)
    valid_loader = torch.utils.data.DataLoader(cifar100_valid_set, batch_size=batch_size, \
                                               pin_memory=pin_memory, num_workers=num_workers, drop_last=drop_last)
    test_loader = torch.utils.data.DataLoader(cifar100_test_set, batch_size=batch_size, \
                                              pin_memory=pin_memory, num_workers=num_workers, drop_last=drop_last)

    dataloaders = {'train' : train_loader,
                   'valid' : valid_loader,
                   'test' : test_loader,}

    dataset_sizes = {'train': len(cifar100_train_set), 'valid' : len(cifar100_valid_set), 'test' : len(cifar100_test_set)}
    
    return dataloaders, dataset_sizes


def tiny_imagenet_setter(batch_size=128, valid_size=10000, pin_memory=False, num_workers=6, add_transforms=None, \
                         root='./data/tiny-imagenet', fixed_valid=True, drop_last=False):    
    if fixed_valid:
        random.seed(10000)
    # Data augmentation and normalization for training
    # Just normalization for validation
    mean = [0.485, 0.456, 0.406]
    stdv = [0.229, 0.224, 0.225]
    
    if add_transforms is None:
        base_transform = transforms.Compose([
            transforms.RandomCrop(64, padding=8),
            transforms.RandomHorizontalFlip()])
        
        train_transforms = join_transform(train_transforms, base_transforms)
    
    else:
        train_transforms = join_transform(train_transforms, add_transforms)
        
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv),
    ])

    train_transforms = transforms.Compose(train_transform_list)
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv),
    ])
    
    batch_size = batch_size
    valid_size = valid_size
    
    # Datasets
    tiny_train_set = TinyImageNet(root, split='train', transform=train_transforms)
    tiny_valid_set = TinyImageNet(root, split='train', transform=test_transforms)
    
    valid_list = random.sample(range(0, len(tiny_train_set)), valid_size)
    train_list = [x for x in range(len(tiny_valid_set))]
    train_list = list(set(train_list)-set(valid_list))

    tiny_train_set = Subset(tiny_train_set, train_list)
    tiny_valid_set = Subset(tiny_valid_set, valid_list)
    tiny_test_set = TinyImageNet(root, split='val', transform=test_transforms)

    train_loader = torch.utils.data.DataLoader(tiny_train_set, batch_size=batch_size, \
                                               shuffle=True, pin_memory=pin_memory, num_workers=num_workers, drop_last=drop_last)
    valid_loader = torch.utils.data.DataLoader(tiny_valid_set, batch_size=batch_size, \
                                               pin_memory=pin_memory, num_workers=num_workers, drop_last=drop_last)
    test_loader = torch.utils.data.DataLoader(tiny_test_set, batch_size=batch_size, \
                                              pin_memory=pin_memory, num_workers=num_workers, drop_last=drop_last)

    dataloaders = {'train' : train_loader,
                   'valid' : valid_loader,
                   'test' : test_loader,}

    dataset_sizes = {'train': len(tiny_train_set), 'valid' : len(tiny_valid_set), 'test' : len(tiny_test_set)}
    
    return dataloaders, dataset_sizes


def imagenet_setter(batch_size=128, valid_size=0, pin_memory=False, num_workers=4, \
                    root='./data/imagenet', fixed_valid=True, drop_last=False):
    """
    Only train and valid set exists for imagenet dataset.
    train/test transfromations are following conventions.
    Imagenet Dataset has same images for Classification Task (2012-2017)
    The valid images are from ImageNet2012. 
    """
    train_dir = os.path.join(root, 'train')
    test_dir = os.path.join(root, 'val')
        
    mean = [0.485, 0.456, 0.406]
    stdv = [0.229, 0.224, 0.225] 

    # Standard train transformation
    train_transform_list = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv)
        ]
    
    test_transform_list = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv),        
    ]

    train_transforms = transforms.Compose(train_transform_list)
    test_transforms = transforms.Compose(test_transform_list)
    
    batch_size = batch_size
    
    # Datasets
    imagenet_train_set = datasets.ImageFolder(train_dir, transform=train_transforms) 
    imagenet_test_set = datasets.ImageFolder(test_dir, transform=test_transforms)

    if valid_size > 0:
        imagenet_valid_set = datasets.ImageFolder(train_dir, transform=test_transforms)
        valid_list = random.sample(range(0, len(imagenet_train_set)), valid_size)
        train_list = [x for x in range(len(imagenet_valid_set))]
        train_list = list(set(train_list) - set(valid_list))

        imagenet_train_set = Subset(imagenet_train_set, train_list)
        imagenet_valid_set = Subset(imagenet_valid_set, valid_list)

        valid_loader = torch.utils.data.DataLoader(imagenet_valid_set, batch_size=batch_size, \
                                                   pin_memory=pin_memory, num_workers=num_workers, drop_last=drop_last)

    train_loader = torch.utils.data.DataLoader(imagenet_train_set, batch_size=batch_size, \
                                               shuffle=True, pin_memory=pin_memory, num_workers=num_workers, drop_last=drop_last)
    test_loader = torch.utils.data.DataLoader(imagenet_test_set, batch_size=100, shuffle=False, \
                                              pin_memory=pin_memory, num_workers=num_workers, drop_last=drop_last)

    dataloaders = {'train' : train_loader, 'test': test_loader}
    dataset_sizes = {'train': len(imagenet_train_set), 'test': len(imagenet_test_set)}
    if valid_size > 0:
        dataloaders['valid'] = valid_loader
        dataset_sizes['valid'] = len(imagenet_valid_set)    
    
    return dataloaders, dataset_sizes
