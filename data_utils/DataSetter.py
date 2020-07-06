import torch, os
from torchvision import datasets
from torch.utils.data import Subset
import torchvision.transforms as transforms
import random
from .custom_transforms import *
from .Datasets import *
from torch.utils.data.distributed import DistributedSampler

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



def data_loader(datasets, root, rep_augment=None, batch_size=128, valid_size=5000, 
                pin_memory=False, num_workers=5, drop_last=True, distributed=False, num_replicas=None, rank=None):
        
    if rep_augment == 'simclr':
        rep_augment = simclr_transform(mean[datasets], std[datasets], image_size[datasets], 1)
        train_transforms = RepLearnTransform(rep_augment)
        test_transforms = RepLearnTransform(rep_augment)
        
    if rep_augment == 'moco':
        rep_augment = moco_transform(mean[datasets], std[datasets], image_size[datasets])
        train_transforms = RepLearnTransform(rep_augment)
        test_transforms = RepLearnTransform(rep_augment)
        
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

    if datasets == 'imagenet':
        transforms_dict = {'pretrain': RepLearnTransform(simclr_transform(mean[datasets], 
                                                                     std[datasets], 
                                                                     image_size[datasets], 1)), 
                      'train': transforms.Compose([transforms.RandomResizedCrop(image_size[datasets]),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean[datasets], std[datasets])]),
                      'valid': transforms.Compose([transforms.Resize(256),
                                                   transforms.CenterCrop(image_size[datasets]),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(mean[datasets], std[datasets])]),
                      'test':  transforms.Compose([transforms.Resize(256),
                                                   transforms.CenterCrop(image_size[datasets]),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(mean[datasets], std[datasets])])}
        dataset_paths = {'train': os.path.join(root, 'train'),
                         'test': os.path.join(root, 'val')}
        
        dataloaders, dataset_sizes = imagenet_dataloader(dataset_paths, transforms_dict, batch_size, 
                                                         pin_memory, num_workers, drop_last, distributed, num_replicas, rank)

        return dataloaders, dataset_sizes
    
    else:
        train_set = DataSet[datasets](root, train=True, transform=train_transforms, download=True)
        valid_set = DataSet[datasets](root, train=True, transform=test_transforms, download=True)

        train_list, valid_list = _valid_sampler(datasets, valid_size)

        probe_train_set = Subset(train_set, train_list)
        probe_valid_set = Subset(valid_set, valid_list)
        test_set = DataSet[datasets](root, train=False, transform=test_transforms, download=True)
        
        if distributed:
            train_sampler = DistributedSampler(train_set, num_replicas, rank)
        else:
            train_sampler = None
        
        pretrain_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=(train_sampler is None), 
                                                   sampler=train_sampler, pin_memory=pin_memory, 
                                                   num_workers=num_workers, drop_last=drop_last)
        train_loader = torch.utils.data.DataLoader(probe_train_set, batch_size=batch_size, shuffle=(train_sampler is None), 
                                                   sampler=train_sampler, pin_memory=pin_memory, 
                                                   num_workers=num_workers, drop_last=drop_last)
        valid_loader = torch.utils.data.DataLoader(probe_valid_set, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, 
                                                   num_workers=num_workers, drop_last=drop_last)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, 
                                                   num_workers=num_workers, drop_last=drop_last)

        dataloaders = {'pretrain': pretrain_loader, 'train': train_loader, 'valid' : valid_loader, 'test' : test_loader}
        dataset_sizes = {'pretrain': len(train_set), 'train': len(probe_train_set), 'valid' : len(probe_valid_set), 'test' : len(test_set)}

        return dataloaders, dataset_sizes


def _valid_sampler(datasets, valid_size):
    random.seed(0)

    valid_size = int(valid_size)
    valid_list = random.sample(range(0, image_num[datasets]), valid_size)

    train_list = [x for x in range(image_num[datasets])]
    train_list = list(set(train_list)-set(valid_list))

    return train_list, valid_list
