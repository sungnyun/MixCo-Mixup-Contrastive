import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder
from utils.utils import sample_weights, random_split_image_folder


def imagenet_dataloader(dataset_paths, transforms, batch_size, pin_memory, num_workers):
    datasets = {i: ImageFolder(root=dataset_paths[i]) for i in ['train', 'test']}
    #f_s_weights = sample_weights(datasets['train'].targets)
    data, labels = random_split_image_folder(data=np.asarray(datasets['train'].samples),
                                             labels=datasets['train'].targets,
                                             n_classes = 1000,
                                             n_samples_per_class = np.repeat(50, 1000).reshape(-1))

    datasets['pretrain'] = CustomDataset(data=np.asarray(datasets['train'].samples),
                                         labels=torch.from_numpy(np.stack(datasets['train'].targets)),
                                         transform=transforms['pretrain'], pretrain=True)
    datasets['train'] = CustomDataset(data=np.asarray(data['train']),
                                      labels=labels['train'],
                                      transform=transforms['train'], pretrain=False)
    datasets['valid'] = CustomDataset(data=np.asarray(data['valid']),
                                      labels=labels['valid'],
                                      transform=transforms['valid'], pretrain=False)
    datasets['test'] = CustomDataset(data=np.asarray(datasets['test'].samples),
                                     labels=torch.from_numpy(np.stack(datasets['test'].targets)),
                                     transform=transforms['test'], pretrain=False)
    
    s_weights = sample_weights(datasets['pretrain'].labels)
    config = {'pretrain': WeightedRandomSampler(s_weights, num_samples=len(s_weights), replacement=True),
              'train': WeightedRandomSampler(s_weights, num_samples=len(s_weights), replacement=True),
              'test': None, 'valid': None}
    
    dataloaders = {i: DataLoader(datasets[i], sampler=config[i], 
                                 batch_size=batch_size, pin_memory=pin_memory, 
                                 num_workers=num_workers) for i in config.keys()}
    dataset_sizes = {i: len(dataloaders[i]) for i in config.keys()}
    
    return dataloaders, dataset_sizes


class CustomDataset(Dataset):
    def __init__(self, data, labels, transform, pretrain):
        # shuffle dataset
        idx = np.random.permutation(data.shape[0])
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        
        self.data = data[idx]
        self.labels = labels
        self.transform = transform
        self.pretrain = pretrain

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        image = Image.open(self.data[index][0]).convert('RGB')
        if self.pretrain:
            img1, img2 = self.transform(image)
            img = torch.cat([img, img2], dim=0)
        else:
            img = self.transform(image)

        return img, self.labels[index].long()

