import os
import numpy as np

import torch


def set_device(args):
    if isinstance(args, int):
        device = str(args)
    elif args.device is not None:
        device = str(args.device[0])
        for i in range(len(args.device) - 1):
            device += (',' + str(args.device[i+1]))

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    print(torch.device('cuda'))
    return torch.device('cuda')


def sample_weights(labels):
    """ Calculates per sample weights. """
    class_sample_count = np.unique(labels, return_counts=True)[1]
    class_weights = 1. / torch.Tensor(class_sample_count)
    return class_weights[list(map(int, labels))]


def random_split_image_folder(data, labels, n_classes, n_samples_per_class):
    """ Creates a class-balanced validation set from a training set.
        Specifically for the image folder class
    """

    train_x, train_y, valid_x, valid_y = [], [], [], []

    if isinstance(labels, list):
        labels = np.array(labels)

    for i in range(n_classes):
        # get indices of all class 'c' samples
        c_idx = (np.array(labels) == i).nonzero()[0]
        # get n unique class 'c' samples
        valid_samples = np.random.choice(c_idx, n_samples_per_class[i], replace=False)
        # get remaining samples of class 'c'
        train_samples = np.setdiff1d(c_idx, valid_samples)
        # assign class c samples to validation, and remaining to training
        train_x.extend(data[train_samples])
        train_y.extend(labels[train_samples])
        valid_x.extend(data[valid_samples])
        valid_y.extend(labels[valid_samples])

    # torch.from_numpy(np.stack(labels)) this takes the list of class ids and turns them to tensor.long

    return {'train': train_x, 'valid': valid_x}, \
        {'train': torch.from_numpy(np.stack(train_y)), 'valid': torch.from_numpy(np.stack(valid_y))}

def random_split(data, labels, n_classes, n_samples_per_class):
    """ Creates a class-balanced validation set from a training set.
    """

    train_x, train_y, valid_x, valid_y = [], [], [], []

    if isinstance(labels, list):
        labels = np.array(labels)

    for i in range(n_classes):
        # get indices of all class 'c' samples
        c_idx = (np.array(labels) == i).nonzero()[0]
        # get n unique class 'c' samples
        valid_samples = np.random.choice(c_idx, n_samples_per_class[i], replace=False)
        # get remaining samples of class 'c'
        train_samples = np.setdiff1d(c_idx, valid_samples)
        # assign class c samples to validation, and remaining to training
        train_x.extend(data[train_samples])
        train_y.extend(labels[train_samples])
        valid_x.extend(data[valid_samples])
        valid_y.extend(labels[valid_samples])

    if isinstance(data, torch.Tensor):
        # torch.stack transforms list of tensors to tensor
        return {'train': torch.stack(train_x), 'valid': torch.stack(valid_x)}, \
            {'train': torch.stack(train_y), 'valid': torch.stack(valid_y)}
    # transforms list of np arrays to tensor
    return {'train': torch.from_numpy(np.stack(train_x)),
            'valid': torch.from_numpy(np.stack(valid_x))}, \
        {'train': torch.from_numpy(np.stack(train_y)),
         'valid': torch.from_numpy(np.stack(valid_y))}
