# Refactored from https://github.com/sthalles/SimCLR
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import random
from PIL import ImageFilter

np.random.seed(0)

__all__ = ['RepLearnTransform', 'simclr_transform', 'moco_transform']


class GaussianBlur_simclr():
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample
    

class GaussianBlur_moco(object):
    # Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

    
class RepLearnTransform():
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        xi = self.transform(sample)
        xj = self.transform(sample)
        return xi, xj

    
def simclr_transform(mean, std, img_size=32, s=1):
    # get a set of data augmentation transformations as described in the SimCLR paper.
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=img_size),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomApply([color_jitter], p=0.8),
                                          transforms.RandomGrayscale(p=0.2),
                                          GaussianBlur_simclr(kernel_size=int(0.1 * img_size)+1),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=mean, std=std)])
    return data_transforms


def moco_transform(mean, std, img_size=32):
    # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
    data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=img_size, scale=(0.2, 1.)),
                                          transforms.RandomApply([color_jitter], p=0.8),
                                          transforms.RandomGrayscale(p=0.2),
                                          transforms.RandomApply([GaussianBlur_moco([.1, 2.])], p=0.5),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=mean, std=std)])
    return data_transforms