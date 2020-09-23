# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
from .tinyimagenet import *

import cv2

import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms

logger = getLogger()


def MultiCropDataset_tinyimg(self,
                            data_path,
                            size_crops,
                            nmb_crops,
                            min_scale_crops,
                            max_scale_crops,
                            size_dataset=-1,
                            return_index=False):
    
    assert len(size_crops) == len(nmb_crops)
    assert len(min_scale_crops) == len(nmb_crops)
    assert len(max_scale_crops) == len(nmb_crops)
    
    self.return_index = return_index

    trans = []
    color_transform = transforms.Compose([get_color_distortion(), RandomGaussianBlur()])
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    for i in range(len(size_crops)):
        randomresizedcrop = transforms.RandomResizedCrop(
            size_crops[i],
            scale=(min_scale_crops[i], max_scale_crops[i]),
        )
        trans.extend([transforms.Compose([
            randomresizedcrop,
            transforms.RandomHorizontalFlip(p=0.5),
            color_transform,
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
        ] * nmb_crops[i])
        
    self.trans = trans
    
    train_transform = CropsTransform(self.trans)
    
    train_dataset = TinyImageNet(data_path, train=True, download=False, transform=train_transform)
    
    return train_dataset
    
    
class CropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        multi_crops = list(map(lambda trans: trans(x), self.base_transform))
        
        return multi_crops
    
    
class RandomGaussianBlur(object):
    def __call__(self, img):
        do_it = np.random.rand() > 0.5
        if not do_it:
            return img
        sigma = np.random.rand() * 1.9 + 0.1
        return cv2.GaussianBlur(np.asarray(img), (23, 23), sigma)


def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort
