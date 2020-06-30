# Refactored from https://github.com/sthalles/SimCLR

import torch
import torch.nn as nn
import torch.nn.functional as F
from .Architectures.ResNet import *
import torchvision.models as models

__all__ = ['SimCLR']

class SimCLR(nn.Module):
    def __init__(self, base_model, out_dim=128, from_small=False):
        super(SimCLR, self).__init__()
        self.encoder_dict = {
            "resnet18": models.resnet18(pretrained=False),
            "resnet50": models.resnet50(pretrained=False)}
            #"resnet18": resnet18(pretrained=False),
            #"resnet50": resnet50(pretrained=False)}


        encoder = self._get_basemodel(base_model)
        num_ftrs = encoder.fc.in_features

        self.features = nn.Sequential(*list(encoder.children())[:-1])

        # projection MLP
        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, out_dim)
        self.relu = nn.ReLU()
        
        self.rep_dim = num_ftrs
        self.out_dim = out_dim

    def _get_basemodel(self, model_name):
        try:
            model = self.encoder_dict[model_name]
            print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file.")

    def forward(self, x):
        h = self.features(x)
        h = h.squeeze()

        x = self.l1(h)
        x = self.relu(x)
        x = self.l2(x)
        return h, x
