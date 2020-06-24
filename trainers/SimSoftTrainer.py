import torch
import torch.nn as nn
import time, copy
from .SimCLRTrainer import SimCLRTrainer
from .LinearProber import LinearProber

class SimSoftTrainer(SimCLRTrainer):
    """SimSoft Trainer"""
    def __init__(self, model, dataloaders, dataset_sizes, criterion,
                optimizer, scheduler, device, use_wandb=False):
        super(SimSoftrainer, self).__init__(model, dataloaders, dataset_sizes, criterion, 
                                            optimizer, scheduler, device, use_wandb)
        self.measure_name = 'Loss'


    def _step(self, inputs, labels, epoch=None):
        # augmented samples i
        xis = inputs[0].to(self.device)
        
        # augmented samples j
        xjs = inputs[1].to(self.device)
            
        # get projections from xis, xjs (not representation!!!)
        zis, zjs = self._inference(xis, xjs)
        
        # calculate loss. note that (batchsize *2)
        loss = self.criterion(zis, zjs) * (inputs[0].size(0) * 2)
        measure = loss
        
        return loss, measure
        
        
    def _inference(self, xis, xjs):            
        # get the representations and the projections
        ris, zis = self.model(xis)  # [N,C]

        # get the representations and the projections
        rjs, zjs = self.model(xjs)  # [N,C]

        return zis, zjs
    
    
        

