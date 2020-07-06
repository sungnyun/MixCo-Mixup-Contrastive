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

    def train(self, num_epochs, probe_freq=None, probe_setup=None):
        # save initial weights & get base criterion to select best model
        best_model_wts = copy.deepcopy(self.model.state_dict())
        
        print('=' * 50)
        device_name = torch.cuda.get_device_name(int(self.device[-1]))
        print('Train start on device: {}'.format(device_name))
        print('=' * 50, '\n')
        
        for epoch in range(1, num_epochs+1):
            epoch_start = time.time()
            train_loss, train_measure = self.epoch_phase('train', epoch)
            epoch_elapse = round(time.time() - epoch_start, 3)
            
            self._print_stat(epoch, num_epochs, 
                                  epoch_elapse,
                                  train_loss, 
                                  train_measure)
            
            best_model_wts = self._get_best_valid()
            result_dict = {
                'Train_Loss': train_loss,
                'epoch': epoch}
            
            self._result_logger(epoch, result_dict)
            
            if (probe_freq is not None) and (epoch % probe_freq == 0):
                prober = self._set_prober(probe_setup)
                t_prob_loss, t_prob_acc1 = result = prober.train()
                result_dict = {
                    'Train_ProbLoss': t_prob_loss,
                    'Train_ProbAccuracy1': t_prob_acc1,
                    'epoch': epoch}
                
                self._result_logger(epoch, result_dict)

    def _step(self, inputs, labels):
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

    
    
def 