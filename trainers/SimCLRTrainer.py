import torch
import time, copy
from .BaseTrainer import BaseTrainer
from .LinearProber import LinearProber

class SimCLRTrainer(BaseTrainer):
    """SimCLR Trainer"""
    def __init__(self, model, dataloaders, dataset_sizes, criterion,
                optimizer, scheduler, device, use_wandb=False):
        super(SimCLRTrainer, self).__init__(model, dataloaders, dataset_sizes, criterion, 
                                            optimizer, scheduler, device, use_wandb)

        self.measure_name = 'Loss'
        self.valid_type='loss'


    def train(self, num_epochs, probe_freq=None, probe_setup=None):
        best_model_wts = copy.deepcopy(self.model.state_dict())
        
        if 'max' in self.valid_type:
            best_criterion = 0.0
        else:
            best_criterion = 100000000
        
        print('=' * 50)
        device_name = torch.cuda.get_device_name(int(self.device[-1]))
        print('Train start on device: {}'.format(device_name))
        print('=' * 50, '\n')
        
        for epoch in range(1, num_epochs+1):
            epoch_start = time.time()
            train_loss, train_measure = self.epoch_phase('train', epoch)
            valid_loss, valid_measure = self.epoch_phase('valid', epoch)
            epoch_elapse = round(time.time() - epoch_start, 3)
            
            self._print_stat(epoch, num_epochs, 
                                  epoch_elapse,
                                  train_loss, 
                                  train_measure, 
                                  valid_loss, 
                                  valid_measure)
            
            best_criterion, best_model_wts = self._get_best_valid(best_model_wts, 
                                                                  best_criterion, 
                                                                  valid_measure,
                                                                  valid_loss)
            result_dict = {
                'Train_Loss': train_loss,
                'Valid_Loss': valid_loss,
                'epoch': epoch}
            
            self._result_logger(epoch, result_dict)
            
            if (probe_freq is not None) and (epoch % probe_freq == 0):
                prober = self._set_prober(probe_setup)
                t_prob_loss, t_prob_acc1, v_prob_loss, v_prob_acc1, v_prob_acc5 = result = prober.train()
                result_dict = {
                    'Train_ProbLoss': t_prob_loss,
                    'Train_ProbAccuracy1': t_prob_acc1,
                    'Valid_ProbLoss': v_prob_loss,
                    'Valid_ProbAccuracy1': v_prob_acc1,
                    'Valid_ProbAccuracy5': v_prob_acc5,
                    'epoch': epoch}
                
                self._result_logger(epoch, result_dict)
                

    def test(self, save_path, do_probe=False, probe_setup=None):
        test_loss, test_measure = self.epoch_phase('test')
        print(('[{}] Loss - {:.4f}, Measure - '.format('Test', test_loss)) \
              + '{:2.2f}%'.format(test_measure))
        
        # linear probing (linear classification test).
        if do_probe:
            prober = self._set_prober(probe_setup)
            t_prob_loss, t_prob_acc1, v_prob_loss, v_prob_acc1, v_prob_acc5 = prober.train()
            test_prob_loss, test_prob_acc1, test_prob_acc5 = prober.test()
        
        result_dict = {
            'Test_Loss': test_loss,
            'Test_ProbLoss': test_prob_loss,
            'Test_ProbAccuracy1': test_prob_acc1,
            'Test_ProbAccuracy5': test_prob_acc5,
            'epcoh': 0}
        
        self._result_logger(epoch, result_dict)
        self._result_saver(save_path)
        self._model_saver(save_path)
        

    def _set_prober(self, prob_setup):
        # make & train linear classifier
        prober = LinearProber(encoder=self.model,
                              dataloaders=prob_setup['dataloaders'], 
                              dataset_sizes=prob_setup['dataset_sizes'],
                              device=self.device,
                              num_classes=prob_setup['num_classes'],
                              rep_channel_dim=self.model.rep_dim)
        self.prober = prober
        return prober
    
        
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
    