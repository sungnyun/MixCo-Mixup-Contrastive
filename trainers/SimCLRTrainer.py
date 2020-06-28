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
        self.measure_name = 'Accuracy'


    def train(self, num_epochs, phase='train'):
        # save initial weights & get base criterion to select best model
        best_model_wts = copy.deepcopy(self.model.state_dict())
        print('=' * 50)
        # device_name = torch.cuda.get_device_name(int(self.device[-1]))
        # print('Train start on device: {}'.format(device_name))
        # print('=' * 50, '\n')
        
        for epoch in range(1, num_epochs+1):
            epoch_start = time.time()
            train_loss, train_measure = self.epoch_phase(phase, epoch)
            epoch_elapse = round(time.time() - epoch_start, 3)
            
            self._print_stat(epoch, num_epochs, 
                                  epoch_elapse,
                                  train_loss, 
                                  train_measure)
            
            # best_model_wts = self._get_best_valid()
            result_dict = {
                'Train_Loss': train_loss,
                'epoch': epoch}
            
            self._result_logger(epoch, result_dict)
            """
            if (probe_freq is not None) and (epoch % probe_freq == 0):
                prober = self._set_prober(probe_setup)
                t_prob_loss, t_prob_acc1 = result = prober.train()
                result_dict = {
                    'Train_ProbLoss': t_prob_loss,
                    'Train_ProbAccuracy1': t_prob_acc1,
                    'epoch': epoch}
                
                self._result_logger(epoch, result_dict)
            """
        best_model_wts = copy.deepcopy(self.model.state_dict())
        self._result_saver(save_path, phase, best_model_wts)

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
        
        #self._result_logger(result_dict)
        #self._result_saver(save_path)
        #self._model_saver(save_path)
        

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
    
        
    def _step(self, inputs, labels, epoch=None):
        # augmented samples i
        xis = inputs[0].to(self.device)
        
        # augmented samples j
        xjs = inputs[1].to(self.device)
            
        # get projections from xis, xjs (not representation!!!)
        zis, zjs = self._inference(xis, xjs)
        
        # calculate loss. note that (batchsize *2)
        loss, measure = self.criterion(zis, zjs)
        
        return loss, measure * 100
        
        
    def _inference(self, xis, xjs):            
        # get the representations and the projections
        import ipdb; ipdb.set_trace(context=15)
        ris, zis = self.model(xis)  # [N,C]

        # get the representations and the projections
        rjs, zjs = self.model(xjs)  # [N,C]
        
        return zis, zjs
    
