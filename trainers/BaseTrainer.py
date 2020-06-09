import torch
import os, copy, time
import json
#import wandb
from .utils import directory_setter

class BaseTrainer():
    """
    A base class to control training process
    """
    def __init__(self, model, dataloaders, dataset_sizes, criterion,
                optimizer, scheduler, device, use_wandb=False):
        self.model = model.to(device)
        self.dataloaders = dataloaders
        self.dataset_sizes = dataset_sizes
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.results = {}
        self.measure_name = 'measure'
        self.valid_type = 'max_measure'
        self.use_wandb = use_wandb
        self.num_epochs = None
    
    def train(self, num_epochs):
        self.num_epochs = num_epochs
        
        # save initial weights
        best_model_wts = copy.deepcopy(self.model.state_dict())
        
        # get base criterion to select best model
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
            
            # run train epoch
            train_loss, train_measure = self.epoch_phase('train', epoch)
            
            # run validation epoch
            valid_loss, valid_measure = self.epoch_phase('valid', epoch)
            
            epoch_elapse = round(time.time() - epoch_start, 3)
            
            # print statistics of current epoch
            self._print_stat(epoch, num_epochs, 
                                  epoch_elapse,
                                  train_loss, 
                                  train_measure, 
                                  valid_loss, 
                                  valid_measure)
            
            # update best model weights
            best_criterion, best_model_wts = self._get_best_valid(best_model_wts, 
                                                                  best_criterion, 
                                                                  valid_measure,
                                                                  valid_loss)
            # result of current epoch
            result_dict = {
                'Train_Loss': train_loss,
                'Train_Accuracy': train_measure,
                'Valid_Loss': valid_loss,
                'Valid_Accuracy': valid_measure,
                'epoch': epoch}
            
            # update epoch results to the train result
            self._result_logger(epoch, result_dict)
            
            if self.use_wandb:
                self._wandb_updater(epoch)
            
    def test(self, save_path):
        # run test epoch
        test_loss, test_measure = self.epoch_phase('test')
        print(('[{}] Loss - {:.4f}, Acc - '.format('Test', test_loss)) \
              + '{:2.2f}%'.format(test_measure))
        

        result_dict = {
            'Test_Loss': test_loss,
            'Test_Accuracy': test_measure,
            'epoch': self.num_epochs}
        
        self._result_logger(epoch, result_dict)
        
        self._result_saver(save_path)
        self._model_saver(save_path)
        
        if self.use_wandb:
            self._wandb_updater(self.num_epochs)
                
    def epoch_phase(self, phase, epoch=None):
        # set model train/eval phase
        self.model.train() if phase == 'train' else self.model.eval()
        
        # define loss and custom measure
        running_loss, running_measure = 0.0, 0.0
        
        for inputs, labels in self.dataloaders[phase]:
            # remove gradients of weights
            self.optimizer.zero_grad() 
            
            # calculate gradients only in train phase
            with torch.set_grad_enabled(phase=='train'): 
                # run a single step
                loss, measure = self._step(inputs, labels)

                # backward pass & update parameters
                if phase == 'train':
                    loss.backward()
                    self.optimizer.step()
                
                # update results from steps
                running_loss += loss.item()
                running_measure += measure.item()
        
        epoch_loss = running_loss / self.dataset_sizes[phase]
        epoch_measure = running_measure / self.dataset_sizes[phase]
        
        # update learning rate if in train phase
        if phase == 'train':
            self.scheduler.step()
        
        return round(epoch_loss, 4), round(epoch_measure, 4)

    def _get_best_valid(self, 
                        best_model_wts, 
                        best_criterion, 
                        valid_measure, 
                        valid_loss):
        
        if self.valid_type == 'max_measure':
            if valid_measure > best_criterion:
                best_criterion = valid_measure
                best_model_wts = copy.deepcopy(self.model.state_dict())
            
        elif self.valid_type == 'min_measure':
            if valild_measure < best_criterion:
                best_criterion = valid_measure
                best_model_wts = copy.deepcopy(self.model.state_dict())
                
        elif self.valid_type == 'loss':
            if valid_loss < best_criterion:
                best_criterion = valid_loss
                best_model_wts = copy.deepcopy(self.model.state_dict())        
        
        return best_criterion, best_model_wts

            
    def _step(self, inputs, labels):
        """To be implemented"""
        raise NotImplementedError
    
    
    def _inference(self, inputs):
        """To be implemented"""
        raise NotImplementedError
    
    def _print_stat(self, epoch, num_epochs, epoch_elapse, train_loss, 
                    train_measure, valid_loss, valid_measure):
        
        for param_group in self.optimizer.param_groups:
            lr = param_group['lr']
            
        print('[Epoch {}/{}] Elapsed {}s/it'.format(epoch, num_epochs, epoch_elapse))
        print(('[{}] Loss - {:.4f}, {} - '.format('Train', train_loss, self.measure_name)) \
              + '{:2.2f} '.format(train_measure) \
              + 'Learning Rate - {:0.6f}'.format(lr))

        print(('[{}] Loss - {:.4f}, {} - '.format('Valid', valid_loss, self.measure_name)) \
              + ('{:2.2f} '.format(valid_measure)))
    
        print('=' * 50)        

    def _result_logger(self, epoch, result_dict):
        if int(epoch) not in self.results.keys():
            self.results[int(epoch)] = result_dict
        else:
            self.results[int(epoch)] = {**self.results[int(epoch)], **result_dict} # merge two dicts
            
    def _result_saver(self, path):
        # if not exist, make directory
        directory_setter(path=path, make_dir=True)
        
        # save results as a json file
        info_path = os.path.join(path, 'result_logs.json')
        with open(path, 'w') as fp:
            json.dump(self.results, fp)
        
    def _model_saver(self, path):
        # if not exist, make directory
        directory_setter(path=path, make_dir=True)
        
        # save model
        torch.save(model.state_dict(), os.path.join(path, 'model.h5'))

    def _wandb_updater(self, epoch):
        # upload to wandb server
        wandb.log(self.results[int(epoch)], step=epoch)
