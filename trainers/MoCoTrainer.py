import torch
import time, copy
from .BaseTrainer import BaseTrainer
from .LinearProber import LinearProber

class MoCoTrainer(BaseTrainer):
    """MoCo Trainer"""
    def __init__(self, model, dataloaders, dataset_sizes, criterion,
                optimizer, scheduler, device, use_wandb=False):
        super(MoCoTrainer, self).__init__(model, dataloaders, dataset_sizes, criterion, 
                                            optimizer, scheduler, device, use_wandb)

        self.measure_name = 'Accuracy'
        self.criterion = torch.nn.CrossEntropyLoss()
        self.prediction = lambda outputs : torch.max(outputs, 1)[1]


    def train(self, num_epochs):        
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
            
            result_dict = {
                'Train_Loss': train_loss,
                'epoch': epoch}
            
            self._result_logger(epoch, result_dict)
            """
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
            """
        best_model_wts = copy.deepcopy(self.model.state_dict())

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
    
    
    def _ntxentloss(self, zis, zjs):
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [zis, zjs]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [zis, self.model.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        # apply temperature
        logits /= self.model.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda().to(self.device)
        
        # dequeue and enqueue
        self.model._dequeue_and_enqueue(zjs)
        
        loss = self.criterion(logits, labels)
        #loss /= logits.shape[1]
        
        pred = self.prediction(logits)
        measure = torch.sum(pred == labels.data)

        return loss, measure * 100

        
    def _step(self, inputs, labels, epoch=None):
        # augmented samples i
        xis = inputs[0].to(self.device)
        
        # augmented samples j
        xjs = inputs[1].to(self.device)
            
        # get projections from xis, xjs (not representation!!!)
        zis, zjs = self._inference(xis, xjs)
        
        # calculate loss. note that (batchsize *2)
        loss, measure = self._ntxentloss(zis, zjs)
        
        return loss, measure
        
        
    def _inference(self, xis, xjs):            
        # get the representations and the projections
        _, [zis, zjs] = self.model(xis, xjs)

        return zis, zjs
    